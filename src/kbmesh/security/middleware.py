"""HTTP 侧的身份入口:中间件验签一次,下游只拿 `TenantContext`。

防线② 的落点。为什么放在中间件而不是每个路由的 `Depends`:

- 一条请求只验一次签,审计只写一次;
- SSE 是长连接,`yield` 型依赖的退出发生在流结束后,ContextVar 的 set/reset 会跨任务错位;
  中间件包住的整条请求与处理任务是同一个 context,`use_context` 才成立。

公共同步点:`authenticate()` 是纯函数式的入口,中间件与测试都调它 —— 越权留痕这件事必须能被
单测覆盖到,而不是只能靠起整个服务。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from kbmesh.core.tenancy import TenantContext, use_context
from kbmesh.db.base import get_session_factory
from kbmesh.db.repositories import record_security_event
from kbmesh.security.tokens import TokenError, context_from_token

#: 不带租户身份的入口路径(健康检查与登录)。上线门禁:新增公开路径必须在这里登记。
PUBLIC_PATHS: frozenset[str] = frozenset({"/healthz", "/readyz", "/api/auth/login", "/api/auth/token"})

BEARER_PREFIX = "Bearer "


async def authenticate(authorization: str | None, *, path: str = "") -> TenantContext:
    """`Authorization: Bearer …` → `TenantContext`;任何失败都留痕后抛 401。

    留痕先于异常:防线④要的是「越权尝试有记录」,不是「客户端拿到什么状态码」。
    """
    if authorization is None or not authorization.strip():
        await _audit(None, "unauthenticated", path)
        raise HTTPException(status_code=401, detail="missing token")
    if not authorization.startswith(BEARER_PREFIX):
        await _audit(None, "bad_token", path, reason="not_bearer")
        raise HTTPException(status_code=401, detail="bearer token required")

    try:
        return context_from_token(authorization.removeprefix(BEARER_PREFIX).strip())
    except TokenError as exc:
        await _audit(None, "bad_token", path, reason=str(exc))
        raise HTTPException(status_code=401, detail="invalid token") from exc


async def _audit(actor: TenantContext | None, event: str, path: str, **detail) -> None:
    factory = get_session_factory()
    async with factory() as session:
        await record_security_event(session, actor=actor, event=event, target="", path=path, detail=detail)
        await session.commit()


class TenantContextMiddleware(BaseHTTPMiddleware):
    """把 `ctx` 同时放进 `request.state` 与 ContextVar,后者供仓储之外的门面读取。"""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        try:
            ctx = await authenticate(request.headers.get("authorization"), path=request.url.path)
        except HTTPException as exc:
            # 自己转响应,不指望异常处理器:裸 ASGI 栈(测试、网关直连)没有那层包装。
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

        request.state.ctx = ctx
        # dispatch 与处理任务是同一条协程链,同步 with 就够;yield 型依赖做不到这点
        with use_context(ctx):
            return await call_next(request)


def get_tenant_context(request: Request) -> TenantContext:
    """技术文档 §6.1 里那个 FastAPI 依赖:值只能来自中间件验签后的结果。

    路由层拿不到第二个来源 —— 想伪造身份得先把中间件换掉,而那不是加参数能做到的。
    """
    ctx = getattr(request.state, "ctx", None)
    if ctx is None:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return ctx
