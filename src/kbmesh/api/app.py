"""FastAPI 应用工厂(#7)。

lifespan 只管两件事:`SaverHandle`(检查点池)开与关、基础 `GraphDeps` 组装。启动期没有
租户上下文,`sweep_stale` 不能在这里跑 —— 扫描点收敛在 `/api/maintenance/expire-stale` 与
`resume` 的懒判(见 `agents/runtime` 的注释)。

注入点:`create_app(deps=…, saver=…)` 供测试换掉模型网关与真引擎;生产路径两者缺省,
全部从 Settings 起。
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text as sa_text

from kbmesh.agents.nodes import GraphDeps
from kbmesh.agents.runtime import CKPT_TABLES, SaverHandle
from kbmesh.api.body_limit import MaxBodySizeMiddleware
from kbmesh.api.routers import admin, auth, chat, documents, evals, feedback, kbs, traces
from kbmesh.core.config import get_settings
from kbmesh.core.errors import KbmError
from kbmesh.db.base import get_session_factory
from kbmesh.security.middleware import TenantContextMiddleware


async def _kbm_error_handler(request: Request, exc: KbmError) -> JSONResponse:
    del request
    return JSONResponse({"detail": str(exc)}, status_code=exc.http_status)


def create_app(*, deps: GraphDeps | None = None, saver: SaverHandle | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        handle = saver or await SaverHandle.open_sync(settings=settings).open()
        app.state.saver_handle = handle
        if deps is not None:
            app.state.graph_deps = deps
        else:
            from kbmesh.agents.wiring import build_graph_deps, close_graph_deps
            from kbmesh.eval.tasks import enqueue_online_eval

            app.state.graph_deps = build_graph_deps(
                settings, online_eval=enqueue_online_eval
            )
        try:
            yield
        finally:
            if saver is None:  # 测试注入的 handle 归测试管,应用只关自己开的
                await handle.close()
            if deps is None:
                close_graph_deps(app.state.graph_deps)

    app = FastAPI(title="kb-mesh", version="0.1.0", lifespan=lifespan)
    # 后加的外层先执行:体积上限必须在 multipart 解析与身份中间件之前。
    app.add_middleware(TenantContextMiddleware)
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_document_bytes)
    app.add_exception_handler(KbmError, _kbm_error_handler)
    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(kbs.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(feedback.router)
    app.include_router(evals.router)
    app.include_router(traces.router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        """就绪 = 业务库可达 + 检查点四表齐全(§12 上线顺序的最后两段)。"""
        components: dict[str, str] = {}
        try:
            async with get_session_factory()() as session:
                await session.execute(sa_text("select 1"))
            components["db"] = "ok"
        except Exception:  # noqa: BLE001 - 就绪探针必须把一切失败折叠成 fail,不能 500
            components["db"] = "fail"
        try:
            tables = await request.app.state.saver_handle.tables()
            have = set(tables.get(settings.checkpoint_schema, []))
            components["checkpoint"] = "ok" if set(CKPT_TABLES) <= have else "fail"
        except Exception:  # noqa: BLE001
            components["checkpoint"] = "fail"
        ready = all(value == "ok" for value in components.values())
        return JSONResponse(
            {"ok": ready, "components": components}, status_code=200 if ready else 503
        )

    return app
