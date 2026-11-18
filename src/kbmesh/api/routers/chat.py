"""问答两条腿(SSE)+ 会话列表 + TTL 维护(#7)。

SSE 的开始语义:流开始**之前**抛的异常走 HTTP 状态码(403/409/422),已经开始之后出的错
收敛为 `error` 帧 —— 前者能进重试逻辑,后者只能进 UI。分界线就是路由层预检取第一帧
(`_prime`):第一帧落在图跑完 `route_intent` 之后,所以到这里还没炸的请求都值得开流。
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from kbmesh.agents.runtime import StreamEvent, sweep_stale
from kbmesh.api import sse
from kbmesh.api.deps import build_runtime, get_ctx, get_db
from kbmesh.api.schemas import AskIn, ResumeIn
from kbmesh.core.errors import PermissionDenied
from kbmesh.core.tenancy import ROLE_PLATFORM_ADMIN, ROLE_TENANT_ADMIN, TenantContext
from kbmesh.db.repositories import TraceRepo

router = APIRouter(prefix="/api", tags=["chat"])


async def _prime(agen: AsyncIterator[StreamEvent]) -> StreamEvent | None:
    """取第一帧;此时抛的异常还没开流,能带状态码出去,不会变成 200 + error 帧。"""
    try:
        return await agen.__anext__()
    except StopAsyncIteration:
        return None
    except BaseException:
        await agen.aclose()
        raise


def _respond(agen: AsyncIterator[StreamEvent], first: StreamEvent | None) -> EventSourceResponse:
    return EventSourceResponse(sse.pump(agen, first))


async def _deny_scoped_kb(
    request: Request, ctx: TenantContext, kb_id: str | None, path: str
) -> None:
    """kb_id 给了就必须在令牌的 kb_scope 里;越界的拒绝照惯例先留痕再抛。"""
    if kb_id is None or ctx.allows_kb(kb_id):
        return
    trace = request.app.state.graph_deps.trace
    if trace is not None:
        await trace.security_event("kb_scope_denied", f"kb:{kb_id}", path=path)
    raise PermissionDenied(f"kb_scope_denied: kb:{kb_id}")


@router.post("/ask")
async def ask(
    body: AskIn,
    request: Request,
    ctx: TenantContext = Depends(get_ctx),
) -> EventSourceResponse:
    await _deny_scoped_kb(request, ctx, body.kb_id, path="/api/ask")
    runtime = build_runtime(request, kb_id=body.kb_id)
    agen = runtime.stream_ask(body.query, body.thread_id)
    return _respond(agen, await _prime(agen))


@router.post("/threads/{thread_id}/resume")
async def resume(
    thread_id: str,
    body: ResumeIn,
    request: Request,
    ctx: TenantContext = Depends(get_ctx),
) -> EventSourceResponse:
    await _deny_scoped_kb(request, ctx, body.kb_id, path=f"/api/threads/{thread_id}/resume")
    runtime = build_runtime(request, kb_id=body.kb_id)
    agen = runtime.stream_resume(thread_id, body.model_dump())
    return _respond(agen, await _prime(agen))


@router.get("/threads")
async def list_threads(
    limit: int = 20,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """会话列表:一次问答的对外可见落点(`KINDS`),`thread_id` 返回裸 id 不回租户前缀。"""
    rows = await TraceRepo(ctx, db).list_runs(max(1, min(limit, 100)))
    out: list[dict[str, Any]] = []
    for row in rows:
        _, sep, bare = str(row.thread_id).partition(":")
        out.append(
            {
                "run_id": row.id,
                "thread_id": bare if sep else row.thread_id,
                "query": row.query,
                "final_status": row.final_status,
                "degraded": row.degraded,
                "latency_ms": row.latency_ms,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )
    return out


@router.post("/maintenance/expire-stale")
async def expire_stale(
    request: Request,
    ctx: TenantContext = Depends(get_ctx),
) -> dict[str, Any]:
    """把「停在 await_confirm 且已过期」的会话落 `expired` 并删检查点。

    启动期没有租户上下文,`sweep_stale` 不能在 lifespan 里跑 —— 扫描点收敛到这里(按租户、
    按需触发)加上 `resume` 里的懒判,共两处,没有常驻定时器。
    """
    if not ctx.has_role(ROLE_TENANT_ADMIN, ROLE_PLATFORM_ADMIN):
        trace = request.app.state.graph_deps.trace
        if trace is not None:
            await trace.security_event("role_denied", "maintenance:expire-stale", path="/api")
        raise PermissionDenied("role_denied: maintenance:expire-stale")
    runtime = build_runtime(request, kb_id=None)
    keys = await sweep_stale(runtime)
    return {"expired": keys}
