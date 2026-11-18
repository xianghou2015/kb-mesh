"""路由层的公共依赖:数据库会话、基础 deps、按请求组装的 Runtime。"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from kbmesh.agents.graph import compile_graph
from kbmesh.agents.nodes import GraphDeps
from kbmesh.agents.runtime import Runtime, SaverHandle
from kbmesh.core.errors import PermissionDenied
from kbmesh.db.base import get_session_factory
from kbmesh.security.middleware import get_tenant_context

__all__ = ["build_runtime", "get_base_deps", "get_ctx", "get_db", "get_saver"]

#: 名字短一点,语义同一份:ctx 的值只能来自中间件验签后的结果(§6.1 的那个依赖)。
get_ctx = get_tenant_context


async def get_db() -> AsyncIterator[AsyncSession]:
    """会话 + 提交语义。

    `PermissionDenied` 先提交再抛:审计行已经 flush,不能跟异常一起回滚 —— 与
    `TraceWriter._tx` 同一个约定,防线④要的是「越权有记录」,不是「客户端拿到 403」。
    """
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except PermissionDenied:
            await session.commit()
            raise


def get_base_deps(request: Request) -> GraphDeps:
    return request.app.state.graph_deps


def get_saver(request: Request) -> SaverHandle:
    return request.app.state.saver_handle


def build_runtime(request: Request, *, kb_id: str | None) -> Runtime:
    """一次问答一个 Runtime:`kb_id` 是检索过滤参数,逐请求取,不能烤死在启动期。

    图重编译一次的成本在毫秒级,而 `deps.kb_id` 是闭包进各节点工厂的 —— 换 kb 必须换图,
    这里不做过早的缓存优化。
    """
    base: GraphDeps = request.app.state.graph_deps
    deps = base if base.kb_id == kb_id else replace(base, kb_id=kb_id)
    graph = compile_graph(deps, checkpointer=request.app.state.saver_handle.saver)
    return Runtime(deps, graph, request.app.state.saver_handle)
