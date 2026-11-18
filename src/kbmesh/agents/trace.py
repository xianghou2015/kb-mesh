"""编排层的留痕(FR-701):一次问答 = 一条 `trace_run` + 若干条按 §11 stage 命名的 `trace_step`。

为什么不用请求里那条 `AsyncSession`,而是每个操作自开一条事务:一次问答会停在 `await_confirm`
上最长 30 分钟(FR-403「进程中断后可续跑」),HTTP 请求级事务早就归还连接了。更要命的是「崩在
中间」这一件事 —— 只有每一步都已真实提交,重启后才知道跑到哪、还剩哪几步没跑。代价是编排层的
写入**不归测试的 SAVEPOINT 兜底管**,验收里必须显式清库(见 `tests/test_graph_orchestration.py`)。

发号权在 State 的 `trace_seq`,不在这里:`interrupt()` 之前的节点体在 resume 时会重跑一遍(实测
`scripts/verify_checkpoint.py` C4c),所以写入必须按 `(run_id, seq)` 幂等 —— 那是
`TraceRepo.step_once` 的职责,这里不重复判断。
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kbmesh.agents.state import STAGES
from kbmesh.core.errors import IllegalTransition, PermissionDenied
from kbmesh.core.tenancy import TenantContext, current_context
from kbmesh.db.models import TraceRun, TraceStep
from kbmesh.db.repositories import TraceRepo, record_security_event


class TraceWriter:
    """`GraphDeps.trace` 的形状:一个方法一条事务,提交点在这里,不在调用方。

    `enabled=False` 只留给「跑图但不落库」的调试与单节点测试:发号与校验照常,写库跳过。生产上
    不许关 —— §10 把检查点持久化列为不可降级项,留痕和它是同一条链。
    """

    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession],
        *,
        ctx_provider: Callable[[], TenantContext] = current_context,
        enabled: bool = True,
    ) -> None:
        self.factory = factory
        self.ctx_provider = ctx_provider
        self.enabled = enabled

    def _ctx(self) -> TenantContext:
        # 上下文照取不误:租户缺失是 bug,连「不写库」这条路都不许把它绕过去
        return self.ctx_provider()

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator[AsyncSession]:
        """一条事务,**异常路径也要把审计留下**。

        `TraceRepo.deny()` 的约定是「先写 SecurityEvent 再抛 `PermissionDenied`」。用
        `session.begin()` 的话,那个异常会把刚写进去的审计一起回滚 —— §9 的「越权尝试必须留痕」
        就退化成一句注释:探测者什么痕迹都不留,而调用方只看到一个 403。
        """
        async with self.factory() as session:
            try:
                yield session
            except PermissionDenied:
                await session.commit()
                raise
            await session.commit()

    async def start(
        self,
        thread_key: str,
        query: str,
        *,
        route_flags: dict[str, Any] | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> str:
        if not self.enabled:
            return ""
        ctx = self._ctx()
        async with self._tx() as session:
            row = await TraceRepo(ctx, session).start_run(
                thread_key, query, route_flags=route_flags, config_snapshot=config_snapshot
            )
            return row.id

    async def step(
        self,
        run_id: str,
        seq: int,
        stage: str,
        *,
        payload: dict[str, Any] | None = None,
        latency_ms: int = 0,
        token_in: int = 0,
        token_out: int = 0,
        model: str = "",
    ) -> None:
        """一步一条,`stage` 必须落在 §11 白名单里。

        白名单在这里挡而不是在库里挡:stage 名是 FR-701 的对账口径,写错一个字母那条 step 就在
        报表里凭空消失 —— 消失的 step 比报错的 step 难查得多。
        """
        if stage not in STAGES:
            raise IllegalTransition(f"stage 不在 §11 白名单里: {stage!r}")
        if seq < 1:
            raise IllegalTransition(f"trace_step.seq 从 1 起,拿到 {seq}")
        if not self.enabled or not run_id:
            return
        ctx = self._ctx()
        async with self._tx() as session:
            await TraceRepo(ctx, session).step_once(
                run_id,
                seq,
                stage,
                payload=payload,
                latency_ms=latency_ms,
                token_in=token_in,
                token_out=token_out,
                model=model,
            )

    async def finish(
        self,
        run_id: str,
        *,
        final_status: str,
        degraded: bool = False,
        latency_ms: int = 0,
        token_in: int = 0,
        token_out: int = 0,
        cost_usd: float = 0,
    ) -> None:
        if not self.enabled or not run_id:
            return
        ctx = self._ctx()
        async with self._tx() as session:
            await TraceRepo(ctx, session).finish_run(
                run_id,
                final_status=final_status,
                degraded=degraded,
                latency_ms=latency_ms,
                token_in=token_in,
                token_out=token_out,
                cost_usd=cost_usd,
            )

    async def security_event(self, event: str, target: str, path: str = "", **detail: Any) -> None:
        if not self.enabled:
            return
        ctx = self._ctx()
        async with self._tx() as session:
            await record_security_event(
                session, actor=ctx, event=event, target=target, path=path, detail=detail
            )

    async def assert_thread(self, thread_key: str) -> None:
        ctx = self._ctx()
        if not self.enabled:
            return
        async with self._tx() as session:
            await TraceRepo(ctx, session).assert_thread(thread_key)

    async def get_run(self, run_id: str) -> TraceRun:
        ctx = self._ctx()
        async with self._tx() as session:
            return await TraceRepo(ctx, session).get_run(run_id)

    async def list_runs(self, limit: int = 20) -> list[TraceRun]:
        """本租户最近若干条 run —— TTL 扫描与 #7 的会话列表都靠它,不给跨租户的口子。"""
        ctx = self._ctx()
        async with self._tx() as session:
            return await TraceRepo(ctx, session).list_runs(limit)

    async def list_steps(self, run_id: str) -> list[TraceStep]:
        """按 seq 升序回读:恢复验收要靠它证明中断前后的 step 是同一份、没有空洞。

        `trace_step` 没有 `tenant_id` 列(归属靠 `run_id` 外键级联),所以租户谓词必须借
        `trace_run` 下推 —— 直接裸查 step 表就是跨租户读。
        """
        ctx = self._ctx()
        async with self._tx() as session:
            await TraceRepo(ctx, session).get_run(run_id)
            stmt = (
                select(TraceStep)
                .join(TraceRun, TraceStep.run_id == TraceRun.id)
                .where(TraceStep.run_id == run_id, TraceRun.tenant_id == ctx.tenant_id)
                .order_by(TraceStep.seq)
            )
            return list((await session.execute(stmt)).scalars())
