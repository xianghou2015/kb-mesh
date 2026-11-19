"""跨进程可恢复的问答运行时(§7.2 / FR-403 / NFR-1)。

这里只管四件事:**检查点连接池**、`thread_key` 的构造、`ask`/`resume` 两条腿、TTL 到期。
判定全在 `guards`,连边全在 `graph`,留痕全在 `trace` —— 混在一起就没法单独问「重启后到底
发生了什么」,而那正是本模块唯一的价值。

三条实测来的硬规矩(`scripts/verify_checkpoint.py`),它们决定了这个文件的形状:

1. **C1**:检查点表由 `saver.setup()` 自建、MIGRATIONS 里表名**不带 schema**,落在连接
   `search_path` 的第一位。复用 `db/base` 那条连接会把表建进 `platform`,与
   `checkpoint_schema="lg"` 对不上,还会让 `alembic check` 要求 DROP。所以这里自造池子,
   kwargs 里写死 `options="-c search_path=<schema>"`。
2. **C3/C5**:停在 `interrupt()` 时检查点**已经落盘**,SIGTERM 之后新进程读得回来 —— FR-403
   的验收靠的就是这个,不需要任何显式 flush。载荷在 `snap.interrupts` 与
   `snap.tasks[i].interrupts` **两处都可能出现**,所以读的时候两处都收。
3. **C6**:`adelete_thread(key)` 之后再发 `Command(resume)` **不报错,而是静默开一个没有历史的
   新 run**。所以到期必须在碰到图之前判完:先读 `trace_run` 定生死,再决定要不要动 saver。
   顺序反了,「一个已经过期的会话」就变成「一次全新问答」—— 用户看到的是答案,审计看到的是
   一条凭空多出来的 run。
"""
from __future__ import annotations

import contextlib
import re
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from kbmesh.agents import guards
from kbmesh.agents.nodes import GraphDeps, _plain
from kbmesh.agents.schemas import ResumeChoice
from kbmesh.agents.state import empty_state, next_seq, utc_now
from kbmesh.core.config import Settings, get_settings
from kbmesh.core.errors import IllegalTransition
from kbmesh.core.tenancy import TenantContext, current_context
from kbmesh.observability.pricing import PRICE_TABLE_VERSION, estimate_cost

#: 探针 C1 数的就是这四张;`setup()` 少建一张,恢复路径会在第一次 `aget_state` 才炸
CKPT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations")

#: 一次问答对外可见的落点。`await_confirm` 不是终态,其余全是;`replay` 是「这条已经答完了」。
KINDS = ("await_confirm", "answered", "refused", "error", "expired", "replay")

#: §7.2 的等待节点名 —— 快照里的 `next` 不是它就不许确认(否则是在往半截状态里注词)
CONFIRM_NODE = "confirm_interrupt"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: `thread_id` 的上限:列宽是 200(`trace_run.thread_id`),但存的是**带租户前缀的整键**,
#: 这里先按裸 id 挡一道过长的,超列宽那种在 `thread_key()` 里再挡第二道。
THREAD_ID_MAX = 96
THREAD_KEY_MAX = 191
#: 检查点池的最大连接数。`saver` 的一次读写只用一条连接,这个数按并发会话数量级给就够,
#: 探针 C2 实测 min=1/max=4 下 8 路并发 `aget_tuple` 无冲突。
POOL_MAX = 4


# --------------------------------------------------------------------- 检查点连接
def serde_for(settings: Settings) -> JsonPlusSerializer:
    """State 里的 `Hit`/`Citation` 是自定义 dataclass,必须显式进 msgpack 白名单。

    默认的 `JsonPlusSerializer()` 今天也能往返,但它只是**放行并 log 警告**,报错文案明说未来
    版本会改成直接拒绝 —— 靠现状活着的那一类,升级当天才会发现。白名单要写全「模块路径 + 类名」,
    少一条就是一条未注册告警。
    """
    del settings  # 目前没有按环境变化的字段;留参数是为了不让调用方猜「要不要传」
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("kbmesh.retrieval.contracts", "Hit"),
            ("kbmesh.retrieval.contracts", "Citation"),
        ]
    )


def checkpoint_dsn(settings: Settings) -> str:
    """把 asyncpg URL 换成 libpq 串,并丢掉 `?` 之后的查询串。

    剥掉 query 段是必须的:search_path 走 `options` 关键字,和 URL 参数撞在一起就变成两个互不
    相识的 search_path,谁赢取决于 psycopg 的解析顺序 —— 这种「看起来连上了其实连错库」最难查。
    """
    url = settings.db_url
    if not url:
        raise IllegalTransition("没有可用的数据库 URL:设 KBMESH_DB_URL")
    return url.replace("postgresql+asyncpg://", "postgresql://", 1).split("?", 1)[0]


def make_pool(dsn: str, settings: Settings) -> AsyncConnectionPool:
    """检查点专用的连接池 —— 不复用 `db/base` 那条,原因见模块 docstring 的 C1。

    三个 kwargs 每一个都在挡一件具体的事,不是样板:
    - `autocommit`:`setup()` 里有 `CREATE INDEX CONCURRENTLY`,包在事务里直接报错;
    - `prepare_threshold=0`:连接被池子轮换复用,语句缓存要关掉,否则 search_path 变了会拿到旧计划;
    - `row_factory=dict_row`:saver 按列名取值,取不到就是 KeyError。
    """
    return AsyncConnectionPool(
        dsn,
        min_size=1,
        max_size=POOL_MAX,
        open=False,
        name=f"kbmesh-ckpt-{settings.checkpoint_schema}",
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": f"-c search_path={settings.checkpoint_schema}",
        },
    )


@dataclass
class SaverHandle:
    """`AsyncPostgresSaver` 的生命周期:`open_sync()` → `await open()` → 用 → `await close()`。

    用法只有两种:进程级 `async with saver_handle() as handle: ...`,或把 `close()` 挂到 #7 的
    lifespan 上。#6 不发明第三种。
    """

    pool: AsyncConnectionPool
    saver: AsyncPostgresSaver
    schema: str
    dsn: str

    @classmethod
    def open_sync(cls, *, settings: Settings | None = None) -> SaverHandle:
        """建池但不连 —— 真正的 `wait=True` 交给 `await handle.open()`。"""
        settings = settings or get_settings()
        schema = settings.checkpoint_schema
        if not _IDENTIFIER.match(schema):
            raise IllegalTransition(f"checkpoint_schema 不是合法标识符: {schema!r}")
        dsn = checkpoint_dsn(settings)
        pool = make_pool(dsn, settings)
        return cls(pool, AsyncPostgresSaver(pool, serde=serde_for(settings)), schema, dsn)

    async def open(self) -> SaverHandle:
        await self.pool.open(wait=True)
        await self.setup()
        return self

    async def setup(self) -> None:
        """schema 必须先存在:saver 的建表语句不带 schema,落点全看连接的 search_path。

        拼进 `CREATE SCHEMA` 的标识符已经在 `open_sync` 按白名单校验过,拼不出注入。仍然**不写
        Alembic revision** —— 检查点表的生死归 `saver.setup()` 的 MIGRATIONS 自管,这不是我们的
        迁移漏了一步(见 `db/migrations/env.py` 的注释)。
        """
        async with self.pool.connection() as conn:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
        await self.saver.setup()

    async def close(self) -> None:
        await self.pool.close()

    async def tables(self) -> dict[str, list[str]]:
        """每张检查点表落在哪个 schema 下 —— C1 的守门口径,#7 的健康检查也读它。"""
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "select table_schema, table_name from information_schema.tables"
                    " where table_name = any(%s)",
                    (list(CKPT_TABLES),),
                )
            ).fetchall()
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(row["table_schema"], []).append(row["table_name"])
        return {key: sorted(value) for key, value in out.items()}


@asynccontextmanager
async def saver_handle(*, settings: Settings | None = None):
    """`async with saver_handle() as handle:` —— 出错也要把连接还给池子。"""
    handle = SaverHandle.open_sync(settings=settings)
    try:
        yield await handle.open()
    finally:
        await handle.close()


# --------------------------------------------------------------------- 线程键
def assert_thread_shape(thread_id: str) -> None:
    """裸 `thread_id` 的形状检查。冒号是**租户前缀的分隔符**,别人不能占。

    `tenant_id` 的字符集是 `[a-z0-9_-]`(`core/tenancy` 已经校验过,它自己不会带冒号),所以键里
    出现第二个冒号只能是调用方塞进来的 —— 那是跨租户读的入口,不是格式问题,直接拒。
    """
    if not thread_id or not thread_id.strip():
        raise IllegalTransition("thread_id 不能为空")
    if len(thread_id) > THREAD_ID_MAX:
        raise IllegalTransition(f"thread_id 超长(> {THREAD_ID_MAX}):{len(thread_id)}")
    if ":" in thread_id:
        raise IllegalTransition(f"thread_id 不得含冒号(它是租户前缀的分隔符): {thread_id!r}")


def thread_key(ctx: TenantContext, thread_id: str) -> str:
    """检查点与 `trace_run` 共用的物理键:**租户 id 作为不透明前缀写进键本身**。

    这是 NFR-1 的第一道闸,也是唯一一道不依赖应用层约定的:键写错不会「查到别人的」,因为前缀
    由验签后的 `ctx` 拼,不由调用方传。不设默认值、不做规范化,只做形状检查。
    """
    assert_thread_shape(thread_id)
    key = f"{ctx.tenant_id}:{thread_id}"
    if len(key) > THREAD_KEY_MAX:
        raise IllegalTransition(f"thread_key 超长(> {THREAD_KEY_MAX}):{len(key)}")
    return key


def split_thread_key(key: str) -> tuple[str, str]:
    """按**第一个**冒号切;`thread_id` 不可能含冒号,所以左半边一定完整。"""
    tenant, sep, rest = key.partition(":")
    if not sep or not tenant or not rest:
        raise IllegalTransition(f"thread_key 形状不合法: {key!r}")
    return tenant, rest


# --------------------------------------------------------------------- 结果
@dataclass(frozen=True)
class Outcome:
    """一次 `ask` / `resume` 的对外结论 —— #7 的响应体可以直接由它映射,不必再判一次状态。"""

    kind: str
    thread_key: str
    run_id: str
    state: Mapping[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] | None = None
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    final_status: str = ""
    reject_code: str = ""
    degraded: list[str] = field(default_factory=list)
    replayed: bool = False

    @property
    def awaiting(self) -> bool:
        return self.kind == "await_confirm"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "thread_key": self.thread_key,
            "run_id": self.run_id,
            "payload": self.payload,
            "answer": self.answer,
            "citations": self.citations,
            "final_status": self.final_status,
            "reject_code": self.reject_code,
            "degraded": self.degraded,
            "replayed": self.replayed,
        }


def interrupt_payload(result: Mapping[str, Any] | None) -> Any | None:
    """从 `ainvoke` 的返回值里取中断载荷(实测 C3b:键名是 `__interrupt__`)。"""
    if not result:
        return None
    stopped = result.get("__interrupt__") or []
    return getattr(stopped[0], "value", stopped[0]) if stopped else None


# --------------------------------------------------------------------- SSE 流(#7)
#: 节点更新 → SSE 中间事件的映射。终态三帧(token/citations/done)与 await_confirm **不**在
#: 这张表里:它们一律由 `_settle` 的 Outcome 统一映射,ask/resume/replay 共用同一条出口。
NODE_EVENTS: dict[str, str] = {
    "route_intent": "route",
    "rewrite_query": "rewrite_candidates",
    "rerank_fused": "retrieval_done",
}


@dataclass(frozen=True)
class StreamEvent:
    """SSE 的一帧。`event` 是九类事件名之一(§7.4 + #7),`data` 进 `data:` 字段的 JSON。"""

    event: str
    data: dict[str, Any]


def snapshot_interrupts(snap: Any) -> list[Any]:
    """`aget_state` 的载荷有**两个**落点,实测两处都要收(探针 C3 报的就是这个二选一)。"""
    out = list(getattr(snap, "interrupts", None) or [])
    for task in getattr(snap, "tasks", None) or []:
        out.extend(getattr(task, "interrupts", None) or [])
    return [getattr(one, "value", one) for one in out]


def snapshot_values(snap: Any) -> dict[str, Any]:
    return dict(getattr(snap, "values", None) or {})


def elapsed_minutes(pending_since: Any, now: datetime) -> float | None:
    """`pending_since` 是随 checkpoint 落盘的 ISO UTC 串 —— 到期判断不依赖 saver 的内部时钟。

    读不出来就返回 None:宁可「判不了就不判」,也不要因为时间格式换了就把全部待确认会话清光。
    """
    try:
        started = datetime.fromisoformat(str(pending_since))
    except (TypeError, ValueError):
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return (now - started).total_seconds() / 60.0


def _tokens(state: Mapping[str, Any]) -> tuple[int, int]:
    token_in = token_out = 0
    for one in state.get("usage") or []:
        data = one if isinstance(one, Mapping) else one.__dict__
        token_in += int(data.get("token_in") or 0)
        token_out += int(data.get("token_out") or 0)
    return token_in, token_out


# --------------------------------------------------------------------- 两条腿
class Runtime:
    """把 `deps` + 编译好的图 + 检查点句柄绑成一次问答的现场。

    **不持有租户身份**:每次调用现取 `ctx_provider()`(默认 `current_context()`),绝不从检查点
    里恢复身份 —— 状态是跨进程读回来的,身份不是。
    """

    def __init__(
        self,
        deps: GraphDeps,
        graph: CompiledStateGraph,
        handle: SaverHandle | None = None,
        *,
        ctx_provider: Callable[[], TenantContext] = current_context,
    ) -> None:
        self.deps = deps
        self.graph = graph
        self.handle = handle
        self.ctx_provider = ctx_provider

    async def ask(self, query: str, thread_id: str) -> Outcome:
        """发起一次问答:建 run → 走到 `confirm_interrupt` 停下 → 把候选交回调用方。

        停下时**不收尾** `trace_run`。注意「等确认」在库里没有专用记号:`final_status` 的
        `server_default` 是 `'error'`(#3 定的兜底:没收尾的 run 一律算失败),所以真判据是
        检查点快照的 `next`,不是这一列 —— `sweep_stale` 就是照这个写的。
        """
        key = thread_key(self._ctx(), thread_id)
        started = utc_now()
        if not str(query or "").strip():
            raise IllegalTransition("query 不能为空")
        run_id = await self._start_run(key, query)
        state = empty_state(
            query=query,
            thread_id=thread_id,
            thread_key=key,
            run_id=run_id,
            route_flags=self.deps.flags_snapshot(),
            config_hash=self.deps.config_hash(),
        )
        return await self._settle(
            await self.graph.ainvoke(state, self._cfg(key)), key=key, run_id=run_id, started=started
        )

    async def resume(self, thread_id: str, decision: ResumeChoice | Mapping[str, Any]) -> Outcome:
        """带着用户的选择回到中断处。

        顺序是定死的:形状检查 → `assert_thread` → 读快照 → **到期判断** → 确认决策预检 → 才动图。
        实测 `adelete_thread` 之后 `Command(resume)` 会静默开一个新 run(C6),所以过期必须在看图
        一眼之前判完;`assert_thread` 放在最前,是因为跨租户的探测也要留痕;决策预检放在动图之前,
        理由见 `_precheck`。
        """
        key = thread_key(self._ctx(), thread_id)
        if self.deps.trace is not None:
            await self.deps.trace.assert_thread(key)
        snap = await self.graph.aget_state(self._cfg(key))
        values = snapshot_values(snap)
        run_id = str(values.get("run_id") or "")
        if not values:
            raise IllegalTransition(f"没有这个会话,或它的检查点已被清: {key}")
        expired = await self._expire_if_stale(key, run_id, values)
        if expired is not None:
            return expired
        if not snap.next:
            return self._replay(key, run_id, values)
        if tuple(snap.next) != (CONFIRM_NODE,):
            raise IllegalTransition(f"该会话停在 {tuple(snap.next)},不是 await_confirm,无法确认")
        await self._precheck(run_id, values, decision)
        return await self._settle(
            await self.graph.ainvoke(Command(resume=decision), self._cfg(key)),
            key=key,
            run_id=run_id,
            started=utc_now(),
        )

    async def _precheck(
        self, run_id: str, values: Mapping[str, Any], decision: Any
    ) -> None:
        """非法确认必须在**碰到图之前**挡掉。

        实测(`test_degrade.py` 钉住):`Command(resume=坏值)` 一旦写进 checkpoint,之后每次
        resume 都会**重放那个坏值** —— 用户改成合法的也出不去,线程永久停在 `confirm_interrupt`。
        所以这里用同一个纯函数 `guards.validate_resume` 先判一遍:节点里那一遍照留(纵深防御,
        直接 `ainvoke` 的调用方还在),但正常入口进来的永远是合法值。

        被拒的确认也要留痕:#7 返回 409,面板要能数出「用户试了几次没对上」。发号仍走
        `next_seq(values)` —— 失败不推进 state 的 `trace_seq`,而 `step_once` 按 `(run_id, seq)`
        幂等,所以重复的坏确认不会堆出无穷多条 step。
        """
        try:
            guards.validate_resume(values, decision)
        except IllegalTransition as exc:
            if self.deps.trace is not None and run_id:
                await self.deps.emit(
                    run_id,
                    next_seq(values),  # type: ignore[arg-type]
                    "rewrite",
                    payload={"kind": "resume_rejected", "reason": str(exc)[:200]},
                )
            raise

    def _cfg(self, key: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": key}}

    def _ctx(self) -> TenantContext:
        return self.ctx_provider()

    async def _start_run(self, key: str, query: str) -> str:
        if self.deps.trace is None:
            return ""
        return await self.deps.trace.start(
            key,
            query,
            route_flags=self.deps.flags_snapshot(),
            config_snapshot={
                "config_hash": self.deps.config_hash(),
                "theta_rel": self.deps.theta_rel,
                "theta_answer": self.deps.theta_answer,
                "max_targeted_retry": self.deps.max_retries,
                "pending_ttl_min": self.deps.settings.pending_confirm_ttl_min,
                # 历史成本不失真:每条 run 记下算它时用的单价表版本。
                "price_table_version": PRICE_TABLE_VERSION,
            },
        )

    async def _settle(
        self,
        result: Mapping[str, Any] | None,
        *,
        key: str,
        run_id: str,
        started: datetime,
    ) -> Outcome:
        payload = interrupt_payload(result)
        state = {k: v for k, v in (result or {}).items() if k != "__interrupt__"}
        if payload is not None:
            return Outcome(
                kind="await_confirm",
                thread_key=key,
                run_id=run_id,
                state=state,
                payload=dict(payload),
            )
        final = str(state.get("final_status") or "")
        if final not in ("answered", "refused"):
            # 图没走到任何终态节点:空图、崩在最后一步之前、或状态被截断。error 是唯一诚实的
            # 落点 —— 写成 answered 会直接污染 FR-406 的「答了多少」口径。
            final = "error"
        outcome = Outcome(
            kind=final,
            thread_key=key,
            run_id=run_id,
            state=state,
            answer=str(state.get("answer") or ""),
            citations=list(state.get("citations") or []),
            final_status=final,
            reject_code=str(state.get("reject_code") or ""),
            degraded=[str(one) for one in (state.get("degraded") or [])],
        )
        token_in, token_out = _tokens(state)
        rerank_calls = 1 if self.deps.use_rerank and state.get("reranked") else 0
        cost_usd = estimate_cost(
            state.get("usage") or [],
            rerank_calls=rerank_calls,
            rerank_model=self.deps.settings.rerank_model,
        )
        if self.deps.trace is not None and run_id:
            await self.deps.trace.finish(
                run_id,
                final_status=final,
                degraded=bool(outcome.degraded),
                latency_ms=int((utc_now() - started).total_seconds() * 1000),
                token_in=token_in,
                token_out=token_out,
                cost_usd=cost_usd,
            )
        if (
            final in ("answered", "refused")
            and self.deps.online_eval is not None
            and self.deps.settings.flags.online_eval
            and run_id
        ):
            # 同步入队(Celery apply_async),模型调用在 worker 里;评估通道故障不阻断问答。
            payload = {
                "run_id": run_id,
                "tenant_id": self._ctx().tenant_id,
                "question": str(state.get("query") or state.get("confirmed_query") or ""),
                "answer": outcome.answer,
                "final_status": final,
                "reject_code": outcome.reject_code,
                "cite_ids": dict(state.get("cite_ids") or {}),
                "evidence_ids": [
                    str(hit.chunk_id) for hit in (state.get("hits") or [])
                ],
                "evidence_texts": [str(hit.text) for hit in (state.get("hits") or [])],
                "citations": [_plain(citation) for citation in outcome.citations],
                "usage": list(state.get("usage") or []),
            }
            # 观测/评估通道不得反向影响主链路
            with contextlib.suppress(Exception):
                self.deps.online_eval(payload)
        return outcome

    async def _expire_if_stale(
        self, key: str, run_id: str, values: Mapping[str, Any]
    ) -> Outcome | None:
        """超过 `pending_confirm_ttl_min` 的待确认会话:落 `expired` + 删检查点。

        删检查点是因为不删就永久占位 —— 这条会话已经不可能再被确认,留着只会让每次 `resume`
        都读回一个推不动的快照。**不删 trace 行**:拒答与到期在 FR-406 的口径里是两个数,都得
        查得到。到期也不是越权,所以不落 SecurityEvent。
        """
        minutes = elapsed_minutes(values.get("pending_since"), utc_now())
        if minutes is None or minutes <= self.deps.settings.pending_confirm_ttl_min:
            return None
        if self.deps.trace is not None and run_id:
            await self.deps.trace.finish(run_id, final_status="expired")
        if self.handle is not None:
            await self.handle.saver.adelete_thread(key)
        return Outcome(
            kind="expired",
            thread_key=key,
            run_id=run_id,
            state=dict(values),
            final_status="expired",
            reject_code=str(values.get("reject_code") or ""),
        )

    def _replay(self, key: str, run_id: str, values: Mapping[str, Any]) -> Outcome:
        """已经跑完的线程再 resume:回放既有答案,不重跑。

        实测「在完成过的线程上重复 `Command(resume)`」是**静默 no-op**,所以必须由 `snap.next`
        显式判;否则调用方拿到一个「成功」的空返回,以为又答了一次。
        """
        final = str(values.get("final_status") or "")
        if final not in ("answered", "refused", "error", "expired"):
            final = "error"
        return Outcome(
            kind="replay",
            thread_key=key,
            run_id=run_id,
            state=dict(values),
            answer=str(values.get("answer") or ""),
            citations=list(values.get("citations") or []),
            final_status=final,
            reject_code=str(values.get("reject_code") or ""),
            degraded=[str(one) for one in (values.get("degraded") or [])],
            replayed=True,
        )

    # ------------------------------------------------------------- 流式两条腿(#7)
    async def stream_ask(self, query: str, thread_id: str) -> AsyncIterator[StreamEvent]:
        """`ask` 的流式形态:同一套留痕与判定,只是把「跑完再给结论」换成「节点一落就推一帧」。

        校验与发号和 `ask` 完全同序(先 `thread_key` 形状,再 query,再 start_run)—— 两条腿的
        审计与幂等口径必须一致,否则 SSE 的 run 在对账时和 REST 的 run 长得不一样。第一帧之前的
        异常原样抛给调用方:那是 HTTP 状态码(403/409)的领地,不是 `error` 事件的。
        """
        key = thread_key(self._ctx(), thread_id)
        if not str(query or "").strip():
            raise IllegalTransition("query 不能为空")
        run_id = await self._start_run(key, query)
        state = empty_state(
            query=query,
            thread_id=thread_id,
            thread_key=key,
            run_id=run_id,
            route_flags=self.deps.flags_snapshot(),
            config_hash=self.deps.config_hash(),
        )
        async for frame in self._stream(state, key=key, run_id=run_id):
            yield frame

    async def stream_resume(
        self, thread_id: str, decision: ResumeChoice | Mapping[str, Any]
    ) -> AsyncIterator[StreamEvent]:
        """`resume` 的流式形态。预检顺序与 `resume` 逐条对应,一条不少:
        形状 → assert_thread → 读快照 → 到期 → 重放/确认预检 → 才动图。
        """
        key = thread_key(self._ctx(), thread_id)
        if self.deps.trace is not None:
            await self.deps.trace.assert_thread(key)
        snap = await self.graph.aget_state(self._cfg(key))
        values = snapshot_values(snap)
        run_id = str(values.get("run_id") or "")
        if not values:
            raise IllegalTransition(f"没有这个会话,或它的检查点已被清: {key}")
        expired = await self._expire_if_stale(key, run_id, values)
        if expired is not None:
            async for frame in self._outcome_events(expired):
                yield frame
            return
        if not snap.next:
            async for frame in self._outcome_events(self._replay(key, run_id, values)):
                yield frame
            return
        if tuple(snap.next) != (CONFIRM_NODE,):
            raise IllegalTransition(f"该会话停在 {tuple(snap.next)},不是 await_confirm,无法确认")
        await self._precheck(run_id, values, decision)
        async for frame in self._stream(Command(resume=decision), key=key, run_id=run_id):
            yield frame

    async def _stream(self, inp: Any, *, key: str, run_id: str) -> AsyncIterator[StreamEvent]:
        """`astream(updates)` → 中间事件;收尾与 `ainvoke` 路径共用 `_settle`。

        `usage` 在 State 里是累加键,updates 补丁里却是各节点自己的那一段 —— 合并时必须先扩展
        再覆写,否则 `_settle` 数的 token 只剩最后一个节点的,trace_run 的成本口径当场失真。
        """
        started = utc_now()
        merged: dict[str, Any] = {}
        interrupted: Any = None
        async for chunk in self.graph.astream(inp, self._cfg(key), stream_mode="updates"):
            if "__interrupt__" in chunk:
                # 存原始 `Interrupt` 对象,交给 `_settle` 统一拆。这里先拆一次会让 `_settle`
                # 对 payload dict 再按 Interrupt 索引(KeyError: 0)。
                interrupted = chunk["__interrupt__"]
                continue
            for node, patch in chunk.items():
                if not isinstance(patch, Mapping):
                    continue
                usage = patch.get("usage")
                if usage:
                    merged.setdefault("usage", []).extend(usage)
                merged.update({k: v for k, v in patch.items() if k != "usage"})
                event = NODE_EVENTS.get(str(node))
                if event is not None:
                    yield StreamEvent(event, self._event_data(event, patch, merged))
                degraded = patch.get("degraded")
                items = degraded if isinstance(degraded, list) else ([degraded] if degraded else [])
                if items:
                    yield StreamEvent("degraded", {"degraded": [str(one) for one in items]})
        result = dict(merged)
        if interrupted is not None:
            result["__interrupt__"] = interrupted
        outcome = await self._settle(result, key=key, run_id=run_id, started=started)
        async for frame in self._outcome_events(outcome):
            yield frame

    async def _outcome_events(self, outcome: Outcome) -> AsyncIterator[StreamEvent]:
        """终态的统一出口:`await_confirm` 一帧;其余 (token) → (citations) → done。

        `citations` 在 State 里是 `Citation` dataclass,出 SSE 边界前必须收敛成原生类型 ——
        `json.dumps(default=str)` 会把它拍成 `"Citation(...)"`,前端拿到的是一串类名。
        """
        if outcome.awaiting:
            yield StreamEvent(
                "await_confirm",
                {"thread_key": outcome.thread_key, "run_id": outcome.run_id, **(outcome.payload or {})},
            )
            return
        if outcome.answer:
            yield StreamEvent("token", {"text": outcome.answer})
        if outcome.citations:
            yield StreamEvent("citations", {"citations": [_plain(c) for c in outcome.citations]})
        done = outcome.as_dict()
        # `done` 与 `citations` 帧必须是同一份原生类型:直接 json.dumps(default=str) 会把
        # Citation dataclass 拍成字符串,前端拿到的是类名而不是 chunk_id。
        done["citations"] = [_plain(c) for c in outcome.citations]
        yield StreamEvent("done", done)

    @staticmethod
    def _event_data(
        event: str, patch: Mapping[str, Any], merged: Mapping[str, Any]
    ) -> dict[str, Any]:
        if event == "route":
            return {"intent": dict(patch.get("intent") or {})}
        if event == "rewrite_candidates":
            return {"candidates": list(patch.get("rewrite_candidates") or [])}
        # retrieval_done:名次摘要之后就是生成,前端拿到条数就够渲染「检索完成」
        return {
            "n_context": len(patch.get("reranked") or []),
            "route_stats": dict(merged.get("route_stats") or {}),
        }


async def sweep_stale(runtime: Runtime, *, limit: int = 50) -> list[str]:
    """把本租户「停在 await_confirm 且已过期」的会话落 `expired` 并删检查点,返回处理过的键。

    扫描点只有两个:#7 启动时那一次,加上 `resume` 里的懒判(那条已经在 `Runtime.resume` 内)。
    **不引入常驻定时器** —— 懒判覆盖了唯一会真实读到这个状态的路径,这里只是把启动那次的漏项
    补齐,顺带让 FR-403 的「过期」在面板上可数。

    待确认的判据取自**检查点快照的 `next`**,不是 `trace_run.final_status`:那一列的
    `server_default` 是 `'error'`(没收尾的 run 一律算失败,是 #3 定的兜底),拿空串当「在等人」
    会永远匹配不上。
    """
    if runtime.deps.trace is None:
        return []
    out: list[str] = []
    for row in await runtime.deps.trace.list_runs(limit):
        key = str(row.thread_id)  # 这一列存的就是带租户前缀的完整键(thread_key 的原样)
        snap = await runtime.graph.aget_state(runtime._cfg(key))
        if tuple(snap.next or ()) != (CONFIRM_NODE,):
            continue
        values = snapshot_values(snap)
        run_id = str(values.get("run_id") or row.id)
        if await runtime._expire_if_stale(key, run_id, values) is not None:
            out.append(key)
    return out
