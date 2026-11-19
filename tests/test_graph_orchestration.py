"""编排层验收:五层防御逐层留痕、中断-确认-续跑、FR-701 的 stage 全覆盖。

跑真 PG(`trace_run` / `trace_step` / `security_event` 落的是真行)。检索层用桩:那一层已经有
`test_retrieval_e2e.py` 拿真 Milvus / 真 Neo4j 验过,这一份要验的是「编排拿到候选之后怎么判定、
怎么留痕」—— 混在一起测,失败时说不清是谁的锅。

**编排层的 trace 是一步一事务真实提交的**(`agents/trace.py` 的 docstring 写了为什么),
`conftest.session` 的 SAVEPOINT 兜不住它 —— 所以本文件自己 purge:`tnt-lg-*` 留下的行不清掉,
下一轮的 seq 对不上,FK 也会拦住租户删除。检查点用 `InMemorySaver`;跨进程恢复归
`test_resume_after_restart.py`(#16)。
"""
from __future__ import annotations

from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from conftest import database_url
from kbmesh.agents import nodes as nodes_mod
from kbmesh.agents.graph import compile_graph, mermaid, node_names, topology_ok
from kbmesh.agents.nodes import (
    ALL_NODES,
    NODE_STAGES,
    REFUSAL_TEXT,
    GraphDeps,
    after_assess,
    after_intent,
    after_retry,
    seed_state,
)
from kbmesh.agents.schemas import IntentOut, RelevanceOut, RewriteOut
from kbmesh.agents.state import REJECT_CODES, STAGES, AgentState, empty_state
from kbmesh.agents.trace import TraceWriter
from kbmesh.core.config import clear_settings_cache, get_settings
from kbmesh.core.errors import IllegalTransition
from kbmesh.core.gateway import ChatResult, ModelUnavailable, RerankHit, Usage
from kbmesh.core.tenancy import ROLE_READER, TenantContext, use_context
from kbmesh.db.base import make_engine
from kbmesh.db.models import FINAL_STATUSES, SecurityEvent, Tenant, TraceRun
from kbmesh.db.repositories import list_security_events
from kbmesh.retrieval.contracts import ROUTES, Citation, Hit
from kbmesh.retrieval.fusion import RetrievalResult

TENANT_A, KB_A = "tnt-lg-a", "kb-lg-a"
TENANT_B, KB_B = "tnt-lg-b", "kb-lg-b"
CTX_A = TenantContext(
    tenant_id=TENANT_A, user_id="usr-lg-a", roles=(ROLE_READER,), kb_scope=frozenset({KB_A})
)
CTX_B = TenantContext(
    tenant_id=TENANT_B, user_id="usr-lg-b", roles=(ROLE_READER,), kb_scope=frozenset({KB_B})
)

BODY = "Milvus 承载向量检索,隔离设计靠租户分区键完成。"
GRAPH = "Neo4j 做受控跳数的邻域扩展,每一跳都重新核对租户。"
QUERY = "隔离 设计 怎么做"
#: 用户在确认框里选第 1 条之后,真正拿去检索的那句话
CHOSEN = "隔离设计怎么做"


def _hit(index: int, text: str, route: str = "body") -> Hit:
    cid = f"chunk-lg-{index}"
    return Hit(
        chunk_id=cid,
        score=0.5,
        source_routes={route},
        citation=Citation(doc_id="doc-lg", chunk_id=cid, heading_path="检索 > 隔离设计", title="设计册"),
        text=text,
        route_scores={route: 0.5},
    )


def _hits() -> list[Hit]:
    return [_hit(1, BODY), _hit(2, GRAPH, "graph")]


def _scores(hits: list[Hit], score: float, *, missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "scores": [{"chunk_id": h.chunk_id, "score": score, "gap": ""} for h in hits],
        "missing_info": list(missing or []),
    }


def _cast(**overrides: Any) -> dict[str, Any]:
    return {
        **empty_state(query=QUERY, thread_id="th-1", thread_key=f"{TENANT_A}:th-1", run_id="run-x"),
        **overrides,
    }


# --------------------------------------------------------------------- 替身
class FakeGateway:
    """按 `purpose` 分派可编程返回:`script[key]` 是「第 N 次调用给什么」,用完重复最后一条。"""

    def __init__(self, **script: Any) -> None:
        self.calls: Counter[str] = Counter()
        self.rerank_calls = 0
        self.rerank_fails = False
        self.chat_fails = False
        self.fail_purposes: set[str] = set(script.pop("fail_purposes", set()))
        self.answer_text = script.pop(
            "answer_text", "隔离靠分区键完成 [1],邻域扩展另有一条 [2]。"
        )
        self.script: dict[str, list[Any]] = {key: list(value) for key, value in script.items()}
        self._cursor: Counter[str] = Counter()
        self.prompts: list[tuple[str, str]] = []

    def _next(self, key: str) -> Any:
        queue = self.script.get(key) or []
        if not queue:
            return None
        value = queue[min(self._cursor[key], len(queue) - 1)]
        self._cursor[key] += 1
        return value

    def embed(self, texts, *, purpose: str = "embed"):
        return [[0.0] * get_settings().embedding_dim for _ in texts], Usage(model="fake-embed")

    def rerank(self, query: str, documents, *, top_n: int | None = None):
        self.rerank_calls += 1
        if self.rerank_fails:
            raise ModelUnavailable("rerank", "mock 配额用尽")
        return [RerankHit(index=i, score=0.9 - 0.05 * i) for i in range(len(documents))][
            :(top_n or len(documents))
        ]

    def chat(self, prompt: str, *, purpose: str = "generate", **kwargs: Any) -> ChatResult:
        self.calls["chat"] += 1
        self.prompts.append(("generate", prompt))
        if self.chat_fails:
            raise ModelUnavailable(f"chat:{purpose}", "mock 不可用")
        text = self._next("answer") if self.script.get("answer") else self.answer_text
        return ChatResult(
            text=text, usage=Usage(token_in=120, token_out=40, latency_ms=9, model="fake-llm")
        )

    def structured(self, prompt: str, schema: type, *, purpose: str = "route", system: str = ""):
        self.calls[purpose] += 1
        self.prompts.append((purpose, prompt))
        if purpose in self.fail_purposes:
            raise ModelUnavailable(f"structured:{purpose}", "mock 不可用")
        usage = Usage(token_in=30, token_out=8, latency_ms=5, model="fake-router")
        if schema is IntentOut:
            payload = self._next("intent") or {
                "verdict": "answerable",
                "confidence": 0.9,
                "reason": "库内问题",
            }
            return IntentOut.model_validate(payload), usage
        if schema is RewriteOut:
            return RewriteOut.model_validate({"candidates": self._next("rewrite") or []}), usage
        if schema is RelevanceOut:
            payload = self._next("assess") or {"scores": [], "missing_info": []}
            return RelevanceOut.model_validate(payload), usage
        raise AssertionError(f"编排层不该问这一类结构化输出:{schema!r}")


class StubRetrieve:
    """换掉 `nodes.retrieve`:编排只消费「候选集 + 逐路 stats + 降级清单」,不给它真引擎。"""

    def __init__(
        self,
        rounds: list[list[Hit]] | None = None,
        *,
        degraded: list[str] | None = None,
        dropped: list[str] | None = None,
    ) -> None:
        self.rounds = rounds if rounds is not None else [_hits()]
        self.calls: list[str] = []
        self.ctx: list[TenantContext] = []
        self.kwargs: list[dict[str, Any]] = []
        self.degraded = list(degraded or [])
        self.dropped = list(dropped or [])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(args[0], args[1], kwargs)

    async def _run(self, query: str, ctx: TenantContext, kwargs: dict[str, Any]) -> RetrievalResult:
        self.calls.append(query)
        self.ctx.append(ctx)
        self.kwargs.append(kwargs)
        hits = self.rounds[min(len(self.calls) - 1, len(self.rounds) - 1)]
        return RetrievalResult(
            query=query,
            hits=list(hits),
            stats={
                route: {
                    "count": len(hits),
                    "fingerprint": f"fp-{route}-{len(self.calls)}",
                    "latency_ms": 3,
                    "error": None,
                    "degraded": False,
                }
                for route in ROUTES
            },
            graph={"mentions": [], "lines": ["Milvus --USES--> 向量检索"]},
            degraded=list(self.degraded),
            dropped=list(self.dropped),
            fused_count=len(hits),
        )


# --------------------------------------------------------------------- 现场
@dataclass
class Run:
    """一次问答:开好 `trace_run`、编译好图、拿到 cfg;`invoke` 与 `resume` 是两条腿。"""

    rig: Rig
    ctx: TenantContext
    thread: str
    run_id: str
    state: dict[str, Any]
    graph: Any
    config: dict[str, Any]
    first: dict[str, Any] | None = None
    final: dict[str, Any] | None = None

    @property
    def payload(self) -> dict[str, Any]:
        assert self.first is not None, "还没 invoke,谈不上 interrupt 载荷"
        stopped = self.first.get("__interrupt__") or []
        assert stopped, "图没有在 await_confirm 停下来,FR-402 的形态就不对了"
        return stopped[0].value

    async def invoke(self) -> dict[str, Any]:
        with use_context(self.ctx):
            self.first = await self.graph.ainvoke(self.state, self.config)
        return self.first

    async def resume(self, decision: Any) -> dict[str, Any]:
        with use_context(self.ctx):
            try:
                self.final = await self.graph.ainvoke(Command(resume=decision), self.config)
            finally:
                await self._finish()
        return self.final

    async def ask(self, decision: Any = None) -> dict[str, Any]:
        await self.invoke()
        return await self.resume({"choice": 1} if decision is None else decision)

    async def _finish(self) -> None:
        """按 runtime(#16)的同一口径收尾,好让 `trace_run` 的列在本文件里也是真值。"""
        status = str((self.final or {}).get("final_status") or "")
        if status not in FINAL_STATUSES:
            return
        usage = self.final.get("usage") or []  # type: ignore[union-attribute]
        await self.rig.deps.trace.finish(
            self.run_id,
            final_status=status,
            degraded=bool(self.final.get("degraded")),
            latency_ms=1,
            token_in=sum(int(one.get("token_in") or 0) for one in usage),
            token_out=sum(int(one.get("token_out") or 0) for one in usage),
        )


@dataclass
class Rig:
    factory: async_sessionmaker[AsyncSession]
    gateway: FakeGateway
    stub: StubRetrieve
    deps: GraphDeps

    async def open(
        self,
        *,
        query: str = QUERY,
        thread: str = "th-1",
        ctx: TenantContext = CTX_A,
        saver: Any = None,
    ) -> Run:
        key = f"{ctx.tenant_id}:{thread}"
        with use_context(ctx):
            run_id = await self.deps.trace.start(
                key,
                query,
                route_flags=self.deps.flags_snapshot(),
                config_snapshot={"config_hash": self.deps.config_hash()},
            )
            state = seed_state(
                query=query, thread_id=thread, thread_key=key, run_id=run_id, deps=self.deps
            )
            graph = compile_graph(self.deps, saver if saver is not None else InMemorySaver())
        return Run(
            rig=self,
            ctx=ctx,
            thread=thread,
            run_id=run_id,
            state=state,
            graph=graph,
            config={"configurable": {"thread_id": key}},
        )

    async def ask(self, *, resume: Any = None, **kwargs: Any) -> Run:
        run = await self.open(**kwargs)
        await run.invoke()
        await run.resume({"choice": 1} if resume is None else resume)
        return run

    async def steps(self, run_id: str, *, ctx: TenantContext = CTX_A) -> list[Any]:
        with use_context(ctx):
            return await self.deps.trace.list_steps(run_id)

    def use_gateway(self, gateway: FakeGateway) -> None:
        """换替身必须同时换掉 deps 里的那一份,否则节点读到的还是旧脚本。"""
        self.gateway = gateway
        self.deps = replace(self.deps, gateway=gateway)

    def use_settings(self, settings: Any) -> None:
        self.deps = replace(self.deps, settings=settings)

    def bare(self) -> GraphDeps:
        """不落库的 deps:单节点 await 时用,发号照常、写库跳过。"""
        return replace(self.deps, trace=None)


async def _purge(factory: async_sessionmaker[AsyncSession]) -> None:
    ids = [TENANT_A, TENANT_B]
    async with factory() as session, session.begin():
        for stmt in (
            "delete from platform.trace_step where run_id in "
            "(select id from platform.trace_run where tenant_id = any(:ts))",
            "delete from platform.trace_run where tenant_id = any(:ts)",
            "delete from platform.sec_event where tenant_id = any(:ts)",
            "delete from platform.kb_tenant where id = any(:ts)",
        ):
            await session.execute(sa_text(stmt), {"ts": ids})


@pytest.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = make_engine(database_url())
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def rig(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Rig, FakeGateway, StubRetrieve]]:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    await _purge(factory)
    async with factory() as session, session.begin():
        for tenant_id in (TENANT_A, TENANT_B):
            session.add(Tenant(id=tenant_id, name=f"{tenant_id}-名字"))
    gateway = FakeGateway()
    stub = StubRetrieve()
    monkeypatch.setattr(nodes_mod, "retrieve", stub)  # type: ignore[arg-type]
    deps = GraphDeps(
        factory=factory,
        gateway=gateway,  # type: ignore[arg-type]
        kb_id=KB_A,
        # 身份只从 ContextVar 读:把 ctx 硬编进 deps 就等于让「谁在问」由 fixture 决定
        trace=TraceWriter(factory),
    )
    handle = Rig(factory=factory, gateway=gateway, stub=stub, deps=deps)
    yield handle, gateway, stub
    await _purge(factory)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    """改环境 → 重建 Settings:阈值和开关的唯一真值是 Settings,不是某个节点里的常量。"""

    def apply(**overrides: Any) -> Any:
        for key, value in overrides.items():
            text = {True: "true", False: "false"}.get(value, str(value))
            monkeypatch.setenv(f"KBMESH_{key}", text)
        clear_settings_cache()
        return get_settings()

    yield apply
    clear_settings_cache()


# --------------------------------------------------------------------- 拓扑(FR-401)
async def test_topology_is_the_documents_shape(rig: tuple) -> None:
    deps = rig[0].deps
    assert set(ALL_NODES) <= set(node_names(deps))
    assert topology_ok(deps)
    for node, stages in NODE_STAGES.items():
        assert node in ALL_NODES
        assert set(stages) <= set(STAGES), f"{node} 的 stage 不在 §11 白名单里"
    drawing = mermaid(deps)
    for edge in (
        "__start__ --> route_intent",
        "route_intent -.-> rewrite_query",
        "route_intent -.-> refuse",
        "rewrite_query --> confirm_interrupt",
        "confirm_interrupt --> retrieve_dual",
        "retrieve_dual --> dedupe",
        "dedupe --> rerank_fused",
        "rerank_fused --> assess_relevance",
        "assess_relevance -.-> retry_targeted",
        "assess_relevance -.-> refuse",
        "assess_relevance -.-> generate_cite",
        "retry_targeted -.-> retrieve_dual",
        "retry_targeted -.-> refuse",
        "generate_cite --> __end__",
    ):
        assert edge in drawing, f"§7.1 的这条边在编译产物里找不到:{edge}"


def test_routers_read_state_only() -> None:
    assert after_intent(_cast(reject_code="")) == "rewrite_query"
    assert after_intent(_cast(reject_code="chitchat")) == "refuse"
    assert after_assess(_cast(reject_code="")) == "generate_cite"
    assert after_assess(_cast(reject_code="below_theta", missing_info=["时间范围"])) == (
        "retry_targeted"
    )
    assert after_assess(_cast(reject_code="below_theta", missing_info=[])) == "refuse"
    assert after_assess(_cast(reject_code="below_theta", missing_info=["x"], retry_attempt=2)) == (
        "refuse"
    )
    assert after_retry(_cast(retry_constraints=["2024"])) == "retrieve_dual"
    assert after_retry(_cast(retry_constraints=[])) == "refuse"


@pytest.mark.parametrize("node", [name for name in ALL_NODES if name != "confirm_interrupt"])
async def test_every_node_is_directly_awaitable(rig: tuple, node: str) -> None:
    """FR-401:节点是工厂产物,不是图的私有物 —— 拿一份 state 直接 await 就能测。"""
    hits = _hits()
    st = _cast(
        confirmed_query=CHOSEN,
        hits=hits,
        reranked=hits,
        rewrite_candidates=[{"index": 1, "text": CHOSEN, "focus": "隔离", "rationale": "口语化"}],
        missing_info=["容量上限对应的时间范围"],
        relevance_scores={h.chunk_id: 0.8 for h in hits},
        cite_ids={"1": hits[0].chunk_id, "2": hits[1].chunk_id},
    )
    factory = getattr(nodes_mod, "make_" + node)
    with use_context(CTX_A):
        patch = await factory(rig[0].bare())(st)
    assert set(patch) <= set(AgentState.__annotations__), f"{node} 写了没登记过的 state 字段"
    assert int(patch.get("trace_seq", 0)) >= int(st.get("trace_seq", 0))


# --------------------------------------------------------------------- 顺利路径
async def test_full_chain_answers_and_every_step_landed(rig: tuple) -> None:
    handle, gateway, stub = rig
    hits = _hits()
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "口语化"}]],
        "assess": [_scores(hits, 0.8)],
    }
    run = await handle.ask()
    final = run.final
    assert final["final_status"] == "answered"
    assert final["answer"].endswith("[2]。")
    assert [c.chunk_id for c in final["citations"]] == [h.chunk_id for h in hits]
    assert final["cite_ids"] == {"1": hits[0].chunk_id, "2": hits[1].chunk_id}
    assert stub.calls == [CHOSEN], "检索必须用确认后的词,而且确认之前一次都不许跑"
    assert stub.kwargs[0]["defer_rerank"] is True, "精排归编排,`retrieve` 要把完整候选集交回来"
    assert gateway.calls["route"] == 1 and gateway.calls["rewrite"] == 1
    assert gateway.calls["assess"] == 2, "crosscheck 默认开:两遍判分,取较低的那个"
    assert gateway.rerank_calls == 1

    stages = [s.stage for s in await handle.steps(run.run_id)]
    assert stages == [
        "route_intent",
        "rewrite",
        "rewrite",
        "retrieve_body",
        "retrieve_heading",
        "retrieve_dense",
        "retrieve_graph",
        "merge",
        "rerank",
        "assess",
        "generate",
        "cite_check",
    ], "留下一步跑一步:编排层的对账口径就是这份顺序"
    row = await _run_row(handle, run.run_id)
    assert row.final_status == "answered" and row.degraded is False
    assert row.token_in > 0 and row.token_out > 0, "FR-703 的成本汇总靠这两列,不能一直是 0"


async def test_interrupt_payload_carries_no_evidence_text(rig: tuple) -> None:
    handle, gateway, stub = rig
    gateway.script["rewrite"] = [[{"text": CHOSEN, "focus": "隔离", "rationale": "口语化"}]]
    run = await handle.open()
    await run.invoke()
    payload = run.payload
    assert stub.calls == [] and stub.kwargs == []
    assert payload["kind"] == "await_confirm"
    assert payload["question"] == QUERY
    assert payload["run_id"] == run.run_id and payload["thread_id"] == "th-1"
    assert payload["rewrite_candidates"][0]["text"] == CHOSEN
    assert payload["edited_allowed"] is True
    assert payload["pending_ttl_min"] == get_settings().pending_confirm_ttl_min
    assert payload["config_hash"] == handle.deps.config_hash()
    ttl = timedelta(seconds=payload["pending_ttl_min"] * 60)
    assert datetime.now(UTC) <= datetime.fromisoformat(payload["expires_at"]) <= datetime.now(UTC) + ttl
    blob = str(payload)
    for body in (BODY, GRAPH):
        assert body not in blob, "确认发生在检索之前,载荷里出现正文就是给探针开门"


@pytest.mark.parametrize(
    "decision",
    [{"choice": 99}, {"choice": 0}, {"edited": ""}, {"edited": "x"}, {"edited": "含\x07控制符"}],
)
async def test_bad_resume_raises_never_falls_back(rig: tuple, decision: dict[str, Any]) -> None:
    handle, gateway, stub = rig
    gateway.script["rewrite"] = [[{"text": CHOSEN, "focus": "隔离", "rationale": "口语化"}]]
    run = await handle.open()
    await run.invoke()
    with pytest.raises(IllegalTransition):
        await run.resume(decision)
    assert stub.calls == [], "确认没通过就去检索,等于跳过 FR-402"
    steps = await handle.steps(run.run_id)
    assert steps[-1].stage == "rewrite" and steps[-1].payload["kind"] == "resume_rejected"


async def test_edited_rewording_wins_over_the_choice(rig: tuple) -> None:
    handle, gateway, stub = rig
    hits = _hits()
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "口语化"}]],
        "assess": [_scores(hits, 0.9)],
    }
    run = await handle.ask(resume={"choice": 1, "edited": "容量上限是多少"})
    assert run.final["user_choice"]["edited"] == "容量上限是多少"
    assert stub.calls == ["容量上限是多少"], "用户改写了,选第几条已经不重要"


# --------------------------------------------------------------------- 层1:意图门禁
@pytest.mark.parametrize(
    ("verdict", "code"),
    [("chitchat", "chitchat"), ("out_of_scope", "out_of_scope"), ("out_of_privilege", "intent_forbidden")],
)
async def test_refused_intent_never_reaches_retrieval(rig: tuple, verdict: str, code: str) -> None:
    handle, _old, stub = rig
    gateway = FakeGateway(intent=[{"verdict": verdict, "confidence": 0.8, "reason": "不该进库"}])
    handle.use_gateway(gateway)
    run = await handle.ask(thread=f"th-{verdict}")
    final = run.final
    assert stub.calls == [], "判死的问题还去查库,越权在这一步就已经发生了"
    assert final["final_status"] == "refused" and final["reject_code"] == code
    assert final["reject_code"] in REJECT_CODES
    assert final["answer"] == REFUSAL_TEXT
    assert final["citations"] == [] and final["cite_ids"] == {}
    assert gateway.calls == Counter(route=1), "闲聊同样终态:不改写、不检索、不生成"
    row = await _run_row(handle, run.run_id)
    assert row.final_status == "refused", "FR-406:拒答样本不许混进 answered 口径"


async def test_only_out_of_privilege_is_an_audit_event(rig: tuple) -> None:
    handle, _old, _stub = rig
    for verdict, expected in (("chitchat", set()), ("out_of_privilege", {"intent_out_of_privilege"})):
        await _drop_events(handle.factory)
        handle.use_gateway(
            FakeGateway(intent=[{"verdict": verdict, "confidence": 0.7, "reason": "r"}])
        )
        await handle.ask(thread=f"th-audit-{verdict}")
        events = await _events(handle)
        assert {e.event for e in events} == expected, f"{verdict} 的审计口径不对"
        for one in events:
            assert one.detail["verdict"] == verdict
            assert QUERY not in str([e.detail for e in events]), "问题原文不进安全流水"


async def test_a_verdict_outside_the_gate_fails_closed(rig: tuple) -> None:
    """纵深防御:`verdict` 是 Literal,现网的模型给不出表外值,但从老检查点读回来的 dict 能给。

    门禁失败的方向只有一个 —— 不放行。这条测的就是「拿到一个看不懂的判定」时不许放行。
    """
    handle, _old, stub = rig

    class WeirdGateway(FakeGateway):
        def structured(self, prompt: str, schema: type, *, purpose: str = "route", system: str = ""):
            if purpose == "route":
                return {"verdict": "maybe", "confidence": 0.5, "reason": "拿不准"}, Usage()
            return super().structured(prompt, schema, purpose=purpose, system=system)

    handle.use_gateway(WeirdGateway())
    run = await handle.ask()
    assert run.final["reject_code"] == "intent_unknown"
    assert run.final["final_status"] == "refused"
    assert stub.calls == []


async def test_intent_outage_is_an_error_not_a_pass(rig: tuple) -> None:
    """§10 把意图路由列为不可降级:模型挂掉宁要 error,也不放一个没过门禁的问题进检索。"""
    handle, _old, stub = rig
    handle.use_gateway(FakeGateway(fail_purposes={"route"}))
    run = await handle.open()
    with pytest.raises(ModelUnavailable):
        await run.invoke()
    assert stub.calls == []
    steps = await handle.steps(run.run_id)
    assert [s.stage for s in steps] == ["route_intent"]
    assert steps[0].payload["fail_closed"] == "error"


# --------------------------------------------------------------------- 层2 + 层4:判分与拒答
async def test_below_theta_answer_refuses_with_the_fixed_text(rig: tuple) -> None:
    handle, gateway, stub = rig
    hits = _hits()
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        # 0.40 过得了 θ_rel(0.35)进上下文,却撑不起 θ_answer(0.45):进上下文 ≠ 够格回答
        "assess": [_scores(hits, 0.40)],
    }
    run = await handle.ask()
    assert run.final["final_status"] == "refused"
    assert run.final["reject_code"] == "below_theta"
    assert run.final["answer"] == REFUSAL_TEXT
    assert gateway.calls["chat"] == 0, "证据不够就不该产生任何文字"
    assert stub.calls == [CHOSEN]
    assert (await _run_row(handle, run.run_id)).final_status == "refused"
    assess = _step(await handle.steps(run.run_id), "assess")
    assert assess.payload["best_score"] == 0.4 and assess.payload["kept"] == 2


async def test_empty_candidate_set_refuses_without_calling_the_model(rig: tuple) -> None:
    handle, gateway, stub = rig
    stub.rounds = [[]]
    gateway.script["rewrite"] = [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]]
    run = await handle.ask()
    assert run.final["reject_code"] == "no_evidence"
    assert gateway.calls["assess"] == 0, "没有候选还去问模型,是把判定交给没有依据的分数"
    assert _step(await handle.steps(run.run_id), "assess").payload["reason"] == "候选集为空"


async def test_unscored_hit_is_dropped_and_cannot_be_cited(rig: tuple) -> None:
    """层2:模型没给分的命中 = 一次没有依据的引用,连编号都不许有(宁缺毋滥)。"""
    handle, gateway, _stub = rig
    hits = _hits()
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(hits[:1], 0.9)],
    }
    gateway.answer_text = "分区键就够了 [1],另一条在 [2]。"
    run = await handle.ask()
    assert run.final["final_status"] == "answered"
    assert "[2]" not in run.final["answer"] and "[1]" in run.final["answer"]
    assert run.final["cite_ids"] == {"1": hits[0].chunk_id}
    assess = _step(await handle.steps(run.run_id), "assess")
    assert assess.payload["kept"] == 1 and hits[1].chunk_id in assess.payload["dropped"]


async def test_invented_chunk_id_is_recorded_not_trusted(rig: tuple) -> None:
    handle, gateway, _stub = rig
    hits = _hits()
    payload = _scores(hits, 0.9)
    payload["scores"].append({"chunk_id": "chunk-不存在", "score": 0.99, "gap": ""})
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [payload],
    }
    run = await handle.ask()
    assess = _step(await handle.steps(run.run_id), "assess")
    assert assess.payload["invented_ids"] == ["chunk-不存在"]
    assert "chunk-不存在" not in run.final["relevance_scores"]
    assert "chunk-不存在" in run.final["dropped"]


async def test_crosscheck_takes_the_lower_of_two_passes(rig: tuple) -> None:
    handle, gateway, _stub = rig
    hits = _hits()
    # 0.9/0.4 取 min=0.4:够 θ_rel(0.35)进上下文,却撑不起 θ_answer(0.45)。用这个台阶而不是
    # 更低的分,是为了让「取的是低的那个」与「分没达标」两件事在同一个断言里分开成立。
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(hits, 0.9), _scores(hits, 0.4)],
    }
    run = await handle.ask()
    assert gateway.calls["assess"] == 2
    assert _step(await handle.steps(run.run_id), "assess").payload["best_score"] == 0.4, (
        "换个顺序就涨分的证据不算数"
    )
    assert run.final["reject_code"] == "below_theta", "证据达标但最高分不够,是「撑不起」不是「没有」"


async def test_crosscheck_off_scores_once(rig: tuple, env) -> None:
    handle, gateway, _stub = rig
    settings = env(FEATURE_CROSSCHECK=False)
    handle.use_settings(settings)
    hits = _hits()
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(hits, 0.9)],
    }
    run = await handle.ask()
    assert gateway.calls["assess"] == 1
    assert run.final["final_status"] == "answered", "关掉增益不影响能不能答,只少一遍分数"
    assert _step(await handle.steps(run.run_id), "assess").payload["crosscheck"] is False


# --------------------------------------------------------------------- 层3:定向重试
async def test_targeted_retry_carries_new_constraints_and_covers_every_stage(rig: tuple) -> None:
    handle, gateway, stub = rig
    first, second = _hits(), [_hit(1, BODY), _hit(3, "容量上限见容量表附录。", "heading")]
    stub.rounds = [first, second]
    gateway.script = {
        "rewrite": [
            [{"text": CHOSEN, "focus": "隔离", "rationale": "口语化"}],
            [{"text": "2024年隔离容量上限", "focus": "时间范围", "rationale": "补年份"}],
        ],
        "assess": [
            _scores(first, 0.2, missing=["容量上限对应的时间范围"]),
            _scores(second, 0.85),
        ],
    }
    run = await handle.ask()
    assert run.final["final_status"] == "answered"
    assert run.final["retry_attempt"] == 1
    assert stub.calls == [CHOSEN, "2024年隔离容量上限"], "重试必须换问法,不许原样重跑"

    steps = await handle.steps(run.run_id)
    seqs = [s.seq for s in steps]
    assert seqs == list(range(1, len(seqs) + 1)), "seq 有空洞 = 有一步没落库"
    assert {s.stage for s in steps} == set(STAGES), "FR-701:§11 的十二个 stage 要一次跑齐"
    retry = _step(steps, "retry")
    assert retry.payload["run"] is True and retry.payload["attempt"] == 1
    assert retry.payload["missing"] == ["容量上限对应的时间范围"]
    assert retry.payload["constraints"], "FR-405:重试请求里必须看得见新增了哪条约束"
    assert retry.payload["evidence_gain"] > 0


async def test_retry_with_the_same_wording_refuses_instead_of_rerunning(rig: tuple) -> None:
    handle, gateway, stub = rig
    hits = _hits()
    gateway.script = {
        "rewrite": [
            [{"text": CHOSEN, "focus": "隔离", "rationale": "r"}],
            [{"text": CHOSEN, "focus": "隔离", "rationale": "原样再问一遍"}],
        ],
        "assess": [_scores(hits, 0.2, missing=["时间范围"])],
    }
    run = await handle.ask()
    assert len(stub.calls) == 1, "同一句话再烧一轮检索,正是 FR-405 要禁的那件事"
    assert run.final["final_status"] == "refused"
    retry = _step(await handle.steps(run.run_id), "retry")
    assert retry.payload["run"] is False
    assert retry.payload["rejected_reason"] == "no_new_constraint"


async def test_retry_budget_is_max_targeted_retry(rig: tuple) -> None:
    handle, gateway, stub = rig
    # 每轮都要捞到一条新证据:`evidence_gain > 0` 是层3 的第四个条件,拿同一批 chunk_id 会让它在
    # 第二轮就死掉,于是这个测试测的是「增益」而不是「预算」—— 两个数都得能单独归零才行。
    rounds = [
        _hits(),
        [_hit(1, BODY), _hit(3, "容量上限见容量表附录。")],
        [_hit(1, BODY), _hit(4, "审批链见流程册。")],
    ]
    stub.rounds = rounds
    gateway.script = {
        "rewrite": [
            [{"text": CHOSEN, "focus": "隔离", "rationale": "r"}],
            [{"text": "2024年隔离容量上限", "focus": "年份", "rationale": "r"}],
            [{"text": "2025年隔离审批链", "focus": "年份", "rationale": "r"}],
            [{"text": "2026年隔离审批链", "focus": "年份", "rationale": "r"}],
        ],
        "assess": [_scores(one, 0.2, missing=["时间范围"]) for one in rounds],
    }
    run = await handle.ask()
    budget = get_settings().max_targeted_retry
    assert len(stub.calls) == 1 + budget
    assert run.final["final_status"] == "refused"
    assert int(run.final["retry_attempt"]) == budget


# --------------------------------------------------------------------- 层5:编号
async def test_out_of_range_citation_is_stripped_and_the_answer_kept(rig: tuple) -> None:
    handle, gateway, _stub = rig
    hits = _hits()
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(hits[:1], 0.9)],
    }
    gateway.answer_text = "分区键完成 [1],另有 [7]。"
    run = await handle.ask()
    assert run.final["final_status"] == "answered"
    assert "[7]" not in run.final["answer"] and "[1]" in run.final["answer"]
    check = _step(await handle.steps(run.run_id), "cite_check")
    assert check.payload["verdict"] == "warn"
    assert check.payload["stripped"] == [7]
    assert check.payload["cited_ids"] == [hits[0].chunk_id]


async def test_answer_without_any_citation_is_refused(rig: tuple) -> None:
    handle, gateway, _stub = rig
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(_hits(), 0.9)],
    }
    gateway.answer_text = "我凭常识回答一下,没有编号。"
    run = await handle.ask()
    assert run.final["final_status"] == "refused"
    assert run.final["reject_code"] == "cite_unparsable"
    assert run.final["answer"] == REFUSAL_TEXT
    assert run.final["citations"] == []


async def test_citations_are_frozen_before_generation(rig: tuple) -> None:
    """编号在调模型之前定死,而且只覆盖过了 θ_rel 的证据 —— 事后重编号等于让模型自己选证据。"""
    handle, gateway, _stub = rig
    hits = _hits()
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(hits[:1], 0.9)],
    }
    await handle.ask()
    prompt = next(p for purpose, p in gateway.prompts if purpose == "generate")
    assert f"[1] {hits[0].citation.label()}" in prompt
    assert "[2]" not in prompt, "只有一条过了 θ_rel,编号就不该发到 2"


async def test_generate_outage_is_an_error_and_leaves_a_step(rig: tuple) -> None:
    handle, gateway, _stub = rig
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(_hits(), 0.9)],
    }
    gateway.chat_fails = True
    run = await handle.open()
    await run.invoke()
    with pytest.raises(ModelUnavailable):
        await run.resume({"choice": 1})
    steps = await handle.steps(run.run_id)
    assert steps[-1].stage == "generate" and steps[-1].payload["fail_closed"] == "error"


# --------------------------------------------------------------------- §10:开关与降级
async def test_rerank_outage_degrades_without_losing_the_answer(rig: tuple) -> None:
    handle, gateway, _stub = rig
    hits = _hits()
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(hits, 0.9)],
    }
    gateway.rerank_fails = True
    run = await handle.ask()
    assert run.final["final_status"] == "answered"
    assert any("精排降级" in d for d in run.final["degraded"])
    rerank = _step(await handle.steps(run.run_id), "rerank")
    assert rerank.payload["note"] and rerank.payload["enabled"] is True
    assert (await _run_row(handle, run.run_id)).degraded is True


async def test_rerank_flag_off_never_calls_the_model(rig: tuple, env) -> None:
    handle, gateway, _stub = rig
    hits = _hits()
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(hits, 0.9)],
    }
    handle.use_settings(env(FEATURE_RERANK=False))
    assert handle.deps.use_rerank is False
    run = await handle.ask()
    assert gateway.rerank_calls == 0
    rerank = _step(await handle.steps(run.run_id), "rerank")
    assert rerank.payload["enabled"] is False
    assert run.final["final_status"] == "answered", "精排是按量的钱,关掉也要给答案"


@pytest.mark.parametrize(
    ("key", "gone"),
    [("FEATURE_ROUTE_GRAPH", "graph"), ("FEATURE_ROUTE_HEADING", "heading")],
)
async def test_gain_switches_drop_routes_from_the_config(rig: tuple, env, key: str, gone: str) -> None:
    """§10 的增益开关要落到路集合上:关掉图谱路 = 那条路根本不在 routes 里。"""
    deps = rig[0].deps
    assert set(deps.effective_config().routes) == set(ROUTES)
    routes = set(replace(deps, settings=env(**{key: False})).effective_config().routes)
    assert gone not in routes
    assert len(routes) == len(ROUTES) - 1


async def test_retrieval_degradation_is_carried_into_the_trace(rig: tuple) -> None:
    handle, gateway, stub = rig
    stub.degraded = ["graph 路降级:Neo4j 不可用"]
    stub.dropped = ["chunk-已删除"]
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(_hits(), 0.9)],
    }
    run = await handle.ask()
    assert "graph 路降级:Neo4j 不可用" in run.final["degraded"]
    steps = await handle.steps(run.run_id)
    assert _step(steps, "merge").payload["dropped_by_authority"] == ["chunk-已删除"]
    assert _step(steps, "retrieve_graph").payload["degraded"] is False, (
        "路级降级由 retrieve 判定,编排不许自己加戏"
    )


async def test_rewrite_outage_leaves_only_the_edit_box(rig: tuple) -> None:
    """§10 没给改写这一行列降级动作 —— 这里落成一笔实测结论,交 #11 回填。"""
    handle, _old, _stub = rig
    handle.use_gateway(FakeGateway(fail_purposes={"rewrite"}))
    run = await handle.open()
    await run.invoke()
    assert run.payload["rewrite_candidates"] == []
    assert run.payload["edited_allowed"] is True
    assert any("改写候选降级" in d for d in run.first["degraded"])
    with pytest.raises(IllegalTransition):
        await run.resume({"choice": 1})


async def test_edited_text_can_still_answer_when_rewrite_is_down(rig: tuple) -> None:
    handle, _old, stub = rig
    gateway = FakeGateway(fail_purposes={"rewrite"})
    gateway.script["assess"] = [_scores(_hits(), 0.9)]
    handle.use_gateway(gateway)
    run = await handle.ask(resume={"edited": "隔离设计用的什么键"})
    assert run.final["final_status"] == "answered"
    assert stub.calls == ["隔离设计用的什么键"]


async def test_assess_outage_refuses_instead_of_passing_evidence(rig: tuple) -> None:
    handle, _old, stub = rig
    gateway = FakeGateway(fail_purposes={"assess"})
    gateway.script["rewrite"] = [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]]
    handle.use_gateway(gateway)
    run = await handle.ask()
    assert run.final["final_status"] == "refused"
    assert run.final["reject_code"] == "no_evidence"
    assert gateway.calls["chat"] == 0, "判分模型挂掉还照样生成答案 = 层2 白设"
    assert any("相关性判定不可用" in d for d in run.final["degraded"])
    assert len(stub.calls) == 1, "拿不到判定依据就别再检索一轮"


# --------------------------------------------------------------------- 身份(§6.1 红线)
async def test_identity_is_read_from_the_contextvar_at_call_time(rig: tuple) -> None:
    handle, gateway, stub = rig
    gateway.script = {
        "rewrite": [[{"text": CHOSEN, "focus": "隔离", "rationale": "r"}]],
        "assess": [_scores(_hits(), 0.9)],
    }
    run = await handle.ask(ctx=CTX_B, thread="th-9")
    assert stub.ctx[-1] is CTX_B
    async with handle.factory() as session:
        owner = (
            await session.execute(select(TraceRun.tenant_id).where(TraceRun.id == run.run_id))
        ).scalar_one()
    assert owner == TENANT_B, "行归属由运行时的 ctx 决定,不是由 state 里的前缀决定"
    assert await handle.steps(run.run_id, ctx=CTX_B)
    with pytest.raises(Exception, match="cross_tenant_read"):
        await handle.steps(run.run_id, ctx=CTX_A)


def test_state_has_no_bare_tenant_field() -> None:
    """§6.1 的红线在 state 这一侧的形态:能被 checkpoint 带走的字段里不许有租户身份。"""
    keys = set(empty_state(query="q", thread_id="t", thread_key=f"{TENANT_A}:t", run_id="r"))
    assert "tenant_id" not in keys
    assert "ctx" not in keys
    assert "thread_key" in keys, "只有组合键,租户前缀由 runtime 从验签后的 ctx 拼"


# --------------------------------------------------------------------- 小工具
def _step(steps: list[Any], stage: str) -> Any:
    found = [s for s in steps if s.stage == stage]
    assert found, f"没有 {stage} 这一步"
    return found[-1]


async def _run_row(handle: Rig, run_id: str) -> TraceRun:
    async with handle.factory() as session:
        return (await session.execute(select(TraceRun).where(TraceRun.id == run_id))).scalar_one()


async def _drop_events(factory: async_sessionmaker[AsyncSession]) -> None:
    async with factory() as session, session.begin():
        await session.execute(
            sa_text("delete from platform.sec_event where tenant_id = any(:ts)"),
            {"ts": [TENANT_A, TENANT_B]},
        )


async def _events(handle: Rig) -> list[SecurityEvent]:
    async with handle.factory() as session:
        return await list_security_events(CTX_A, session)
