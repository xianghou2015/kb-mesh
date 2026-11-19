"""§10 开关矩阵的编排侧验收:**能关的关掉仍出答案,不能关的连关的位置都没有**。

分工先说清楚,免得两份文件互相抄:检索层的降级(某一路挂掉、精排把顺序改坏、图谱指针指到缺失
切片)已经在 `test_retrieval_e2e.py` 拿真 Milvus / 真 Neo4j 验过。这里验的是**编排拿到那些降级
之后干了什么**:开关有没有落到路集合上、降级有没有留痕、`trace_run.degraded` 有没有被抬起来,
以及最要紧的一条 —— 不可降级项挂了图是不是**停住**而不是继续给答案。

跑的是 `Runtime.ask/resume` 这两条腿,不是自己拼 `graph.ainvoke`:`trace_run` 的 `degraded` 与
`final_status` 两列是 `Runtime._settle` 收尾时写的,绕开它就等于把「面板上看到的那一行」换成测试
自己写的临时值 —— 而那正是本文件要验的东西。

反向断言同样重要:§10 把意图路由、正文 BM25、检查点持久化列为必需项。「必需」在这个代码库里不是
一个注释,而是一个可测的事实 —— `Flags` 上根本没有对应的字段,关掉它得先改代码;而没有
checkpointer 的图连一个可恢复的快照都给不出来。

跑真 PG(`trace_run` / `trace_step` 落真行)。**编排层的 trace 是一步一事务真实提交的**
(`agents/trace.py` 的 docstring 写了为什么),`conftest.session` 的 SAVEPOINT 兜不住,所以本文件
自己 purge `tnt-dg-*`。检查点用 `InMemorySaver`:跨进程恢复归 `test_resume_after_restart.py`
(任务 #16),这里不重复。
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from conftest import database_url
from kbmesh.agents import nodes as nodes_mod
from kbmesh.agents.graph import compile_graph
from kbmesh.agents.nodes import ALL_NODES, REFUSAL_TEXT, GraphDeps
from kbmesh.agents.runtime import Outcome, Runtime
from kbmesh.agents.schemas import IntentOut, RelevanceOut, RewriteOut
from kbmesh.agents.state import empty_state
from kbmesh.agents.trace import TraceWriter
from kbmesh.core.config import Flags, clear_settings_cache, get_settings
from kbmesh.core.errors import IllegalTransition
from kbmesh.core.gateway import ChatResult, ModelUnavailable, RerankHit, Usage
from kbmesh.core.tenancy import ROLE_READER, TenantContext, use_context
from kbmesh.db.base import make_engine
from kbmesh.db.models import Tenant, TraceRun
from kbmesh.retrieval.contracts import Citation, Hit
from kbmesh.retrieval.fusion import RetrievalResult

TENANT, KB = "tnt-dg-a", "kb-dg-a"
CTX = TenantContext(
    tenant_id=TENANT, user_id="usr-dg-a", roles=(ROLE_READER,), kb_scope=frozenset({KB})
)
QUERY = "隔离 设计 怎么做"
CHOSEN = "隔离设计怎么做"
ANSWER = "隔离靠租户分区键完成 [1]。"


def _hit(index: int, snippet: str, route: str = "body") -> Hit:
    cid = f"chunk-dg-{index}"
    return Hit(
        chunk_id=cid,
        score=0.5,
        source_routes={route},
        citation=Citation(
            doc_id="doc-dg", chunk_id=cid, heading_path="检索 > 隔离设计", title="设计册"
        ),
        text=snippet,
        route_scores={route: 0.5},
    )


def _hits() -> list[Hit]:
    return [
        _hit(1, "Milvus 承载向量检索,隔离设计靠租户分区键完成。", "body"),
        _hit(2, "Neo4j 做受控跳数的邻域扩展。", "graph"),
    ]


# --------------------------------------------------------------------- 替身
class FlakyGateway:
    """四类结构化输出 + 生成 + 精排,每一个都能单独挂掉。

    挂的形状只有 `ModelUnavailable` 一种,那是网关对外唯一的失败语言,编排层不许猜别的。
    """

    def __init__(self, *, score: float = 0.9, candidates: list[str] | None = None) -> None:
        self.score = score
        self.candidates = candidates or [CHOSEN]
        self.answer = ANSWER
        self.fails: set[str] = set()
        self.rerank_fails = False
        self.chat_fails = False
        self.rerank_calls = 0
        self.chat_calls = 0
        self.purposes: list[str] = []

    def _bump(self, purpose: str) -> None:
        self.purposes.append(purpose)
        if purpose in self.fails:
            raise ModelUnavailable(f"structured:{purpose}", "mock 不可用")

    def structured(self, prompt: str, schema: type, *, purpose: str = "route", system: str = ""):
        self._bump(purpose)
        usage = Usage(token_in=30, token_out=8, latency_ms=5, model="fake")
        if schema is IntentOut:
            return IntentOut(verdict="answerable", confidence=0.9, reason="库内问题"), usage
        if schema is RewriteOut:
            return (
                RewriteOut.model_validate(
                    {
                        "candidates": [
                            {"text": one, "focus": "隔离", "rationale": "去掉口语"}
                            for one in self.candidates
                        ]
                    }
                ),
                usage,
            )
        if schema is RelevanceOut:
            return (
                RelevanceOut.model_validate(
                    {
                        "scores": [
                            {"chunk_id": one.chunk_id, "score": self.score, "gap": ""}
                            for one in _hits()
                        ],
                        "missing_info": [],
                    }
                ),
                usage,
            )
        raise AssertionError(f"编排层不该问这一类结构化输出:{schema!r}")

    def chat(self, prompt: str, *, purpose: str = "generate", **kwargs: Any) -> ChatResult:
        self._bump(purpose)
        self.chat_calls += 1
        if self.chat_fails:
            raise ModelUnavailable(f"chat:{purpose}", "mock 不可用")
        return ChatResult(text=self.answer, usage=Usage(token_in=90, token_out=30, model="fake"))

    def rerank(self, query: str, documents: list[str], *, top_n: int | None = None):
        del query
        self.rerank_calls += 1
        if self.rerank_fails:
            raise ModelUnavailable("rerank", "mock 配额用尽")
        return [RerankHit(index=i, score=0.9 - 0.05 * i) for i in range(len(documents))][
            : (top_n or len(documents))
        ]


class Recorder:
    """换掉 `nodes.retrieve`:只记「被问到哪几句」和「拿到的 config 里有哪些路」。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.kwargs: list[dict[str, Any]] = []
        self.degraded: list[str] = []

    async def __call__(self, query: str, ctx: TenantContext, **kwargs: Any) -> RetrievalResult:
        del ctx
        self.calls.append(query)
        self.kwargs.append(kwargs)
        routes = list(getattr(kwargs["config"], "routes", ()))
        hits = _hits()
        return RetrievalResult(
            query=query,
            hits=hits,
            stats={
                route: {
                    "count": len(hits),
                    "fingerprint": f"fp-{route}-{len(self.calls)}",
                    "latency_ms": 2,
                    "error": None,
                    "degraded": bool(self.degraded),
                }
                for route in routes
            },
            graph={"mentions": [], "lines": ["Milvus --USES--> 向量检索"]},
            degraded=list(self.degraded),
            fused_count=len(hits),
        )


# --------------------------------------------------------------------- 现场
async def _purge(factory: async_sessionmaker[AsyncSession]) -> None:
    async with factory() as session, session.begin():
        for stmt in (
            "delete from platform.trace_step where run_id in "
            "(select id from platform.trace_run where tenant_id = any(:ts))",
            "delete from platform.trace_run where tenant_id = any(:ts)",
            "delete from platform.sec_event where tenant_id = any(:ts)",
            "delete from platform.kb_tenant where id = any(:ts)",
        ):
            await session.execute(sa_text(stmt), {"ts": [TENANT]})


@pytest.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = make_engine(database_url())
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def site(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[GraphDeps, FlakyGateway, Recorder]]:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    await _purge(factory)
    async with factory() as session, session.begin():
        session.add(Tenant(id=TENANT, name="tnt-dg-a-名字"))
    gateway = FlakyGateway()
    recorder = Recorder()
    monkeypatch.setattr(nodes_mod, "retrieve", recorder)  # type: ignore[arg-type]
    deps = GraphDeps(
        factory=factory,  # type: ignore[arg-type]
        gateway=gateway,  # type: ignore[arg-type]
        kb_id=KB,
        trace=TraceWriter(factory),
    )
    yield deps, gateway, recorder
    await _purge(factory)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    """改 `KBMESH_FEATURE_*` → 重建 Settings:开关的唯一真值是环境,不是节点里的常量。"""

    def apply(**overrides: Any) -> Any:
        for key, value in overrides.items():
            text = {True: "true", False: "false"}.get(value, str(value))
            monkeypatch.setenv(f"KBMESH_FEATURE_{key}", text)
        clear_settings_cache()
        return get_settings()

    yield apply
    clear_settings_cache()


def _runtime(deps: GraphDeps) -> Runtime:
    return Runtime(deps, compile_graph(deps, InMemorySaver()))


async def _ask(deps: GraphDeps, *, thread: str = "th-dg", decision: Any = None) -> Outcome:
    """问一次并把改写确认回来,返回收尾那一条 Outcome。"""
    rt = _runtime(deps)
    resume_with = {"choice": 1} if decision is None else decision
    with use_context(CTX):
        await rt.ask(QUERY, thread)
        return await rt.resume(thread, resume_with)


async def _row(deps: GraphDeps, run_id: str) -> TraceRun:
    """`trace_run` 回读:降级那一列的真值只能从库里查,state 里的是过程。"""
    with use_context(CTX):
        return await deps.trace.get_run(run_id)  # type: ignore[union-attr]


async def _steps(deps: GraphDeps, run_id: str) -> list[Any]:
    with use_context(CTX):
        return await deps.trace.list_steps(run_id)  # type: ignore[union-attr]


def _bare_deps(gateway: Any = None, **overrides: Any) -> GraphDeps:
    """不落库的 deps:纯配置/纯拓扑的断言用,`factory` 在那类测试里根本不会被碰。"""
    return GraphDeps(factory=None, gateway=gateway, kb_id=KB, trace=None, **overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------- 能关的:关掉仍出答案
@pytest.mark.parametrize(
    ("flag", "gone"),
    [
        ("ROUTE_HEADING", "heading"),
        ("ROUTE_GRAPH", "graph"),
        ("CROSSCHECK", None),
        ("RERANK", None),
    ],
)
async def test_a_gain_switch_off_still_answers(
    site: tuple, env: Any, flag: str, gone: str | None
) -> None:
    deps, gateway, recorder = site
    deps = replace(deps, settings=env(**{flag: False}))
    final = await _ask(deps, thread=f"th-dg-off-{flag.lower()}")
    assert final.kind == "answered", f"关掉 {flag} 之后这次问答不该变成拒答"
    assert final.answer == ANSWER
    assert not final.degraded, "关开关是配置,不是降级 —— 混在一起面板就分不清了"
    row = await _row(deps, final.run_id)
    assert row.final_status == "answered" and row.degraded is False
    if gone is not None:
        assert gone not in recorder.kwargs[0]["config"].routes
    if flag == "RERANK":
        assert gateway.rerank_calls == 0, "开关写着关、模型还在被打,那是两份钱"


async def test_switches_are_read_at_call_time(site: tuple, env: Any) -> None:
    """同一份 deps 换个环境就该换行为,而且**不污染**别的 deps:`Flags` 读的是 Settings 实例。"""
    deps, _gateway, _recorder = site
    assert set(deps.effective_config().routes) == {"body", "heading", "dense", "graph"}
    off = replace(deps, settings=env(ROUTE_GRAPH=False))
    assert "graph" not in off.effective_config().routes
    assert "graph" in deps.effective_config().routes, "旧 deps 不该被新环境改掉"


async def test_disabled_route_gets_no_trace_step(site: tuple, env: Any) -> None:
    """关掉图谱路,`retrieve_graph` 这一步就不该存在 —— FR-205 不许出现一路「0 条且没跑」。"""
    deps, _gateway, _recorder = site
    deps = replace(deps, settings=env(ROUTE_GRAPH=False))
    final = await _ask(deps, thread="th-dg-nostep")
    stages = [one.stage for one in await _steps(deps, final.run_id)]
    assert "retrieve_graph" not in stages
    assert "retrieve_body" in stages


# --------------------------------------------------------------------- 能挂的:挂了要留痕
async def test_rerank_outage_degrades_but_still_answers(site: tuple) -> None:
    """精排是按量计费的增益项:配额用尽要照答,但**必须留痕**,不许静默按 RRF 原序截断。"""
    deps, gateway, _recorder = site
    gateway.rerank_fails = True
    final = await _ask(deps, thread="th-dg-rerank")
    assert final.kind == "answered"
    assert any("精排降级" in one for one in final.degraded)
    row = await _row(deps, final.run_id)
    assert row.degraded is True and row.final_status == "answered"


async def test_a_route_outage_from_retrieval_is_carried_not_swallowed(site: tuple) -> None:
    """检索层交上来的降级,编排的职责是**搬运 + 抬旗**,不是自己再判一次。

    `retrieve_dual` 把 `result.degraded` 并进 state,`Runtime._settle` 据此写 `trace_run.degraded`;
    丢掉这一笔,FR-701 的面板就永远看不见它 —— 而「哪一路挂了」恰恰是唯一还能定位的信息。
    """
    deps, _gateway, recorder = site
    recorder.degraded = ["稠密路降级:embed 不可用"]
    final = await _ask(deps, thread="th-dg-route")
    assert "稠密路降级:embed 不可用" in final.degraded
    row = await _row(deps, final.run_id)
    assert row.final_status == "answered" and row.degraded is True
    payload = {one.stage: one.payload for one in await _steps(deps, final.run_id)}
    assert payload["retrieve_dense"]["degraded"] is True


async def test_rewrite_outage_leaves_only_edited(site: tuple) -> None:
    """§10 没列改写节点(回填点)。这里定的方向:改写挂 → 候选清空、确认框只剩「自己改写」。

    它不影响能不能答,所以是降级而不是失败;但**不许静默回退原问** —— 回退等于跳过整个确认环节,
    FR-402 白做。所以候选为空时 `{"choice": 1}` 必须报错,只有 `edited` 走得通。

    第二条断言是本文件顺手钉下的实测:非法确认**不能污染这条线程**。坏值一旦被 `Command(resume)`
    写进 checkpoint,之后每次续跑都会重放它,合法的那次也白搭 —— 所以 `Runtime.resume` 必须在
    碰图之前先判完(`runtime.py` 的 `_precheck`)。
    """
    deps, gateway, _recorder = site
    gateway.fails.add("rewrite")
    rt = _runtime(deps)
    with use_context(CTX):
        first = await rt.ask(QUERY, "th-dg-rewrite")
        assert first.kind == "await_confirm"
        assert first.payload["rewrite_candidates"] == []
        assert first.payload["edited_allowed"] is True
        with pytest.raises(IllegalTransition):
            await rt.resume("th-dg-rewrite", {"choice": 1})
        final = await rt.resume("th-dg-rewrite", {"edited": "隔离设计的分区键到底怎么配的"})
        row = await deps.trace.get_run(final.run_id)  # type: ignore[union-attr]
        steps = await deps.trace.list_steps(final.run_id)  # type: ignore[union-attr]
    assert final.state["confirmed_query"] == "隔离设计的分区键到底怎么配的"
    assert any("改写候选降级" in one for one in final.degraded)
    assert final.kind == "answered"
    assert row.degraded is True
    rejected = [one for one in steps if one.payload.get("kind") == "resume_rejected"]
    assert len(rejected) == 1, "被拒的确认要留痕,但幂等键不许它堆成两条"


async def test_assess_outage_refuses_instead_of_guessing(site: tuple) -> None:
    """打分模型挂掉**不是**降级放行:层2 与层4 的判据就是它,没有分数就没有「够不够格回答」。

    方向:拒答 + `degraded` 留痕。既不静默放行(那等于把五层防御的两层关掉),也不抛异常(用户
    至少要知道「这次没答上」)。
    """
    deps, gateway, recorder = site
    gateway.fails.add("assess")
    final = await _ask(deps, thread="th-dg-assess")
    assert final.kind == "refused"
    assert final.reject_code == "no_evidence"
    assert final.answer == REFUSAL_TEXT
    assert any("相关性判定不可用" in one for one in final.degraded)
    row = await _row(deps, final.run_id)
    assert row.degraded is True
    assert recorder.calls == [CHOSEN], "拒答之后不许再烧一轮检索"
    assert gateway.chat_calls == 0, "证据都没定下来,不该已经花钱生成正文"


# --------------------------------------------------------------------- 不可降级项:挂了要停住
async def test_intent_outage_stops_the_run(site: tuple) -> None:
    """层1 挂 = 整条问答停住。§10 列为不可降级:判不出意图还继续查库,越权问题就拿到了证据。"""
    deps, gateway, recorder = site
    gateway.fails.add("route")
    with pytest.raises(ModelUnavailable):
        await _ask(deps, thread="th-dg-intent")
    assert recorder.calls == [], "门禁没判就检索,等于用问答接口探库"
    assert gateway.chat_calls == 0


async def test_generate_outage_is_an_error_not_an_empty_answer(site: tuple) -> None:
    """生成挂 → 异常交回调用方,`trace_run` **不收尾**,读回来是 #3 定的兜底 `error`。

    这里不能自作聪明地写一条 `refused`:「证据充分但模型没回话」和「证据不足」是两个数,把它们混
    进拒答口径,拒答率会在模型抖动的那天虚高。
    """
    deps, gateway, _recorder = site
    gateway.chat_fails = True
    rt = _runtime(deps)
    with use_context(CTX):
        first = await rt.ask(QUERY, "th-dg-generate")
        with pytest.raises(ModelUnavailable):
            await rt.resume("th-dg-generate", {"choice": 1})
        row = await deps.trace.get_run(first.run_id)  # type: ignore[union-attr]
    assert row.final_status == "error" and row.degraded is False


# --------------------------------------------------------------------- 不能关的:位置都不存在
@pytest.mark.parametrize(
    "forbidden",
    [
        "route_body",
        "body",
        "bm25_body",
        "intent",
        "route_intent",
        "checkpoint",
        "checkpointer",
        "persist",
        "trace",
        "tenancy",
        "tenant_filter",
    ],
)
def test_no_switch_exists_for_the_required_pillars(forbidden: str, env: Any) -> None:
    """§10 的必需项在代码里没有关的位置:要关,得先有人改 `Flags`。

    字段名黑名单比「有没有对应的 env 变量」更好测 —— 后者会把 `KBMESH_FEATURE_BODY=false` 这种
    「拼了一个不存在的名字」也算通过。所以第二条真的去设那个环境变量:它必须什么也不改。
    """
    assert forbidden not in set(Flags.model_fields)
    settings = env(**{forbidden.upper(): False})
    assert settings.flags.model_dump() == Flags().model_dump(), f"{forbidden} 竟然能关"


@pytest.mark.parametrize("flag", ["ROUTE_HEADING", "ROUTE_GRAPH"])
def test_body_and_dense_are_not_switchable(flag: str, env: Any) -> None:
    """路集合可以少掉增益路,不能少掉「有证据」那两条。"""
    off = _bare_deps(settings=env(**{flag: False}))
    assert {"body", "dense"} <= set(off.effective_config().routes)


async def test_no_checkpointer_means_no_resume_point(monkeypatch) -> None:
    """「把检查点持久化关掉」在图上不成立:中断可以冒出来,但**没有地方落盘**,确认无处可去。

    实测口径(别照着「`interrupt()` 会报错」写,那不是真的):无 checkpointer 时 `ainvoke` 照样把
    `__interrupt__` 交回调用方,只是快照不存在 —— 真正的可测依据是 `aget_state` 抛
    `ValueError("No checkpointer set")`,而 `Runtime.resume` 的第一步就是它。所以 §10 把检查点列为
    不可降级项靠的是运行时行为:想跑 FR-402/FR-403 就必须持久化,没有「先不持久化试试」这条路。
    """
    monkeypatch.setattr(nodes_mod, "retrieve", Recorder())  # 不该被走到,挡住以防万一
    deps = _bare_deps(gateway=FlakyGateway())
    rt = Runtime(deps, compile_graph(deps, None))
    key = f"{TENANT}:th-dg-nockpt"
    state = empty_state(query=QUERY, thread_id="th-dg-nockpt", thread_key=key, run_id="")
    with use_context(CTX):
        stopped = await rt.graph.ainvoke(state, {"configurable": {"thread_id": key}})
        assert "__interrupt__" in stopped, "图连中断都没给,那本文件其余部分验的就不是同一条链"
        with pytest.raises(ValueError, match="No checkpointer set"):
            await rt.resume("th-dg-nockpt", {"choice": 1})


async def test_every_stage_is_still_written_when_switches_are_off(site: tuple, env: Any) -> None:
    """开关不许让留痕变短:FR-701 的 stage 覆盖按「跑到了哪一步」算,不按「开了几路」。"""
    deps, _gateway, _recorder = site
    deps = replace(deps, settings=env(ROUTE_HEADING=False, ROUTE_GRAPH=False, RERANK=False))
    final = await _ask(deps, thread="th-dg-stages")
    steps = await _steps(deps, final.run_id)
    stages = {one.stage for one in steps}
    assert {"route_intent", "rewrite", "retrieve_body", "merge", "assess", "generate"} <= stages
    assert [one.seq for one in steps] == list(range(1, len(steps) + 1)), "发号不许留空洞"
    assert (await _row(deps, final.run_id)).final_status == "answered"
    assert set(ALL_NODES) == {
        "route_intent",
        "rewrite_query",
        "confirm_interrupt",
        "retrieve_dual",
        "dedupe",
        "rerank_fused",
        "assess_relevance",
        "retry_targeted",
        "refuse",
        "generate_cite",
    }


# --------------------------------------------------------------------- 降级与拒答是两个维度
async def test_degraded_and_refused_are_separate_columns(site: tuple) -> None:
    """FR-406:精排抖动那天「答了多少」不许突然掉一截 —— `degraded` 不是 `final_status` 的别名。

    顺带钉住收尾口径:`final_status` 只认 `FINAL_STATUSES` 那四个值,写进别的字符串一律
    `IllegalTransition`,否则「降级」就有了第二个藏身之处。
    """
    deps, gateway, _recorder = site
    gateway.rerank_fails = True
    final = await _ask(deps, thread="th-dg-answered-degraded")
    row = await _row(deps, final.run_id)
    assert row.degraded is True and row.final_status == "answered"

    gateway.score = 0.2  # 全部低于 θ_rel=0.35,层2 就一条不留
    refused = await _ask(deps, thread="th-dg-refused")
    assert (await _row(deps, refused.run_id)).final_status == "refused"
    assert refused.reject_code == "no_evidence"
    assert refused.answer == REFUSAL_TEXT
    with use_context(CTX), pytest.raises(IllegalTransition):
        await deps.trace.finish(refused.run_id, final_status="answered_but_please")
