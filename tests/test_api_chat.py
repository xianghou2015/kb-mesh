"""#7 API 验收:问答两条腿的 SSE 契约 + 越权(真库断言)。

跑真 PG(检查点在 `lg`,trace/audit 在 `platform`),模型与检索按 orchestration 的同一套桩
替换 —— 这一层要验的是「HTTP/SSE 通道的形状与隔离」,不是模型行为,也不是引擎召回。
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from conftest import database_url, purge_tenant_data
from kbmesh.agents import nodes as nodes_mod
from kbmesh.agents.nodes import GraphDeps
from kbmesh.agents.runtime import SaverHandle
from kbmesh.agents.schemas import IntentOut, RelevanceOut, RewriteOut
from kbmesh.agents.trace import TraceWriter
from kbmesh.api import sse
from kbmesh.api.app import create_app
from kbmesh.core.config import clear_settings_cache
from kbmesh.core.gateway import ChatResult, ModelUnavailable, RerankHit, Usage
from kbmesh.core.tenancy import ROLE_READER, ROLE_TENANT_ADMIN, TenantContext
from kbmesh.db.base import make_engine, set_engine
from kbmesh.db.models import SecurityEvent, Tenant, TraceRun
from kbmesh.retrieval.contracts import ROUTES, Citation, Hit
from kbmesh.retrieval.fusion import RetrievalResult

TENANT_A, KB_A = "tnt-api-a", "kb-api-a"
TENANT_B, KB_B = "tnt-api-b", "kb-api-b"
CTX_A = TenantContext(
    tenant_id=TENANT_A, user_id="usr-api-a", roles=(ROLE_READER,), kb_scope=frozenset({KB_A})
)
CTX_B = TenantContext(
    tenant_id=TENANT_B, user_id="usr-api-b", roles=(ROLE_READER,), kb_scope=frozenset({KB_B})
)
CTX_ADMIN = TenantContext(
    tenant_id=TENANT_A, user_id="usr-api-admin", roles=(ROLE_TENANT_ADMIN,), kb_scope=frozenset()
)

BODY = "Milvus 承载向量检索,隔离设计靠租户分区键完成。"
GRAPH = "Neo4j 做受控跳数的邻域扩展,每一跳都重新核对租户。"
QUERY = "隔离 设计 怎么做"
CHOSEN = "隔离设计怎么做"


def _hit(index: int, text: str, route: str = "body") -> Hit:
    cid = f"chunk-api-{index}"
    return Hit(
        chunk_id=cid,
        score=0.5,
        source_routes={route},
        citation=Citation(doc_id="doc-api", chunk_id=cid, heading_path="检索 > 隔离设计", title="设计册"),
        text=text,
        route_scores={route: 0.5},
    )


def _hits() -> list[Hit]:
    return [_hit(1, BODY), _hit(2, GRAPH, "graph")]


def _scores(hits: list[Hit], score: float) -> dict[str, object]:
    return {
        "scores": [{"chunk_id": h.chunk_id, "score": score, "gap": ""} for h in hits],
        "missing_info": [],
    }


class FakeGateway:
    """与 orchestration 同形的替身:按 purpose 分派;`rerank_fails` 撑起 degraded 事件。"""

    def __init__(self, **script: object) -> None:
        self.rerank_fails = False
        self.answer_text = str(script.pop("answer_text", "隔离靠分区键完成 [1]。"))
        self.script: dict[str, list[object]] = {k: list(v) for k, v in script.items()}  # type: ignore[arg-type]
        self._cursor = 0

    def _next(self, key: str) -> object:
        queue = self.script.get(key) or []
        if not queue:
            return None
        value = queue[min(self._cursor, len(queue) - 1)]
        self._cursor += 1
        return value

    def embed(self, texts, *, purpose: str = "embed"):
        return [[0.0] * 8 for _ in texts], Usage(model="fake-embed")

    def rerank(self, query: str, documents, *, top_n: int | None = None):
        if self.rerank_fails:
            raise ModelUnavailable("rerank", "mock 配额用尽")
        return [RerankHit(index=i, score=0.9 - 0.05 * i) for i in range(len(documents))][
            : (top_n or len(documents))
        ]

    def chat(self, prompt: str, *, purpose: str = "generate", **kwargs: object) -> ChatResult:
        return ChatResult(
            text=self.answer_text,
            usage=Usage(token_in=120, token_out=40, latency_ms=9, model="fake-llm"),
        )

    def structured(self, prompt: str, schema: type, *, purpose: str = "route", system: str = ""):
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
            return RelevanceOut.model_validate(self._next("assess") or _scores([], 0.9)), usage
        raise AssertionError(f"不该问这一类结构化输出:{schema!r}")


class StubRetrieve:
    """换掉 `nodes.retrieve`:SSE 层只消费「候选集 + stats + 降级清单」,不给它真引擎。"""

    def __init__(self, *, degraded: list[str] | None = None) -> None:
        self.calls: list[str] = []
        self.degraded = list(degraded or [])

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._run(str(args[0]))

    async def _run(self, query: str) -> RetrievalResult:
        self.calls.append(query)
        hits = _hits()
        return RetrievalResult(
            query=query,
            hits=hits,
            stats={
                route: {
                    "count": len(hits),
                    "fingerprint": f"fp-{route}",
                    "latency_ms": 3,
                    "error": None,
                    "degraded": False,
                }
                for route in ROUTES
            },
            graph={"mentions": [], "lines": []},
            degraded=list(self.degraded),
            dropped=[],
            fused_count=len(hits),
        )


def _rewrite() -> list[dict[str, str]]:
    return [{"text": CHOSEN, "focus": "隔离", "rationale": "口语化"}]


def bearer(ctx: TenantContext) -> dict[str, str]:
    from kbmesh.security.tokens import issue_access_token

    return {"Authorization": f"Bearer {issue_access_token(ctx)}"}


async def sse_events(client: httpx.AsyncClient, method: str, url: str, **kw: object):
    """发请求并解析 SSE 帧序列;200 的断言放在这里,调用方只看事件。"""
    async with client.stream(method, url, **kw) as resp:  # type: ignore[arg-type]
        assert resp.status_code == 200, await resp.aread()
        body = (await resp.aread()).decode()
    frames = sse.parse_sse(body)
    assert frames, f"一个事件都没流出:{body!r}"
    return frames


async def _purge(engine: AsyncEngine) -> None:
    await purge_tenant_data(engine)


@pytest.fixture
async def rig(monkeypatch: pytest.MonkeyPatch):
    """环境 → 测试库 → 真检查点引擎 + 桩网关的应用。"""
    monkeypatch.setenv("KBMESH_ENV", "test")
    clear_settings_cache()
    engine = make_engine(database_url())
    set_engine(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    await _purge(engine)
    async with factory() as session, session.begin():
        for tenant_id in (TENANT_A, TENANT_B):
            session.add(Tenant(id=tenant_id, name=f"{tenant_id}-名字"))
    gateway = FakeGateway()
    stub = StubRetrieve()
    monkeypatch.setattr(nodes_mod, "retrieve", stub)
    deps = GraphDeps(factory=factory, gateway=gateway, kb_id=KB_A, trace=TraceWriter(factory))
    handle = await SaverHandle.open_sync().open()
    app = create_app(deps=deps, saver=handle)
    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                yield SimpleNamespace(
                    app=app, gateway=gateway, stub=stub, client=client, factory=factory
                )
    finally:
        await handle.close()
        await engine.dispose()
        await _purge(engine)
        clear_settings_cache()


async def _stop_at_confirm(rig, thread: str):
    """把会话停在 await_confirm;候选与相关性都按「可答」布好。"""
    rig.gateway.script["rewrite"] = [_rewrite()]
    rig.gateway.script["assess"] = [_scores(_hits(), 0.8)]
    return await sse_events(
        rig.client,
        "POST",
        "/api/ask",
        json={"query": QUERY, "thread_id": thread},
        headers=bearer(CTX_A),
    )


# --------------------------------------------------------------------- SSE 契约
async def test_ask_streams_route_rewrite_then_awaits_confirm(rig):
    events = await _stop_at_confirm(rig, "th-ask-1")
    names = [event for event, _ in events]
    assert names == ["route", "rewrite_candidates", "await_confirm"]
    assert events[0][1]["intent"]["verdict"] == "answerable"
    assert [c["text"] for c in events[1][1]["candidates"]] == [CHOSEN]
    payload = events[2][1]
    assert payload["kind"] == "await_confirm"
    assert payload["expires_at"], "TTL 到期时间必须可渲染"
    # 库里:run 已开、尚未收尾 —— final_status 的 server_default 是兜底口径(#3)
    async with rig.factory() as session:
        row = (
            (await session.execute(sa.select(TraceRun).where(TraceRun.tenant_id == TENANT_A)))
            .scalars()
            .one()
        )
    assert row.final_status == "error"


async def test_resume_streams_retrieval_token_citations_done(rig):
    thread = "th-resume-1"
    await _stop_at_confirm(rig, thread)
    events = await sse_events(
        rig.client,
        "POST",
        f"/api/threads/{thread}/resume",
        json={"choice": 1},
        headers=bearer(CTX_A),
    )
    names = [event for event, _ in events]
    assert names[0] == "retrieval_done"
    assert "degraded" not in names
    assert names[-3:] == ["token", "citations", "done"]
    done = events[-1][1]
    assert done["kind"] == "answered"
    assert done["answer"] == "隔离靠分区键完成 [1]。"
    assert done["citations"][0]["chunk_id"] == "chunk-api-1"
    assert done["replayed"] is False
    async with rig.factory() as session:
        row = (
            (
                await session.execute(
                    sa.select(TraceRun).where(TraceRun.thread_id == f"{TENANT_A}:{thread}")
                )
            )
            .scalars()
            .one()
        )
    assert row.final_status == "answered"
    assert row.token_in > 0 and row.token_out > 0, "usage 累加键不能在流式合并里丢"


async def test_online_eval_hook_receives_terminal_payload(rig):
    captured: list[dict] = []
    rig.app.state.graph_deps.online_eval = captured.append
    thread = "th-online-1"
    await _stop_at_confirm(rig, thread)
    await sse_events(
        rig.client,
        "POST",
        f"/api/threads/{thread}/resume",
        json={"choice": 1},
        headers=bearer(CTX_A),
    )
    assert len(captured) == 1
    payload = captured[0]
    assert payload["tenant_id"] == TENANT_A
    assert payload["final_status"] == "answered"
    assert payload["answer"].endswith("[1]。")
    assert payload["run_id"]


async def test_runtime_case_runner_bridges_graph_to_eval_outcome(rig):
    from kbmesh.db.models import EvalCase
    from kbmesh.eval.runtime_case import RuntimeCaseRunner

    rig.gateway.script["rewrite"] = [_rewrite()]
    rig.gateway.script["assess"] = [_scores(_hits(), 0.8)]
    case = EvalCase(
        id="case-bridge-1",
        set_id="set-not-persisted",
        question=QUERY,
        gold_refs=[h.chunk_id for h in _hits()],
        group_key="factual_single",
    )
    runner = RuntimeCaseRunner(
        deps=rig.app.state.graph_deps, handle=rig.app.state.saver_handle, ctx=CTX_A
    )
    outcome = await runner(case)
    assert outcome.final_status == "answered"
    assert outcome.cite_ids, "评估要能核对引用映射"
    assert outcome.retrieved_ids == [h.chunk_id for h in _hits()]
    assert outcome.user_choice.get("choice") == 1
    assert outcome.token_in > 0 and outcome.token_out > 0


async def test_resume_records_estimated_cost_and_price_version(rig, monkeypatch):
    import kbmesh.agents.runtime as runtime_mod
    from kbmesh.observability.pricing import PRICE_TABLE_VERSION

    monkeypatch.setattr(runtime_mod, "estimate_cost", lambda *a, **k: 1.25)
    thread = "th-cost-1"
    await _stop_at_confirm(rig, thread)
    await sse_events(
        rig.client,
        "POST",
        f"/api/threads/{thread}/resume",
        json={"choice": 1},
        headers=bearer(CTX_A),
    )
    async with rig.factory() as session:
        row = (
            (
                await session.execute(
                    sa.select(TraceRun).where(TraceRun.thread_id == f"{TENANT_A}:{thread}")
                )
            )
            .scalars()
            .one()
        )
    assert row.cost_usd == 1.25, "收尾必须按单价表估成本,不能永远写 0"
    assert row.config_snapshot["price_table_version"] == PRICE_TABLE_VERSION


async def test_degraded_event_when_rerank_fails(rig):
    thread = "th-degrade-1"
    await _stop_at_confirm(rig, thread)
    rig.gateway.rerank_fails = True
    events = await sse_events(
        rig.client,
        "POST",
        f"/api/threads/{thread}/resume",
        json={"choice": 1},
        headers=bearer(CTX_A),
    )
    names = [event for event, _ in events]
    assert "degraded" in names, "重排失败必须以独立事件出现,不是只塞进 done"
    assert names[-1] == "done"
    assert events[-1][1]["kind"] == "answered"  # 降级不停服:回退融合序继续生成
    assert events[-1][1]["degraded"]


async def test_resume_after_done_replays_without_rerun(rig):
    thread = "th-replay-1"
    await _stop_at_confirm(rig, thread)
    await sse_events(
        rig.client,
        "POST",
        f"/api/threads/{thread}/resume",
        json={"choice": 1},
        headers=bearer(CTX_A),
    )
    done_calls = len(rig.stub.calls)
    events = await sse_events(
        rig.client,
        "POST",
        f"/api/threads/{thread}/resume",
        json={"choice": 1},
        headers=bearer(CTX_A),
    )
    names = [event for event, _ in events]
    assert names == ["token", "citations", "done"]
    assert events[-1][1]["kind"] == "replay"
    assert events[-1][1]["replayed"] is True
    assert len(rig.stub.calls) == done_calls, "重放不许再跑检索"


async def test_resume_after_ttl_expiry_returns_expired(rig):
    thread = "th-expire-1"
    await _stop_at_confirm(rig, thread)
    # TTL 判定读 deps.settings;把阈值压到 0,「停过一秒」就算过期
    rig.app.state.graph_deps.settings.pending_confirm_ttl_min = 0
    events = await sse_events(
        rig.client,
        "POST",
        f"/api/threads/{thread}/resume",
        json={"choice": 1},
        headers=bearer(CTX_A),
    )
    assert [event for event, _ in events] == ["done"]
    assert events[-1][1]["kind"] == "expired"
    # 检查点已删:再 resume 是「没有这个会话」,409 而不是新的问答
    resp = await rig.client.post(
        f"/api/threads/{thread}/resume", json={"choice": 1}, headers=bearer(CTX_A)
    )
    assert resp.status_code == 409


async def test_stream_survives_observability_outage(rig):
    """观测通道拔掉(enabled=False):SSE 照常走完全程 —— 分通道互不阻塞的形态。"""
    rig.app.state.graph_deps.trace.enabled = False
    events = await _stop_at_confirm(rig, "th-obs-1")
    assert [event for event, _ in events] == ["route", "rewrite_candidates", "await_confirm"]
    async with rig.factory() as session:
        rows = (
            (await session.execute(sa.select(TraceRun).where(TraceRun.tenant_id == TENANT_A)))
            .scalars()
            .all()
        )
    assert rows == [], "观测关闭时不得留下 run 行"


# --------------------------------------------------------------------- 越权
async def test_cross_tenant_resume_is_403_and_audited(rig):
    thread = "th-cross-1"
    await _stop_at_confirm(rig, thread)
    resp = await rig.client.post(
        f"/api/threads/{thread}/resume", json={"choice": 1}, headers=bearer(CTX_B)
    )
    assert resp.status_code == 403
    # 防线④:真数据断言,不是看函数签名
    async with rig.factory() as session:
        rows = (
            (
                await session.execute(
                    sa.select(SecurityEvent).where(
                        SecurityEvent.tenant_id == TENANT_B,
                        SecurityEvent.event == "cross_tenant_read",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert rows, "越权 resume 必须留痕"


async def test_ask_with_out_of_scope_kb_is_403(rig):
    scoped = TenantContext(
        tenant_id=TENANT_A, user_id="usr-api-s", roles=(ROLE_READER,), kb_scope=frozenset({KB_B})
    )
    resp = await rig.client.post(
        "/api/ask",
        json={"query": QUERY, "thread_id": "th-scope-1", "kb_id": KB_A},
        headers=bearer(scoped),
    )
    assert resp.status_code == 403
    async with rig.factory() as session:
        rows = (
            (
                await session.execute(
                    sa.select(SecurityEvent).where(SecurityEvent.event == "kb_scope_denied")
                )
            )
            .scalars()
            .all()
        )
    assert rows


async def test_thread_listing_is_scoped_and_bare(rig):
    await _stop_at_confirm(rig, "th-list-1")
    resp = await rig.client.get("/api/threads", headers=bearer(CTX_A))
    assert resp.status_code == 200
    rows = resp.json()
    assert [row["thread_id"] for row in rows] == ["th-list-1"]
    assert ":" not in rows[0]["thread_id"], "租户前缀是物理键,不许外泄到接口层"
    resp = await rig.client.get("/api/threads", headers=bearer(CTX_B))
    assert resp.status_code == 200 and resp.json() == []


async def test_maintenance_expire_stale_requires_admin(rig):
    thread = "th-sweep-1"
    await _stop_at_confirm(rig, thread)
    # 扫描只处理「已过期」:默认 30 min 的窗口里刚停下的会话不该被清,和 resume 的懒判同一口径。
    rig.app.state.graph_deps.settings.pending_confirm_ttl_min = 0
    resp = await rig.client.post("/api/maintenance/expire-stale", headers=bearer(CTX_A))
    assert resp.status_code == 403
    resp = await rig.client.post("/api/maintenance/expire-stale", headers=bearer(CTX_ADMIN))
    assert resp.status_code == 200
    assert f"{TENANT_A}:{thread}" in resp.json()["expired"]
