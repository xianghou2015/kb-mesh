"""检索层落地验收:三路 + RRF + 融合 + 权威表回填,跑真 Milvus / 真 Neo4j / 真 PG。

模型是假的,原因和索引侧一样:结论必须可复现,验收不能烧钱碰运气。
**假向量用词哈希生成**,所以 dense 路在这里证明的是「查询向量确实进了 Milvus、过滤与
排序按契约工作」,不是「同义改写能召回」—— 后一条要真模型,见 §9 的模型侧验收。
"""
from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any

import pytest
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from conftest import database_url
from kbmesh.core.config import get_settings
from kbmesh.core.gateway import ModelUnavailable, RerankHit
from kbmesh.core.tenancy import worker_context
from kbmesh.db.base import make_engine
from kbmesh.db.models import Document, KnowledgeBase, Tenant
from kbmesh.db.repositories import ChunkRepo, GraphRepo
from kbmesh.ingest.chunking import ChunkPart, chunk_key
from kbmesh.ingest.graph_extract import entity_key as gk
from kbmesh.ingest.writer import rows_from_parts
from kbmesh.retrieval import graph_route as gr
from kbmesh.retrieval.contracts import Hit, RetrievalConfig
from kbmesh.retrieval.fusion import apply_rerank, backfill, retrieve
from kbmesh.retrieval.routes import ROUTES, run_route
from kbmesh.storage import milvus as mv
from kbmesh.storage import neo4j as n4

DIM = get_settings().embedding_dim
COLLECTION = get_settings().milvus_collection
TENANT_A, KB_A, DOC_A = "tnt-rt-a", "kb-rt-a", "doc-rt-a"
TENANT_B, KB_B, DOC_B = "tnt-rt-b", "kb-rt-b", "doc-rt-b"

#: 「拒答策略」只出现在标题里 —— 标题路与正文路必须能各自独立命中,才算两路
PAIRS_A: list[tuple[str, str]] = [
    ("检索 > 隔离设计", "Milvus 承载向量检索,稀疏路走 BM25。隔离设计靠租户分区键完成。"),
    ("检索 > 拒答策略", "低于阈值时系统不回答,并把这一轮记进审计流水。"),
    ("图谱 > 邻域扩展", "Neo4j 做受控跳数的邻域扩展,每一跳都要重新核对租户。"),
]
PAIRS_B: list[tuple[str, str]] = [
    ("检索 > 隔离设计", "别家租户也谈隔离设计,但这条内容不该被本租户检索到。"),
]

MILVUS_K, RETRIEVAL_K = gk("component", "Milvus"), gk("method", "向量检索")


def _tokens(text: str) -> set[str]:
    words = {w.lower() for w in _split_ascii(text)}
    cjk = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    bigrams = {"".join(cjk[i : i + 2]) for i in range(len(cjk) - 1)}
    return {t for t in words | bigrams if t}


def _split_ascii(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isascii() and ch.isalnum():
            buf.append(ch)
        elif buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


def fake_vector(text: str) -> list[float]:
    """词哈希向量:词面重叠 → 余弦相似。它只证明管线,不证明语义。"""
    vec = [0.0] * DIM
    for token in _tokens(text):
        bucket = int.from_bytes(token.encode()[:8].ljust(8, b"\0"), "big")
        vec[bucket % DIM] += 1.0 if bucket % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@dataclass
class Corpus:
    factory: async_sessionmaker
    driver: Any
    client: Any
    gateway: Any

    @property
    def ctx_a(self) -> Any:
        return worker_context(TENANT_A, KB_A)

    @property
    def ctx_b(self) -> Any:
        return worker_context(TENANT_B, KB_B)


class FakeGateway:
    """三个方法:embed(词哈希)、rerank(按词面重叠打分)、structured(点名)。"""

    def __init__(self) -> None:
        self.embed_calls: list[str] = []
        self.rerank_calls = 0
        self.rerank_fails = False
        self.mentions: list[Any] = [gr.MentionOut(label="Milvus", etype="component")]

    def embed(self, texts, *, purpose: str = "embed"):
        self.embed_calls.extend(texts)
        return [fake_vector(t) for t in texts], None

    def rerank(self, query: str, documents, *, top_n: int | None = None):
        self.rerank_calls += 1
        if self.rerank_fails:
            raise ModelUnavailable("rerank", "mock 配额用尽")
        q = _tokens(query)
        scored = sorted(
            ((-len(q & _tokens(doc)) / (len(q) or 1), index) for index, doc in enumerate(documents)),
            key=lambda pair: (pair[0], pair[1]),
        )
        return [RerankHit(index=i, score=1.0 + s) for s, i in scored][:(top_n or len(scored))]

    def structured(self, prompt: str, schema: type, **kwargs: Any):
        return gr.MentionsPayload(mentions=list(self.mentions)), None


async def _purge(factory: async_sessionmaker) -> None:
    ids = [TENANT_A, TENANT_B]
    statements = (
        "delete from platform.relation_edge where tenant_id = any(:ts)",
        "delete from platform.security_event where tenant_id = any(:ts)",
        "delete from platform.chunk where tenant_id = any(:ts)",
        "delete from platform.entity where tenant_id = any(:ts)",
        "delete from platform.document where tenant_id = any(:ts)",
        "delete from platform.kb where tenant_id = any(:ts)",
        "delete from platform.tenant where id = any(:ts)",
    )
    async with factory() as session, session.begin():
        for stmt in statements:
            await session.execute(sa_text(stmt), {"ts": ids})


def _parts(doc_id: str, pairs: list[tuple[str, str]]) -> list[ChunkPart]:
    out: list[ChunkPart] = []
    for position, (heading, text) in enumerate(pairs):
        part = ChunkPart(
            text=text,
            heading_path=heading,
            level=2,
            page_no=None,
            token_len=len(text),
            image_refs=[],
        )
        part.position = position
        part.chunk_key = chunk_key(doc_id, heading, text)
        out.append(part)
    return out


@pytest.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = make_engine(database_url())
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def corpus(engine: AsyncEngine) -> AsyncIterator[Corpus]:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    client = mv.get_client()
    driver = n4.get_driver()
    gateway = FakeGateway()
    await _purge(factory)
    for tenant_id, kb_id in ((TENANT_A, KB_A), (TENANT_B, KB_B)):
        n4.delete_tenant_subgraph(driver, worker_context(tenant_id, kb_id))

    async with factory() as session, session.begin():
        for tenant_id in (TENANT_A, TENANT_B):
            session.add(Tenant(id=tenant_id, name=f"fixture {tenant_id}"))
        await session.flush()
        for tenant_id, kb_id in ((TENANT_A, KB_A), (TENANT_B, KB_B)):
            session.add(
                KnowledgeBase(
                    id=kb_id,
                    tenant_id=tenant_id,
                    name="rt-kb",
                    embedding_model="fake",
                    chunk_policy={},
                )
            )
        await session.flush()
        for tenant_id, kb_id, doc_id in ((TENANT_A, KB_A, DOC_A), (TENANT_B, KB_B, DOC_B)):
            session.add(
                Document(
                    id=doc_id,
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    title=f"文档 {doc_id}",
                    source_uri="s3://bucket/rt.md",
                    mime="text/markdown",
                    sha256="0" * 64,
                    index_version=1,
                )
            )
        await session.flush()

        for tenant_id, kb_id, doc_id, pairs in (
            (TENANT_A, KB_A, DOC_A, PAIRS_A),
            (TENANT_B, KB_B, DOC_B, PAIRS_B),
        ):
            ctx = worker_context(tenant_id, kb_id)
            rows = rows_from_parts(
                ctx, _parts(doc_id, pairs), kb_id=kb_id, doc_id=doc_id, index_version=1
            )
            await ChunkRepo(ctx, session).insert_many_ignoring(rows)
            mv.upsert_rows(
                [
                    mv.to_row(
                        ctx,
                        pk=r["milvus_pk"],
                        chunk_id=r["id"],
                        doc_id=doc_id,
                        kb_id=kb_id,
                        page_no=None,
                        heading_path=r["heading_path"],
                        body_text=r["text"],
                        dense=fake_vector(r["text"]),
                    )
                    for r in rows
                ],
                client=client,
            )

        ctx_a = worker_context(TENANT_A, KB_A)
        anchor = (await ChunkRepo(ctx_a, session).list_for_doc(DOC_A))[0]
        await GraphRepo(ctx_a, session).upsert_entities(
            KB_A,
            [
                {"entity_key": MILVUS_K, "etype": "component", "label": "Milvus", "aliases": ["向量库"]},
                {"entity_key": RETRIEVAL_K, "etype": "method", "label": "向量检索", "aliases": []},
            ],
        )
        await GraphRepo(ctx_a, session).upsert_edges(
            KB_A,
            [
                {
                    "src_key": MILVUS_K,
                    "dst_key": RETRIEVAL_K,
                    "rtype": "USES",
                    "evidence": "Milvus 承载向量检索",
                    "chunk_id": anchor.id,
                    "confidence": 0.9,
                }
            ],
        )
        chunk_ids = [c.id for c in await ChunkRepo(ctx_a, session).list_for_doc(DOC_A)]

    n4.ensure_schema(driver)
    ctx_a = worker_context(TENANT_A, KB_A)
    n4.merge_entities(
        driver,
        ctx_a,
        KB_A,
        [
            {"entity_key": MILVUS_K, "etype": "component", "label": "Milvus", "aliases": ["向量库"]},
            {"entity_key": RETRIEVAL_K, "etype": "method", "label": "向量检索", "aliases": []},
        ],
    )
    n4.merge_chunkrefs(
        driver, ctx_a, KB_A, [{"chunk_id": c, "doc_id": DOC_A, "kb_id": KB_A} for c in chunk_ids]
    )
    n4.merge_relations(
        driver,
        ctx_a,
        KB_A,
        [
            {
                "src_key": MILVUS_K,
                "dst_key": RETRIEVAL_K,
                "rtype": "USES",
                "evidence": "Milvus 承载向量检索",
                "chunk_id": chunk_ids[0],
                "confidence": 0.9,
            }
        ],
    )
    n4.merge_mentions(
        driver,
        ctx_a,
        KB_A,
        [{"entity_key": MILVUS_K, "chunk_id": chunk_ids[0], "doc_id": DOC_A, "kb_id": KB_A}],
    )
    try:
        yield Corpus(factory=factory, driver=driver, client=client, gateway=gateway)
    finally:
        await _purge(factory)
        for tenant_id, kb_id in ((TENANT_A, KB_A), (TENANT_B, KB_B)):
            n4.delete_tenant_subgraph(driver, worker_context(tenant_id, kb_id))
        driver.close()


def _cfg(**over: Any) -> RetrievalConfig:
    base = RetrievalConfig(
        routes=ROUTES,
        top_k_route=20,
        top_k_fusion=20,
        top_n_context=8,
        rrf_k=60,
        use_rerank=False,
        graph_max_hops=2,
        graph_max_edges=30,
    )
    return replace(base, **over) if over else base


async def _retrieve(corpus: Corpus, query: str, ctx=None, **over: Any):
    config = over.pop("config", None)
    defer = over.pop("defer_rerank", False)
    return await retrieve(
        query,
        ctx or corpus.ctx_a,
        factory=corpus.factory,
        gateway=corpus.gateway,
        client=corpus.client,
        collection=over.pop("collection", COLLECTION),
        config=config or _cfg(**over),
        driver=over.pop("driver", None),
        kb_id=over.pop("kb_id", None),
        defer_rerank=defer,
    )


# --------------------------------------------------------------------- FR-201 / 202 / 203


def test_body_route_alone_finds_the_exact_keyword(corpus: Corpus) -> None:
    outcome = run_route("body", "隔离设计", corpus.ctx_a, top_k=10, client=corpus.client, name=COLLECTION)
    assert not outcome.degraded, outcome.error
    assert outcome.count >= 1, "正文路对精确关键词必须命中"
    assert "隔离设计" in outcome.hits[0].text


def test_heading_route_is_not_a_shadow_of_the_body_route(corpus: Corpus) -> None:
    """FR-202 + FR-205:「拒答」只在标题里,标题路命中而正文路命不中,才证明是独立一路。"""
    head = run_route("heading", "拒答", corpus.ctx_a, top_k=10, client=corpus.client, name=COLLECTION)
    body = run_route("body", "拒答", corpus.ctx_a, top_k=10, client=corpus.client, name=COLLECTION)
    assert head.count >= 1, "标题里的词必须能被标题路召回"
    assert "拒答" in head.hits[0].citation.heading_path
    assert all("拒答" not in h.text for h in body.hits), "正文路不该看见只在标题里出现的词"


def test_dense_route_uses_the_query_vector_it_was_given(corpus: Corpus) -> None:
    """FR-203 的管线部分:查询向量进了 Milvus,结果按相似度降序。"""
    vector = fake_vector("向量检索 BM25 稀疏")
    outcome = run_route(
        "dense",
        "向量检索 BM25 稀疏",
        corpus.ctx_a,
        vector=vector,
        top_k=10,
        client=corpus.client,
        name=COLLECTION,
    )
    assert outcome.count >= 1 and not outcome.degraded
    scores = [h.score for h in outcome.hits]
    assert scores == sorted(scores, reverse=True)
    with pytest.raises(ValueError, match="未知检索路"):
        run_route("nope", "x", corpus.ctx_a, client=corpus.client)


def test_dense_route_without_a_vector_degrades_instead_of_guessing(corpus: Corpus) -> None:
    outcome = run_route("dense", "向量检索", corpus.ctx_a, vector=None, client=corpus.client)
    assert outcome.degraded and "查询向量" in (outcome.error or "")


# --------------------------------------------------------------------- FR-204 / 205 / 210 / 211


async def test_single_route_mode_is_selectable_per_request(corpus: Corpus) -> None:
    result = await _retrieve(corpus, "隔离设计", config=_cfg(routes=("body",)))
    assert set(result.stats) == {"body"}
    assert result.hits and all(h.source_routes == {"body"} for h in result.hits)


async def test_three_routes_fuse_into_one_hit_per_chunk(corpus: Corpus) -> None:
    result = await _retrieve(corpus, "隔离设计 租户")
    ids = result.chunk_ids
    assert len(ids) == len(set(ids)), "融合后同一 chunk 出现两次就是没去重"
    multi = [h for h in result.hits if len(h.source_routes) > 1]
    assert multi, "同一个词在三路里至少该有两路命中同一切片"
    assert all(r >= 1 for r in multi[0].ranks.values())


async def test_route_stats_report_every_route_independently(corpus: Corpus) -> None:
    result = await _retrieve(corpus, "邻域扩展 租户")
    assert set(result.stats) == set(ROUTES)
    for name, stat in result.stats.items():
        assert {"count", "fingerprint", "latency_ms", "error", "degraded"} <= set(stat), name


async def test_a_broken_collection_degrades_without_failing_the_request(corpus: Corpus) -> None:
    """§10:集合不存在时三路各自降级,请求仍然返回,并且降级留痕。"""
    result = await _retrieve(corpus, "隔离设计", collection="kb_no_such_collection")
    assert all(stat["degraded"] for stat in result.stats.values())
    assert result.degraded, "降级要留痕,不许静默"


async def test_rerank_overrides_the_rrf_order_and_truncates_by_score(corpus: Corpus) -> None:
    """FR-210 统一精排 + FR-211 按相关性截断。"""
    result = await _retrieve(
        corpus, "审计 流水", config=_cfg(routes=("body",), use_rerank=True, top_n_context=2), kb_id=KB_A
    )
    assert corpus.gateway.rerank_calls == 1
    assert len(result.hits) <= 2
    assert "rerank" in result.hits[0].route_scores
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)


async def test_rerank_outage_falls_back_to_fusion_order(corpus: Corpus) -> None:
    corpus.gateway.rerank_fails = True
    result = await _retrieve(corpus, "隔离设计", config=_cfg(routes=("body",), use_rerank=True))
    assert any("精排降级" in d for d in result.degraded)
    assert result.hits, "精排挂了也要按 RRF 顺序给出上下文"


async def test_defer_rerank_hands_the_full_candidate_set_to_the_caller(corpus: Corpus) -> None:
    """#6 的接缝(§7.1 `rerank_fused` 节点):精排可以交给编排层,但融合逻辑不许有第二套。

    断言的是三件事:一次网络调用都没发、候选没被 `top_n_context` 截断、`route_scores` 里干净。
    少了这三条,编排层拿到的就是「已经被截断过的 8 条」,θ_rel 打分与「精排前后名次」的 trace
    (FR-701)都没有对象可写。
    """
    cfg = _cfg(routes=("body", "heading"), use_rerank=True, top_n_context=1)
    deferred = await _retrieve(corpus, "检索", config=cfg, defer_rerank=True)
    assert corpus.gateway.rerank_calls == 0, "defer 就是把精排原封不动交回调用方"
    assert len(deferred.hits) > 1
    assert all("rerank" not in h.route_scores for h in deferred.hits)
    assert not any("精排" in d for d in deferred.degraded)

    truncated = await _retrieve(corpus, "检索", config=cfg)
    assert len(truncated.hits) == 1, "同一份候选,不 defer 就被上下文预算截掉"
    assert corpus.gateway.rerank_calls == 1

    ranked, note = apply_rerank("检索", deferred.hits, corpus.gateway, top_n=1, enabled=True)
    assert note is None and corpus.gateway.rerank_calls == 2
    assert len(ranked) == 1 and "rerank" in ranked[0].route_scores


# --------------------------------------------------------------------- 图谱路(真 Neo4j)


async def test_graph_route_lands_on_a_citable_chunk(corpus: Corpus) -> None:
    result = await _retrieve(
        corpus,
        "Milvus 怎么做向量检索",
        config=_cfg(routes=("graph",)),
        driver=corpus.driver,
        kb_id=KB_A,
    )
    graph = [h for h in result.hits if "graph" in h.source_routes]
    assert graph, "图谱路一个命中都没有,说明五步里断了链"
    assert all(h.evidence for h in graph), "FR-209:图谱命中必须带原文依据"
    assert all(h.text for h in graph), "回落后的正文必须来自权威表"
    assert result.graph and result.graph["lines"][0].startswith("Milvus --USES-->")


def test_graph_route_walks_only_inside_the_tenant(corpus: Corpus) -> None:
    assert n4.expand_neighborhood(corpus.driver, corpus.ctx_b, [MILVUS_K]) == []


def test_hops_outside_the_whitelist_are_refused(corpus: Corpus) -> None:
    with pytest.raises(n4.IllegalGraphIdentifier):
        n4.expand_neighborhood(corpus.driver, corpus.ctx_a, [MILVUS_K], hops=7)


async def test_a_graph_pointer_to_a_missing_chunk_is_dropped(corpus: Corpus) -> None:
    """投影里有条引用、权威表没有这条切片 —— 它不能进上下文(FR-209 + §4.1)。"""
    ghost = Hit(chunk_id="chk-ghost", score=0.9, source_routes={"graph"}, evidence="边依据")
    kept, dropped = await backfill([ghost], corpus.ctx_a, corpus.factory)
    assert kept == [] and dropped == ["chk-ghost"]


async def test_a_malicious_label_cannot_delete_the_graph(corpus: Corpus) -> None:
    """§6.4:label 走参数、类型走白名单,拼不进 `DETACH DELETE`。"""
    before = _counts(corpus.driver)
    corpus.gateway.mentions = [gr.MentionOut(label="REL] MATCH (n) DETACH DELETE n; //", etype="component")]
    await _retrieve(corpus, "删掉整张图", config=_cfg(routes=("graph",)), driver=corpus.driver)
    assert _counts(corpus.driver) == before


# --------------------------------------------------------------------- 租户四道防线(检索侧)


def test_the_other_tenant_is_invisible_on_every_route(corpus: Corpus) -> None:
    """同一句「隔离设计」:B 租户三路都只能看见自己的切片。"""
    for route in ROUTES:
        outcome = run_route(
            route,
            "隔离设计",
            corpus.ctx_b,
            vector=fake_vector("隔离设计"),
            top_k=10,
            client=corpus.client,
            name=COLLECTION,
        )
        assert outcome.count >= 1, route
        for hit in outcome.hits:
            assert hit.citation.doc_id == DOC_B, f"{route} 路漏了租户过滤"


def test_kb_scope_narrows_results_within_the_tenant(corpus: Corpus) -> None:
    ctx = worker_context(TENANT_A, "kb-not-seeded")
    outcome = run_route("body", "隔离设计", ctx, top_k=10, client=corpus.client, name=COLLECTION)
    assert outcome.hits == [], "令牌 kb_scope 之外的切片不能出现"


def _counts(driver) -> dict[str, int]:
    with n4.session(driver) as s:
        out: dict[str, int] = {}
        for label in ("Entity", "ChunkRef", "Document"):
            rec = s.run(
                f"MATCH (n:{label}) WHERE n.tenant_id = $tenant RETURN count(n) AS c",
                tenant=TENANT_A,
            ).single()
            out[label.lower()] = int(rec["c"])
        return out
