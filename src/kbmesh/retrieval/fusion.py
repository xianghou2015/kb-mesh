"""检索门面:三路 + 图谱路 → RRF → 精排 → 权威表回填(FR-204 / FR-210 / FR-211)。

两条贯穿全模块的规矩:

1. **Postgres 是权威,Milvus/Neo4j 是投影。** 融合完的每条命中都要回 PG 领一次正文:
   领不到正文的命中(切片已删、图上的孤儿引用)直接丢,不进上下文。
   图谱路尤其依赖这一步 —— 它给出的只有 chunk 指针,原文得从权威表取(FR-209)。
2. **同步客户端不进事件循环。** pymilvus / neo4j / gateway 都是阻塞的,统一 `asyncio.to_thread`,
   和 §5.4 三段式同一个理由。
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kbmesh.core.config import Settings, get_settings
from kbmesh.core.gateway import Gateway, ModelUnavailable
from kbmesh.core.tenancy import TenantContext
from kbmesh.db.models import Document
from kbmesh.db.repositories import ChunkRepo
from kbmesh.ingest.writer import indexed_body
from kbmesh.retrieval.contracts import Citation, Hit, RetrievalConfig, RouteOutcome
from kbmesh.retrieval.graph_route import run_graph_route
from kbmesh.retrieval.routes import ROUTES, route_stats, run_route
from kbmesh.retrieval.rrf import fuse


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit] = field(default_factory=list)
    #: FR-205:逐路候选数、命中集合指纹、延迟、失败原因
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: FR-206:图谱五步的中间结果,断点查看与 trace_step 都用它
    graph: dict[str, Any] | None = None
    degraded: list[str] = field(default_factory=list)
    #: 被权威表挡下的命中(投影里还有一条 PG 已删的切片)
    dropped: list[str] = field(default_factory=list)
    latency_ms: int = 0
    fused_count: int = 0

    @property
    def chunk_ids(self) -> list[str]:
        return [h.chunk_id for h in self.hits]


def _config(config: RetrievalConfig | None, settings: Settings, *, graph: bool) -> RetrievalConfig:
    base = config or RetrievalConfig(
        routes=ROUTES,
        top_k_route=settings.top_k_route,
        top_k_fusion=settings.top_k_fusion,
        top_n_context=settings.top_n_context,
        rrf_k=settings.rrf_k,
        use_rerank=settings.flags.rerank,
        graph_max_hops=settings.graph_max_hops,
        graph_max_edges=settings.graph_max_edges,
        theta_rel=settings.theta_rel,
    )
    #: 图谱路要有驱动才开得起来;没图就把 graph 从路集合里摘掉,别留一条注定降级的路
    if graph and "graph" not in base.routes:
        return base.with_graph(True)
    if not graph and "graph" in base.routes:
        return replace(base, routes=tuple(r for r in base.routes if r != "graph"))
    return base


async def _vector_route(
    route: str,
    query: str,
    ctx: TenantContext,
    *,
    vector: Sequence[float] | None,
    top_k: int,
    client,
    name: str | None,
) -> RouteOutcome:
    return await asyncio.to_thread(
        run_route, route, query, ctx, vector=vector, top_k=top_k, client=client, name=name
    )


async def backfill(
    hits: Sequence[Hit],
    ctx: TenantContext,
    factory: async_sessionmaker[AsyncSession],
    *,
    kb_id: str | None = None,
) -> tuple[list[Hit], list[str]]:
    """回权威表领正文与引用;领不到的命中丢掉并记名(FR-209 / 投影不可信)。"""
    if not hits:
        return [], []
    async with factory() as session:
        repo = ChunkRepo(ctx, session)
        rows = {c.id: c for c in await repo.by_ids([h.chunk_id for h in hits])}
        titles: dict[str, str] = {}
        if rows:
            doc_ids = sorted({c.doc_id for c in rows.values()})
            titles = {
                d.id: d.title
                for d in (
                    await session.execute(select(Document).where(Document.id.in_(doc_ids)))
                ).scalars()
            }
    kept: list[Hit] = []
    dropped: list[str] = []
    for hit in hits:
        chunk = rows.get(hit.chunk_id)
        if chunk is None or (kb_id and chunk.kb_id != kb_id):
            dropped.append(hit.chunk_id)
            continue
        hit.text = indexed_body(chunk)
        hit.citation = Citation(
            doc_id=chunk.doc_id,
            chunk_id=chunk.id,
            heading_path=chunk.heading_path,
            page_no=chunk.page_no,
            title=titles.get(chunk.doc_id, ""),
        )
        kept.append(hit)
    return kept, dropped


def apply_rerank(
    query: str, hits: list[Hit], gateway: Gateway | None, *, top_n: int, enabled: bool
) -> tuple[list[Hit], str | None]:
    """精排:cross-encoder 只看权威表正文,RRF 名次就此被覆盖(FR-210「统一精排」)。

    开关来自 `Flags.rerank`(§10 降级矩阵):关掉时按 RRF 顺序截断,而不是报错 ——
    精排档位是按量计费的,大促或超预算时要有这条退路。

    公开是因为编排层的 `rerank_fused` 节点复用同一个函数(§7.1):精排逻辑两份迟早会漂,
    漂出来的差别只有跑评估时才看得见。挂掉时同样返回原序 + 一条 note,不抛。
    """
    if not enabled:
        return hits[:top_n], None
    if gateway is None:
        return hits[:top_n], "精排跳过:未配置模型"
    if not hits:
        return [], None
    try:
        ranked = gateway.rerank(query, [h.text for h in hits], top_n=top_n)
    except ModelUnavailable as exc:
        return hits[:top_n], f"精排降级:{exc}"
    out: list[Hit] = []
    for one in ranked:
        if 0 <= one.index < len(hits):
            hit = hits[one.index]
            hit.score = float(one.score)
            hit.route_scores["rerank"] = float(one.score)
            out.append(hit)
    return out, None


async def retrieve(
    query: str,
    ctx: TenantContext,
    *,
    factory: async_sessionmaker[AsyncSession],
    gateway: Gateway | None = None,
    driver=None,
    client=None,
    collection: str | None = None,
    config: RetrievalConfig | None = None,
    kb_id: str | None = None,
    defer_rerank: bool = False,
) -> RetrievalResult:
    """一次检索的全部工作。入参只有 `ctx`:租户与 kb_scope 由它带入,函数签名里没有 tenant_id。

    `defer_rerank=True` 把精排与 `top_n_context` 截断都交回调用方(§7.1 的 `rerank_fused` 节点
    要拿精排前的完整候选集打分与留痕),返回的就是融合 + 回表后的 `top_k_fusion` 条。
    """
    settings = get_settings()
    cfg = _config(config, settings, graph=driver is not None)
    started = time.perf_counter()
    degraded: list[str] = []

    vector: Sequence[float] | None = None
    if "dense" in cfg.routes:
        if gateway is None:
            degraded.append("dense 路降级:未配置模型")
        else:
            try:
                vectors, _usage = await asyncio.to_thread(gateway.embed, [query])
                vector = vectors[0] if vectors else None
            except ModelUnavailable as exc:
                degraded.append(f"dense 路降级:{exc}")

    outcomes = await asyncio.gather(
        *(
            _vector_route(
                route,
                query,
                ctx,
                vector=vector,
                top_k=cfg.top_k_route,
                client=client,
                name=collection,
            )
            for route in cfg.routes
            if route in ROUTES
        )
    )
    graph_trace: dict[str, Any] | None = None
    if "graph" in cfg.routes and driver is not None and gateway is not None:
        outcome, trace = await asyncio.to_thread(
            run_graph_route,
            query,
            ctx,
            gateway=gateway,
            driver=driver,
            config_hops=cfg.graph_max_hops,
            top_k=settings.top_k_graph,
            max_edges=cfg.graph_max_edges,
        )
        outcomes.append(outcome)
        graph_trace = {
            "mentions": trace.mentions,
            "linked": trace.linked,
            "edges": trace.edges,
            "lines": trace.lines,
            "dropped_no_chunk": trace.dropped_no_chunk,
            "degraded_reason": trace.degraded_reason,
        }
        if outcome.degraded:
            degraded.append(f"图谱路降级:{outcome.error or trace.degraded_reason}")
    for one in outcomes:
        #: 图谱路在上面已经按中文口径报过了,再走一遍会重复留痕
        if one.route == "graph":
            continue
        if one.degraded and one.error:
            degraded.append(f"{one.route} 路降级:{one.error}")

    #: 三路默认等权(§6.2);图谱路只是第四路,权重不额外加戏
    fused = fuse(outcomes, k=cfg.rrf_k, limit=cfg.top_k_fusion)
    kept, dropped = await backfill(fused, ctx, factory, kb_id=kb_id)
    ranked, rerank_note = kept, None
    if not defer_rerank:
        ranked, rerank_note = await asyncio.to_thread(
            apply_rerank, query, kept, gateway, top_n=cfg.top_n_context, enabled=cfg.use_rerank
        )
    if rerank_note:
        degraded.append(rerank_note)
    return RetrievalResult(
        query=query,
        hits=ranked,
        stats=route_stats(outcomes),
        graph=graph_trace,
        degraded=degraded,
        dropped=dropped,
        latency_ms=int((time.perf_counter() - started) * 1000),
        fused_count=len(fused),
    )
