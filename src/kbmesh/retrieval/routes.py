"""三路召回(FR-201 / FR-202 / FR-203)与逐路可观测(FR-205)。

每路一次独立 RPC,而不是一次 `hybrid_search`:

- 单路开关与单路指标要能各自成立 —— 融合在服务端做完就把「标题路其实没生效」这件事永久藏起来了;
- 任一路挂掉只降级自己(§10 降级矩阵),不必整次检索失败。

租户谓词在这里下发,不在调用方:`filter` 只由 `combined_filter(ctx)` 产生,
三路共用同一个出口,忘加过滤这件事在代码上没有位置(§6.1 防线③)。
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from typing import Any

from pymilvus import MilvusClient

from kbmesh.core.config import get_settings
from kbmesh.core.tenancy import TenantContext
from kbmesh.retrieval.contracts import Citation, Hit, RouteOutcome
from kbmesh.storage import milvus as mv

#: 路由名 → 集合里的向量字段(§4.2:两个 BM25 Function 是两个独立稀疏字段)
ROUTE_FIELD: dict[str, str] = {"body": "sparse_body", "heading": "sparse_heading", "dense": "dense"}

#: BM25 字段直接吃查询文本(稀疏向量由 Function 在服务端算),dense 字段只吃向量
TEXT_ROUTES = frozenset({"body", "heading"})

ROUTES: tuple[str, ...] = ("body", "heading", "dense")


def _as_hit(one: dict[str, Any], route: str) -> Hit:
    """`search()` 把 output_fields 摊平在 hit 上;`hybrid_search()` 包在 entity 里,两种都收。"""
    entity = one.get("entity") if isinstance(one.get("entity"), dict) else one
    chunk_id = mv.hit_chunk_id(one)
    score = float(one["distance"])
    return Hit(
        chunk_id=chunk_id,
        score=score,
        source_routes={route},
        citation=Citation(
            doc_id=str(entity.get("doc_id", "")),
            chunk_id=chunk_id,
            heading_path=str(entity.get("heading_path", "")),
            page_no=entity.get("page_no"),
        ),
        text=str(entity.get("body_text", "")),
        route_scores={route: score},
    )


def run_route(
    route: str,
    query: str,
    ctx: TenantContext,
    *,
    vector: Sequence[float] | None = None,
    top_k: int | None = None,
    client: MilvusClient | None = None,
    name: str | None = None,
) -> RouteOutcome:
    """单路召回。返回 `RouteOutcome`,路内失败不抛出 —— 降级由调用方按 §10 的矩阵决定。"""
    if route not in ROUTE_FIELD:
        raise ValueError(f"未知检索路: {route!r},可选 {sorted(ROUTE_FIELD)}")
    settings = get_settings()
    payload: Any = query if route in TEXT_ROUTES else vector
    if payload is None:
        return RouteOutcome(route=route, error="dense 路需要查询向量", degraded=True)

    started = time.perf_counter()
    try:
        client = client or mv.get_client()
        raw = client.search(
            name or settings.milvus_collection,
            data=[payload],
            anns_field=ROUTE_FIELD[route],
            limit=top_k or settings.top_k_route,
            filter=mv.combined_filter(ctx),
            output_fields=list(mv.OUTPUT_FIELDS),
        )
    except Exception as exc:  # noqa: BLE001 — 外部引擎的失败形态不可枚举,降级优先于报错
        return RouteOutcome(
            route=route,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
            degraded=True,
        )
    hits = [_as_hit(one, route) for one in (raw[0] if raw else [])]
    return RouteOutcome(route=route, hits=hits, latency_ms=int((time.perf_counter() - started) * 1000))


def run_routes(
    query: str,
    ctx: TenantContext,
    *,
    routes: Iterable[str] = ROUTES,
    vector: Sequence[float] | None = None,
    top_k: int | None = None,
    client: MilvusClient | None = None,
    name: str | None = None,
) -> list[RouteOutcome]:
    """按调用方给定的路集合逐路跑(评估运行器的 `routes=["body"]` 单路模式走这里)。"""
    return [
        run_route(
            route, query, ctx, vector=vector, top_k=top_k, client=client, name=name
        )
        for route in routes
    ]


def fingerprint(hits: Sequence[Hit]) -> str:
    """命中集合的稳定指纹:同一路两次跑出同一串就说明结果没漂。"""
    joined = "|".join(sorted(h.chunk_id for h in hits))
    return hashlib.sha1(joined.encode()).hexdigest()[:12] if joined else ""


def route_stats(outcomes: Sequence[RouteOutcome]) -> dict[str, dict[str, Any]]:
    """FR-205 要求的面板数据:候选数、命中集合指纹、延迟、失败原因、是否静默。"""
    return {
        one.route: {
            "count": one.count,
            "fingerprint": fingerprint(one.hits),
            "latency_ms": one.latency_ms,
            "error": one.error,
            "degraded": one.degraded,
        }
        for one in outcomes
    }
