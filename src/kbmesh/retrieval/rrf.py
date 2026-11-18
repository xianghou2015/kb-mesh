"""RRF 融合(FR-204 / FR-210)。

`score(d) = Σ_r weight_r / (k + rank_r(d))`,`k=60`(业界默认)。用名次而不是原始分,
是因为 BM25 与 COSINE 的量纲不可比:把两路的距离直接相加等于拿米和斤做算术。

去重键是 `chunk_id`(稳定主键),不是文本 hash —— 两路以不同渲染命中同一切片时必须塌成一条(FR-210)。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from kbmesh.retrieval.contracts import Hit, RouteOutcome

#: RRF 的 k:名次差 1 名带来的分差由它决定,业界默认 60,先不玄学调参(§6.2)
RRF_K = 60


def rrf_score(
    ranks: Mapping[str, int], *, k: int = RRF_K, weights: Mapping[str, float] | None = None
) -> float:
    """单条文档的 RRF 分。`ranks` 是 `路名 → 1-based 名次`,名次缺失即该路未命中。"""
    total = 0.0
    for route, rank in ranks.items():
        weight = (weights or {}).get(route, 1.0)
        total += weight / (k + rank)
    return total


def fuse(
    outcomes: Sequence[RouteOutcome],
    *,
    k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[Hit]:
    """多路结果 → 一份去重并排序好的列表。空路、整路为空都不算错,只是没贡献。"""
    merged: dict[str, Hit] = {}
    for outcome in outcomes:
        for rank, hit in enumerate(outcome.hits, start=1):
            base = merged.get(hit.chunk_id)
            if base is None:
                merged[hit.chunk_id] = hit
                base = hit
            else:
                base.merge_from(hit)
            base.ranks[outcome.route] = rank

    scored = [
        (
            hit,
            rrf_score(hit.ranks, k=k, weights=weights),
        )
        for hit in merged.values()
    ]
    for hit, score in scored:
        hit.score = score
    # 分数并列时按 chunk_id 定序:融合结果必须可复现,否则评估跑两次得到的上下文不同,指标就没法比
    scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
    hits = [hit for hit, _ in scored]
    return hits[:limit] if limit else hits
