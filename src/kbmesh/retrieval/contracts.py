"""Retrieval contracts shared by the vector route, the graph route and the fusion layer.

The single dedup key is `chunk_id`, which is a stable primary key — never a text hash, because
two routes reaching the same chunk through different renderings must collapse into one hit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Route = Literal["body", "heading", "dense", "graph"]
ROUTES: tuple[str, ...] = ("body", "heading", "dense", "graph")


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_id: str
    heading_path: str = ""
    page_no: int | None = None
    title: str = ""
    image_ref: str | None = None

    def label(self) -> str:
        parts = [p for p in (self.title or self.doc_id, self.heading_path) if p]
        tail = f" p.{self.page_no}" if self.page_no else ""
        return " / ".join(parts) + tail


@dataclass
class Hit:
    chunk_id: str
    score: float
    source_routes: set[str] = field(default_factory=set)
    citation: Citation | None = None
    text: str = ""
    #: 图谱路必须回填原文依据;缺依据的图结果不允许进入融合(见 §6.3)
    evidence: str | None = None
    route_scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)

    def merge_from(self, other: Hit) -> None:
        """Union two hits for the same chunk_id: keep both routes, take the better score."""
        self.source_routes |= other.source_routes
        self.route_scores.update(other.route_scores)
        self.ranks.update(other.ranks)
        if other.evidence and not self.evidence:
            self.evidence = other.evidence
        if other.citation and not self.citation:
            self.citation = other.citation


@dataclass
class RouteOutcome:
    route: str
    hits: list[Hit] = field(default_factory=list)
    latency_ms: int = 0
    error: str | None = None
    degraded: bool = False

    @property
    def count(self) -> int:
        return len(self.hits)


@dataclass(frozen=True)
class GraphEdge:
    """遍历出来的一条边:方向保留(FR-212),依据与出处切片一起带回来(FR-209)。"""

    src_key: str
    src_label: str
    rtype: str
    dst_key: str
    dst_label: str
    evidence: str = ""
    chunk_id: str = ""
    confidence: float = 1.0
    hop: int = 1

    def dedupe_key(self) -> tuple[str, str, str]:
        return (self.src_key, self.rtype, self.dst_key)


@dataclass(frozen=True)
class RetrievalConfig:
    """Request-level overrides; defaults come from Settings. Fingerprintable for eval runs."""

    routes: tuple[str, ...] = ("body", "heading", "dense")
    top_k_route: int = 50
    top_k_fusion: int = 30
    top_n_context: int = 8
    rrf_k: int = 60
    use_rerank: bool = True
    graph_max_hops: int = 2
    graph_max_edges: int = 30
    theta_rel: float = 0.35

    def with_graph(self, enabled: bool) -> RetrievalConfig:
        routes = self.routes + ("graph",) if enabled and "graph" not in self.routes else self.routes
        return RetrievalConfig(
            routes=tuple(routes), top_k_route=self.top_k_route, top_k_fusion=self.top_k_fusion,
            top_n_context=self.top_n_context, rrf_k=self.rrf_k, use_rerank=self.use_rerank,
            graph_max_hops=self.graph_max_hops, graph_max_edges=self.graph_max_edges,
            theta_rel=self.theta_rel,
        )
