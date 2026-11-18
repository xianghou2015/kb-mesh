"""图谱路五步(FR-206 / FR-207 / FR-208 / FR-209 / FR-212)。

五步各自是一个可单独调用的函数,中间结果一并返回 —— FR-206 要的「断点查看」不是加日志,
而是把每一步的产物做成数据结构:`trace = {"mentions": [...], "linked": [...], "edges": [...], ...}`。

抽取白名单与 `ingest.graph_extract` 同源(`EType` / `BUSINESS_REL_TYPES`):查询侧和写入侧
各持一份清单,迟早会漂成两套,而图谱路的注入面正好长在清单上。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field

from kbmesh.core.gateway import Gateway, ModelUnavailable
from kbmesh.core.tenancy import TenantContext
from kbmesh.ingest.graph_extract import ENTITY_TYPES, EType, entity_key, normalize_label
from kbmesh.retrieval.contracts import GraphEdge, Hit, RouteOutcome
from kbmesh.storage import neo4j as n4

#: L2 字符匹配的接住阈值(§6.3:normalize + 编辑距离阈值 0.8)
FUZZY_THRESHOLD = 0.8

#: 模糊链接超过这个条数就是「整句被当成了实体」,图谱路宁可不开,也不要给出看似有结构的噪声
MAX_FUZZY_LINKS = 2

#: 模型控制的列表必须有上限:每个 mention 都可能触发一次候选查询,不封顶就是外部查询扇出。
MAX_MENTIONS = 5
#: 每个 mention 会贡献 entity_key + label 两个检索词,所以上限是 mentions 的两倍(仍封顶)。
MAX_LINK_TERMS = MAX_MENTIONS * 2
MAX_LABEL_CHARS = 128


class MentionOut(BaseModel):
    """查询里的实体点名;`Literal` 类型把白名单写进 JSON Schema,模型侧先生效(FR-303)。"""

    label: str = Field(min_length=1, max_length=MAX_LABEL_CHARS)
    etype: EType


class MentionsPayload(BaseModel):
    mentions: list[MentionOut] = Field(default_factory=list, max_length=MAX_MENTIONS)


MENTION_PROMPT = """你是实体链接器。从给定问题里点出**可能在知识图谱里存在**的实体,只输出符合 Schema 的 JSON。

规则:
1. 实体类型只能取:{entities}
2. label 用问题里的原词(中文保持中文),不要改写、不要翻译、不要补全成更大的概念。
3. 问句里的疑问词、程度词、动作不是实体;拿不准就不点,空列表是正确答案。
4. 最多点 5 个,按重要性排序。

问题:
<<<
{query}
>>>
"""


def mentions_prompt(query: str) -> str:
    return MENTION_PROMPT.format(entities=", ".join(sorted(ENTITY_TYPES)), query=query)


@dataclass(frozen=True)
class Linked:
    """一个点名对应的链接结果:命中哪个实体、靠哪一级接住的、相似度多少。"""

    term: str
    entity_key: str
    label: str
    etype: str
    level: str  # exact | fuzzy | none
    similarity: float
    candidates: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphTrace:
    """五步中间结果(FR-206),序列化后进 trace_step 与调试响应。"""

    mentions: list[dict[str, Any]] = field(default_factory=list)
    linked: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    dropped_no_chunk: int = 0
    degraded_reason: str | None = None


def char_similarity(a: str, b: str) -> float:
    """归一化后的字符相似度(0~1):别名、错字、全半角差异都在这一步被接住。"""
    x, y = normalize_label(a), normalize_label(b)
    if not x or not y:
        return 0.0
    if x == y:
        return 1.0
    return SequenceMatcher(None, x, y).ratio()


def _similarities(term: str, row: dict[str, Any]) -> list[float]:
    """点名与一个候选的全部称呼(label + 别名)逐个比,取最高分。"""
    names = [str(row.get("label", "")), *[str(a) for a in (row.get("aliases") or [])]]
    return [char_similarity(term, name) for name in names if name]


def judge_link(
    term: str, rows: Sequence[dict[str, Any]], *, threshold: float = FUZZY_THRESHOLD
) -> Linked | None:
    """纯函数:给一批候选节点,判这个点名能不能接上(FR-207 的两级都在这)。

    先按 `entity_key`/`label`/别名严格等值(L1),落空再按字符相似度取最优(L2)。
    两级都留证据:level 与 similarity 是后面「低置信降级」和评估口径的输入。
    """
    if not rows:
        return None
    wanted = normalize_label(term)
    for row in rows:
        names = {str(row.get("label", "")), *[str(a) for a in (row.get("aliases") or [])]}
        if term in {str(row.get("entity_key", "")), *names} or wanted in {normalize_label(n) for n in names}:
            return Linked(
                term=term,
                entity_key=str(row["entity_key"]),
                label=str(row.get("label", "")),
                etype=str(row.get("etype", "")),
                level="exact",
                similarity=1.0,
            )
    ranked = sorted(
        ((max(_similarities(term, row)), row) for row in rows),
        key=lambda pair: (-pair[0], str(pair[1].get("entity_key", ""))),
    )
    best_score, best = ranked[0]
    if best_score < threshold:
        return None
    tied = sum(1 for score, _ in ranked if score >= threshold)
    return Linked(
        term=term,
        entity_key=str(best["entity_key"]),
        label=str(best.get("label", "")),
        etype=str(best.get("etype", "")),
        level="fuzzy",
        similarity=round(best_score, 4),
        candidates=tied,
    )


def link_terms(
    driver,
    ctx: TenantContext,
    terms: Iterable[str],
    *,
    threshold: float = FUZZY_THRESHOLD,
    max_fuzzy: int = MAX_FUZZY_LINKS,
) -> tuple[list[Linked], list[Linked], int]:
    """② 实体链接:先一次批量严格匹配,落空的再用全文索引生成候选。

    返回 `(全部判定, 通过的判定, 模糊通过数)`;模糊通过过多时由调用方决定整路降级。
    """
    wanted: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        term = str(raw or "").strip()[:MAX_LABEL_CHARS]
        if not term or term in seen:
            continue
        seen.add(term)
        wanted.append(term)
        if len(wanted) >= MAX_LINK_TERMS:
            break
    if not wanted:
        return [], [], 0
    exact_rows = n4.find_entities(driver, ctx, wanted)
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in exact_rows:
        for bucket in (str(row.get("entity_key", "")), str(row.get("label", ""))):
            by_key.setdefault(bucket, []).append(row)
        for alias in row.get("aliases") or []:
            by_key.setdefault(str(alias), []).append(row)

    judged: list[Linked] = []
    for term in wanted:
        rowset = by_key.get(term)
        one = judge_link(term, rowset or [])
        if one is None:
            candidates = n4.candidate_entities(driver, ctx, term)
            one = judge_link(term, candidates, threshold=threshold)
        if one is not None:
            judged.append(one)
    passed = [j for j in judged if j.level in ("exact", "fuzzy")]
    fuzzy_count = sum(1 for j in passed if j.level == "fuzzy")
    return judged, passed, fuzzy_count


def to_edges(rows: Sequence[dict[str, Any]]) -> list[GraphEdge]:
    return [
        GraphEdge(
            src_key=str(r["src_key"]),
            src_label=str(r.get("src_label", "")),
            rtype=str(r["rtype"]),
            dst_key=str(r["dst_key"]),
            dst_label=str(r.get("dst_label", "")),
            evidence=str(r.get("evidence", "")),
            chunk_id=str(r.get("chunk_id", "")),
            confidence=float(r.get("confidence", 1.0)),
            hop=int(r.get("hop", 1)),
        )
        for r in rows
    ]


def linearize(edges: Sequence[GraphEdge], *, max_edges: int = 30) -> list[str]:
    """④ 子图线性化:每条边一行,**方向与依据都在行里**(FR-212)。

    遍历用无向 `-[r:REL*1..n]-` 才能双向扩邻域,但渲染必须按 `start_node → end_node` 的原始方向,
    否则「A 依赖 B」会被写成「B 依赖 A」。
    """
    lines: list[str] = []
    for edge in list(edges)[:max_edges]:
        basis = f"  (依据: {edge.evidence[:60]})" if edge.evidence else ""
        lines.append(f"{edge.src_label} --{edge.rtype}--> {edge.dst_label}{basis}")
    return lines


def graph_hits(
    edges: Sequence[GraphEdge],
    mentions: Sequence[dict[str, Any]],
    labels: dict[str, str],
    *,
    top_k: int = 30,
) -> tuple[list[Hit], int]:
    """⑤ 回落切片:图上命中一律变成「原文切片的指针」,没有指针的结果直接丢(FR-209)。

    两个来源:业务边自带的依据切片(`REL.chunk_id`),以及实体 → `MENTIONED_IN` → 切片的桥接边。
    后者没有边依据,但切片本身就是原文,交给融合层从权威表回填正文。
    """
    pool: dict[str, Hit] = {}
    dropped = 0
    for edge in edges:
        if not edge.chunk_id:
            dropped += 1
            continue
        score = round(edge.confidence - 0.1 * max(0, edge.hop - 1), 4)
        hit = pool.get(edge.chunk_id)
        if hit is None:
            hit = pool[edge.chunk_id] = Hit(chunk_id=edge.chunk_id, score=0.0, source_routes=set())
        hit.source_routes.add("graph")
        hit.route_scores["graph"] = max(hit.route_scores.get("graph", 0.0), score)
        if edge.evidence and edge.evidence not in (hit.evidence or ""):
            hit.evidence = f"{hit.evidence} {edge.evidence}".strip() if hit.evidence else edge.evidence
    for row in mentions:
        chunk_id = str(row.get("chunk_id", ""))
        key = str(row.get("entity_key", ""))
        if not chunk_id:
            continue
        hit = pool.get(chunk_id)
        if hit is None:
            hit = pool[chunk_id] = Hit(chunk_id=chunk_id, score=0.0, source_routes=set())
        hit.source_routes.add("graph")
        hit.route_scores.setdefault("graph", 0.0)
        label = labels.get(key, key)
        basis = f"实体「{label}」出现在此切片"
        hit.evidence = basis if not hit.evidence else f"{hit.evidence};{basis}"

    ranked = sorted(pool.values(), key=lambda h: (-h.route_scores["graph"], h.chunk_id))[:top_k]
    for hit in ranked:
        hit.score = hit.route_scores["graph"]
    return ranked, dropped


def run_graph_route(
    query: str,
    ctx: TenantContext,
    *,
    gateway: Gateway,
    driver=None,
    config_hops: int = 2,
    top_k: int = 30,
    max_edges: int = 30,
) -> tuple[RouteOutcome, GraphTrace]:
    """五步串起来。任何一步失败都只降级图谱路,不影响三路向量召回(§10)。"""
    trace = GraphTrace()
    outcome = RouteOutcome(route="graph")
    try:
        payload, _usage = gateway.structured(
            mentions_prompt(query), MentionsPayload, purpose="graph_mention"
        )
    except ModelUnavailable as exc:
        trace.degraded_reason = f"模型不可用:{exc}"
        outcome.degraded = True
        outcome.error = trace.degraded_reason
        return outcome, trace

    # schema 已限长;这里再 clamp 一次,防 `model_construct` 或非 Pydantic 替身绕过校验。
    mentions = [m for m in payload.mentions if m.label.strip()][:MAX_MENTIONS]
    trace.mentions = [{"label": m.label, "etype": m.etype} for m in mentions]
    if not mentions:
        trace.degraded_reason = "问题里没点出实体"
        outcome.degraded = True
        return outcome, trace

    terms = [entity_key(m.etype, m.label) for m in mentions] + [m.label for m in mentions]
    judged, passed, fuzzy = link_terms(driver, ctx, terms)
    # 严格匹配只给了候选,判定仍按 label 归一化重跑一遍,所以 exact 也来自同一函数
    trace.linked = [one.as_dict() for one in judged]
    if fuzzy > MAX_FUZZY_LINKS:
        trace.degraded_reason = f"模糊链接 {fuzzy} 条超过阈值 {MAX_FUZZY_LINKS},图谱路不启用"
        outcome.degraded = True
        return outcome, trace
    if not passed:
        trace.degraded_reason = "实体链接全部落空"
        outcome.degraded = True
        return outcome, trace

    keys = [one.entity_key for one in passed]
    edges = to_edges(
        n4.expand_neighborhood(driver, ctx, keys, hops=config_hops, max_edges=max_edges)
    )
    trace.edges = [asdict(e) for e in edges]
    trace.lines = linearize(edges, max_edges=max_edges)

    labels = {one.entity_key: one.label for one in passed}
    mention_rows = n4.mentioned_chunks(driver, ctx, [*keys, *[e.dst_key for e in edges]])
    hits, dropped = graph_hits(edges, mention_rows, labels, top_k=top_k)
    trace.dropped_no_chunk = dropped
    outcome.hits = hits
    return outcome, trace
