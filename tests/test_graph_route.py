"""图谱路五步的验收(FR-206 / FR-207 / FR-208 / FR-209 / FR-212)。

链接、线性化、回落这三步是纯函数,恶意输入与边界都能在这里断言;
真正打 Neo4j 的遍历放在 `test_retrieval_e2e.py`,那里的结论是「图上爬不到别租户的边」。
"""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from kbmesh.core.gateway import ModelUnavailable
from kbmesh.core.tenancy import worker_context
from kbmesh.retrieval import graph_route as gr
from kbmesh.retrieval.contracts import RouteOutcome

CTX = worker_context("tnt-route", "kb-route")
MILVUS = {
    "entity_key": "component:milvus",
    "label": "Milvus",
    "etype": "component",
    "aliases": ["向量库"],
}
RRF = {"entity_key": "method:rrf", "label": "RRF", "etype": "method", "aliases": ["倒数排名融合"]}


class FakeGateway:
    """只实现图谱路用到的那一个方法:structured。"""

    def __init__(self, payload: Any = None, error: Exception | None = None) -> None:
        self.payload = payload if payload is not None else gr.MentionsPayload(mentions=[])
        self.error = error
        self.calls: list[str] = []

    def structured(self, prompt: str, schema: type, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return self.payload, None


def _edge(
    src: str,
    dst: str,
    *,
    rtype: str = "USES",
    chunk: str = "chk-1",
    evidence: str = "依据",
    hop: int = 1,
    conf: float = 0.9,
) -> gr.GraphEdge:
    return gr.GraphEdge(
        src_key=src,
        src_label=src.split(":", 1)[-1],
        rtype=rtype,
        dst_key=dst,
        dst_label=dst.split(":", 1)[-1],
        evidence=evidence,
        chunk_id=chunk,
        confidence=conf,
        hop=hop,
    )


# --------------------------------------------------------------------- ① 实体点名

def test_mention_prompt_pins_the_etype_whitelist() -> None:
    prompt = gr.mentions_prompt("Milvus 和 Neo4j 怎么做融合")
    for etype in gr.ENTITY_TYPES:
        assert etype in prompt
    assert "Milvus 和 Neo4j 怎么做融合" in prompt


def test_schema_enum_keeps_the_whitelist_on_the_model_side() -> None:
    """FR-303 的查询侧版本:白名单进 JSON Schema,模型没机会点出清单外的类型。"""
    text = repr(gr.MentionsPayload.model_json_schema())
    assert "component" in text and "concept" in text
    with pytest.raises(ValidationError):
        gr.MentionOut(label="X", etype="not-a-type")  # type: ignore[arg-type]


def test_mention_schema_bounds_model_fanout() -> None:
    with pytest.raises(ValidationError):
        gr.MentionsPayload(
            mentions=[
                gr.MentionOut(label=f"实体{i}", etype="concept")
                for i in range(gr.MAX_MENTIONS + 1)
            ]
        )
    with pytest.raises(ValidationError):
        gr.MentionOut(label="x" * (gr.MAX_LABEL_CHARS + 1), etype="concept")


def test_link_terms_clamps_and_dedupes_model_controlled_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(gr.n4, "find_entities", lambda d, c, t: calls.append(list(t)) or [])
    monkeypatch.setattr(
        gr.n4, "candidate_entities", lambda d, c, t, **k: calls.append([t]) or []
    )
    terms = [f"实体{i}" for i in range(gr.MAX_LINK_TERMS + 10)] + ["实体0"]
    gr.link_terms(object(), CTX, terms)
    assert len(calls[0]) == gr.MAX_LINK_TERMS, "批量严格匹配只带白名单内的上限条数"
    assert len(calls) - 1 <= gr.MAX_LINK_TERMS, "每个 term 至多一次候选查询"


def test_a_question_without_entities_degrades_without_raising() -> None:
    gw = FakeGateway(gr.MentionsPayload(mentions=[]))
    outcome, trace = gr.run_graph_route("今天天气", CTX, gateway=gw)
    assert len(gw.calls) == 1
    assert outcome.degraded and trace.degraded_reason == "问题里没点出实体"
    assert outcome.hits == []


def test_model_outage_degrades_the_route_instead_of_failing_the_query() -> None:
    """§10 降级矩阵:图谱路依赖模型,模型挂了向量三路照常。"""
    gw = FakeGateway(error=ModelUnavailable("structured:graph_mention", "quota"))
    outcome, trace = gr.run_graph_route("Milvus 用什么检索", CTX, gateway=gw)
    assert outcome.degraded and "模型不可用" in (trace.degraded_reason or "")


# --------------------------------------------------------------------- ② 实体链接

def test_l1_exact_match_on_label_alias_and_key() -> None:
    for term in ("Milvus", "向量库", "component:milvus"):
        one = gr.judge_link(term, [MILVUS, RRF])
        assert one is not None and one.level == "exact" and one.similarity == 1.0
        assert one.entity_key == "component:milvus"


def test_l1_is_preferred_over_a_fuzzy_candidate() -> None:
    near = {"entity_key": "component:milvusx", "label": "Milvus", "etype": "component", "aliases": []}
    one = gr.judge_link("Milvus", [near, MILVUS])
    assert one is not None and one.level == "exact"


def test_an_extra_tail_is_caught_by_character_similarity() -> None:
    """FR-207:严格匹配失败(多个后缀)时,字符匹配要能把这个点名接住。"""
    one = gr.judge_link("倒数排名融合方法", [RRF])
    assert one is not None and one.level == "fuzzy"
    assert one.similarity >= gr.FUZZY_THRESHOLD


def test_a_distant_term_is_not_forced_into_an_entity() -> None:
    assert gr.judge_link("今天天气怎么样", [MILVUS, RRF]) is None


def test_threshold_is_a_floor_not_a_preference() -> None:
    rows = [{"entity_key": "concept:图谱", "label": "图谱", "etype": "concept", "aliases": []}]
    assert gr.judge_link("图谱召回", rows, threshold=0.99) is None
    assert gr.judge_link("图谱召回", rows, threshold=0.30) is not None


def test_ambiguity_is_counted_so_the_route_can_back_off() -> None:
    rows = [
        {"entity_key": "method:rrf", "label": "RRF融合", "etype": "method", "aliases": []},
        {"entity_key": "method:rrfg", "label": "RRF融合", "etype": "method", "aliases": []},
    ]
    one = gr.judge_link("RRF融合技术", rows)
    assert one is not None and one.level == "fuzzy"
    assert one.candidates == 2, "并列候选数要留给降级判断"


def test_tied_candidates_resolve_by_entity_key() -> None:
    rows = [
        {"entity_key": "z:RRF融合", "label": "RRF融合", "etype": "method", "aliases": []},
        {"entity_key": "a:RRF融合", "label": "RRF融合", "etype": "method", "aliases": []},
    ]
    one = gr.judge_link("RRF融合技术", rows)
    assert one is not None and one.entity_key == "a:RRF融合", "并列时也要有确定顺序"


def test_char_similarity_normalizes_case_width_and_spaces() -> None:
    assert gr.char_similarity("Ｍｉｌｖｕｓ", "milvus") == 1.0
    assert gr.char_similarity("", "Milvus") == 0.0
    assert gr.char_similarity("Milvus", "Milvus ") == 1.0


# --------------------------------------------------------------------- ③④ 邻域扩展与线性化

def test_linearize_keeps_the_original_direction() -> None:
    """FR-212:遍历是无向的,渲染必须按边的 start→end。"""
    lines = gr.linearize([_edge("a:A", "b:B", rtype="DEPENDS_ON", evidence="A 依赖 B")])
    assert lines == ["A --DEPENDS_ON--> B  (依据: A 依赖 B)"]


def test_linearize_truncates_to_the_budget() -> None:
    edges = [_edge("a:A", f"b:{i}", chunk=f"chk-{i}") for i in range(50)]
    assert len(gr.linearize(edges, max_edges=30)) == 30


def test_linearize_drops_the_basis_clause_when_there_is_none() -> None:
    assert gr.linearize([_edge("a:A", "b:B", evidence="")]) == ["A --USES--> B"]


def test_to_edges_maps_the_driver_rows_into_a_stable_shape() -> None:
    rows = [
        {
            "src_key": "a:A",
            "src_label": "A",
            "rtype": "USES",
            "dst_key": "b:B",
            "dst_label": "B",
            "evidence": "A 使用 B",
            "chunk_id": "chk-1",
            "confidence": 0.8,
            "hop": 2,
        }
    ]
    edges = gr.to_edges(rows)
    assert edges[0].hop == 2 and edges[0].confidence == 0.8
    assert edges[0].dedupe_key() == ("a:A", "USES", "b:B")


# --------------------------------------------------------------------- ⑤ 回落切片

def test_an_edge_without_a_chunk_pointer_is_dropped_not_answered() -> None:
    """FR-209:没有原文指针的裸三元组不能当答案依据,丢弃并计数。"""
    hits, dropped = gr.graph_hits([_edge("a:A", "b:B", chunk="")], [], {})
    assert hits == [] and dropped == 1


def test_graph_hit_carries_the_edge_evidence_and_pointer() -> None:
    hits, dropped = gr.graph_hits([_edge("a:A", "b:B")], [], {})
    assert [h.chunk_id for h in hits] == ["chk-1"]
    assert hits[0].evidence == "依据"
    assert hits[0].source_routes == {"graph"}
    assert dropped == 0


def test_two_edges_into_one_chunk_collapse_and_keep_both_bases() -> None:
    hits, _ = gr.graph_hits(
        [
            _edge("a:A", "b:B", evidence="第一条依据"),
            _edge("a:A", "b:C", rtype="IMPROVES", evidence="第二条依据"),
        ],
        [],
        {},
    )
    assert len(hits) == 1
    assert "第一条依据" in hits[0].evidence and "第二条依据" in hits[0].evidence


def test_hop_distance_discounts_the_route_score() -> None:
    hits, _ = gr.graph_hits(
        [_edge("a:A", "b:B", chunk="chk-1", hop=1), _edge("a:A", "b:C", chunk="chk-2", hop=2)],
        [],
        {},
    )
    scores = {h.chunk_id: h.route_scores["graph"] for h in hits}
    assert scores["chk-1"] > scores["chk-2"], "越远的边越不可信"


def test_mention_bridge_backfills_chunks_and_labels_the_basis() -> None:
    rows = [{"chunk_id": "chk-9", "entity_key": "component:milvus"}]
    hits, _ = gr.graph_hits([], rows, {"component:milvus": "Milvus"})
    assert [h.chunk_id for h in hits] == ["chk-9"]
    assert "Milvus" in (hits[0].evidence or "")


def test_edges_and_mentions_union_without_duplicating_a_chunk() -> None:
    hits, _ = gr.graph_hits(
        [_edge("a:A", "b:B", chunk="chk-1", evidence="边依据")],
        [{"chunk_id": "chk-1", "entity_key": "a:A"}],
        {"a:A": "A"},
    )
    assert len(hits) == 1
    assert "边依据" in (hits[0].evidence or "") and "A" in (hits[0].evidence or "")


def test_top_k_bounds_the_graph_route() -> None:
    edges = [_edge("a:A", f"b:{i}", chunk=f"chk-{i}") for i in range(10)]
    hits, _ = gr.graph_hits(edges, [], {}, top_k=4)
    assert len(hits) == 4


# --------------------------------------------------------------------- 五步串起来(FR-206)


def _fake_expand(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return [
        {
            "src_key": "component:milvus",
            "src_label": "Milvus",
            "rtype": "USES",
            "dst_key": "method:向量检索",
            "dst_label": "向量检索",
            "evidence": "Milvus 用向量检索",
            "chunk_id": "chk-1",
            "confidence": 0.9,
            "hop": 1,
        }
    ]


def test_five_steps_expose_their_intermediates(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = gr.MentionsPayload(mentions=[gr.MentionOut(label="Milvus", etype="component")])
    monkeypatch.setattr(gr.n4, "find_entities", lambda d, c, t: [MILVUS])
    monkeypatch.setattr(gr.n4, "candidate_entities", lambda d, c, t, **k: [])
    monkeypatch.setattr(gr.n4, "expand_neighborhood", _fake_expand)
    monkeypatch.setattr(
        gr.n4,
        "mentioned_chunks",
        lambda d, c, k, **kw: [{"chunk_id": "chk-2", "entity_key": "component:milvus"}],
    )
    outcome, trace = gr.run_graph_route(
        "Milvus 怎么做检索", CTX, gateway=FakeGateway(payload), driver=object()
    )
    assert [h.chunk_id for h in outcome.hits] == ["chk-1", "chk-2"]
    assert trace.mentions == [{"label": "Milvus", "etype": "component"}]
    assert trace.linked[0]["level"] == "exact"
    assert trace.lines == ["Milvus --USES--> 向量检索  (依据: Milvus 用向量检索)"]
    assert not outcome.degraded and trace.degraded_reason is None


def test_too_many_fuzzy_links_switches_the_route_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """§6.3:模糊命中过多说明整句被当成了实体 —— 宁可不开图谱路。"""
    payload = gr.MentionsPayload(
        mentions=[
            gr.MentionOut(label=f"倒数排名融合方{i}", etype="method") for i in range(4)
        ]
    )
    fuzzy = {
        "entity_key": "method:rrf",
        "label": "倒数排名融合",
        "etype": "method",
        "aliases": [],
    }
    monkeypatch.setattr(gr.n4, "find_entities", lambda d, c, t: [])
    monkeypatch.setattr(gr.n4, "candidate_entities", lambda d, c, t, **k: [fuzzy])
    outcome, trace = gr.run_graph_route(
        "倒数排名融合方0", CTX, gateway=FakeGateway(payload), driver=object()
    )
    assert outcome.degraded and "超过阈值" in (trace.degraded_reason or "")
    assert outcome.hits == []


def test_link_miss_does_not_query_the_neighborhood(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = gr.MentionsPayload(mentions=[gr.MentionOut(label="不存在的实体", etype="concept")])
    monkeypatch.setattr(gr.n4, "find_entities", lambda d, c, t: [])
    monkeypatch.setattr(gr.n4, "candidate_entities", lambda d, c, t, **k: [])
    called: list[str] = []
    monkeypatch.setattr(
        gr.n4, "expand_neighborhood", lambda *a, **k: called.append("expand") or []
    )
    outcome, trace = gr.run_graph_route("X", CTX, gateway=FakeGateway(payload), driver=object())
    assert outcome.degraded and trace.degraded_reason == "实体链接全部落空"
    assert called == [], "没链接上就不该发起遍历"


def test_a_route_outcome_shape_the_fusion_layer_can_consume() -> None:
    outcome = RouteOutcome(route="graph", hits=[], degraded=True, error="x")
    assert outcome.count == 0 and outcome.degraded
