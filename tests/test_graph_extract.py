"""三元组抽取的两道白名单闸门(FR-301 / FR-302 / FR-303 / FR-305)。

这个模块是纯函数,所以恶意样本可以在**没有模型、没有图库**的情况下直接喂进来:
FR-302 要求「抽取层」与「写入层」两道校验不许合并,这里就用同一组样本各打一遍,
证明它们拦的是同一个入口而不是两份会漂移的规则副本。
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from kbmesh.ingest import graph_extract as ge
from kbmesh.storage import neo4j as n4

SOURCE = "RAGFlow 用 Milvus 做向量检索。\nNeo4j 支撑图谱召回,RRF 负责融合。\n"

VALID: dict[str, Any] = {
    "src_label": "Milvus",
    "src_type": "component",
    "rtype": "USES",
    "dst_label": "向量检索",
    "dst_type": "method",
    "evidence": "Milvus 做向量检索",
    "confidence": 0.9,
}


def _item(**overrides: Any) -> dict[str, Any]:
    return {**VALID, **overrides}


def _payload(*items: dict[str, Any]) -> dict[str, Any]:
    return {"triplets": list(items)}


def _kept(payload: Any, *, source: str = SOURCE) -> list[ge.Triplet]:
    return ge.validate_triplets(payload, chunk_id="chk-1", source_text=source)


# ---------------------------------------------------------------- Schema 与 prompt


def test_prompt_pins_both_whitelists() -> None:
    prompt = ge.prompt_for(SOURCE)
    for etype in ge.ENTITY_TYPES:
        assert etype in prompt, etype
    for rtype in ge.BUSINESS_REL_TYPES:
        assert rtype in prompt, rtype


def test_prompt_never_offers_pipeline_owned_edges() -> None:
    """MENTIONED_IN / HAS_CHUNK 是管道自己维护的桥接边:让模型产出这两种等于授权它伪造引用。"""
    prompt = ge.prompt_for(SOURCE)
    for forged in ("MENTIONED_IN", "HAS_CHUNK"):
        assert forged not in prompt


@pytest.mark.parametrize("bad_rtype", ["REL] MATCH (n) DETACH DELETE n; //", "USES) OR (1=1", ""])
def test_schema_rejects_off_whitelist_before_we_ever_see_it(bad_rtype: str) -> None:
    """FR-303:枚举进 JSON Schema,不合规的取值在模型侧就过不去,不靠事后清洗兜底。"""
    with pytest.raises(ValidationError):
        ge.TripletOut(**_item(rtype=bad_rtype))


def test_enum_lands_in_the_json_schema() -> None:
    schema = json.dumps(ge.TripletPayload.model_json_schema(), ensure_ascii=False)
    for value in ("USES", "concept", "process"):
        assert value in schema, value


# ---------------------------------------------------------------- 归一化与键


def test_entity_key_absorbs_case_width_and_spacing() -> None:
    assert ge.entity_key("model", "RAGFlow") == ge.entity_key("model", "ragflow")
    assert ge.entity_key("concept", "ＭＬ") == ge.entity_key("concept", "ML")
    assert ge.entity_key("concept", "  向量\n检索  ") == ge.entity_key("concept", "向量 检索")


def test_intra_label_space_is_collapsed_not_erased() -> None:
    """空白只折叠成单个空格,不整体删除:删干净会把「甲 乙」与「甲乙」并成一个节点。

    真正要合并的写法变体走别名表(FR-305),不归一化函数管。
    """
    assert ge.entity_key("concept", "向量检索") != ge.entity_key("concept", "向量 检索")


def test_type_is_part_of_the_key() -> None:
    """「奖励模型」既可能是 model 也可能是 concept:并成一个节点会连出荒谬的边。"""
    assert ge.entity_key("model", "奖励模型") != ge.entity_key("concept", "奖励模型")


def test_key_is_stable_across_runs() -> None:
    """重跑必须算出同一个键:键不稳 ⇒ MERGE 不稳 ⇒ 每次重索引都长出新节点。"""
    first = _kept(_payload(_item()))[0]
    second = _kept(_payload(_item()))[0]
    assert (first.src_key, first.dst_key, first.rtype, first.evidence) == (
        second.src_key,
        second.dst_key,
        second.rtype,
        second.evidence,
    )


# ---------------------------------------------------------------- 校验通过的形态


def test_valid_triplet_keeps_every_field() -> None:
    one = _kept(_payload(_item()))
    assert len(one) == 1
    trip = one[0]
    assert trip.chunk_id == "chk-1"
    assert trip.src_key == "component:milvus"
    assert trip.dst_key == "method:向量检索"
    assert trip.confidence == pytest.approx(0.9)
    assert {e.entity_key for e in trip.entities()} == {trip.src_key, trip.dst_key}


def test_pydantic_instances_and_dicts_take_the_same_path() -> None:
    """重放与线上共用一份代码:模型返回的对象与手工喂进来的 dict 必须得到同样的结果。"""
    from_dict = _kept(_payload(_item()))
    from_model = _kept(ge.TripletPayload(triplets=[ge.TripletOut(**_item())]))
    assert from_dict == from_model


def test_evidence_matching_ignores_whitespace() -> None:
    """原文换行/续行造成的空格差异不该把合法依据判成幻觉。"""
    assert len(_kept(_payload(_item(evidence="RAGFlow 用 Milvus\n做向量检索")))) == 1


def test_overlong_evidence_is_checked_by_prefix() -> None:
    long_src = "隔离" * 200
    item = _item(evidence=long_src[:150], src_label="隔离甲", dst_label="隔离乙")
    assert len(_kept(_payload(item), source=long_src)) == 1


def test_bad_items_are_dropped_without_taking_the_good_one_down() -> None:
    stat = ge.DropStat()
    kept = ge.validate_triplets(
        _payload(_item(), _item(rtype="USES) OR (1=1"), _item(confidence=0.1)),
        chunk_id="chk-1",
        source_text=SOURCE,
        stat=stat,
    )
    assert len(kept) == 1
    assert stat.total == 2


# ---------------------------------------------------------------- 逐条丢弃原因


@pytest.mark.parametrize(
    "payload,reason",
    [
        pytest.param({"triplets": None}, "payload_not_a_list", id="payload-null"),
        pytest.param({"triplets": "abc"}, "payload_not_a_list", id="payload-string"),
        pytest.param([_item()], "payload_not_a_list", id="payload-is-list"),
        pytest.param(_payload("一条字符串"), "item_not_an_object", id="item-not-object"),
        pytest.param(_payload(_item(src_type="product")), "etype_not_whitelisted", id="etype"),
        pytest.param(_payload(_item(dst_type="")), "etype_not_whitelisted", id="etype-blank"),
        pytest.param(
            _payload(_item(rtype="REL] MATCH (n) DETACH DELETE n; //")),
            "rtype_not_whitelisted",
            id="rtype-injection",
        ),
        pytest.param(_payload(_item(rtype="MENTIONED_IN")), "rtype_not_whitelisted", id="rtype-bridge"),
        pytest.param(_payload(_item(rtype="has_chunk")), "rtype_not_whitelisted", id="rtype-case"),
        pytest.param(_payload(_item(src_label="它")), "label_out_of_range", id="label-short"),
        pytest.param(_payload(_item(dst_label="长" * 60)), "label_out_of_range", id="label-long"),
        pytest.param(_payload(_item(src_label="Mil\x00vus")), "label_out_of_range", id="label-control"),
        pytest.param(
            _payload(_item(dst_label="Milvus", dst_type="component")), "self_loop", id="self-loop"
        ),
        pytest.param(_payload(_item(evidence="")), "missing_evidence", id="no-evidence"),
        pytest.param(
            _payload(_item(evidence="这句原文里根本没有")), "evidence_not_in_source", id="hallucination"
        ),
        pytest.param(_payload(_item(confidence="很高")), "confidence_not_a_number", id="confidence-text"),
        pytest.param(_payload(_item(confidence=0.54)), "low_confidence", id="low-confidence"),
    ],
)
def test_malformed_extraction_is_dropped_and_counted(payload: Any, reason: str) -> None:
    stat = ge.DropStat()
    kept = ge.validate_triplets(payload, chunk_id="chk-1", source_text=SOURCE, stat=stat)
    assert kept == []
    assert stat.drops == {reason: 1}, stat.drops
    assert stat.samples[0]["reason"] == reason


def test_dropping_is_counted_not_silent() -> None:
    """FR-301:被丢掉的抽取要能报出「丢了几条、为什么」,否则模型退化看不出来。"""
    stat = ge.DropStat()
    ge.validate_triplets(
        _payload(_item(rtype="FOO"), _item(rtype="BAR"), _item(confidence=0.1)),
        chunk_id="chk-1",
        source_text=SOURCE,
        stat=stat,
    )
    assert stat.drops == {"rtype_not_whitelisted": 2, "low_confidence": 1}


# ---------------------------------------------------------------- 第二道闸门


INJECTED_REL_TYPES = (
    "REL] MATCH (n) DETACH DELETE n; //",
    'MERGE (e:Entity {entity_key: "x"}) //',
    "USES) OR (1=1",
    "*1..10",
    "",
)


@pytest.mark.parametrize("rtype", INJECTED_REL_TYPES)
def test_the_same_payload_hits_both_gates(rtype: str) -> None:
    """抽取层丢掉的样本,写入层单独也拦得住 —— 少了任何一道都不许放行。"""
    assert _kept(_payload(_item(rtype=rtype))) == []
    with pytest.raises(n4.IllegalGraphIdentifier):
        n4.assert_rel_type(rtype)


@pytest.mark.parametrize("bridge", ["MENTIONED_IN", "HAS_CHUNK"])
def test_bridge_rels_are_pipeline_owned_not_model_owned(bridge: str) -> None:
    """这两种边由 `link` 阶段自己写,所以图层的白名单收它、抽取层的白名单不收它。

    两道闸门的差集正是设计意图:模型没有伪造引用的通道,管道自己还有。
    """
    assert _kept(_payload(_item(rtype=bridge))) == []
    with pytest.raises(ValidationError):
        ge.TripletOut(**_item(rtype=bridge))
    assert n4.assert_rel_type(bridge) == bridge


def test_every_whitelisted_rtype_passes_both_gates() -> None:
    for rtype in sorted(ge.BUSINESS_REL_TYPES):
        assert len(_kept(_payload(_item(rtype=rtype)))) == 1, rtype
        assert n4.assert_rel_type(rtype) == rtype


# ---------------------------------------------------------------- 合并


def test_dedupe_keeps_the_strongest_edge() -> None:
    a, b, c = (
        ge.Triplet(
            src_key="component:milvus",
            dst_key="method:向量检索",
            rtype="USES",
            evidence=evid,
            chunk_id=f"chk-{i}",
            src_label="Milvus",
            dst_label="向量检索",
            src_type="component",
            dst_type="method",
            confidence=conf,
        )
        for i, (evid, conf) in enumerate((("短依据", 0.7), ("更长的依据内容", 0.9), ("第三条", 0.95)))
    )
    merged = ge.dedupe([a, b, c])
    assert len(merged) == 1
    assert merged[0].confidence == pytest.approx(0.95)


def test_dedupe_breaks_ties_by_evidence_length() -> None:
    def one(evidence: str) -> ge.Triplet:
        return ge.Triplet(
            src_key="a:x",
            dst_key="b:y",
            rtype="USES",
            evidence=evidence,
            chunk_id="chk-1",
            src_label="x",
            dst_label="y",
            src_type="a",
            dst_type="b",
            confidence=0.8,
        )

    assert ge.dedupe([one("甲"), one("甲乙丙")])[0].evidence == "甲乙丙"


def test_dedupe_keeps_different_relations_apart() -> None:
    base = _kept(_payload(_item()))[0]
    other = ge.Triplet(**{**base.__dict__, "rtype": "PART_OF"})
    assert len(ge.dedupe([base, other])) == 2
