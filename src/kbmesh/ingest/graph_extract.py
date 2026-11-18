"""三元组抽取的 Schema、归一化与白名单校验(FR-301 / FR-302 / FR-303 / FR-304)。

这个模块**不做 I/O**:模型调用在 `pipeline.py`,落库在 `writer.py`。校验函数是纯函数,
所以「抽取层第一道」和「写入层第二道」可以用同一份代码、同一组恶意样本各跑一遍 ——
FR-302 要求两道不可合并,靠的就是把同一个 `validate_triplets()` 调两次,而不是复制两份规则。

白名单与 `storage.neo4j` 同源:图上的 `rtype` 只能是 `REL_TYPES` 里的值,
因为 Cypher 的关系类型不可参数化(§4.3 坑 2),白名单是唯一能安全渲染成字面量的来源。
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from kbmesh.storage.neo4j import BUSINESS_REL_TYPES

#: 实体类型双白名单的一半;清单本身按 Q-5 待评审,先固化成可改的常量而不是散落在 prompt 里
ENTITY_TYPES: frozenset[str] = frozenset(
    {"concept", "model", "method", "dataset", "metric", "component", "process"}
)

#: 低于这个置信度的三元组不进图:宁可图上少几条边,也不要一条说不清依据的边
MIN_CONFIDENCE = 0.55

#: 归一化后 label 的长度区间:太短没有区分度(「它」),太长是整句被当成了实体
MIN_LABEL_CHARS = 2
MAX_LABEL_CHARS = 48

EType = Literal[
    "concept", "model", "method", "dataset", "metric", "component", "process"
]
RType = Literal[
    "USES", "DEPENDS_ON", "PART_OF", "PRODUCES", "EVALUATED_BY", "IMPROVES", "LIMITED_BY", "RELATED_TO"
]


class TripletOut(BaseModel):
    """一条抽取结果;`Literal` 枚举会进 JSON Schema,所以白名单在**模型侧**先生效(FR-303)。"""

    src_label: str
    src_type: EType
    rtype: RType
    dst_label: str
    dst_type: EType
    evidence: str
    confidence: float = 1.0


class TripletPayload(BaseModel):
    triplets: list[TripletOut] = Field(default_factory=list)


PROMPT_HEADER = """你是知识图谱抽取器。只输出满足 Schema 的 JSON,不要解释、不要代码围栏。

规则:
1. 实体类型只能取:{entities}
2. 关系类型只能取:{relations}
3. evidence 必须是给定原文里**逐字出现**的一小段(不超过 120 字),不得改写、不得拼接。
4. label 用原文里的称呼(中文保持中文),不要翻译成术语大类;拿不准就不抽。
5. confidence 取 0 到 1;低于 0.55 的猜测不要提交。
6. 抽不出符合规则的关系时返回 {{"triplets": []}},这是正确答案而不是失败。
"""


def prompt_for(source_text: str) -> str:
    relations = ", ".join(sorted(BUSINESS_REL_TYPES))
    entities = ", ".join(sorted(ENTITY_TYPES))
    return (
        PROMPT_HEADER.format(entities=entities, relations=relations) + f"\n原文:\n<<<\n{source_text}\n>>>\n"
    )


@dataclass(frozen=True)
class Triplet:
    src_key: str
    dst_key: str
    rtype: str
    evidence: str
    chunk_id: str
    src_label: str
    dst_label: str
    src_type: str
    dst_type: str
    confidence: float = 1.0

    def entities(self) -> list[ExtractedEntity]:
        return [
            ExtractedEntity(self.src_key, self.src_type, self.src_label),
            ExtractedEntity(self.dst_key, self.dst_type, self.dst_label),
        ]


@dataclass(frozen=True)
class ExtractedEntity:
    entity_key: str
    etype: str
    label: str


@dataclass
class DropStat:
    """被丢弃的抽取结果按原因计数(FR-301 要求「计入统计」而不是静默扔掉)。"""

    drops: dict[str, int] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def add(self, reason: str, raw: Any) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1
        if len(self.samples) < 20:
            self.samples.append({"reason": reason, "raw": _brief(raw)})

    @property
    def total(self) -> int:
        return sum(self.drops.values())


def _brief(raw: Any) -> Any:
    one = _as_dict(raw)
    if one:
        return {k: one.get(k) for k in ("src_label", "rtype", "dst_label", "confidence")}
    return str(raw)[:120]


def _as_dict(one: Any) -> dict[str, Any] | None:
    """三种来路一起收:模型返回的 pydantic 实例、直接喂进来的 dict(测试与重放)、其他。"""
    if isinstance(one, BaseModel):
        return one.model_dump()
    if isinstance(one, dict):
        return one
    return None


def normalize_label(label: str) -> str:
    """NFKC + 折叠空白 + ASCII 小写:同一实体的「大小写/全半角/多余空格」变体归到同一个键。"""
    text = unicodedata.normalize("NFKC", label or "").strip()
    text = " ".join(text.split())
    return text.lower() if text.isascii() else text


def entity_key(etype: str, label: str) -> str:
    #: 类型进键:「奖励模型」既可能是 model 也可能是 concept,合并成一个节点会连出荒谬的边
    return f"{etype}:{normalize_label(label)}"


def _is_clean_label(label: str) -> bool:
    if not (MIN_LABEL_CHARS <= len(label) <= MAX_LABEL_CHARS):
        return False
    return not any(ord(c) < 0x20 for c in label)


def validate_triplets(
    raw: Any,
    *,
    chunk_id: str,
    source_text: str,
    etype_whitelist: frozenset[str] = ENTITY_TYPES,
    rtype_whitelist: frozenset[str] = BUSINESS_REL_TYPES,
    stat: DropStat | None = None,
) -> list[Triplet]:
    """第二道校验(§5.1 写入层再验一次):任何一项不合格就丢弃并计数,绝不部分放行。"""
    bucket = stat if stat is not None else DropStat()
    items = getattr(raw, "triplets", None)
    if items is None:
        as_dict = _as_dict(raw)
        items = as_dict.get("triplets") if as_dict is not None else None
    if not isinstance(items, list):
        bucket.add("payload_not_a_list", raw)
        return []

    source_norm = normalize_label(source_text)
    kept: list[Triplet] = []
    for one in items:
        item = _as_dict(one)
        if item is None:
            bucket.add("item_not_an_object", one)
            continue
        src_type = str(item.get("src_type", ""))
        dst_type = str(item.get("dst_type", ""))
        rtype = str(item.get("rtype", ""))
        if src_type not in etype_whitelist or dst_type not in etype_whitelist:
            bucket.add("etype_not_whitelisted", item)
            continue
        # 关系类型一旦越过白名单,渲染进 Cypher 就是注入面(FR-306):这里拦住的是字面量拼接的入口
        if rtype not in rtype_whitelist:
            bucket.add("rtype_not_whitelisted", item)
            continue
        src_label = normalize_label(str(item.get("src_label", "")))
        dst_label = normalize_label(str(item.get("dst_label", "")))
        if not (_is_clean_label(src_label) and _is_clean_label(dst_label)):
            bucket.add("label_out_of_range", item)
            continue
        if entity_key(src_type, src_label) == entity_key(dst_type, dst_label):
            bucket.add("self_loop", item)
            continue
        evidence = " ".join(str(item.get("evidence", "")).split())
        if not evidence:
            bucket.add("missing_evidence", item)
            continue
        # 依据必须是原文子串:这是图谱路的幻觉防线,编不出依据的边不能进图
        if normalize_label(evidence)[:120] not in source_norm:
            bucket.add("evidence_not_in_source", item)
            continue
        try:
            confidence = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            bucket.add("confidence_not_a_number", item)
            continue
        if confidence < MIN_CONFIDENCE:
            bucket.add("low_confidence", item)
            continue
        kept.append(
            Triplet(
                src_key=entity_key(src_type, src_label),
                dst_key=entity_key(dst_type, dst_label),
                rtype=rtype,
                evidence=evidence,
                chunk_id=chunk_id,
                src_label=src_label,
                dst_label=dst_label,
                src_type=src_type,
                dst_type=dst_type,
                confidence=confidence,
            )
        )
    return kept


def dedupe(triplets: list[Triplet]) -> list[Triplet]:
    """同 (src, rtype, dst) 只留一条,保留置信度最高、evidence 最长的那条(FR-305 全量合并)。"""
    best: dict[tuple[str, str, str], Triplet] = {}
    for one in triplets:
        mark = (one.src_key, one.rtype, one.dst_key)
        current = best.get(mark)
        rank = (one.confidence, len(one.evidence))
        if current is None or rank > (current.confidence, len(current.evidence)):
            best[mark] = one
    return list(best.values())
