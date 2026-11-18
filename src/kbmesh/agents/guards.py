"""五层幻觉防御的可测纯函数(FR-404 / FR-405 / FR-408)。

这一层不 import langgraph、不碰网络、不读 Settings:阈值全部从参数进,所以同一套判定在
「模型挂了 / 引擎挂了 / 根本没配置」三种现场里都是同一个结果。阈值也因此**只能由节点从
Settings 取一次再传进来** —— 层2 若顺手读 `RetrievalConfig.theta_rel`,就会出现两个 θ,
而这两个值谁赢取决于调用点,不是取决于设计。

每条判定的边界都在 docstring 里写死方向(「低于」= 严格小于,等于阈值算达标保留),因为
0.35 这一条线上站着的是「这条证据算不算数」。
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kbmesh.core.errors import IllegalTransition
from kbmesh.ingest.chunking import normalize_text
from kbmesh.retrieval.contracts import Hit

EDITED_MIN, EDITED_MAX = 2, 512

#: 层1 判定 → (final_status, reject_code)。`answerable` 之外一律终态,连检索都不进:
#: 「答案必须有证据」的前提是先给证据划范围,让越权问题去检索一轮再拒,等于已经查了库。
_INTENT_TABLE: dict[str, tuple[str, str, str | None]] = {
    "answerable": ("", "", None),
    "out_of_privilege": ("refused", "intent_forbidden", "intent_out_of_privilege"),
    "out_of_scope": ("refused", "out_of_scope", None),
    "chitchat": ("refused", "chitchat", None),
}

# --- 引用编号:1-based,方括号内含全角;正文一个字都不改写,只做「剥离」 --------------------------------------
_CITE_SPAN = re.compile(r"([\u3010\uFF3B\[])([^\u3011\uFF3D\uFF3B\[\]{}]{1,24})([\u3011\uFF3D\]])")
_SPAN_SPLIT = re.compile(r"[,;\s\u3001\uff0c\uff1b]+")
_PURE_NUM = re.compile(r"^[0-9]+$")
_FULLWIDTH = str.maketrans("\uFF10\uFF11\uFF12\uFF13\uFF14\uFF15\uFF16\uFF17\uFF18\uFF19", "0123456789")
#: 控制字符零容忍:换行/制表/零宽字符都可能是「把确认框当成注入面」的第一步
_CONTROL = re.compile(r"[\x00-\x1f\x7f\u200b\u2060\ufeff]")

# --- 归一分词:ASCII 词 + 中日韩双字(与 §6.3 第②级的字符级匹配同一口径) ------------------------------------
_TOKEN = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """既收 pydantic 模型也收 dict:检查点里落的是 dict,刚调完模型拿到的是模型。"""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _numbers(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for one in values:
        try:
            out.append(float(one))
        except (TypeError, ValueError):
            continue
    return out


# --- 层1:意图门禁 --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Gate:
    allow: bool
    verdict: str
    final_status: str
    reject_code: str
    confidence: float = 0.0
    #: 非 None 就必须写 SecurityEvent —— 越权是审计事件,拒答本身不是
    security_event: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "allow": self.allow,
            "verdict": self.verdict,
            "final_status": self.final_status,
            "reject_code": self.reject_code,
            "confidence": self.confidence,
            "security_event": self.security_event,
        }


def intent_gate(intent: Any) -> Gate:
    """把 `IntentOut` 的四档判定翻成「放行 / 终态 + 分类 + 是否留痕」。

    判不出(空 verdict、Schema 外的值)按 `intent_unknown` 终态拒答:门禁失败的方向只有一个 ——
    不放行。宁可让一个可答的问题被误拒(会进 trace 的拒答口径,评估阶段能看见),也不能让一个
    越权问题被误放(它拿到的答案根本无从审计)。
    """
    verdict = str(_field(intent, "verdict", "") or "")
    confidence = (_numbers([_field(intent, "confidence", 0.0)]) or [0.0])[0]
    if verdict not in _INTENT_TABLE:
        return Gate(
            allow=False,
            verdict=verdict,
            final_status="refused",
            reject_code="intent_unknown",
            confidence=confidence,
        )
    final_status, reject_code, event = _INTENT_TABLE[verdict]
    return Gate(
        allow=not final_status,
        verdict=verdict,
        final_status=final_status,
        reject_code=reject_code,
        confidence=confidence,
        security_event=event,
    )


# --- 层2:相关性过滤 -------------------------------------------------------------------------------------
def dedupe_hits(hits: Sequence[Hit], *, limit: int) -> tuple[list[Hit], list[str]]:
    """按 `chunk_id` 去重后截断;返回保留序列与被丢弃的 id(重复的 + 超预算的)。

    去重键只认 `chunk_id`,不认正文文本:`contracts.Hit` 的注释钉死了它是稳定主键,而两路召回
    同一切片时渲染方式本来就不一样,按文本去重等于让同一个证据占两个编号。
    """
    kept: list[Hit] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.chunk_id in seen:
            dropped.append(hit.chunk_id)
            continue
        seen.add(hit.chunk_id)
        kept.append(hit)
    if limit >= 0:
        dropped.extend(h.chunk_id for h in kept[limit:])
        del kept[limit:]
    return kept, dropped


def unknown_ids(scores: Mapping[str, Any], known: Iterable[str]) -> list[str]:
    """模型抄写的 chunk_id 里,不在证据集的那些。

    这是层2 的第二重:评分表本身也可能被幻觉出来。这些 id 直接丢,并且要进 trace —— 模型开始
    编 chunk_id 是管道要响的信号,不是「忽略就好」。
    """
    evidence = set(known)
    out: list[str] = []
    for chunk_id in scores:
        if chunk_id not in evidence and chunk_id not in out:
            out.append(str(chunk_id))
    return out


def filter_by_theta(
    hits: Sequence[Hit], scores: Mapping[str, Any], *, theta_rel: float
) -> tuple[list[Hit], list[str]]:
    """θ_rel 过滤:严格小于阈值的丢,等于阈值保留(§7.3 层2 写的是「低于」)。

    **没被打分的命中也被丢**:层2 的依据是「这条证据被评过、且分数达标」,一条模型没给分的
    命中就是一次没有依据的引用。宁缺毋滥,否则「答案必须有证据」就退化成「答案大概有证据」。
    """
    kept: list[Hit] = []
    dropped: list[str] = []
    for hit in hits:
        scored = _numbers([scores.get(hit.chunk_id)])
        if not scored or scored[0] < theta_rel:
            dropped.append(hit.chunk_id)
            continue
        kept.append(hit)
    return kept, dropped


# --- 层3:定向重试 --------------------------------------------------------------------------------------
def norm_tokens(text: str) -> list[str]:
    """归一分词:小写、折叠空白,ASCII 词 ≥2 字符,中日韩按双字滑窗。

    单字母是噪声("a"、"x"),单字 CJK 是有效语素(「奖」「型」),所以两条路的长度下限不一样。
    """
    out: list[str] = []
    for piece in _TOKEN.findall(normalize_text(text)):
        if piece.isascii():
            if len(piece) >= 2:
                out.append(piece)
        elif len(piece) == 1:
            out.append(piece)
        else:
            out.extend(piece[i : i + 2] for i in range(len(piece) - 1))
    return list(dict.fromkeys(out))


def constraints_of(before: str, after: str) -> list[str]:
    """改写后新增的约束(归一 token 差集),顺序按新问题里的出现顺序,不去排序。

    顺序有意义:重试 payload 要把「这轮补了什么」原样展示给用户(FR-405),按字典序排会把
    「2024年 营收」拆成不相邻的两块,看不出是在补时间范围。
    """
    before_tokens = set(norm_tokens(before))
    return [t for t in norm_tokens(after) if t not in before_tokens]


@dataclass(frozen=True)
class RetryPlan:
    query: str
    constraints: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    attempt: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "constraints": list(self.constraints),
            "missing": list(self.missing),
            "attempt": self.attempt,
        }


def plan_retry(
    *,
    query: str,
    missing: Sequence[str],
    queries_tried: Sequence[str],
    new_constraints: Sequence[str],
    evidence_gain: int,
    attempt: int,
    max_attempts: int,
) -> RetryPlan | None:
    """给得出重试计划才重跑,否则返回 None(转拒答)。四个条件缺一个都不许跑:

    1. 还有预算(`attempt < max_attempts`);
    2. 有缺失项 —— 没有 `missing_info` 的重跑就是「随机再跑一次」,正是 FR-405 禁止的那件事;
    3. 归一后有**真正新**的词(在已跑过的问题里都没出现过),且整句归一后不等于任何跑过的问题
       —— 换语序、换标点都不算新查询;
    4. 上一轮确实新增了证据(`evidence_gain > 0`),一条新证据都没捞到,说明这个方向是死的。
    """
    if attempt >= max_attempts:
        return None
    if not [m for m in missing if str(m).strip()]:
        return None
    tried_tokens = [frozenset(norm_tokens(str(one))) for one in queries_tried]
    #: 整句归一后完全相同 = 换了语序或标点,依然是一次没有新信息的重跑。这一条比下面的
    #: 「新增约束」更严:双字滑窗会因为数字打断中文连写而把重排误报成新词(实测「2024年营收」
    #: vs「营收2024年」),只靠差集挡不住。
    if frozenset(norm_tokens(query)) in tried_tokens:
        return None
    tried: set[str] = set().union(*tried_tokens) if tried_tokens else set()
    fresh = [c for c in new_constraints if c and c not in tried]
    if not fresh:
        return None
    if evidence_gain <= 0:
        return None
    return RetryPlan(
        query=query,
        constraints=fresh,
        missing=[str(m) for m in missing if str(m).strip()],
        attempt=attempt + 1,
    )


# --- 层4:明确拒答 --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Refusal:
    refuse: bool
    reject_code: str = ""
    best_score: float = 0.0
    evidence: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "refuse": self.refuse,
            "reject_code": self.reject_code,
            "best_score": self.best_score,
            "evidence": self.evidence,
        }


def decide_refusal(
    kept: Sequence[Hit],
    scores: Mapping[str, Any],
    *,
    theta_answer: float,
    min_evidence: int = 1,
) -> Refusal:
    """过完 θ_rel 的证据还撑不起一个答案,就在这里判:证据数 < min_evidence,或最高分 < θ_answer。

    θ_answer(0.45)比 θ_rel(0.35)高是刻意的:进上下文 ≠ 够格回答。方向同样是严格小于,
    最高分正好等于 0.45 算达标。
    """
    best = max(_numbers(scores.get(h.chunk_id) for h in kept), default=0.0)
    if len(kept) < min_evidence:
        return Refusal(True, "no_evidence", best, len(kept))
    if best < theta_answer:
        return Refusal(True, "below_theta", best, len(kept))
    return Refusal(False, "", best, len(kept))


# --- 层5:引用编号程序侧硬校验 -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CiteCheck:
    text: str
    cited_ids: list[str] = field(default_factory=list)
    #: 被剥离的越界编号,原样进 trace:模型开始报不存在的编号是回归的信号
    stripped: list[int] = field(default_factory=list)
    #: 答案里所有被识别为引用的编号总数 / 其中指向真证据的数量。#8 的 citation_validity
    #: 是「有效编号占比」,不能拿去重后的 chunk_id 数当分母(多个编号可指同一 chunk)。
    numbers_total: int = 0
    numbers_valid: int = 0
    #: ok | warn(有剥离但仍有合法引用)| refuse
    verdict: str = "ok"

    @property
    def refused(self) -> bool:
        return self.verdict == "refuse"


def check_citations(
    answer: str,
    cite_ids: Mapping[str, str],
    *,
    evidence_ids: Iterable[str] | None = None,
) -> CiteCheck:
    """解析 `[n]` 并核对编号确实指向证据集里的切片(FR-408「程序侧硬校验,非模型自证」)。

    四条口径(文档只给了方向,这几处必须定死才测得动,已列入回填清单):

    1. 编号 **1-based**;全角方括号 `【1】` 与全角数字 `［１］` 都算(模型爱这么输出),
       `[1,2]` 拆成两个编号。
    2. 只有「方括号内全部是数字」才算引用候选;`[图1]`、`[详见第3章]`、markdown 链接不算,
       不剥离也不因此拒答 —— 它们是正文,不是伪造的引用。
    3. 越界(编号 > len(cite_ids))、映射缺失、或映射到的 chunk_id 不在证据集里 → 从正文
       **剥离并记录**,不重编号、不改写其余正文。
    4. 「零个可解析的引用」与「剥离后归零」都算 `refuse`:文档未写「缺失」算不算,保守按算 ——
       没有证据指针的答案正是这套防御要拦的东西。
    """
    known = {str(k): str(v) for k, v in cite_ids.items()}
    evidence = set(evidence_ids) if evidence_ids is not None else set(known.values())

    cited: list[str] = []
    stripped: list[int] = []
    candidates = 0
    numbers_total = 0
    numbers_valid = 0

    def _rewrite(match: re.Match[str]) -> str:
        nonlocal candidates, numbers_total, numbers_valid
        open_ch, inner, close_ch = match.group(1), match.group(2), match.group(3)
        parts = [p for p in _SPAN_SPLIT.split(inner.strip()) if p]
        digits = [p.translate(_FULLWIDTH) for p in parts]
        if not parts or not all(_PURE_NUM.match(d) for d in digits):
            return match.group(0)  # 不是引用样式:正文原样留着
        candidates += 1
        keep: list[str] = []
        for part, digits_ in zip(parts, digits, strict=True):
            numbers_total += 1
            num = int(digits_)
            chunk_id = known.get(digits_)
            if chunk_id is None or (evidence and chunk_id not in evidence):
                stripped.append(num)
                continue
            numbers_valid += 1
            keep.append(part)
            if chunk_id not in cited:
                cited.append(chunk_id)
        if not keep:
            return ""
        return f"{open_ch}{', '.join(keep)}{close_ch}" if len(keep) != len(parts) else match.group(0)

    text = _CITE_SPAN.sub(_rewrite, answer)

    if not candidates or not cited:
        verdict = "refuse"
    elif stripped:
        verdict = "warn"
    else:
        verdict = "ok"
    return CiteCheck(
        text=text,
        cited_ids=cited,
        stripped=stripped,
        numbers_total=numbers_total,
        numbers_valid=numbers_valid,
        verdict=verdict,
    )


# --- 中断确认:resume 入参 -------------------------------------------------------------------------------
def validate_resume(state: Mapping[str, Any], payload: Any) -> str:
    """把 `Command(resume=…)` 换成这一轮真正要检索的那句话。

    `edited` 非空就覆盖 `choice`(用户改写了,选哪条已经不重要);两者都不可用时抛
    `IllegalTransition`,**绝不静默回退原问** —— 静默回退等于跳过整个确认环节,FR-402 白做。
    """
    candidates = list(state.get("rewrite_candidates") or [])
    edited = str(_field(payload, "edited", "") or "").strip()
    if edited:
        if len(edited) < EDITED_MIN or len(edited) > EDITED_MAX:
            raise IllegalTransition(f"edited 长度必须在 {EDITED_MIN}..{EDITED_MAX} 之间")
        if _CONTROL.search(edited):
            raise IllegalTransition("edited 含控制字符")
        return edited

    raw_choice = _field(payload, "choice", 0)
    try:
        index = int(raw_choice)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise IllegalTransition(f"choice 不是整数: {raw_choice!r}") from exc
    if not 1 <= index <= len(candidates):
        raise IllegalTransition(f"choice 越界: {index} 不在 1..{len(candidates)}")
    text = str(_field(candidates[index - 1], "text", "") or "").strip()
    if not text:
        raise IllegalTransition(f"第 {index} 条改写候选是空串,无法确认")
    return text
