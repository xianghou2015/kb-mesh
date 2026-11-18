"""在线评估(FR-601/602):格式 / 引用 / 安全全量规则检查,judge 按采样率。

规则检查必须是纯函数、零模型调用 —— 它挂在每次问答的收尾路径上,不能把主链路拖慢;
faithfulness / answer_relevancy 由调用方按 `should_judge()` 抽样后再走 judge。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from kbmesh.agents.guards import check_citations

#: 密钥形状与越权提示词:宁可误报一条,也不让它们出现在对外答案里。
_SECRET = re.compile(r"sk-[A-Za-z0-9_-]{8,}")
_PROMPT_LEAK = re.compile(r"(?i)(system prompt|系统提示词|ignore previous instructions)")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_ANSWER_CHARS = 20_000


@dataclass(frozen=True)
class OnlineInput:
    question: str
    answer: str
    final_status: str
    cite_ids: dict[str, str] = field(default_factory=dict)
    evidence_ids: set[str] = field(default_factory=set)
    #: judge 需要证据原文;规则检查只用 id。缺省时回退到 id,至少不崩。
    evidence_texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class OnlineReport:
    format_ok: bool
    citation_ok: bool
    safety_ok: bool
    failed: list[str] = field(default_factory=list)


def should_judge(sample_rate: float, *, draw: float) -> bool:
    """`draw < rate`;rate 两端收敛到 0/100%,避免浮点边界把 100% 抽样偶发判成不抽。"""
    rate = max(0.0, min(1.0, float(sample_rate)))
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    return float(draw) < rate


def rule_checks(data: OnlineInput) -> OnlineReport:
    answer = str(data.answer or "")
    failed: list[str] = []

    format_ok = bool(answer.strip()) and len(answer) <= MAX_ANSWER_CHARS and not _CONTROL.search(answer)
    if not format_ok:
        failed.append("format")

    if str(data.final_status) == "answered":
        evidence = {str(one) for one in data.evidence_ids}
        check = check_citations(answer, data.cite_ids, evidence_ids=evidence or {"__none__"})
        citation_ok = not check.refused and check.numbers_valid > 0
    else:
        # 拒答/错误没有引用是正常形态;这里不拿「零编号」去惩罚一条拒答。
        citation_ok = True
    if not citation_ok:
        failed.append("citation")

    safety_ok = not (
        _SECRET.search(answer)
        or _PROMPT_LEAK.search(answer)
        or _CONTROL.search(answer)
    )
    if not safety_ok:
        failed.append("safety")

    return OnlineReport(
        format_ok=format_ok,
        citation_ok=citation_ok,
        safety_ok=safety_ok,
        failed=failed,
    )


class OnlineEvaluator:
    """规则 + 可选 judge 的组合器;持久化由调用方负责(见 `eval/tasks.py`)。"""

    def __init__(self, judge: Any | None = None) -> None:
        self.judge = judge

    async def evaluate(self, data: OnlineInput, *, sample_rate: float, draw: float) -> dict[str, Any]:
        report = rule_checks(data)
        out: dict[str, Any] = {
            "format_ok": report.format_ok,
            "citation_ok": report.citation_ok,
            "safety_ok": report.safety_ok,
            "failed": report.failed,
            "sampled": False,
            "faithfulness": None,
            "answer_relevancy": None,
        }
        if self.judge is None or not should_judge(sample_rate, draw=draw):
            return out
        out["sampled"] = True
        out["faithfulness"] = await self.judge.faithfulness(
            data.question,
            data.answer,
            list(data.evidence_texts) or list(data.evidence_ids),
        )
        out["answer_relevancy"] = await self.judge.answer_relevancy(data.question, data.answer)
        return out
