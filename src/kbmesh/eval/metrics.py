"""四层指标的**唯一计算口径**(FR-603,技术文档 §8.1)。

每个函数只吃纯数据,不连库、不调模型:检索层看 gold 与名次,答案层看引用编号,
Agent 层看确认/重试/拒答信号,系统层看 trace_run 的列。LLM judge 在 `judge.py`,
这里不做「差不多算通过」的模糊判定。
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kbmesh.agents.guards import check_citations
from kbmesh.observability.aggregate import percentile


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    ndcg_at_10: float


def retrieval_metrics(
    gold_refs: Sequence[str], retrieved_ids: Sequence[str], *, k: int
) -> RetrievalMetrics | None:
    """`gold = eval_case.gold_refs`;`k` 取精排后进入上下文的 top-N。

    gold 为空(反馈导入的题目还没有人工标注)时返回 None:这种样本不进召回均值,
    否则「没有标准答案」会伪装成 0 分。
    """
    gold = {str(one) for one in gold_refs if str(one)}
    if not gold:
        return None
    ranked = [str(one) for one in retrieved_ids]
    top_k = ranked[: max(1, int(k))]
    recall = len(gold & set(top_k)) / len(gold)

    mrr = 0.0
    for index, chunk_id in enumerate(ranked, start=1):
        if chunk_id in gold:
            mrr = 1.0 / index
            break

    dcg = 0.0
    seen: set[str] = set()
    for index, chunk_id in enumerate(ranked[:10], start=1):
        if chunk_id in gold and chunk_id not in seen:
            seen.add(chunk_id)
            dcg += 1.0 / math.log2(index + 1)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(len(gold), 10) + 1))
    ndcg = dcg / ideal if ideal else 0.0
    return RetrievalMetrics(recall_at_k=recall, mrr=mrr, ndcg_at_10=ndcg)


def citation_validity(
    answer: str,
    cite_ids: Mapping[str, str],
    *,
    evidence_ids: Iterable[str],
) -> float:
    """程序侧可解析且指向真证据的**编号**占比(不是去重后的 chunk 数占比)。

    证据集为空时答案里的任何编号都不可能合法,直接 0;没有引用编号同样是 0。
    """
    evidence = {str(one) for one in evidence_ids}
    if not evidence:
        return 0.0
    check = check_citations(answer, cite_ids, evidence_ids=evidence)
    if not check.numbers_total:
        return 0.0
    return check.numbers_valid / check.numbers_total


@dataclass(frozen=True)
class AgentCase:
    """Agent 层指标的输入信号,通常由 `Outcome.state` 直接映射。"""

    expected_refusal: bool
    final_status: str
    reject_code: str = ""
    rewrite_candidates: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    user_choice: Mapping[str, Any] = field(default_factory=dict)
    retry_attempt: int = 0


def _rewrite_accepted(case: AgentCase) -> bool | None:
    candidates = list(case.rewrite_candidates)
    if not candidates or not case.user_choice:
        return None  # 没给候选、或用户没做确认:两种都不进分母
    edited = str(case.user_choice.get("edited") or "").strip()
    if edited:
        return False  # 用户改了自己的词 = 没有接受任何候选
    try:
        choice = int(case.user_choice.get("choice") or 0)
    except (TypeError, ValueError):
        return False
    return 1 <= choice <= len(candidates)


def _retry_succeeded(case: AgentCase) -> bool | None:
    if int(case.retry_attempt or 0) <= 0:
        return None
    return str(case.final_status) == "answered"


def _refuse_correct(case: AgentCase) -> bool | None:
    if not case.expected_refusal:
        return None
    return str(case.final_status) == "refused"


def _mean(values: Sequence[bool]) -> float:
    return (sum(1 for one in values if one) / len(values)) if values else 0.0


def agent_case_metrics(case: AgentCase) -> dict[str, bool | None]:
    """单 case 的三个布尔信号;None = 这个 case 不适用,不写行也不进分母。"""
    return {
        "rewrite_accept": _rewrite_accepted(case),
        "retry_success": _retry_succeeded(case),
        "refuse_correct": _refuse_correct(case),
    }


def agent_metrics(cases: Sequence[AgentCase]) -> dict[str, float]:
    """三个比率各自的分母只算「适用样本」:没有候选 / 没有重试 / 不应拒答都不进分母。"""
    per_case = [agent_case_metrics(case) for case in cases]
    return {
        "rewrite_accept_rate": _mean(
            [row["rewrite_accept"] for row in per_case if row["rewrite_accept"] is not None]
        ),
        "retry_success_rate": _mean(
            [row["retry_success"] for row in per_case if row["retry_success"] is not None]
        ),
        "refuse_correctness": _mean(
            [row["refuse_correct"] for row in per_case if row["refuse_correct"] is not None]
        ),
    }


def system_metrics(
    *,
    latencies: Sequence[float],
    token_in: Sequence[int],
    token_out: Sequence[int],
    cost_usd: Sequence[float],
    statuses: Sequence[str],
    degraded: Sequence[bool],
) -> dict[str, Any]:
    """按 run 聚合的系统层:量、P50/P95、token、成本、错误率、降级率。

    `refusal_rate` 的分母是全部 run —— 与 #9 的租户视图同一口径,两处不能各算各的。
    """
    total = len(statuses)
    refused = sum(1 for one in statuses if one == "refused")
    errors = sum(1 for one in statuses if one == "error")
    degraded_n = sum(1 for one in degraded if one)

    def rate(n: int) -> float:
        return (n / total) if total else 0.0

    return {
        "runs": total,
        "refusal_rate": rate(refused),
        "error_rate": rate(errors),
        "degraded_rate": rate(degraded_n),
        "latency_ms": {
            "p50": percentile(list(latencies), 50),
            "p95": percentile(list(latencies), 95),
        },
        "token_in": sum(int(one) for one in token_in),
        "token_out": sum(int(one) for one in token_out),
        "cost_usd": sum(float(one) for one in cost_usd),
    }
