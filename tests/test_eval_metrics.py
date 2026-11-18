"""#8 四层指标的唯一算法(FR-603):检索 / 答案 / Agent / 系统。

每个断言都钉死一个口径边界(空 gold、重复 chunk、越界引用号、无重试样本),
避免「总分看着还行,换个人算就是另一个数」。
"""
from __future__ import annotations

import math

import pytest

from kbmesh.eval.metrics import (
    AgentCase,
    agent_metrics,
    citation_validity,
    retrieval_metrics,
    system_metrics,
)


# --------------------------------------------------------------------- 检索层
def test_retrieval_metrics_use_ranked_top_k() -> None:
    metrics = retrieval_metrics(["a", "b"], ["x", "a", "b"], k=2)
    assert metrics is not None
    assert metrics.recall_at_k == pytest.approx(0.5)  # top-2 只捞到 a
    assert metrics.mrr == pytest.approx(0.5)  # 第一个相关项在第 2 位
    # DCG = 1/log2(3) + 1/log2(4);IDCG = 1 + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_10 == pytest.approx(expected)


def test_retrieval_metrics_without_gold_is_unmeasurable() -> None:
    assert retrieval_metrics([], ["a"], k=5) is None


def test_retrieval_metrics_rank_full_recall() -> None:
    metrics = retrieval_metrics(["a"], ["a", "b"], k=5)
    assert metrics is not None
    assert metrics.recall_at_k == 1.0 and metrics.mrr == 1.0 and metrics.ndcg_at_10 == 1.0


# --------------------------------------------------------------------- 答案层
def test_citation_validity_counts_only_real_evidence() -> None:
    score = citation_validity(
        "隔离靠分区键 [1][2][9]。",
        {"1": "c1", "2": "c2"},
        evidence_ids={"c1", "c2"},
    )
    assert score == pytest.approx(2 / 3)


def test_citation_validity_zero_when_answer_has_no_parseable_citation() -> None:
    assert citation_validity("没有编号的答案。", {"1": "c1"}, evidence_ids={"c1"}) == 0.0


# --------------------------------------------------------------------- Agent 层
def _agent(**over: object) -> AgentCase:
    base = {
        "expected_refusal": False,
        "final_status": "answered",
        "reject_code": "",
        "rewrite_candidates": [{"index": 1, "text": "q"}],
        "user_choice": {"choice": 1},
        "retry_attempt": 0,
    }
    base.update(over)
    return AgentCase(**base)  # type: ignore[arg-type]


def test_agent_metrics_denominators_exclude_inapplicable_cases() -> None:
    cases = [
        _agent(),  # 接受改写
        _agent(user_choice={"edited": "自己写的问题"}),  # 拒绝候选,改用自己的词
        _agent(rewrite_candidates=[], user_choice={"edited": "q"}),  # 没候选,不进分母
        _agent(retry_attempt=1, final_status="answered", user_choice={}),  # 重试成功
        _agent(retry_attempt=2, final_status="refused", user_choice={}),  # 重试失败
        _agent(expected_refusal=True, final_status="refused", user_choice={}),  # 拒答正确
        _agent(expected_refusal=True, final_status="answered", user_choice={}),  # 拒答错误
    ]
    metrics = agent_metrics(cases)
    assert metrics["rewrite_accept_rate"] == pytest.approx(0.5)  # 1 接受 / 2 有候选
    assert metrics["retry_success_rate"] == pytest.approx(0.5)  # 1 成功 / 2 重试
    assert metrics["refuse_correctness"] == pytest.approx(0.5)  # 1 正确 / 2 应拒答


def test_agent_metrics_with_no_applicable_cases_are_zero() -> None:
    metrics = agent_metrics([_agent(rewrite_candidates=[])])
    assert metrics == {
        "rewrite_accept_rate": 0.0,
        "retry_success_rate": 0.0,
        "refuse_correctness": 0.0,
    }


# --------------------------------------------------------------------- 系统层
def test_system_metrics_rates_and_percentiles() -> None:
    summary = system_metrics(
        latencies=[100, 200, 300, 400],
        token_in=[10, 20, 0, 0],
        token_out=[5, 0, 0, 0],
        cost_usd=[0.01, 0.02, 0.0, 0.0],
        statuses=["answered", "refused", "error", "expired"],
        degraded=[False, False, True, False],
    )
    assert summary["runs"] == 4
    assert summary["refusal_rate"] == pytest.approx(0.25)
    assert summary["error_rate"] == pytest.approx(0.25)
    assert summary["degraded_rate"] == pytest.approx(0.25)
    assert summary["latency_ms"] == {"p50": pytest.approx(250.0), "p95": pytest.approx(385.0)}
    assert summary["token_in"] == 30 and summary["token_out"] == 5
    assert summary["cost_usd"] == pytest.approx(0.03)
