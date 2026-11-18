"""#8 在线规则检查(FR-601/602):格式 / 引用 / 安全全量跑,judge 抽样。"""
from __future__ import annotations

from kbmesh.eval.online import OnlineInput, rule_checks, should_judge


def _input(**over: object) -> OnlineInput:
    base = {
        "question": "隔离怎么做?",
        "answer": "隔离靠分区键 [1]。",
        "final_status": "answered",
        "cite_ids": {"1": "c1"},
        "evidence_ids": {"c1"},
    }
    base.update(over)
    return OnlineInput(**base)  # type: ignore[arg-type]


def test_answered_with_valid_citation_passes_all_rules() -> None:
    report = rule_checks(_input())
    assert report.failed == []
    assert report.format_ok and report.citation_ok and report.safety_ok


def test_answered_without_citation_fails_citation_rule() -> None:
    report = rule_checks(_input(answer="没有编号的答案。"))
    assert report.citation_ok is False
    assert "citation" in report.failed


def test_refused_answer_does_not_need_citations() -> None:
    report = rule_checks(_input(answer="证据不足", final_status="refused", cite_ids={}, evidence_ids=set()))
    assert report.citation_ok is True
    assert report.failed == []


def test_empty_answer_fails_format_rule() -> None:
    report = rule_checks(_input(answer="   "))
    assert report.format_ok is False
    assert "format" in report.failed


def test_secret_leak_fails_safety_rule() -> None:
    report = rule_checks(_input(answer="答案 [1]，key=sk-test-only-secret"))
    assert report.safety_ok is False
    assert "safety" in report.failed


def test_sampling_boundaries() -> None:
    assert should_judge(0.0, draw=0.0) is False
    assert should_judge(1.0, draw=0.999) is True
    assert should_judge(0.1, draw=0.05) is True
    assert should_judge(0.1, draw=0.2) is False
