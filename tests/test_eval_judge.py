"""#8 LLM judge:双次采样一致才计入,模型不可用返回 None(样本跳过而不是记 0 分)。"""
from __future__ import annotations

import pytest

from kbmesh.core.gateway import ModelUnavailable, Usage
from kbmesh.eval.judge import GatewayJudge


class FakeGateway:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def structured(self, prompt: str, schema: type, *, purpose: str = "", system: str = ""):
        del prompt, purpose, system
        self.calls += 1
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            item = schema(**item)
        return item, Usage()


async def test_faithfulness_takes_ratio_when_two_samples_agree() -> None:
    from kbmesh.eval.judge import FaithfulnessOut

    gateway = FakeGateway(
        [
            FaithfulnessOut(supported=3, total=4, reason="a"),
            FaithfulnessOut(supported=3, total=4, reason="b"),
        ]
    )
    score = await GatewayJudge(gateway).faithfulness("q", "answer", ["e1", "e2"])
    assert score == pytest.approx(0.75)
    assert gateway.calls == 2


async def test_faithfulness_is_none_when_samples_disagree() -> None:
    from kbmesh.eval.judge import FaithfulnessOut

    gateway = FakeGateway(
        [
            FaithfulnessOut(supported=1, total=4, reason="a"),
            FaithfulnessOut(supported=4, total=4, reason="b"),
        ]
    )
    assert await GatewayJudge(gateway).faithfulness("q", "answer", ["e"]) is None


async def test_answer_relevancy_is_none_when_model_unavailable() -> None:
    gateway = FakeGateway([ModelUnavailable("structured:assess", "quota")])
    assert await GatewayJudge(gateway).answer_relevancy("q", "answer") is None


async def test_answer_relevancy_uses_mean_of_consistent_samples() -> None:
    from kbmesh.eval.judge import AnswerRelevancyOut

    gateway = FakeGateway(
        [
            AnswerRelevancyOut(score=0.8, reason="a"),
            AnswerRelevancyOut(score=0.8, reason="b"),
        ]
    )
    assert await GatewayJudge(gateway).answer_relevancy("q", "answer") == pytest.approx(0.8)
