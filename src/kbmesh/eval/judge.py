"""LLM judge(FR-603):faithfulness / answer_relevancy。

两条硬口径:
- **双次采样一致才计入**(技术文档 §8.1):两次分数差超过 0.1 视为 judge 不稳定,该样本返回
  None,不写成 0 —— 「判不了」和「判成 0 分」在报告里必须是两件事。
- **judge 不可用不阻断评估**:`ModelUnavailable` 一律收敛成 None,运行器跳过该指标。
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from kbmesh.core.gateway import Gateway, ModelUnavailable

CONSISTENCY_TOLERANCE = 0.1


class FaithfulnessOut(BaseModel):
    supported: int = Field(ge=0)
    total: int = Field(ge=0)
    reason: str = Field(max_length=500)


class AnswerRelevancyOut(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=500)


def _truncate(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


class GatewayJudge:
    def __init__(self, gateway: Gateway | Any) -> None:
        self.gateway = gateway

    async def _twice(self, prompt: str, schema: type[BaseModel]) -> list[BaseModel] | None:
        out: list[BaseModel] = []
        for _ in range(2):
            try:
                parsed, _usage = await asyncio.to_thread(
                    self.gateway.structured, prompt, schema, purpose="judge"
                )
            except ModelUnavailable:
                return None
            out.append(parsed)
        return out

    async def faithfulness(
        self, question: str, answer: str, evidence: Sequence[str]
    ) -> float | None:
        """faithfulness = 断言中被证据支持的比例;两次采样的比例差 ≤ 0.1 才计入。"""
        joined = "\n".join(f"[{index}] {_truncate(text, 800)}" for index, text in enumerate(evidence, 1))
        prompt = (
            "判断答案里的每个事实断言是否都能被给定证据支持。只数断言,不要评价风格。\n"
            f"问题:{_truncate(question)}\n答案:{_truncate(answer)}\n证据:\n{joined or '(无)'}\n"
            "输出 supported(被支持的断言数)/ total(断言总数)/ reason。"
        )
        samples = await self._twice(prompt, FaithfulnessOut)
        if samples is None:
            return None
        first, second = samples
        assert isinstance(first, FaithfulnessOut) and isinstance(second, FaithfulnessOut)
        ratios = [
            (one.supported / one.total) if one.total else 0.0 for one in (first, second)
        ]
        if abs(ratios[0] - ratios[1]) > CONSISTENCY_TOLERANCE:
            return None
        return (ratios[0] + ratios[1]) / 2

    async def answer_relevancy(self, question: str, answer: str) -> float | None:
        prompt = (
            "判断答案是否直接回答了问题(0=答非所问,1=完全切题)。不要因为答案长就给高分。\n"
            f"问题:{_truncate(question)}\n答案:{_truncate(answer)}\n"
            "输出 score(0~1)/ reason。"
        )
        samples = await self._twice(prompt, AnswerRelevancyOut)
        if samples is None:
            return None
        first, second = samples
        assert isinstance(first, AnswerRelevancyOut) and isinstance(second, AnswerRelevancyOut)
        if abs(first.score - second.score) > CONSISTENCY_TOLERANCE:
            return None
        return (first.score + second.score) / 2
