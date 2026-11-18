"""FR-407 的四类结构化输出 Schema。

网关 `Gateway.structured()` 会把这里的 `model_json_schema()` 原样塞进 strict json_schema,
所以字段形状有两条硬规矩,写错不是「模型不听话」而是「接口直接报错」:

1. 枚举一律 `Literal`,不用 `str` + 运行时校验:白名单进 Schema 之后模型侧就学不会越界值
   (图谱路的 `MentionsPayload` 同一个套路,见 `retrieval/graph_route.py`)。
2. 标量一律 required 无默认。数组字段带 `default_factory` 是**待实测项**:strict 模式是否接受
   非 required 字段,由 `scripts/verify_models.py` 的 C7 探针给结论,结论不合规就改成 required。

数值边界也写进 Schema(`ge/le`),不靠节点里再 clamp 一次:模型给的 1.7 分必须当场被拒,
而不是悄悄变成一个「刚好满足 θ」的分数。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: §7.3 层1 的四档判定。可答之外全都不进检索。
Verdict = Literal["answerable", "chitchat", "out_of_privilege", "out_of_scope"]


class IntentOut(BaseModel):
    """意图路由:越权与超范围在这里就被挡下,不给它任何靠近证据集的机会。"""

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=120)


class RewriteCandidate(BaseModel):
    text: str = Field(min_length=2, max_length=512)
    #: 这条候选在补哪一块(时间范围/主体/口径),用户看这个决定选哪条
    focus: str = Field(max_length=80)
    rationale: str = Field(max_length=120)


class RewriteOut(BaseModel):
    """查询改写候选,也复用作定向重试的新查询来源(§7.3 层3 的「带新增约束」就看它的 text)。"""

    candidates: list[RewriteCandidate] = Field(default_factory=list, max_length=4)


class RelScore(BaseModel):
    chunk_id: str = Field(max_length=64)
    score: float = Field(ge=0.0, le=1.0)
    #: 这条证据缺什么(层3 的输入)。注意它不参与 θ 判定,只是给重试指方向。
    gap: str = Field(max_length=120)


class RelevanceOut(BaseModel):
    scores: list[RelScore] = Field(default_factory=list, max_length=32)
    #: 本轮汇总出来的缺失项:缺哪一类事实、缺哪个时间段(FR-405 的「重试请求中能看到缺失项」)
    missing_info: list[str] = Field(default_factory=list, max_length=8)


class ResumeChoice(BaseModel):
    """`Command(resume=…)` 的入参形状(#7 的 API 层直接用它校验请求体)。

    这不是给模型看的 Schema,所以允许默认值:不选就是 `choice=0`,不改写就是空串,
    两者同时不可用时由 `guards.validate_resume()` 抛,而不是静默回退原问。
    """

    choice: int = Field(default=0, ge=0)
    edited: str = Field(default="", max_length=512)
