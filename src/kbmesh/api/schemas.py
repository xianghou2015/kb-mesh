"""API 请求体 Schema(#7)。

校验规则尽量前移到这里:422 的错误比 409 便宜,也比让 runtime 抛 IllegalTransition 便宜。
`thread_id` 的字符集在这里挡(`runtime.assert_thread_shape` 还会再挡一遍 —— 纵深防御,
两个入口都进得来的东西才敢依赖)。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LoginIn(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=64)
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=200)


class AskIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    thread_id: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_-]+$")
    #: 检索的 kb 收窄项。给了就必须在令牌的 kb_scope 里;不给 = 令牌允许的整个租户范围。
    kb_id: str | None = None


class ResumeIn(BaseModel):
    choice: int = Field(default=0, ge=0)
    edited: str = Field(default="", max_length=512)
    #: 与 `ask` 的 kb_id 同口径:重试路(retry_targeted)会用同一个 kb 再检索一轮,
    #: 断线重连后必须带上同一个值,否则第二轮召回的过滤条件和第一轮不一致。
    kb_id: str | None = None


class FeedbackIn(BaseModel):
    #: -1 踩 / 0 中性 / 1 赞。口径先收窄 —— 宽了评估集导入时没法分层(#8)。
    score: int = Field(ge=-1, le=1)
    corrected_answer: str | None = Field(default=None, max_length=4000)


class EvalCaseIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    reference_answer: str | None = Field(default=None, max_length=4000)
    gold_refs: list[str] = Field(default_factory=list, max_length=32)
    group_key: str = Field(default="default", max_length=40)
    expected_refusal: bool = False


class EvalSetIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    cases: list[EvalCaseIn] = Field(default_factory=list, max_length=200)


class EvalSetFromFeedbackIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    #: None = 全部反馈;-1 = 只导入踩(负样本优先人工确认)。
    max_score: int | None = Field(default=None, ge=-1, le=1)
    limit: int = Field(default=200, ge=1, le=500)


class EvalRunIn(BaseModel):
    set_id: str = Field(min_length=1)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    baseline_run_id: str | None = None
