"""模型网关:chat / embedding / rerank / vision 的唯一出口。

三件事在这里固化,不允许每个调用点自己拼(FR-407):

1. **思考模式与工具/结构化输出互斥**:凡是走严格 schema 或带 tools 的调用,一律显式
   `enable_thinking=False`。实测 qwen3.8-flash 默认开思考,一句两字回复计费 29 个 reasoning
   token、输入从 30 涨到 66 —— 意图路由每条问题都要跑,不关就是全线延迟与成本翻倍。
2. **并发单独收口**:`llm_max_concurrency` 的小并发 + 超时 + 重试,worker 数量放大不会打爆配额。
3. **失败要说得出原因**:重试耗尽抛 `ModelUnavailable`,上层按 §10 的降级矩阵标记 degraded,
   而不是拿到空结果继续跑。

同步实现:Celery worker 与 LangGraph 的 `invoke` 都是同步栈;异步调用方用
`fastapi.concurrency.run_in_threadpool` 包一层,不要在网关里再维护第二个客户端。
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from kbmesh.core.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

RERANK_NATIVE_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"


class ModelUnavailable(RuntimeError):
    """重试耗尽或配置缺失 —— 调用方必须记 degraded,不能当空结果继续。"""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail


@dataclass
class Usage:
    token_in: int = 0
    token_out: int = 0
    latency_ms: int = 0
    model: str = ""

    def merge(self, other: Usage) -> Usage:
        return Usage(
            self.token_in + other.token_in,
            self.token_out + other.token_out,
            self.latency_ms + other.latency_ms,
            other.model or self.model,
        )


@dataclass
class ChatResult:
    text: str
    usage: Usage
    parsed: Any = None
    attempts: int = 1


@dataclass
class RerankHit:
    index: int
    score: float


def _no_thinking() -> dict[str, Any]:
    return {"enable_thinking": False}


class Gateway:
    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._gate = threading.BoundedSemaphore(max(1, self.settings.llm_max_concurrency))
        self._last_rerank_shape: str | None = None

    # ------------------------------------------------------------------ clients
    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.settings.dashscope_api_key or self.settings.dashscope_api_key == "sk-xxx":
                raise ModelUnavailable("gateway", "DASHSCOPE_API_KEY 未配置(占位 sk-xxx 不可用)")
            self._client = OpenAI(
                api_key=self.settings.dashscope_api_key,
                base_url=self.settings.dashscope_base_url,
                timeout=self.settings.llm_timeout_s,
                max_retries=0,  # 重试由下面这一层管,不要两套叠加成乘法
            )
        return self._client

    def _retry(self, stage: str, fn: Callable[[], Any]) -> Any:
        """统一重试:限流与超时退避,其他 APIError 直接抛(参数错重试一百次也是错)。"""
        delay = 1.0
        last = ""
        for attempt in range(self.settings.llm_max_retries + 1):
            with self._gate:
                try:
                    return fn()
                except (RateLimitError, APITimeoutError, httpx.HTTPError) as exc:
                    last = f"{type(exc).__name__}: {str(exc)[:160]}"
                except APIError as exc:
                    raise ModelUnavailable(stage, f"{type(exc).__name__}: {str(exc)[:160]}") from exc
            if attempt < self.settings.llm_max_retries:
                time.sleep(delay)
                delay *= 2
        raise ModelUnavailable(stage, last or "unknown")

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        prompt: str,
        *,
        system: str = "",
        purpose: str = "generate",
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> ChatResult:
        model = self._model_for(purpose)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        extra: dict[str, Any] = {}
        if not self.settings.llm_enable_thinking:
            extra["extra_body"] = _no_thinking()

        def call():
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )

        t0 = time.perf_counter()
        resp = self._retry(f"chat:{purpose}", call)
        return ChatResult(
            text=(resp.choices[0].message.content or "").strip(),
            usage=self._usage(resp, model, t0),
        )

    def structured(
        self, prompt: str, schema: type[T], *, purpose: str = "route", system: str = ""
    ) -> tuple[T, Usage]:
        """四类必用结构化输出的节点都走这里(§7.1):严格 schema + 自修复重试,不解析散文。

        思考模式在这条路上永远关掉:带 tools 或要严格 JSON 时叠加思考会掉进「解释写得漂亮但
        schema 不合规」的坑。

        现网实测(`scripts/verify_models.py` C7,2026-11-06):四类一次通过、没触发自修复重试 ——
        包括 `RewriteOut.candidates`、`RelevanceOut.scores` 这种「数组字段带 default」的写法,
        strict json_schema 照样收。单条 0.8–3.8 秒,判分档位实测落在 0.6–0.7(θ_rel=0.35 /
        θ_answer=0.45 的门槛据此仍在有效区间内)。
        """
        envelope = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }
        base_system = (
            "只输出符合给定 JSON Schema 的一个 JSON 对象,不要输出解释、代码块或多余字段。"
            f"\nSchema:\n{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        )
        messages = [{"role": "system", "content": (system + "\n" + base_system).strip()}]
        messages.append({"role": "user", "content": prompt})

        usage = Usage(model=self._model_for(purpose))
        last_error = ""
        for _round in range(self.settings.llm_max_retries + 1):
            pinned = list(messages)

            def call(messages=pinned):
                return self.client.chat.completions.create(
                    model=self._model_for(purpose),
                    messages=messages,
                    temperature=0,
                    response_format=envelope,
                    extra_body=_no_thinking(),
                )

            t0 = time.perf_counter()
            resp = self._retry(f"structured:{purpose}", call)
            usage = usage.merge(self._usage(resp, usage.model, t0))
            raw = (resp.choices[0].message.content or "").strip()
            try:
                return schema.model_validate_json(_strip_fences(raw)), usage
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:400]
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {"role": "user", "content": f"上一次输出不符合 schema:{last_error}\n只重发合法 JSON。"}
                )
        raise ModelUnavailable(
            f"structured:{purpose}",
            f"schema 自修复 {self.settings.llm_max_retries} 次仍失败: {last_error}",
        )

    # ------------------------------------------------------------------ embedding
    def embed(self, texts: Sequence[str], *, purpose: str = "embed") -> tuple[list[list[float]], Usage]:
        """分批向量化。批大小是配置项:不同 embedding 档位的单请求上限不一样。"""
        if not texts:
            return [], Usage(model=self.settings.embedding_model)
        out: list[list[float]] = []
        usage = Usage(model=self.settings.embedding_model)
        batch = max(1, self.settings.embedding_batch_size)
        for start in range(0, len(texts), batch):
            piece = list(texts[start : start + batch])

            def call(batch=piece):
                return self.client.embeddings.create(model=self.settings.embedding_model, input=batch)

            t0 = time.perf_counter()
            resp = self._retry("embed", call)
            out.extend(list(item.embedding) for item in resp.data)
            usage = usage.merge(
                Usage(
                    token_in=int(getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0),
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    model=self.settings.embedding_model,
                )
            )

        # 维度在建集合时定死,跑错档位的向量会整库重放 —— 在这里响,不要等 Milvus 报
        if out and len(out[0]) != self.settings.embedding_dim:
            raise ModelUnavailable(
                "embed",
                f"维度不符:模型给 {len(out[0])},配置 {self.settings.embedding_dim};"
                "换 embedding 模型必须同步重索引(见 KBMESH_EMBEDDING_DIM)",
            )
        return out, usage

    # ------------------------------------------------------------------ rerank
    def rerank(self, query: str, documents: Sequence[str], *, top_n: int | None = None) -> list[RerankHit]:
        """精排。两条真实形状按实测顺序试,成功的形状记下来供后续请求直接走。"""
        if not documents:
            return []
        key = self.settings.dashscope_api_key
        if not key or key == "sk-xxx":
            raise ModelUnavailable("rerank", "DASHSCOPE_API_KEY 未配置")
        n = top_n or len(documents)
        shapes = [
            (
                "native",
                RERANK_NATIVE_URL,
                {
                    "model": self.settings.rerank_model,
                    "input": {"query": query, "documents": list(documents)},
                    "parameters": {"top_n": n, "return_documents": False},
                },
                lambda j: j["output"]["results"],
            ),
            (
                "compat",
                RERANK_COMPAT_URL,
                {
                    "model": self.settings.rerank_model,
                    "query": query,
                    "documents": list(documents),
                    "top_n": n,
                },
                lambda j: j["results"],
            ),
        ]
        if self._last_rerank_shape:
            shapes.sort(key=lambda s: 0 if s[0] == self._last_rerank_shape else 1)

        errors: list[str] = []
        for label, url, body, extract in shapes:
            def call(target=url, payload=body):
                return httpx.post(
                    target,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.settings.llm_timeout_s,
                )

            try:
                with self._gate:
                    resp = self._retry(f"rerank:{label}", call)
                if resp.status_code != 200:
                    errors.append(f"{label}: HTTP {resp.status_code} {resp.text[:120]}")
                    continue
                results = extract(resp.json())
            except (ModelUnavailable, KeyError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{label}: {str(exc)[:120]}")
                continue

            self._last_rerank_shape = label
            return [
                RerankHit(
                    index=int(r.get("index", r.get("document", {}).get("idx", -1))),
                    score=float(r.get("relevance_score", r.get("score", 0.0))),
                )
                for r in results
            ]
        raise ModelUnavailable("rerank", "; ".join(errors) or "两种接口形状都不可用")

    # ------------------------------------------------------------------ vision
    def vision(self, image_url: str, prompt: str, *, purpose: str = "vision") -> ChatResult:
        """图片描述/OCR 回填只走这一条:omni 档位,非流式实测可用。"""
        model = self._model_for(purpose)

        def call():
            return self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                **({} if self.settings.llm_enable_thinking else {"extra_body": _no_thinking()}),
            )

        t0 = time.perf_counter()
        resp = self._retry(f"vision:{purpose}", call)
        return ChatResult(
            text=(resp.choices[0].message.content or "").strip(),
            usage=self._usage(resp, model, t0),
        )

    # ------------------------------------------------------------------ helpers
    def _model_for(self, purpose: str) -> str:
        return {
            "route": self.settings.router_model,
            "assess": self.settings.router_model,
            "rewrite": self.settings.router_model,
            "vision": self.settings.vision_model,
            "embed": self.settings.embedding_model,
        }.get(purpose, self.settings.llm_model)

    @staticmethod
    def _usage(resp: Any, model: str, started: float) -> Usage:
        u = getattr(resp, "usage", None)
        return Usage(
            token_in=int(getattr(u, "prompt_tokens", 0) or 0),
            token_out=int(getattr(u, "completion_tokens", 0) or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=model,
        )


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[1:] if text.startswith("json") else text
    return text.strip()


_default: Gateway | None = None


def get_gateway() -> Gateway:
    global _default
    if _default is None:
        _default = Gateway()
    return _default
