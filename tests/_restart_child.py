"""`test_resume_after_restart.py` 的子进程:一个文件跑两段,父进程负责把它们隔开。

    mode=a  走到 `await_confirm` 停住 → 打印 `READY {json}` → **自我 SIGTERM**
    mode=b  全新的连接池 + 全新的进程,`Command(resume)` 续跑 → 打印 `RESUMED {json}`

为什么非要两个进程:`§9` 的验收原文是「停在 await_confirm → SIGTERM 重启服务 →
Command(resume) 续跑成功」。同一进程里 `proc.kill()` 只是它的一个不太可信的类比 —— 进程内
重启永远带着旧的内存态,而「恢复」要防的恰恰是内存没了。

检索与模型都是**桩**:要验的是检查点能不能跨进程把状态带回来,不是三路召回
(`test_retrieval_e2e.py`)或结构化输出(`verify_models.py`)各自的验收。桩越薄,失败时越确定
是恢复这条链断了。
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: E402

from kbmesh.agents import nodes as nodes_mod  # noqa: E402
from kbmesh.agents.graph import compile_graph  # noqa: E402
from kbmesh.agents.nodes import GraphDeps  # noqa: E402
from kbmesh.agents.runtime import Runtime, saver_handle  # noqa: E402
from kbmesh.agents.schemas import IntentOut, RelevanceOut, RewriteOut  # noqa: E402
from kbmesh.agents.trace import TraceWriter  # noqa: E402
from kbmesh.core.config import get_settings  # noqa: E402
from kbmesh.core.gateway import ChatResult, RerankHit, Usage  # noqa: E402
from kbmesh.core.tenancy import ROLE_READER, TenantContext, use_context  # noqa: E402
from kbmesh.db.base import make_engine  # noqa: E402
from kbmesh.db.models import Tenant  # noqa: E402
from kbmesh.retrieval.contracts import Citation, Hit  # noqa: E402
from kbmesh.retrieval.fusion import RetrievalResult  # noqa: E402

TENANT, KB = "tnt-rs-a", "kb-rs-a"
CTX = TenantContext(
    tenant_id=TENANT, user_id="usr-rs-a", roles=(ROLE_READER,), kb_scope=frozenset({KB})
)
BODY = "Milvus 承载向量检索,隔离设计靠租户分区键完成。"
QUERY = "隔离 设计 怎么做"
CHOSEN = "隔离设计怎么做"
ANSWER = "隔离靠租户分区键在检索与图谱两处同时下推 [1]。"


def _hits() -> list[Hit]:
    return [
        Hit(
            chunk_id="chunk-rs-1",
            score=0.5,
            source_routes={"body"},
            citation=Citation(
                doc_id="doc-rs",
                chunk_id="chunk-rs-1",
                heading_path="检索 > 隔离设计",
                title="设计册",
            ),
            text=BODY,
            route_scores={"body": 0.5},
        )
    ]


class StubRetrieve:
    """编排只需要「候选集 + 逐路 stats + 降级清单」,这里给一套最小能用的。"""

    async def __call__(self, query: str, ctx: TenantContext, **kwargs: Any) -> RetrievalResult:
        del ctx, kwargs
        return RetrievalResult(
            query=query,
            hits=_hits(),
            stats={
                route: {
                    "count": 1,
                    "fingerprint": f"fp-{route}",
                    "latency_ms": 2,
                    "error": None,
                    "degraded": False,
                }
                for route in ("body", "heading", "dense", "graph")
            },
            graph={"mentions": [], "lines": ["Milvus --USES--> 向量检索"]},
            fused_count=1,
        )


class StubGateway:
    """四个 purpose 各给一条写死的返回:够把图推到 `answered`,不多做任何判定。"""

    def structured(self, prompt: str, schema: type, *, purpose: str = "route", system: str = ""):
        del prompt, system, purpose
        usage = Usage(token_in=30, token_out=8, latency_ms=5, model="stub")
        if schema is IntentOut:
            return IntentOut(verdict="answerable", confidence=0.9, reason="库内问题"), usage
        if schema is RewriteOut:
            return (
                RewriteOut(candidates=[{"text": CHOSEN, "focus": "隔离", "rationale": "口语化"}]),
                usage,
            )
        if schema is RelevanceOut:
            return (
                RelevanceOut(scores=[{"chunk_id": "chunk-rs-1", "score": 0.9, "gap": ""}]),
                usage,
            )
        raise AssertionError(f"编排层不该问这一类结构化输出:{schema!r}")

    def chat(self, prompt: str, *, purpose: str = "generate", **kwargs: Any) -> ChatResult:
        del prompt, purpose, kwargs
        return ChatResult(text=ANSWER, usage=Usage(token_in=100, token_out=30, model="stub"))

    def rerank(self, query: str, documents: Sequence[Any], *, top_n: int | None = None):
        del query
        return [RerankHit(index=i, score=0.9 - 0.05 * i) for i in range(len(documents))][
            : (top_n or len(documents))
        ]


def _url() -> str:
    settings = get_settings()
    return settings.db_url


async def _factory() -> async_sessionmaker:
    """租户行必须先在:被杀掉的那一段没法补,重启那一段也不该去补。"""
    maker = async_sessionmaker(make_engine(_url()), expire_on_commit=False)
    async with maker() as session, session.begin():
        if await session.get(Tenant, TENANT) is None:
            session.add(Tenant(id=TENANT, name="tnt-rs-a-名字"))
    return maker


def _deps(factory: async_sessionmaker) -> GraphDeps:
    nodes_mod.retrieve = StubRetrieve()  # type: ignore[assignment]
    return GraphDeps(
        factory=factory,  # type: ignore[arg-type]
        gateway=StubGateway(),  # type: ignore[arg-type]
        kb_id=KB,
        trace=TraceWriter(factory),
    )


async def mode_a(thread: str) -> int:
    deps = _deps(await _factory())
    async with saver_handle() as handle:
        runtime = Runtime(deps, compile_graph(deps, handle.saver), handle)
        with use_context(CTX):
            outcome = await runtime.ask(QUERY, thread)
        print(
            "READY "
            + json.dumps(
                {
                    "run_id": outcome.run_id,
                    "kind": outcome.kind,
                    "candidates": [
                        one["text"] for one in (outcome.payload or {}).get("rewrite_candidates") or []
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        # 不给 pool 任何 close 的机会:FR-403 要验的就是「崩得最难看的一次也留得住中断态」
        os.kill(os.getpid(), signal.SIGTERM)
    return 0


async def mode_b(thread: str, decision: Any = None) -> int:
    deps = _deps(await _factory())
    async with saver_handle() as handle:
        runtime = Runtime(deps, compile_graph(deps, handle.saver), handle)
        with use_context(CTX):
            outcome = await runtime.resume(thread, {"choice": 1} if decision is None else decision)
        print(
            "RESUMED "
            + json.dumps(
                {
                    "kind": outcome.kind,
                    "run_id": outcome.run_id,
                    "final_status": outcome.final_status,
                    "answer": outcome.answer,
                    "citations": [
                        {"chunk_id": one.chunk_id, "heading_path": one.heading_path}
                        for one in outcome.citations
                    ],
                    "confirmed_query": outcome.state.get("confirmed_query"),
                    "stages": [one.get("stage") for one in (outcome.state.get("usage") or [])],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


def main(argv: Sequence[str]) -> int:
    mode, thread = argv[1], argv[2]
    if mode == "a":
        return asyncio.run(mode_a(thread))
    if mode == "b":
        return asyncio.run(mode_b(thread, json.loads(argv[3]) if len(argv) > 3 else None))
    raise SystemExit(f"未知 mode: {mode!r}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
