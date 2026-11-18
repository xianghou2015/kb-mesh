"""把一次真实问答桥接成评估器要的 `CaseOutcome`(FR-607/608 的执行侧)。

评估运行器不直接起图:它只认 `CaseAnswerer`。生产用这个类 —— 同一套 Runtime/检查点/
五层防御,自动确认选第一条改写候选(没有候选时使用原问),然后把 State 里的证据、
引用映射与 trace 用量收敛成纯数据。
"""
from __future__ import annotations

import uuid
from typing import Any

from kbmesh.agents.graph import compile_graph
from kbmesh.agents.nodes import GraphDeps
from kbmesh.agents.runtime import Runtime, SaverHandle
from kbmesh.core.tenancy import TenantContext, use_context
from kbmesh.db.models import EvalCase
from kbmesh.eval.runner import CaseOutcome


class RuntimeCaseRunner:
    def __init__(
        self, *, deps: GraphDeps, handle: SaverHandle, ctx: TenantContext
    ) -> None:
        self.deps = deps
        self.handle = handle
        self.ctx = ctx
        self.graph = compile_graph(deps, checkpointer=handle.saver)
        self.runtime = Runtime(deps, self.graph, handle)

    async def __call__(self, case: EvalCase) -> CaseOutcome:
        # 每次执行一条新线程:同一题在两次评估运行之间可以复用 set,但检查点不能互相污染。
        thread_id = f"eval-{case.id}-{uuid.uuid4().hex[:8]}"
        with use_context(self.ctx):
            outcome = await self.runtime.ask(case.question, thread_id)
            if outcome.awaiting:
                candidates = list((outcome.payload or {}).get("rewrite_candidates") or [])
                if candidates:
                    decision: dict[str, Any] = {"choice": 1}
                else:
                    # 改写降级时没有候选;`validate_resume` 要求 edited 至少 2 字符。
                    edited = case.question if len(case.question) >= 2 else case.question + "  "
                    decision = {"edited": edited}
                outcome = await self.runtime.resume(thread_id, decision)
            state = dict(outcome.state or {})
            row = None
            if self.deps.trace is not None and outcome.run_id:
                row = await self.deps.trace.get_run(outcome.run_id)

        reranked = list(state.get("reranked") or [])
        evidence = list(state.get("hits") or [])
        return CaseOutcome(
            answer=outcome.answer,
            final_status=outcome.final_status or outcome.kind,
            reject_code=outcome.reject_code,
            retrieved_ids=[str(hit.chunk_id) for hit in reranked],
            cite_ids={str(k): str(v) for k, v in (state.get("cite_ids") or {}).items()},
            evidence_ids=[str(hit.chunk_id) for hit in evidence],
            evidence_texts=[str(hit.text) for hit in evidence],
            rewrite_candidates=list(state.get("rewrite_candidates") or []),
            user_choice=dict(state.get("user_choice") or {}),
            retry_attempt=int(state.get("retry_attempt") or 0),
            latency_ms=int(getattr(row, "latency_ms", 0) or 0),
            token_in=int(getattr(row, "token_in", 0) or 0),
            token_out=int(getattr(row, "token_out", 0) or 0),
            cost_usd=float(getattr(row, "cost_usd", 0) or 0),
            degraded=bool(outcome.degraded),
        )
