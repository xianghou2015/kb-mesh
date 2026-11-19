"""Trace 查询(FR-401/404 的只读面):run 列表、单 run 及其 steps、拓扑可视化。

纯只读,不带写角色校验 —— 但租户谓词一道不少(`TraceRepo` 的 `_Scoped` 兜着)。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from kbmesh.api.deps import get_ctx, get_db
from kbmesh.core.errors import NotFound
from kbmesh.core.tenancy import TenantContext
from kbmesh.db.models import OnlineEval, TraceRun, TraceStep
from kbmesh.db.repositories import OnlineEvalRepo, TraceRepo
from kbmesh.observability.aggregate import summarize_runs

router = APIRouter(prefix="/api", tags=["traces"])


def _run_payload(row: TraceRun) -> dict[str, Any]:
    return {
        "id": row.id,
        "thread_id": row.thread_id,
        "query": row.query,
        "final_status": row.final_status,
        "degraded": row.degraded,
        "latency_ms": row.latency_ms,
        "token_in": row.token_in,
        "token_out": row.token_out,
        "route_flags": row.route_flags,
        "config_snapshot": row.config_snapshot,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _step_payload(row: TraceStep) -> dict[str, Any]:
    return {
        "seq": row.seq,
        "stage": row.stage,
        "payload": row.payload,
        "latency_ms": row.latency_ms,
        "token_in": row.token_in,
        "token_out": row.token_out,
        "model": row.model,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/traces/runs")
async def list_runs(
    limit: int = 20,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    rows = await TraceRepo(ctx, db).list_runs(max(1, min(limit, 100)))
    return [_run_payload(row) for row in rows]


def _online_payload(row: OnlineEval) -> dict[str, Any]:
    return {
        "id": row.id,
        "run_id": row.run_id,
        "format_ok": row.format_ok,
        "citation_ok": row.citation_ok,
        "safety_ok": row.safety_ok,
        "failed": row.failed,
        "sampled": row.sampled,
        "faithfulness": float(row.faithfulness) if row.faithfulness is not None else None,
        "answer_relevancy": (
            float(row.answer_relevancy) if row.answer_relevancy is not None else None
        ),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/traces/runs/{run_id}/online-eval")
async def get_online_eval(
    run_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    row = await OnlineEvalRepo(ctx, db).get_for_run(run_id)
    if row is None:
        raise NotFound(f"run 没有在线评估结果: {run_id}")
    return _online_payload(row)


@router.get("/online-evals/failures")
async def list_online_failures(
    limit: int = 100,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """FR-601:失败样本可查 —— 只返回本租户、且规则检查有失败项的记录。"""
    return [
        _online_payload(row)
        for row in await OnlineEvalRepo(ctx, db).list_failures(limit)
    ]


@router.get("/traces/stats")
async def stats(
    limit: int = 50,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """FR-704:当前租户一个窗口内的量、P50/P95、token、成本、拒答/降级率、路命中分布。"""
    repo = TraceRepo(ctx, db)
    runs = await repo.list_runs(max(1, min(limit, 200)))
    steps_by_run = {run.id: await repo.list_steps(run.id) for run in runs}
    return summarize_runs(runs, steps_by_run=steps_by_run)


@router.get("/traces/runs/{run_id}")
async def get_run(
    run_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    repo = TraceRepo(ctx, db)
    run = await repo.get_run(run_id)
    return {**_run_payload(run), "steps": [_step_payload(step) for step in await repo.list_steps(run_id)]}


@router.get("/graph/topology")
async def graph_topology(request: Request) -> dict[str, Any]:
    """FR-401:状态图拓扑可视化,返回 mermaid 文本与节点列表。"""
    from kbmesh.agents.graph import mermaid, node_names

    deps = request.app.state.graph_deps
    return {"mermaid": mermaid(deps), "nodes": node_names(deps)}
