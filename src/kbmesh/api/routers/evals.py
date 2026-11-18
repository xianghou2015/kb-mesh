"""评估集与评估运行的端点(#7):建集、冻结、建 run、查状态与结果。

指标逻辑与执行器归 #8/#9 —— 这里只负责把 run 的账开出来(status=pending),端点形状先于
指标实现稳定下来,避免 #8 反过来改接口。
"""
from __future__ import annotations

import contextlib
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from kbmesh.api.deps import get_ctx, get_db
from kbmesh.api.schemas import EvalRunIn, EvalSetFromFeedbackIn, EvalSetIn
from kbmesh.core.errors import PermissionDenied
from kbmesh.core.tenancy import ROLE_PLATFORM_ADMIN, ROLE_TENANT_ADMIN, TenantContext
from kbmesh.db.models import EvalRun, EvalSet
from kbmesh.db.repositories import EvalRepo, record_security_event
from kbmesh.eval.runner import CaseOutcome, EvalRunner

router = APIRouter(prefix="/api", tags=["evals"])


async def _no_answer(_case: object) -> CaseOutcome:
    raise RuntimeError("这个端点只读报告,不执行 case")


def _runner(request: Request, ctx: TenantContext) -> EvalRunner:
    return EvalRunner(
        factory=request.app.state.graph_deps.factory,
        ctx=ctx,
        answer_case=_no_answer,  # type: ignore[arg-type]
    )


def _set_payload(row: EvalSet, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": row.id,
        "name": row.name,
        "version": row.version,
        "frozen_at": row.frozen_at.isoformat() if row.frozen_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
    data.update(extra)
    return data


def _run_payload(row: EvalRun, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": row.id,
        "set_id": row.set_id,
        "status": row.status,
        "cursor_case_id": row.cursor_case_id,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "config_snapshot": row.config_snapshot,
        "baseline_run_id": row.baseline_run_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
    data.update(extra)
    return data


@router.post("/eval-sets", status_code=201)
async def create_eval_set(
    body: EvalSetIn,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    row = await EvalRepo(ctx, db).create_set(
        body.name, cases=[one.model_dump() for one in body.cases]
    )
    return _set_payload(row, cases_total=len(body.cases))


@router.post("/eval-sets/from-feedback", status_code=201)
async def import_eval_set_from_feedback(
    body: EvalSetFromFeedbackIn,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """FR-609:反馈池 → 候选评估集。题目取原问,修正答案进 reference_answer。"""
    repo = EvalRepo(ctx, db)
    row = await repo.import_feedback(body.name, max_score=body.max_score, limit=body.limit)
    return _set_payload(row, cases_total=await repo.case_count(row.id))


@router.get("/eval-sets")
async def list_eval_sets(
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    return [_set_payload(row) for row in await EvalRepo(ctx, db).list_sets()]


@router.get("/eval-sets/{set_id}")
async def get_eval_set(
    set_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    repo = EvalRepo(ctx, db)
    row = await repo.require_set(set_id)
    return _set_payload(row, cases_total=await repo.case_count(set_id))


@router.post("/eval-sets/{set_id}/freeze")
async def freeze_eval_set(
    set_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """FR-606:冻结后内容不可变(触发器兜底),评估才可复现。"""
    return _set_payload(await EvalRepo(ctx, db).freeze_set(set_id))


@router.post("/eval-runs", status_code=201)
async def create_eval_run(
    body: EvalRunIn,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    repo = EvalRepo(ctx, db)
    run = await repo.create_run(
        body.set_id,
        config_snapshot=body.config_snapshot,
        baseline_run_id=body.baseline_run_id,
    )
    return _run_payload(run, cases_total=await repo.case_count(run.set_id))


@router.post("/eval-runs/{run_id}/execute", status_code=202)
async def execute_eval_run(
    run_id: Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")],
    limit: int | None = None,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """FR-607:把离线评估投给 worker。先验 run 归属,避免替别人跑一轮。"""
    if not ctx.has_role(ROLE_TENANT_ADMIN, ROLE_PLATFORM_ADMIN):
        # 与其它 role_denied 门禁同口径:先用请求会话留痕,再抛;get_db 会在
        # PermissionDenied 分支提交这条审计。审计失败不能把 403 换成 500。
        with contextlib.suppress(Exception):
            await record_security_event(
                db,
                actor=ctx,
                event="role_denied",
                target=f"eval-run:{run_id}",
                path=f"/api/eval-runs/{run_id}/execute",
            )
        raise PermissionDenied("role_denied: eval-run:execute")
    await EvalRepo(ctx, db).get_run(run_id)
    from kbmesh.eval.tasks import enqueue_eval_run

    return {"run_id": run_id, "submitted": enqueue_eval_run(run_id, ctx.tenant_id, limit=limit)}


@router.get("/eval-runs/{run_id}/report")
async def eval_run_report(
    run_id: str,
    request: Request,
    ctx: TenantContext = Depends(get_ctx),
) -> dict[str, Any]:
    """四层指标按 group 切片 + 总量;运行器负责写入,报告端点只读。"""
    return await _runner(request, ctx).report(run_id)


@router.get("/eval-runs/{run_id}/regressions")
async def eval_run_regressions(
    run_id: str,
    baseline_run_id: str,
    request: Request,
    ctx: TenantContext = Depends(get_ctx),
) -> dict[str, Any]:
    """FR-605/608:当前 run 对基线的逐 group 差值 + 回退样本。"""
    return await _runner(request, ctx).compare(run_id, baseline_run_id)


@router.get("/eval-runs/{run_id}/export.csv", response_class=PlainTextResponse)
async def eval_run_export(
    run_id: str,
    baseline_run_id: str,
    request: Request,
    ctx: TenantContext = Depends(get_ctx),
) -> PlainTextResponse:
    text = await _runner(request, ctx).regression_csv(run_id, baseline_run_id)
    return PlainTextResponse(
        text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="eval-{run_id}-regressions.csv"'},
    )


@router.get("/eval-runs")
async def list_eval_runs(
    set_id: str | None = None,
    limit: int = 20,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    rows = await EvalRepo(ctx, db).list_runs(set_id=set_id, limit=max(1, min(limit, 100)))
    return [_run_payload(row) for row in rows]


@router.get("/eval-runs/{run_id}")
async def get_eval_run(
    run_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """进度口径:`status` + `cases_total`(结果数归 #8 的执行器回填)。"""
    repo = EvalRepo(ctx, db)
    run = await repo.get_run(run_id)
    return _run_payload(run, cases_total=await repo.case_count(run.set_id))


@router.get("/eval-runs/{run_id}/results")
async def list_eval_results(
    run_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """每个指标一行(`uq_eval_result_run_case_metric`),#8 的执行器按 case×metric 回填。"""
    rows = await EvalRepo(ctx, db).results(run_id)
    return [
        {
            "id": row.id,
            "eval_run_id": row.eval_run_id,
            "case_id": row.case_id,
            "metric": row.metric,
            "value": float(row.value),
            "detail": row.detail,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]
