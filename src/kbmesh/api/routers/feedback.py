"""反馈回流(FR-609):评分 + 修正答案进反馈池,#8 的评估集从这里导入。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from kbmesh.api.deps import get_ctx, get_db
from kbmesh.api.schemas import FeedbackIn
from kbmesh.core.tenancy import TenantContext
from kbmesh.db.repositories import FeedbackRepo

router = APIRouter(prefix="/api", tags=["feedback"])


def _payload(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "run_id": row.run_id,
        "user_id": row.user_id,
        "score": row.score,
        "corrected_answer": row.corrected_answer,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.post("/runs/{run_id}/feedback", status_code=201)
async def add_feedback(
    run_id: str,
    body: FeedbackIn,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    row = await FeedbackRepo(ctx, db).add(
        run_id, score=body.score, corrected_answer=body.corrected_answer
    )
    return _payload(row)


@router.get("/runs/{run_id}/feedback")
async def list_feedback(
    run_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    return [_payload(row) for row in await FeedbackRepo(ctx, db).list_for_run(run_id)]
