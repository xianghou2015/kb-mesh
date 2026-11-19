"""知识库管理(FR-506):建库、列库、删库。写操作限 tenant_admin+。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from kbmesh.api.deps import get_ctx, get_db
from kbmesh.api.schemas import KBIn
from kbmesh.core.config import get_settings
from kbmesh.core.errors import NotFound
from kbmesh.core.tenancy import TenantContext
from kbmesh.db.repositories import KBRepo

router = APIRouter(prefix="/api", tags=["kbs"])


def _payload(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "embedding_model": row.embedding_model,
        "chunk_policy": row.chunk_policy,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.post("/kbs", status_code=201)
async def create_kb(
    body: KBIn,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    model = body.embedding_model or get_settings().embedding_model
    row = await KBRepo(ctx, db).create(body.name, model, body.chunk_policy)
    return _payload(row)


@router.get("/kbs")
async def list_kbs(
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    return [_payload(row) for row in await KBRepo(ctx, db).list()]


@router.get("/kbs/{kb_id}")
async def get_kb(
    kb_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    row = await KBRepo(ctx, db).get(kb_id)
    if row is None:
        raise NotFound(f"知识库不存在: {kb_id}")
    return _payload(row)


@router.delete("/kbs/{kb_id}", status_code=204)
async def delete_kb(
    kb_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> None:
    await KBRepo(ctx, db).delete(kb_id)