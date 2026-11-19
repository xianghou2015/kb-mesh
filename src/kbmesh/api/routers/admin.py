"""租户与用户管理(FR-506)。租户 CRUD 限 platform_admin;用户 CRUD 限 tenant_admin+。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from kbmesh.api.deps import get_ctx, get_db
from kbmesh.api.schemas import TenantIn, UserIn
from kbmesh.core.errors import NotFound, PermissionDenied
from kbmesh.core.tenancy import (
    ALL_ROLES,
    ROLE_PLATFORM_ADMIN,
    ROLE_TENANT_ADMIN,
    TenantContext,
)
from kbmesh.db.repositories import TenantAdminRepo, UserRepo, record_security_event
from kbmesh.security.passwords import hash_password

router = APIRouter(prefix="/api/admin", tags=["admin"])


async def _require_platform_admin(ctx: TenantContext, db: AsyncSession, target: str) -> None:
    if not ctx.has_role(ROLE_PLATFORM_ADMIN):
        await record_security_event(
            db, actor=ctx, event="role_denied", target=target, path="/api/admin"
        )
        raise PermissionDenied(f"role_denied: {target}")


async def _require_tenant_admin(ctx: TenantContext, db: AsyncSession, target: str) -> None:
    if not ctx.has_role(ROLE_TENANT_ADMIN, ROLE_PLATFORM_ADMIN):
        await record_security_event(
            db, actor=ctx, event="role_denied", target=target, path="/api/admin"
        )
        raise PermissionDenied(f"role_denied: {target}")


def _tenant_payload(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _user_payload(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "username": row.username,
        "role": row.role,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.post("/tenants", status_code=201)
async def create_tenant(
    body: TenantIn,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    await _require_platform_admin(ctx, db, "tenant:create")
    row = await TenantAdminRepo(ctx, db).create_tenant(body.name)
    return _tenant_payload(row)


@router.get("/tenants")
async def list_tenants(
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    await _require_platform_admin(ctx, db, "tenant:list")
    return [_tenant_payload(r) for r in await TenantAdminRepo(ctx, db).list_tenants()]


@router.delete("/tenants/{target_id}", status_code=204)
async def delete_tenant(
    target_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> None:
    await _require_platform_admin(ctx, db, f"tenant:{target_id}")
    await TenantAdminRepo(ctx, db).delete_tenant(target_id)


@router.post("/users", status_code=201)
async def create_user(
    body: UserIn,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    await _require_tenant_admin(ctx, db, "user:create")
    if body.role not in ALL_ROLES:
        raise NotFound(f"未知角色: {body.role}")
    row = await UserRepo(ctx, db).create(body.username, hash_password(body.password), body.role)
    return _user_payload(row)


@router.get("/users")
async def list_users(
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    await _require_tenant_admin(ctx, db, "user:list")
    return [_user_payload(r) for r in await UserRepo(ctx, db).list()]


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> None:
    await _require_tenant_admin(ctx, db, f"user:{user_id}")
    await UserRepo(ctx, db).delete(user_id)