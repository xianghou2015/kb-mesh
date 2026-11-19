"""种子数据:创建一个演示租户 + 管理员 + 读者,供前端联调。

用法: .venv/bin/python scripts/seed_demo.py
幂等:已存在同名租户/用户时跳过。
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from kbmesh.core.config import get_settings
from kbmesh.db.base import get_session_factory
from kbmesh.db.models import AppUser, Tenant
from kbmesh.security.passwords import hash_password


async def main() -> None:
    get_settings()
    async with get_session_factory()() as session, session.begin():
        existing = await session.scalar(
            text("SELECT id FROM platform.kb_tenant WHERE name = :n"), {"n": "demo"}
        )
        if existing:
            tenant_id = existing
            print(f"租户 demo 已存在: {tenant_id}")
        else:
            tenant = Tenant(name="demo")
            session.add(tenant)
            await session.flush()
            tenant_id = tenant.id
            print(f"创建租户: {tenant_id}")

        for username, role, pwd in [
            ("admin", "platform_admin", "admin123"),
            ("manager", "tenant_admin", "manager123"),
            ("reader", "reader", "reader123"),
        ]:
            exists = await session.scalar(
                text("SELECT id FROM platform.kb_user WHERE tenant_id = :t AND username = :u"),
                {"t": tenant_id, "u": username},
            )
            if exists:
                print(f"  用户 {username} 已存在,跳过")
                continue
            session.add(AppUser(
                tenant_id=tenant_id, username=username, role=role,
                password_hash=hash_password(pwd),
            ))
            print(f"  创建用户: {username} / {role}")

    print(f"\n登录租户ID={tenant_id};账号 admin/manager/reader,演示口令见本脚本源码,勿用于生产。")


if __name__ == "__main__":
    asyncio.run(main())