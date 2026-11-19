"""防线②在 HTTP 层的落点:没有合法令牌进不了路由,失败必须留痕。"""
from __future__ import annotations

import httpx
import pytest
from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from kbmesh.core.tenancy import ROLE_READER, TenantContext, active_context
from kbmesh.db.base import set_engine
from kbmesh.db.repositories import record_security_event
from kbmesh.security.middleware import PUBLIC_PATHS, TenantContextMiddleware, get_tenant_context
from kbmesh.security.tokens import issue_access_token

CTX = TenantContext(tenant_id="tnt-mw", user_id="u-mw", roles=(ROLE_READER,))
AUDIT_PATH = "/api/mw-probe"


def build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(TenantContextMiddleware)

    @app.get("/api/mw-probe")
    async def probe(ctx: TenantContext = Depends(get_tenant_context)) -> dict[str, object]:
        return {
            "tenant": ctx.tenant_id,
            "roles": list(ctx.roles),
            # 门面/存储层在路由体内要能从 ContextVar 读到同一条身份
            "var_visible": active_context() is ctx,
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"ok": "yes"}

    return app


async def audit_rows(engine: AsyncEngine, path: str) -> list[tuple[str, str | None]]:
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT event, tenant_id FROM platform.sec_event WHERE path = :p"), {"p": path}
        )
        return [(row[0], row[1]) for row in result]


@pytest.fixture
async def client(engine: AsyncEngine) -> httpx.AsyncClient:
    """审计走独立连接、真提交,所以这个 fixture 负责前后清库,不给共享测试库留残渣。"""
    set_engine(engine)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM platform.sec_event WHERE path LIKE '/api/mw-%'"))
    transport = httpx.ASGITransport(app=build_app())
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            yield c
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM platform.sec_event WHERE path LIKE '/api/mw-%'"))
        # 故意不还原全局引擎:置空会让后续代码按 settings.db_url 重建,而那个指向 dev 库。
        # 留在测试库上才是安全的默认。


async def test_missing_token_is_401_and_audited(client: httpx.AsyncClient, engine: AsyncEngine) -> None:
    r = await client.get(AUDIT_PATH)
    assert r.status_code == 401
    assert await audit_rows(engine, AUDIT_PATH) == [("unauthenticated", None)]


async def test_non_bearer_header_is_401(client: httpx.AsyncClient, engine: AsyncEngine) -> None:
    r = await client.get(AUDIT_PATH, headers={"Authorization": "Basic zzz"})
    assert r.status_code == 401
    assert [row[0] for row in await audit_rows(engine, AUDIT_PATH)] == ["bad_token"]


async def test_garbage_token_is_401(client: httpx.AsyncClient, engine: AsyncEngine) -> None:
    r = await client.get(AUDIT_PATH, headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401
    assert [row[0] for row in await audit_rows(engine, AUDIT_PATH)] == ["bad_token"]


async def test_valid_token_reaches_the_route(client: httpx.AsyncClient, engine: AsyncEngine) -> None:
    r = await client.get(AUDIT_PATH, headers={"Authorization": f"Bearer {issue_access_token(CTX)}"})
    assert r.status_code == 200
    assert r.json() == {"tenant": "tnt-mw", "roles": ["reader"], "var_visible": True}
    assert await audit_rows(engine, AUDIT_PATH) == []


async def test_public_paths_stay_open(client: httpx.AsyncClient) -> None:
    assert "/healthz" in PUBLIC_PATHS
    assert (await client.get("/healthz")).status_code == 200


async def test_unauthenticated_event_has_null_tenant(engine: AsyncEngine) -> None:
    """这张表是防线①唯一的例外(未认证请求没有租户可归属),必须有测试钉住而不是靠注释。"""
    try:
        async with engine.begin() as conn:  # begin() 退出即提交,所以下面必须自己删干净
            session = AsyncSession(bind=conn)
            await record_security_event(session, actor=None, event="unauthenticated", path="/api/mw-null")
            await session.flush()
            count = await session.scalar(
                text(
                    "SELECT count(*) FROM platform.sec_event "
                    "WHERE path = '/api/mw-null' AND tenant_id IS NULL"
                )
            )
            assert count == 1
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM platform.sec_event WHERE path = '/api/mw-null'"))
