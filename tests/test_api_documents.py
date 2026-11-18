"""文档入库/重索引/查询的 HTTP 面(FR-106/107)。

MinIO 与 Celery 是外部边界:对象写入与任务投递按模块符号整体替换,PG 行为照旧打真库 ——
租户隔离断言全部落在权威行上,不落在替身上。
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from conftest import database_url, purge_tenant_data
from kbmesh.agents.nodes import GraphDeps
from kbmesh.api.app import create_app
from kbmesh.core.config import clear_settings_cache
from kbmesh.core.tenancy import ROLE_READER, ROLE_TENANT_ADMIN, TenantContext
from kbmesh.db.base import make_engine, set_engine
from kbmesh.db.models import Document, KnowledgeBase, SecurityEvent, Tenant
from kbmesh.ingest import tasks as ingest_tasks
from kbmesh.security.tokens import issue_access_token
from kbmesh.storage import s3 as storage

TENANT_A, KB_A = "tnt-api-a", "kb-api-a"
TENANT_B = "tnt-api-b"
CTX_INDEX = TenantContext(
    tenant_id=TENANT_A, user_id="usr-api-i", roles=(ROLE_TENANT_ADMIN,), kb_scope=frozenset({KB_A})
)
CTX_READ = TenantContext(
    tenant_id=TENANT_A, user_id="usr-api-r", roles=(ROLE_READER,), kb_scope=frozenset({KB_A})
)
CTX_B = TenantContext(
    tenant_id=TENANT_B, user_id="usr-api-b", roles=(ROLE_TENANT_ADMIN,), kb_scope=frozenset()
)
UPLOAD_BYTES = "# hi\n隔离设计".encode()


def bearer(ctx: TenantContext) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_access_token(ctx)}"}


async def _purge(engine: AsyncEngine) -> None:
    await purge_tenant_data(engine)


@pytest.fixture
async def rig(engine, monkeypatch: pytest.MonkeyPatch):
    """env=test + 真库 + 替身存储/Celery 的应用;KB_A 由 INDEXER 建好。"""
    monkeypatch.setenv("KBMESH_ENV", "test")
    clear_settings_cache()
    eng = make_engine(database_url())
    set_engine(eng)
    factory = async_sessionmaker(eng, expire_on_commit=False)
    await _purge(eng)
    async with factory() as session, session.begin():
        session.add(Tenant(id=TENANT_A, name="api-a"))
        session.add(Tenant(id=TENANT_B, name="api-b"))
        # tenant 必须先落库:kb 的 FK 指向它,而两个 ORM 对象之间没有 relationship 可让
        # unit of work 推导顺序。
        await session.flush()
        session.add(
            KnowledgeBase(
                tenant_id=TENANT_A, id=KB_A, name="api-kb", embedding_model="fake-embed"
            )
        )
    puts: list[dict[str, object]] = []
    submissions: list[tuple[str, int]] = []

    def fake_put(data: bytes, *, key: str, mime: str) -> str:
        puts.append({"key": key, "size": len(data), "mime": mime})
        return f"s3://kbmesh-artifact/{key}"

    def fake_submit(doc_id: str, index_version: int) -> dict[str, object]:
        submissions.append((doc_id, index_version))
        return {"submitted": True, "doc_id": doc_id, "index_version": index_version}

    monkeypatch.setattr(storage, "put_bytes", fake_put)
    monkeypatch.setattr(ingest_tasks, "submit_document", fake_submit)
    deps = GraphDeps(factory=factory, trace=None)
    app = create_app(deps=deps, saver=None)
    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                yield SimpleNamespace(
                    app=app, client=client, factory=factory, puts=puts, submissions=submissions
                )
    finally:
        await eng.dispose()
        await _purge(eng)
        clear_settings_cache()


async def _upload(client: httpx.AsyncClient, *, content: bytes = UPLOAD_BYTES):
    files = {"file": ("设计册.md", content, "text/markdown")}
    return await client.post(f"/api/kbs/{KB_A}/documents", files=files, headers=bearer(CTX_INDEX))


async def test_upload_rejects_oversized_payload(rig):
    rig.app.state.graph_deps.settings.max_document_bytes = 10
    resp = await _upload(rig.client, content=b"01234567890")
    assert resp.status_code == 413, resp.text
    assert rig.puts == [] and rig.submissions == [], "超限请求不得写对象或投链"


async def test_upload_creates_row_and_submits_pipeline(rig):
    resp = await _upload(rig.client)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["parse_status"] == "PENDING"
    assert body["duplicate"] is False
    assert body["source_uri"].startswith("s3://kbmesh-artifact/")
    import hashlib

    assert body["sha256"] == hashlib.sha256(UPLOAD_BYTES).hexdigest()
    assert len(rig.puts) == 1 and rig.puts[0]["key"].endswith(".md")
    assert len(rig.submissions) == 1
    doc_id, version = rig.submissions[0]
    assert doc_id == body["id"] and version == 1
    # 权威行落在真库,且归属租户正确
    async with rig.factory() as session:
        row = (await session.execute(sa.select(Document).where(Document.id == doc_id))).scalars().one()
    assert row.tenant_id == TENANT_A and row.kb_id == KB_A


async def test_duplicate_upload_returns_existing_without_resubmit(rig):
    first = await _upload(rig.client)
    second = await _upload(rig.client)
    assert first.status_code == 202 and second.status_code == 202
    assert second.json()["duplicate"] is True
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["submitted"] is False
    assert len(rig.submissions) == 1, "同内容重复入库不投第二遍链"
    assert len(rig.puts) == 1


async def test_reindex_bumps_index_version_and_resubmits(rig):
    doc = (await _upload(rig.client)).json()
    resp = await rig.client.post(
        f"/api/documents/{doc['id']}/reindex", headers=bearer(CTX_INDEX)
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["index_version"] == 2
    assert body["parse_status"] == "PENDING"
    assert rig.submissions == [(doc["id"], 1), (doc["id"], 2)]


async def test_reindex_unknown_or_foreign_doc_is_403_and_audited(rig):
    doc = (await _upload(rig.client)).json()
    # 别的租户:存在性都不许区分,一律 deny 留痕
    resp = await rig.client.post(f"/api/documents/{doc['id']}/reindex", headers=bearer(CTX_B))
    assert resp.status_code == 403
    async with rig.factory() as session:
        rows = (
            (
                await session.execute(
                    sa.select(SecurityEvent).where(
                        SecurityEvent.tenant_id == TENANT_B,
                        SecurityEvent.event == "cross_tenant_read",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert rows


async def test_reader_cannot_upload(rig):
    files = {"file": ("设计册.md", b"x", "text/markdown")}
    resp = await rig.client.post(
        f"/api/kbs/{KB_A}/documents", files=files, headers=bearer(CTX_READ)
    )
    assert resp.status_code == 403
    async with rig.factory() as session:
        rows = (
            (
                await session.execute(
                    sa.select(SecurityEvent).where(SecurityEvent.event == "role_denied")
                )
            )
            .scalars()
            .all()
        )
    assert rows
    assert rig.submissions == []


async def test_upload_to_kb_outside_scope_is_403(rig):
    scoped = TenantContext(
        tenant_id=TENANT_A,
        user_id="usr-api-s",
        roles=(ROLE_TENANT_ADMIN,),
        kb_scope=frozenset({"kb-other"}),
    )
    files = {"file": ("设计册.md", b"x", "text/markdown")}
    resp = await rig.client.post(f"/api/kbs/{KB_A}/documents", files=files, headers=bearer(scoped))
    assert resp.status_code == 403  # kb_scope 不含 KB_A,上传入口直接挡下
    assert rig.submissions == []


async def test_document_listing_is_scoped(rig):
    await _upload(rig.client)
    resp = await rig.client.get(f"/api/kbs/{KB_A}/documents", headers=bearer(CTX_READ))
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    resp = await rig.client.get(f"/api/kbs/{KB_A}/documents", headers=bearer(CTX_B))
    assert resp.status_code == 403  # B 连 KB 的存在性都摸不到


async def test_document_progress_is_readable(rig):
    doc = (await _upload(rig.client)).json()
    resp = await rig.client.get(f"/api/documents/{doc['id']}", headers=bearer(CTX_READ))
    assert resp.status_code == 200
    assert resp.json()["parse_status"] == "PENDING"
    resp = await rig.client.get(f"/api/documents/{doc['id']}", headers=bearer(CTX_B))
    assert resp.status_code == 403
