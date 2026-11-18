"""Test fixtures bound to the real Postgres test database.

数据层与隔离验收一律跑真 PG(技术文档 §9:不接受靠 mock 或「看函数签名」通过),所以缺
`KBMESH_TEST_DB_URL` 时**直接 fail 而不是 skip** —— 静默跳过等于让一条没连库的 CI 报全绿,
而那正是这套测试要防的事。
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from kbmesh.core.config import get_settings
from kbmesh.db.base import make_engine
from kbmesh.db.models import Chunk, Document, EvalCase, EvalSet, KnowledgeBase, Tenant


def database_url() -> str:
    """走 Settings,而不是裸 os.environ:`.env` 里的值只进 pydantic,不进进程环境。"""
    url = get_settings().test_db_url
    if not url:
        pytest.fail(
            "KBMESH_TEST_DB_URL 未设置:数据层测试必须连真实 PG。"
            "跑法见 AGENTS.md(`uv run pytest`),不要 mock 掉这一层。"
        )
    return url


@pytest_asyncio.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = make_engine(database_url())
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Outer transaction + savepoint-bound session: every test rolls back, never commits.

    `join_transaction_mode="create_savepoint"` 让测试内的 `session.commit()` 变成 SAVEPOINT
    释放,触发器与 FK 检查照常执行,而外层事务回滚后库里不留残渣 —— 既保住真实语义,又不用
    每个测试自己清扫。
    """
    async with engine.connect() as conn:
        trans = await conn.begin()
        try:
            yield AsyncSession(
                bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
            )
        finally:
            await trans.rollback()


@pytest_asyncio.fixture
async def tenant(session: AsyncSession) -> Tenant:
    row = Tenant(name="fixture-tenant")
    session.add(row)
    await session.flush()
    return row


@pytest_asyncio.fixture
async def kb(session: AsyncSession, tenant: Tenant) -> KnowledgeBase:
    row = KnowledgeBase(tenant_id=tenant.id, name="fixture-kb", embedding_model="fixture-model")
    session.add(row)
    await session.flush()
    return row


@pytest_asyncio.fixture
async def doc(session: AsyncSession, tenant: Tenant, kb: KnowledgeBase) -> Document:
    row = Document(
        tenant_id=tenant.id,
        kb_id=kb.id,
        source_uri="s3://fixture/a.pdf",
        sha256="0" * 64,
        title="fixture doc",
    )
    session.add(row)
    await session.flush()
    return row


@pytest_asyncio.fixture
async def chunk(session: AsyncSession, tenant: Tenant, kb: KnowledgeBase, doc: Document) -> Chunk:
    row = Chunk(
        tenant_id=tenant.id,
        kb_id=kb.id,
        doc_id=doc.id,
        chunk_key="0",
        text="fixture chunk",
        milvus_pk=1,
    )
    session.add(row)
    await session.flush()
    return row


async def purge_tenant_data(engine: AsyncEngine, *, prefix: str = "tnt-api-") -> None:
    """清扫某个前缀下的测试租户数据,依赖顺序从叶子到根。

    API 测试自己提交事务(fixture 的 SAVEPOINT 兜不住),所以每个用例前后都要显式清扫;
    三份 API 测试曾经各写一份,漏掉新表就会互相污染。冻结评估集的触发器会拒绝 DELETE,
    测试库没有需要保护的真实数据,清扫期间直接关掉,事务回滚会一并还原。
    """
    pattern = f"{prefix}%"
    async with engine.begin() as conn:
        await conn.execute(sa.text("ALTER TABLE platform.eval_set DISABLE TRIGGER USER"))
        await conn.execute(sa.text("ALTER TABLE platform.eval_case DISABLE TRIGGER USER"))
        try:
            for stmt in (
                "delete from platform.eval_result where eval_run_id in "
                "(select id from platform.eval_run where tenant_id like :p) or case_id in "
                "(select id from platform.eval_case where set_id in "
                "(select id from platform.eval_set where tenant_id like :p))",
                "delete from platform.eval_case where set_id in "
                "(select id from platform.eval_set where tenant_id like :p)",
                "delete from platform.eval_run where tenant_id like :p",
                "delete from platform.eval_set where tenant_id like :p",
            ):
                await conn.execute(sa.text(stmt), {"p": pattern})
        finally:
            await conn.execute(sa.text("ALTER TABLE platform.eval_set ENABLE TRIGGER USER"))
            await conn.execute(sa.text("ALTER TABLE platform.eval_case ENABLE TRIGGER USER"))
        for stmt in (
            "delete from platform.feedback where tenant_id like :p",
            "delete from platform.trace_step where run_id in "
            "(select id from platform.trace_run where tenant_id like :p)",
            "delete from platform.trace_run where tenant_id like :p",
            "delete from platform.online_eval where tenant_id like :p",
            "delete from platform.chunk where tenant_id like :p",
            "delete from platform.document where tenant_id like :p",
            "delete from platform.entity where tenant_id like :p",
            "delete from platform.relation_edge where tenant_id like :p",
            "delete from platform.kb where tenant_id like :p",
            "delete from platform.app_user where tenant_id like :p",
            "delete from platform.security_event where tenant_id like :p",
            # 登录失败/节流事件的 actor=None(未认证),tenant_id 为空;不清会把限速状态带进下一个用例。
            "delete from platform.security_event where tenant_id is null and path like '/api/auth/%'",
            "delete from platform.tenant where id like :p",
        ):
            await conn.execute(sa.text(stmt), {"p": pattern})


async def make_eval_set(session: AsyncSession, tenant_id: str, *, cases: int = 2) -> EvalSet:
    # 走 relationship 而不是裸 set_id:异步 session 之后无法 lazy-load `evset.cases`
    row = EvalSet(tenant_id=tenant_id, name="fixture-set", version=1)
    row.cases = [EvalCase(question=f"q{i}", gold_refs=[], group_key="factual_single") for i in range(cases)]
    session.add(row)
    await session.flush()
    return row
