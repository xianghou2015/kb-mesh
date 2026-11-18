"""迁移落地的真实形状:租户列不可空、冻结只进不退。

这两件事都由**数据库**保证而不是靠应用层约定,所以断言必须打真库:任何「看起来对但没连库」
的测试都挡不住防线①和 §8.3 的失效。
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from conftest import make_eval_set
from kbmesh.db.models import Chunk, EvalRun, EvalSet

#: 审计流水:未认证请求没有租户可归属,是唯一允许 tenant_id 为空的表(见 models.SecurityEvent)。
NULLABLE_TENANT_TABLES = {"security_event"}

#: 业务表(不含 tenant 自身):防线①要求每一张都带非空 tenant_id。
TENANT_SCOPED_TABLES = [
    "app_user",
    "kb",
    "document",
    "chunk",
    "entity",
    "relation_edge",
    "trace_run",
    "feedback",
    "online_eval",
    "eval_set",
    "eval_run",
]

#: 唯一约束/复合索引至少覆盖到的列对(§6.1:漏了它,带租户过滤的扫描照样全表)。
REQUIRED_COMPOSITES = {
    "app_user": ("tenant_id", "username"),
    "kb": ("tenant_id", "name"),
    "document": ("tenant_id", "kb_id", "parse_status"),
    "chunk": ("tenant_id", "chunk_key"),
    "entity": ("tenant_id", "entity_key"),
    "relation_edge": ("tenant_id", "src_key", "dst_key", "rtype"),
    "trace_run": ("tenant_id", "created_at"),
    "feedback": ("tenant_id", "created_at"),
    "online_eval": ("tenant_id", "created_at"),
    "eval_set": ("tenant_id", "name", "version"),
}


async def test_every_business_table_has_non_null_tenant_id(session: AsyncSession) -> None:
    rows = (
        await session.execute(
            text(
                "SELECT table_name, is_nullable FROM information_schema.columns "
                "WHERE table_schema = 'platform' AND column_name = 'tenant_id'"
            )
        )
    ).all()
    found = {table for table, _ in rows}
    missing = sorted(set(TENANT_SCOPED_TABLES) - found)
    assert not missing, f"这些表没有 tenant_id 列: {missing}"
    nullable = sorted(table for table, is_null in rows if is_null == "YES")
    assert set(nullable) == NULLABLE_TENANT_TABLES, f"tenant_id 可空的不止审计表: {nullable}"


async def test_tenant_composite_indexes_exist(session: AsyncSession) -> None:
    rows = (
        await session.execute(
            text("SELECT tablename, indexdef FROM pg_indexes WHERE schemaname = 'platform'")
        )
    ).all()
    columns_by_table: dict[str, list[tuple[str, ...]]] = {}
    for tablename, indexdef in rows:
        cols = re.search(r"\(([^)]*)\)", indexdef)
        if cols:
            columns_by_table.setdefault(tablename, []).append(
                tuple(c.strip() for c in cols.group(1).split(","))
            )

    for table, expected in REQUIRED_COMPOSITES.items():
        actual = columns_by_table.get(table, [])
        assert expected in actual, f"{table} 缺复合索引 {expected};实有 {actual}"


async def test_unfrozen_eval_set_is_editable(session: AsyncSession, tenant) -> None:
    evset = await make_eval_set(session, tenant.id)
    evset.name = "renamed-before-freeze"
    await session.flush()
    assert (await session.get(EvalSet, evset.id)).name == "renamed-before-freeze"

    case = evset.cases[0]
    await session.delete(case)
    await session.flush()
    remaining = await session.scalar(
        text("SELECT count(*) FROM platform.eval_case WHERE set_id = :sid"), {"sid": evset.id}
    )
    assert remaining == 1


async def test_frozen_set_rejects_update(session: AsyncSession, tenant) -> None:
    evset = await make_eval_set(session, tenant.id)
    evset.frozen_at = datetime.now(UTC)
    await session.flush()

    with pytest.raises(Exception, match="is frozen"):
        evset.name = "sneaky"
        await session.flush()


async def test_frozen_set_rejects_unfreeze(session: AsyncSession, tenant) -> None:
    """解冻本身也是一次 UPDATE,被同一道触发器挡死 —— 这是故意的。

    版本可比性不允许「先冻结跑两轮、再回头改题目」;要变更就新增 version,而不是复活冻结集。
    """
    evset = await make_eval_set(session, tenant.id)
    evset.frozen_at = datetime.now(UTC)
    await session.flush()

    with pytest.raises(Exception, match="is frozen"):
        await session.execute(
            text("UPDATE platform.eval_set SET frozen_at = NULL WHERE id = :sid"), {"sid": evset.id}
        )


async def test_frozen_set_rejects_delete(session: AsyncSession, tenant) -> None:
    evset = await make_eval_set(session, tenant.id)
    evset.frozen_at = datetime.now(UTC)
    await session.flush()

    with pytest.raises(Exception, match="is frozen"):
        await session.execute(text("DELETE FROM platform.eval_set WHERE id = :sid"), {"sid": evset.id})


async def test_frozen_set_case_update_rejected(session: AsyncSession, tenant) -> None:
    evset = await make_eval_set(session, tenant.id)
    case_id = evset.cases[0].id
    evset.frozen_at = datetime.now(UTC)
    await session.flush()

    with pytest.raises(Exception, match="is frozen"):
        await session.execute(
            text("UPDATE platform.eval_case SET question = 'edited' WHERE id = :cid"), {"cid": case_id}
        )


async def test_frozen_set_case_delete_rejected(session: AsyncSession, tenant) -> None:
    evset = await make_eval_set(session, tenant.id)
    case_id = evset.cases[0].id
    evset.frozen_at = datetime.now(UTC)
    await session.flush()

    with pytest.raises(Exception, match="is frozen"):
        await session.execute(text("DELETE FROM platform.eval_case WHERE id = :cid"), {"cid": case_id})


async def test_frozen_set_blocks_cascade_delete_from_parent(session: AsyncSession, tenant) -> None:
    """ORM 走级联删除(先删子再删父)也必须被拦 —— 触发器不认路径,只认操作。"""
    evset = await make_eval_set(session, tenant.id, cases=2)
    evset.frozen_at = datetime.now(UTC)
    await session.flush()

    with pytest.raises(Exception, match="is frozen"):
        await session.delete(evset)
        await session.flush()


async def test_frozen_set_still_accepts_new_version_and_runs(session: AsyncSession, tenant) -> None:
    """冻结只锁内容,不锁「被引用」:迭代走新版本,评估照常往冻结集上跑。"""
    evset = await make_eval_set(session, tenant.id)
    evset.frozen_at = datetime.now(UTC)
    await session.flush()

    v2 = EvalSet(tenant_id=tenant.id, name=evset.name, version=2)
    session.add(v2)
    await session.flush()
    assert v2.frozen_at is None

    run = EvalRun(tenant_id=tenant.id, set_id=evset.id, config_snapshot={"routes": ["body"]})
    session.add(run)
    await session.flush()
    assert run.status == "pending"


async def test_chunk_idempotency_key_is_per_tenant(
    session: AsyncSession, tenant, kb, doc, chunk
) -> None:
    """§5.3 幂等第一层:`(tenant_id, chunk_key)` 唯一,重跑同一文档不会翻倍。"""
    assert chunk.doc_id == doc.id
    with pytest.raises(IntegrityError):
        session.add(
            Chunk(
                tenant_id=tenant.id,
                kb_id=kb.id,
                doc_id=doc.id,
                chunk_key=chunk.chunk_key,
                text="duplicate run of the same slice",
                milvus_pk=chunk.milvus_pk,
            )
        )
        await session.flush()
