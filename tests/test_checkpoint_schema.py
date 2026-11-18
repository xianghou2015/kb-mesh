"""探针 C1 的应用侧守门:检查点四表必须落在 `checkpoint_schema`,一张都不许漏进 `platform`。

为什么值得单独立一个文件:`saver.setup()` 的 MIGRATIONS 里表名**不带 schema**,建到哪张 schema
完全由连接 `search_path` 的第一位决定。落错地方的后果不是「读不到」而是 `alembic check` 第二天
要求 DROP 掉那三张表 —— 也就是**把别人的检查点当漂移删掉**。这条错误没有任何一行代码会报错,
所以只能靠一个专门盯着落点的用例挡。

跑真 PG。用例幂等:`setup()` 自己就是幂等的,重复调用不产生新行。
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from kbmesh.agents.runtime import CKPT_TABLES, SaverHandle, serde_for
from kbmesh.core.config import get_settings
from kbmesh.core.errors import IllegalTransition


@pytest.fixture(scope="session")
async def handle() -> AsyncIterator[SaverHandle]:
    handler = SaverHandle.open_sync()
    await handler.open()
    try:
        yield handler
    finally:
        await handler.close()


async def test_checkpoint_tables_live_only_in_checkpoint_schema(handle: SaverHandle) -> None:
    settings = get_settings()
    where = await handle.tables()
    assert sorted(where.get(settings.checkpoint_schema, [])) == sorted(CKPT_TABLES), (
        f"检查点表不齐:实际落在 {where}"
    )
    # platform 是 Alembic 的地盘。同名表出现在这里 = 下一次 `alembic check` 会要求删掉它
    assert not set(where.get("platform", [])) & set(CKPT_TABLES), (
        f"检查点表漏进了 platform:{sorted(where['platform'])}"
    )


async def test_setup_is_idempotent(handle: SaverHandle) -> None:
    """`setup()` 要能被每次进程启动重复调用:恢复路径的第一句就是它,不能是「第二次就炸」。"""
    before = await handle.tables()
    await handle.setup()
    assert await handle.tables() == before


async def test_pool_pins_search_path(handle: SaverHandle) -> None:
    """连接上读到的 `search_path` 第一位必须是 checkpoint_schema —— 这是 C1 成立的唯一机制。"""
    async with handle.pool.connection() as conn:
        row = await (await conn.execute("show search_path")).fetchone()
    first = str(row["search_path"]).split(",")[0].strip()
    assert first == get_settings().checkpoint_schema, f"search_path={row['search_path']}"


async def test_bad_checkpoint_schema_never_reaches_the_database() -> None:
    """schema 名要拼进 `CREATE SCHEMA`,所以它在建池之前就必须是合法标识符。

    带空格、带引号、带分号的一律拒 —— 不是为了风格:`test` 环境允许改这个值,而它是唯一一个
    「配错了不会立刻报错、只会把表建到别人的地盘」的字段。
    """
    settings = get_settings().model_copy(update={"checkpoint_schema": "lg; drop table x"})
    with pytest.raises(IllegalTransition):
        SaverHandle.open_sync(settings=settings)


async def test_serde_whitelists_the_state_payload() -> None:
    """State 里的 `Hit`/`Citation` 走 msgpack:往返必须相等。

    默认的 `JsonPlusSerializer()` 今天也放行未注册类型,只是 log 警告,而报错文案明说未来会
    改成拒绝。这里不查日志,只查白名单确实生效那一件事:往返回来还是原类型、字段没退化成 dict
    —— 退化成 dict 的话 `guards._field` 照样读得动,坏掉的是「谁来兜住这个形状」。
    """
    from kbmesh.retrieval.contracts import Citation, Hit

    probe = Hit(
        chunk_id="chunk-ck-1",
        score=0.5,
        source_routes={"body"},
        citation=Citation(doc_id="doc-ck", chunk_id="chunk-ck-1", heading_path="检索 > 隔离"),
        text="隔离靠分区键完成。",
        route_scores={"body": 0.5},
    )
    back = serde_for(get_settings()).loads_typed(serde_for(get_settings()).dumps_typed(probe))
    assert back == probe
    assert isinstance(back, Hit) and isinstance(back.citation, Citation)
