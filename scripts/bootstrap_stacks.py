"""上线前的三段式初始化:PG 迁移 → Milvus 集合 → Neo4j 约束(技术文档 §12 的上线顺序)。

跑法:
    uv run python scripts/bootstrap_stacks.py            # 全跑
    uv run python scripts/bootstrap_stacks.py --check     # 只校验,不建

为什么要一个脚本把它们串起来:这三步顺序错了不会立刻报错 —— 索引没建好就放 worker 跑存量文档,
Milvus 会收下数据但 BM25/分词全不生效,等到问答上线才暴露。错误必须在这里就响。

检查点表(`lg` schema)只有 schema 归这里建、表本身不归这里:`saver.setup()` 建的表名不带 schema,
落在连接的 search_path 第一位,所以 schema 必须先存在;而表本身由 LangGraph 自己迁移,故意不写
Alembic revision(见 db/migrations/env.py 的 include_object)。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text as sa_text

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kbmesh.core.config import get_settings  # noqa: E402
from kbmesh.db.base import make_engine  # noqa: E402
from kbmesh.storage import milvus as mv  # noqa: E402
from kbmesh.storage import neo4j as n4  # noqa: E402


async def _checkpoint_schema(url: str, *, check_only: bool) -> str:
    name = get_settings().checkpoint_schema
    if not name.isidentifier():
        raise SystemExit(f"KBMESH_CHECKPOINT_SCHEMA 不是合法标识符: {name!r}")
    engine = make_engine(url)
    try:
        async with engine.begin() as conn:
            if check_only:
                found = await conn.execute(
                    sa_text("select 1 from information_schema.schemata where schema_name = :s"),
                    {"s": name},
                )
                if not found.first():
                    raise SystemExit(f"[pg] 检查点 schema 不存在: {name}")
            else:
                await conn.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{name}"')
    finally:
        await engine.dispose()
    return name


def bootstrap_postgres(check_only: bool) -> str:
    settings = get_settings()
    url = settings.db_url
    if not url:
        raise SystemExit("没有数据库 URL:设 KBMESH_DB_URL")

    cfg = Config(str(REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO / "src" / "kbmesh" / "db" / "migrations"))
    if check_only:
        print(f"[pg] current revision: {command.current(cfg) or '(无)'}")
    else:
        command.upgrade(cfg, "head")
    schema = asyncio.run(_checkpoint_schema(url, check_only=check_only))
    # 只报库名:整串 URL 带口令,打进日志就等于把口令贴进 CI/终端记录
    print(f"[pg] {'checked' if check_only else 'upgraded to head'}, database={_db_name(url)} "
          f"checkpoint_schema={schema}")
    return url


def _db_name(url: str) -> str:
    tail = url.rsplit("/", 1)[-1].split("?", 1)[0]
    host = url.split("@")[-1].rsplit("/", 1)[0] if "@" in url else ""
    return f"{tail}@{host}" if host else tail


def bootstrap_milvus(check_only: bool) -> dict:
    settings = get_settings()
    client = mv.get_client()
    try:
        if check_only:
            if not mv.collection_ready(client, settings.milvus_collection):
                raise SystemExit(f"[milvus] 集合不存在: {settings.milvus_collection}")
            info = mv.verify_collection(client, settings.milvus_collection)
        else:
            name = mv.ensure_collection(client, settings.milvus_collection)
            info = mv.verify_collection(client, name)
    finally:
        client.close()
    print(f"[milvus] {info['collection']} 就绪: functions={info['functions']} dim={info['dense_dim']}")
    return info


def bootstrap_neo4j(check_only: bool) -> dict:
    driver = n4.get_driver()
    try:
        if check_only:
            with n4.session(driver) as s:
                names = {r["name"] for r in s.run("SHOW CONSTRAINTS YIELD name")}
                names |= {r["name"] for r in s.run("SHOW INDEXES YIELD name")}
            expected = {n for n, _ in n4.CONSTRAINTS + n4.INDEXES}
            missing = expected - names
            if missing:
                raise SystemExit(f"[neo4j] 缺约束/索引: {sorted(missing)}")
            info = {"constraints": sorted(names)}
        else:
            info = n4.ensure_schema(driver)
    finally:
        driver.close()
    ready = sorted(set(info.get("constraints", [])) | set(info.get("indexes", [])))
    print(f"[neo4j] 约束与索引就绪: {ready}")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验现状,不创建任何东西")
    args = ap.parse_args()

    bootstrap_postgres(args.check)
    bootstrap_milvus(args.check)
    bootstrap_neo4j(args.check)
    print("\n三段式初始化完成;下一步:放 worker 跑存量文档,最后再开在线评估抽样(§12)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
