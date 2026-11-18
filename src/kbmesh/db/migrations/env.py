"""Alembic environment (async).

URL resolution order: `-x db_url=` > `KBMESH_TEST_DB_URL` when `KBMESH_ENV=test` > `KBMESH_DB_URL`.
Nothing here reads `alembic.ini` for the URL, so a stale credential can never be committed into
the config file by accident.

Two traps this file has to survive, both learned on the sibling project:
- On an async connection `context.begin_transaction()` only opens a SAVEPOINT, so a plain
  `run_sync` that creates objects can exit 0 having created nothing. The explicit commit below is
  what makes `upgrade head` real.
- The LangGraph checkpointer owns schema `lg` in the same database and creates its tables itself.
  Autogenerate must not see it, or every revision proposes dropping the checkpoint tables.
"""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from kbmesh.core.config import get_settings
from kbmesh.db import models  # noqa: F401  (registers every table on Base.metadata)
from kbmesh.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

SCHEMA = "platform"


def _url() -> str:
    x_url = context.get_x_argument(as_dictionary=True).get("db_url")
    if x_url:
        return x_url
    settings = get_settings()
    if os.getenv("KBMESH_ENV", settings.env) == "test" and settings.test_db_url:
        return settings.test_db_url
    if not settings.db_url:
        raise RuntimeError("no database URL: set KBMESH_DB_URL or pass -x db_url=…")
    return settings.db_url


def include_object(obj, name, type_, reflected, compare_to) -> bool:  # noqa: ANN001
    if type_ == "table":
        return obj.schema in (None, SCHEMA)
    if type_ == "index" and getattr(obj, "table", None) is not None:
        return obj.table.schema in (None, SCHEMA)
    if type_ == "schema":
        return name in (None, SCHEMA)
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        include_schemas=True,
        version_table_schema=SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        include_schemas=True,
        version_table_schema=SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        # Deliberately *not* base.connect_args(): `platform` in the search_path makes reflection
        # report every table with schema=None, and autogenerate then "sees" the whole platform
        # schema twice — it proposes dropping and re-adding every FK, and the version table with
        # it. Migrations only need `extensions` visible, for the `vector` type name to resolve.
        connect_args={"server_settings": {"search_path": "extensions, public"}},
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
        # SAVEPOINT semantics: without this the transaction is rolled back when the connection
        # closes and `upgrade head` reports success while creating nothing.
        await connection.commit()
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
