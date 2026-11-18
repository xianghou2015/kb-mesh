from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from kbmesh.core.config import get_settings


class Base(DeclarativeBase):
    """All business tables live in schema `platform`; the checkpointer lives in `lg`.

    Schema-qualified instead of relying on `search_path`: a per-role search_path would leak across
    databases served by the same Postgres role.
    """

    type_annotation_map = {}


def connect_args() -> dict:
    """Per-session `search_path`, shared by the app engine and by Alembic's env.py.

    DDL is schema-qualified everywhere, so the only thing this buys is type resolution: `vector`
    lives in schema `extensions` and is invisible at the default search_path.
    Keeping the two engines on one helper means a migration can never succeed where the app then
    fails to read the column back.
    """
    return {"server_settings": {"search_path": get_settings().pg_search_path}}


def make_engine(url: str | None = None) -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(
        url or settings.db_url,
        connect_args=connect_args(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )


_engine: AsyncEngine | None = None
_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _factory
    if _factory is None:
        _factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _factory


def set_engine(engine: AsyncEngine) -> None:
    """Test/CLI hook: swap the process-wide engine and reset the session factory."""
    global _engine, _factory
    _engine = engine
    _factory = async_sessionmaker(engine, expire_on_commit=False) if engine else None


async def reset_singletons() -> None:
    global _engine, _factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _factory = None


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with get_session_factory()() as session:
        yield session
