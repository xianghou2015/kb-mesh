"""chunk captions

Revision ID: b6fe5ca80989
Revises: 39d874131dd5
Create Date: 2026-11-05 23:25:59.306541

`server_default` 不是装饰:表里已有切片时,NOT NULL 列没有默认值会让 ALTER 直接失败,
而 autogenerate 只看模型(模型侧的 default 是 Python 端的),不会替你想这一步。
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b6fe5ca80989"
down_revision: str | None = "39d874131dd5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunk",
        sa.Column(
            "captions",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        schema="platform",
    )


def downgrade() -> None:
    op.drop_column("chunk", "captions", schema="platform")
