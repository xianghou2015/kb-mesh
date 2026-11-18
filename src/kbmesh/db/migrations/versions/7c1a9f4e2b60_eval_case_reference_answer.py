"""eval_case reference_answer

Revision ID: 7c1a9f4e2b60
Revises: b6fe5ca80989
Create Date: 2026-11-06

反馈池导入时用户提交的正确答案需要随 case 冻结,judge 才能拿它做答案层对照;
gold_refs 是 chunk_id 集合,不能把一段文本塞进去。
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7c1a9f4e2b60"
down_revision: str | None = "b6fe5ca80989"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "eval_case",
        sa.Column("reference_answer", sa.Text(), nullable=True),
        schema="platform",
    )


def downgrade() -> None:
    op.drop_column("eval_case", "reference_answer", schema="platform")
