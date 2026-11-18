"""online_eval

Revision ID: 9f3d2c1b7a80
Revises: 7c1a9f4e2b60
Create Date: 2026-11-06

在线评估结果落库:每次问答一行(规则全量 + 抽样 judge),失败样本按 `failed` 非空查询。
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "9f3d2c1b7a80"
down_revision: str | None = "7c1a9f4e2b60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "online_eval",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("format_ok", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("citation_ok", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("safety_ok", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("failed", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("sampled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("faithfulness", sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column("answer_relevancy", sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["platform.tenant.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["platform.trace_run.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("run_id", name="uq_online_eval_run"),
        schema="platform",
    )
    op.create_index(
        "ix_online_eval_tenant_created",
        "online_eval",
        ["tenant_id", "created_at"],
        schema="platform",
    )
    # 与模型里的 `index=True` 对齐;autogenerate 会逐列要求这两个索引。
    op.create_index(
        "ix_platform_online_eval_tenant_id", "online_eval", ["tenant_id"], schema="platform"
    )
    op.create_index(
        "ix_platform_online_eval_run_id", "online_eval", ["run_id"], schema="platform"
    )


def downgrade() -> None:
    op.drop_index("ix_platform_online_eval_run_id", table_name="online_eval", schema="platform")
    op.drop_index("ix_platform_online_eval_tenant_id", table_name="online_eval", schema="platform")
    op.drop_index("ix_online_eval_tenant_created", table_name="online_eval", schema="platform")
    op.drop_table("online_eval", schema="platform")
