"""rename tables to domain prefixes

Revision ID: a1b2c3d4e5f6
Revises: 9f3d2c1b7a80
Create Date: 2026-11-19

kb_: tenant/app_user/kb/document/chunk/entity/relation_edge/feedback
sec_: security_event
eval_: online_eval
trace_/eval_set/eval_case/eval_run/eval_result 不变。
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "9f3d2c1b7a80"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = "platform"

_TABLE_RENAMES: list[tuple[str, str]] = [
    ("tenant", "kb_tenant"),
    ("app_user", "kb_user"),
    ("kb", "kb_base"),
    ("document", "kb_document"),
    ("chunk", "kb_chunk"),
    ("entity", "kb_entity"),
    ("relation_edge", "kb_edge"),
    ("security_event", "sec_event"),
    ("feedback", "kb_feedback"),
    ("online_eval", "eval_online"),
]

_CONSTRAINT_RENAMES: list[tuple[str, str, str]] = [
    ("kb_user", "uq_app_user_tenant_username", "uq_kb_user_tenant_username"),
    ("kb_base", "uq_kb_tenant_name", "uq_kb_base_tenant_name"),
    ("kb_document", "uq_document_tenant_kb_sha256", "uq_kb_document_tenant_kb_sha256"),
    ("kb_chunk", "uq_chunk_tenant_key", "uq_kb_chunk_tenant_key"),
    ("kb_entity", "uq_entity_tenant_key", "uq_kb_entity_tenant_key"),
    ("kb_edge", "uq_edge_tenant_src_dst_type", "uq_kb_edge_tenant_src_dst_type"),
    ("eval_online", "uq_online_eval_run", "uq_eval_online_run"),
]

_INDEX_RENAMES: list[tuple[str, str]] = [
    ("ix_app_user_tenant_username", "ix_kb_user_tenant_username"),
    ("ix_platform_app_user_tenant_id", "ix_platform_kb_user_tenant_id"),
    ("ix_kb_tenant_name", "ix_kb_base_tenant_name"),
    ("ix_platform_kb_tenant_id", "ix_platform_kb_base_tenant_id"),
    ("ix_document_tenant_kb_status", "ix_kb_document_tenant_kb_status"),
    ("ix_platform_document_kb_id", "ix_platform_kb_document_kb_id"),
    ("ix_platform_document_tenant_id", "ix_platform_kb_document_tenant_id"),
    ("ix_entity_tenant_kb", "ix_kb_entity_tenant_kb"),
    ("ix_platform_entity_kb_id", "ix_platform_kb_entity_kb_id"),
    ("ix_platform_entity_tenant_id", "ix_platform_kb_entity_tenant_id"),
    ("ix_chunk_tenant_doc", "ix_kb_chunk_tenant_doc"),
    ("ix_chunk_tenant_kb_doc", "ix_kb_chunk_tenant_kb_doc"),
    ("ix_platform_chunk_doc_id", "ix_platform_kb_chunk_doc_id"),
    ("ix_platform_chunk_kb_id", "ix_platform_kb_chunk_kb_id"),
    ("ix_platform_chunk_milvus_pk", "ix_platform_kb_chunk_milvus_pk"),
    ("ix_platform_chunk_tenant_id", "ix_platform_kb_chunk_tenant_id"),
    ("ix_edge_tenant_kb", "ix_kb_edge_tenant_kb"),
    ("ix_platform_relation_edge_chunk_id", "ix_platform_kb_edge_chunk_id"),
    ("ix_platform_relation_edge_kb_id", "ix_platform_kb_edge_kb_id"),
    ("ix_platform_relation_edge_tenant_id", "ix_platform_kb_edge_tenant_id"),
    ("ix_feedback_tenant_created", "ix_kb_feedback_tenant_created"),
    ("ix_platform_feedback_run_id", "ix_platform_kb_feedback_run_id"),
    ("ix_platform_feedback_tenant_id", "ix_platform_kb_feedback_tenant_id"),
    ("ix_security_event_tenant_created", "ix_sec_event_tenant_created"),
    ("ix_platform_security_event_tenant_id", "ix_platform_sec_event_tenant_id"),
    ("ix_online_eval_tenant_created", "ix_eval_online_tenant_created"),
    ("ix_platform_online_eval_tenant_id", "ix_platform_eval_online_tenant_id"),
    ("ix_platform_online_eval_run_id", "ix_platform_eval_online_run_id"),
]


def upgrade() -> None:
    for old, new in _TABLE_RENAMES:
        op.rename_table(old, new, schema=S)
    for table, old, new in _CONSTRAINT_RENAMES:
        op.execute(f'ALTER TABLE {S}.{table} RENAME CONSTRAINT "{old}" TO "{new}"')
    for old, new in _INDEX_RENAMES:
        op.execute(f'ALTER INDEX {S}."{old}" RENAME TO "{new}"')


def downgrade() -> None:
    for old, new in reversed(_INDEX_RENAMES):
        op.execute(f'ALTER INDEX {S}."{new}" RENAME TO "{old}"')
    for table, old, new in reversed(_CONSTRAINT_RENAMES):
        op.execute(f'ALTER TABLE {S}.{table} RENAME CONSTRAINT "{new}" TO "{old}"')
    for old, new in reversed(_TABLE_RENAMES):
        op.rename_table(new, old, schema=S)