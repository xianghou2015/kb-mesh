"""Authoritative Postgres tables (schema `platform`).

Milvus and Neo4j are *projections* of these rows, not the source of truth: a corrupted index is
rebuilt by replaying PG, so every field either engine needs must be recoverable from here — which
is why `chunk.embedding` is stored even though search runs in Milvus.

Every business table carries `tenant_id NOT NULL` plus a `(tenant_id, …)` composite index
(§6.1 防线①). A query that forgets the tenant column cannot use those indexes, which is how the
storage layer makes the application-layer filter cheap rather than optional.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kbmesh.core.config import get_settings
from kbmesh.core.tenancy import ALL_ROLES
from kbmesh.db.base import Base

SCHEMA = "platform"


def new_id(prefix: str = "") -> str:
    """String ids with a readable prefix: they appear in logs, citations and Milvus `chunk_id`."""
    return f"{prefix}{uuid.uuid4().hex}" if prefix else uuid.uuid4().hex


def milvus_pk_for(chunk_id: str) -> int:
    """Stable INT64 primary key inside Milvus, derived from `chunk_id`.

    Milvus needs an int pk and upsert idempotency depends on the same chunk always landing on the
    same pk — so the value is a hash of the immutable `chunk_id`, computed here and nowhere else.
    62 bits keeps it inside INT64 and non-negative.
    """
    digest = uuid.uuid5(uuid.NAMESPACE_OID, chunk_id).hex
    return int(digest[:16], 16) & (2**62 - 1)


class ParseStage(enum.StrEnum):
    """`document.parse_status` state machine (§5.2). Transitions move forward or retry in place."""

    PENDING = "PENDING"
    PARSING = "PARSING"
    CHUNKED = "CHUNKED"
    EMBEDDED = "EMBEDDED"
    GRAPH_QUEUED = "GRAPH_QUEUED"
    GRAPH_DONE = "GRAPH_DONE"
    DONE = "DONE"
    FAILED = "FAILED"


#: Allowed forward moves. Anything else is a bug in the pipeline, so the repo raises instead of
#: quietly accepting a skipped stage — a re-run has to be decidable (§5.2 "禁止跳跃").
STAGE_ORDER: tuple[ParseStage, ...] = (
    ParseStage.PENDING,
    ParseStage.PARSING,
    ParseStage.CHUNKED,
    ParseStage.EMBEDDED,
    ParseStage.GRAPH_QUEUED,
    ParseStage.GRAPH_DONE,
    ParseStage.DONE,
)

FINAL_STATUSES = ("answered", "refused", "error", "expired")


class Tenant(Base):
    __tablename__ = "kb_tenant"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("tnt-"))
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), default="active", server_default="'active'")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AppUser(Base):
    __tablename__ = "kb_user"
    __table_args__ = (
        UniqueConstraint("tenant_id", "username", name="uq_kb_user_tenant_username"),
        Index("ix_kb_user_tenant_username", "tenant_id", "username"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    username: Mapped[str] = mapped_column(String(120))
    #: Role names come from `tenancy.ALL_ROLES`; the token claim is the authority, this column is
    #: what the management UI shows and what issued tokens are checked against on rotation.
    role: Mapped[str] = mapped_column(String(32), default=ALL_ROLES[-1])
    password_hash: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeBase(Base):
    __tablename__ = "kb_base"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_kb_base_tenant_name"),
        Index("ix_kb_base_tenant_name", "tenant_id", "name"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("kb-"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    #: Recorded per KB because the embedding model is a rebuild input: a KB whose model differs
    #: from the current default must be re-indexed with *its own* model, not the new one.
    embedding_model: Mapped[str] = mapped_column(String(120))
    chunk_policy: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    __tablename__ = "kb_document"
    __table_args__ = (
        Index("ix_kb_document_tenant_kb_status", "tenant_id", "kb_id", "parse_status"),
        UniqueConstraint("tenant_id", "kb_id", "sha256", name="uq_kb_document_tenant_kb_sha256"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("doc-"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    kb_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_base.id"), index=True)
    source_uri: Mapped[str] = mapped_column(String(600))
    mime: Mapped[str] = mapped_column(String(120), default="")
    sha256: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(400), default="")
    parse_status: Mapped[str] = mapped_column(
        String(32), default=ParseStage.PENDING.value, server_default=f"'{ParseStage.PENDING.value}'"
    )
    #: Bumped when the chunk policy or the embedding model changes; it is part of every Celery
    #: task id, so a re-index of an already-processed document cannot collide with the old run.
    index_version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    failure: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Chunk(Base):
    __tablename__ = "kb_chunk"
    __table_args__ = (
        # §5.3 幂等第一层:切片键唯一,重跑走 ON CONFLICT DO NOTHING
        UniqueConstraint("tenant_id", "chunk_key", name="uq_kb_chunk_tenant_key"),
        Index("ix_kb_chunk_tenant_kb_doc", "tenant_id", "kb_id", "doc_id"),
        Index("ix_kb_chunk_tenant_doc", "tenant_id", "doc_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("chk-"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    kb_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_base.id"), index=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_document.id"), index=True)
    chunk_key: Mapped[str] = mapped_column(String(64))
    heading_path: Mapped[str] = mapped_column(String(600), default="")
    level: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    position: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    page_no: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_len: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    image_refs: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    #: 视觉回填(FR-104):`[{"ref": 原图引用, "text": 描述}]`。
    #: 单独一列而不是并进 `text`——`chunk_key` 只由源文决定,模型输出进键里会让每次重跑都长出新切片。
    captions: Mapped[list[Any]] = mapped_column(JSONB, default=list, server_default=sa_text("'[]'::jsonb"))
    milvus_pk: Mapped[int] = mapped_column(BigInteger, index=True)
    graph_linked: Mapped[bool] = mapped_column(default=False, server_default="false")
    #: Kept so Milvus can be rebuilt without paying the embedding API again.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(get_settings().embedding_dim))
    index_version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Entity(Base):
    __tablename__ = "kb_entity"
    __table_args__ = (
        UniqueConstraint("tenant_id", "entity_key", name="uq_kb_entity_tenant_key"),
        Index("ix_kb_entity_tenant_kb", "tenant_id", "kb_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ent-"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    kb_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_base.id"), index=True)
    #: Normalized label + alias-table collapse, computed before the write. Neo4j MERGE keys on it,
    #: so an unstable key means duplicate nodes on every re-index.
    entity_key: Mapped[str] = mapped_column(String(200))
    etype: Mapped[str] = mapped_column(String(60))
    label: Mapped[str] = mapped_column(String(300))
    aliases: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RelationEdge(Base):
    __tablename__ = "kb_edge"
    __table_args__ = (
        UniqueConstraint("tenant_id", "src_key", "dst_key", "rtype", name="uq_kb_edge_tenant_src_dst_type"),
        Index("ix_kb_edge_tenant_kb", "tenant_id", "kb_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    kb_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_base.id"), index=True)
    src_key: Mapped[str] = mapped_column(String(200))
    dst_key: Mapped[str] = mapped_column(String(200))
    #: Whitelisted relation name. Neo4j cannot parameterize a relationship type, so the same value
    #: lives as a property on `[:REL]` and the graph layer filters it with `r.rtype IN $types`.
    rtype: Mapped[str] = mapped_column(String(60))
    evidence: Mapped[str] = mapped_column(Text)
    chunk_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_chunk.id"), index=True)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), default=1.0, server_default="1.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SecurityEvent(Base):
    """越权尝试与认证失败的审计流水(§9:断言「空结果 **且** 审计有记录」)。

    这是唯一一张 `tenant_id` 可空的表 —— 未认证的请求没有租户可归属,而「谁在拿 A 的令牌查 B」
    恰恰是最需要留痕的一条。读取侧仍然只按 `ctx.tenant_id` 过滤,平台管理员可跨租户查。
    """

    __tablename__ = "sec_event"
    __table_args__ = (
        Index("ix_sec_event_tenant_created", "tenant_id", "created_at"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("sec-"))
    tenant_id: Mapped[str | None] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(64), default="")
    #: `unauthenticated` / `bad_token` / `cross_tenant_read` / `kb_scope_denied` / `role_denied`
    event: Mapped[str] = mapped_column(String(40))
    #: 被触碰的资源标识,如 `kb:kb-abc` / `thread:tnt-1:th-9`;只记标识不记内容。
    target: Mapped[str] = mapped_column(String(300), default="")
    path: Mapped[str] = mapped_column(String(300), default="")
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TraceRun(Base):
    __tablename__ = "trace_run"
    __table_args__ = (
        Index("ix_trace_run_tenant_created", "tenant_id", "created_at"),
        Index("ix_trace_run_tenant_thread", "tenant_id", "thread_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("run-"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    thread_id: Mapped[str] = mapped_column(String(200))
    user_id: Mapped[str] = mapped_column(String(64), default="")
    query: Mapped[str] = mapped_column(Text)
    final_status: Mapped[str] = mapped_column(String(32), default="error", server_default="'error'")
    degraded: Mapped[bool] = mapped_column(default=False, server_default="false")
    route_flags: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    token_in: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    token_out: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    steps: Mapped[list[TraceStep]] = relationship(back_populates="run", cascade="all, delete-orphan")


class TraceStep(Base):
    __tablename__ = "trace_step"
    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_trace_step_run_seq"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.trace_run.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    stage: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    token_in: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    token_out: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    model: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[TraceRun] = relationship(back_populates="steps")


class Feedback(Base):
    __tablename__ = "kb_feedback"
    __table_args__ = (
        Index("ix_kb_feedback_tenant_created", "tenant_id", "created_at"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.trace_run.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(64), default="")
    score: Mapped[int] = mapped_column(Integer)
    corrected_answer: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvalSet(Base):
    __tablename__ = "eval_set"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", "version", name="uq_eval_set_tenant_name_version"),
        Index("ix_eval_set_tenant_name", "tenant_id", "name"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("set-"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    #: Once frozen, a trigger rejects UPDATE/DELETE on this row and on its cases (§8.3): a frozen
    #: eval set is the only way two runs are comparable, and a silently edited set lies forever.
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    cases: Mapped[list[EvalCase]] = relationship(back_populates="set", cascade="all, delete-orphan")


class EvalCase(Base):
    __tablename__ = "eval_case"
    __table_args__ = (
        Index("ix_eval_case_set_group", "set_id", "group_key"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("case-"))
    set_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.eval_set.id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(Text)
    #: 反馈池导入时用户提交的参考答案;人工确认后作为答案层 judge 的对照。程序不拿它当 gold
    #: chunk(那是 gold_refs 的职责),两者语义不同,不能混用。
    reference_answer: Mapped[str | None] = mapped_column(Text)
    #: chunk_id list (and optionally doc-level expectations) the answer must be able to cite.
    gold_refs: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    #: Slicing dimension for §8.2 — the gate compares per group, never only on the total.
    group_key: Mapped[str] = mapped_column(String(40))
    expected_refusal: Mapped[bool] = mapped_column(default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    set: Mapped[EvalSet] = relationship(back_populates="cases")


class EvalRun(Base):
    __tablename__ = "eval_run"
    __table_args__ = (
        Index("ix_eval_run_set_status", "set_id", "status"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("evr-"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    set_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.eval_set.id"), index=True)
    #: The config under test, fingerprinted and stored — §8.4 attributes a delta to a change only
    #: because the run records what produced it.
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    baseline_run_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending", server_default="'pending'")
    #: Last completed case, so an interrupted run resumes without re-running scored cases (§8.3).
    cursor_case_id: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvalResult(Base):
    __tablename__ = "eval_result"
    __table_args__ = (
        UniqueConstraint("eval_run_id", "case_id", "metric", name="uq_eval_result_run_case_metric"),
        Index("ix_eval_result_run_metric", "eval_run_id", "metric"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    eval_run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.eval_run.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.eval_case.id"), index=True)
    metric: Mapped[str] = mapped_column(String(60))
    value: Mapped[float] = mapped_column(Numeric(8, 6))
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OnlineEval(Base):
    """一次问答的在线评估结果(FR-601)。一次 run 只有一行,citation/format/safety 全量,
    judge 两项按抽样率可能为空。失败样本靠 `failed` 非空筛出来。"""

    __tablename__ = "eval_online"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_eval_online_run"),
        Index("ix_eval_online_tenant_created", "tenant_id", "created_at"),
        {"schema": SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("oev-"))
    tenant_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.kb_tenant.id"), index=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.trace_run.id", ondelete="CASCADE"), index=True
    )
    format_ok: Mapped[bool] = mapped_column(default=True, server_default="true")
    citation_ok: Mapped[bool] = mapped_column(default=True, server_default="true")
    safety_ok: Mapped[bool] = mapped_column(default=True, server_default="true")
    failed: Mapped[list[str]] = mapped_column(JSONB, default=list)
    sampled: Mapped[bool] = mapped_column(default=False, server_default="false")
    faithfulness: Mapped[float | None] = mapped_column(Numeric(8, 6))
    answer_relevancy: Mapped[float | None] = mapped_column(Numeric(8, 6))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
