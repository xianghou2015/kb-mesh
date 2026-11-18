"""切片写入:Postgres 权威记录 + Milvus 向量投影(§5.3 幂等第二层)。

Postgres 是权威,两个引擎是**可重建的投影** —— 所以这里的顺序永远是「先 PG 落定,
再投影出去」,并且投影失败可以整库重放,不会丢数据(§4.1)。

主键与 Milvus `pk` 都由 `chunk_key` 派生:重跑时同一份内容算出同一个 id,
不需要「先查有没有再插入」这种会在并发下出错的读-改-写。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pymilvus import MilvusClient

from kbmesh.core.tenancy import TenantContext
from kbmesh.db.models import Chunk
from kbmesh.ingest.chunking import ChunkPart
from kbmesh.storage import milvus as mv


def chunk_id_of(chunk_key: str) -> str:
    return f"chk-{chunk_key[:24]}"


def milvus_pk_of(chunk_key: str) -> int:
    """60 位正值:sha256 前 15 个十六进制位,落进 INT64 且稳定可重放。"""
    return int(chunk_key[:15], 16)


def indexed_body(chunk: Chunk | ChunkView) -> str:
    """进 BM25 与向量的正文 = 源切片 + 视觉回填(FR-104「图片内容进入切片正文并参与检索」)。"""
    captions = [str(c.get("text", "")) for c in chunk.captions or [] if c.get("text")]
    return "\n".join(x for x in (chunk.text, *captions) if x)


@dataclass
class ChunkView:
    """跨阶段传递的切片快照(纯数据)。

    不用 ORM 对象穿越阶段:§5.4 要求外部 I/O 期间不持有事务,而 ORM 实例一脱离它的事务就
    读不到属性了。把要用的字段显式拷出来,既是那条规则的落地,也顺带写清了阶段间的数据契约。
    """

    id: str
    tenant_id: str
    kb_id: str
    doc_id: str
    chunk_key: str
    heading_path: str
    page_no: int | None
    text: str
    token_len: int
    image_refs: list[str]
    captions: list[dict[str, Any]]
    milvus_pk: int
    embedding: list[float] | None
    graph_linked: bool

    @property
    def needs_caption(self) -> bool:
        return bool(self.image_refs) and not self.captions

    def body(self) -> str:
        return indexed_body(self)


def view_of(row: Chunk) -> ChunkView:
    return ChunkView(
        id=row.id,
        tenant_id=row.tenant_id,
        kb_id=row.kb_id,
        doc_id=row.doc_id,
        chunk_key=row.chunk_key,
        heading_path=row.heading_path,
        page_no=row.page_no,
        text=row.text,
        token_len=row.token_len,
        image_refs=list(row.image_refs or []),
        captions=list(row.captions or []),
        milvus_pk=row.milvus_pk,
        embedding=list(row.embedding) if row.embedding is not None else None,
        graph_linked=row.graph_linked,
    )


def rows_from_parts(
    ctx: TenantContext,
    parts: Sequence[ChunkPart],
    *,
    kb_id: str,
    doc_id: str,
    index_version: int,
) -> list[dict[str, Any]]:
    """把切片产物换成 `chunk` 表的插入字典(幂等键 `(tenant_id, chunk_key)`)。"""
    rows: list[dict[str, Any]] = []
    for part in parts:
        key = part.chunk_key
        rows.append(
            {
                "id": chunk_id_of(key),
                "tenant_id": ctx.tenant_id,
                "kb_id": kb_id,
                "doc_id": doc_id,
                "chunk_key": key,
                "heading_path": mv.truncate_utf8(part.heading_path, mv.HEADING_TEXT_MAX),
                "level": part.level,
                "position": part.position,
                "page_no": part.page_no,
                "text": part.text,
                "token_len": part.token_len,
                "image_refs": list(part.image_refs),
                "captions": [],
                "milvus_pk": milvus_pk_of(key),
                "graph_linked": False,
                "index_version": index_version,
            }
        )
    return rows


def index_vectors(
    ctx: TenantContext,
    doc_id: str,
    chunks: Sequence[ChunkView],
    embeddings: Mapping[str, Sequence[float]],
    *,
    client: MilvusClient | None = None,
) -> dict[str, int]:
    """upsert 本批 + 删孤儿:切片策略变了,旧向量不能继续被召回。

    `keep` 取的是**该文档当前全部**切片的 pk,不是本批的:分批嵌入时,
    前一批刚写进去的行如果当批就当成孤儿删掉,索引会被自己清空。
    """
    if not chunks:
        return {"upserted": 0, "deleted": 0}
    keep = {c.milvus_pk for c in chunks}
    rows = [
        mv.to_row(
            ctx,
            pk=c.milvus_pk,
            chunk_id=c.id,
            doc_id=c.doc_id,
            kb_id=c.kb_id,
            page_no=c.page_no,
            heading_path=c.heading_path,
            body_text=indexed_body(c),
            dense=list(embeddings[c.id]),
        )
        for c in chunks
        if embeddings.get(c.id)
    ]
    upserted = mv.upsert_rows(rows, client=client)
    orphans = mv.pks_for_doc(ctx, doc_id, client=client) - keep
    deleted = mv.delete_orphan_pks(orphans, client=client)
    return {"upserted": upserted, "deleted": deleted}
