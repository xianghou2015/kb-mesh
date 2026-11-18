"""离线索引管道(§5.1 的六个阶段)。

三条规则贯穿整个模块:

1. **三段式(§5.4)**:读 → 无锁外部 I/O(模型、Milvus、Neo4j、对象存储)→ 短事务写。
   事务里绝不调模型;为此阶段之间传的是 `ChunkView` 纯数据,不是 ORM 对象。
2. **幂等(§5.3)**:切片键 upsert + 孤儿清理;向量先 upsert 再按 pk 删旧;图谱走 MERGE。
   任何阶段都可以原样重跑,重跑五次与跑一次的落库结果相同。
3. **身份来自权威行(防线②的离线形态)**:worker 只按主键取 `document`,租户与知识库从这行读出来,
   之后的每次仓储访问都被这个上下文收窄到本租户、本知识库。

阶段失败不在这里吞:交给 `run_stage()` 统一写 `document.failure(stage, error, retry_count)`,
状态机进 FAILED,重跑从失败阶段原地继续。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kbmesh.core.config import get_settings
from kbmesh.core.errors import NotFound
from kbmesh.core.gateway import Gateway, ModelUnavailable, get_gateway
from kbmesh.core.tenancy import TenantContext, use_context, worker_context
from kbmesh.db.models import Chunk, Document, Entity, KnowledgeBase, ParseStage, RelationEdge
from kbmesh.db.repositories import ChunkRepo, DocumentRepo, GraphRepo
from kbmesh.ingest import graph_extract as ge
from kbmesh.ingest import lease
from kbmesh.ingest.chunking import ChunkPolicy, split
from kbmesh.ingest.extract import extract, render_page_png
from kbmesh.ingest.writer import ChunkView, index_vectors, rows_from_parts, view_of
from kbmesh.storage import milvus as mv
from kbmesh.storage import neo4j as n4
from kbmesh.storage import s3

#: §5.1 的 DAG 顺序;`parse` 含解析与切片两步(它们共用一次对象存储读取)
PIPELINE_STAGES = ("parse", "caption", "embed", "extract", "graph", "link")

CAPTION_PROMPT = "描述这张图:图里的文字逐字转写,图表则给出结论与关键数值。不超过 200 字。"
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass
class DocJob:
    """文档快照 + 由它推导出的租户上下文(纯数据,可跨事务与跨阶段传递)。"""

    doc_id: str
    kb_id: str
    tenant_id: str
    index_version: int
    source_uri: str
    mime: str
    filename: str
    chunk_policy: dict[str, Any]
    ctx: TenantContext

    @property
    def is_pdf(self) -> bool:
        return "pdf" in (self.mime or "").lower() or self.filename.lower().endswith(".pdf")


@dataclass
class StageReport:
    stage: str
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "counts": self.counts, "warnings": self.warnings}


def _factory(session_factory: async_sessionmaker[AsyncSession] | None) -> async_sessionmaker[AsyncSession]:
    if session_factory is not None:
        return session_factory
    from kbmesh.db.base import get_session_factory

    return get_session_factory()


async def load_job(factory: async_sessionmaker[AsyncSession], doc_id: str) -> DocJob:
    """防线② 的离线形态:租户不是传进来的,是从权威行读出来的。

    这是唯一一次「不带租户谓词」的读取 —— 上下文本身要由这行产生,先有鸡还是先有蛋。
    拿到 `DocJob` 之后,每一次仓储调用都必须经由 `ctx`,跨租户的数据一步都出不去。
    """
    async with factory() as session, session.begin():
        row = await session.get(Document, doc_id)
        if row is None:
            raise NotFound(f"document {doc_id} 不存在")
        kb = await session.get(KnowledgeBase, row.kb_id)
        if kb is None:
            raise NotFound(f"知识库 {row.kb_id} 不存在")
        return DocJob(
            doc_id=row.id,
            kb_id=row.kb_id,
            tenant_id=row.tenant_id,
            index_version=row.index_version,
            source_uri=row.source_uri,
            mime=row.mime,
            filename=row.source_uri.rsplit("/", 1)[-1],
            chunk_policy=dict(kb.chunk_policy or {}),
            ctx=worker_context(row.tenant_id, row.kb_id),
        )


async def _views(job: DocJob, factory: async_sessionmaker[AsyncSession]) -> list[ChunkView]:
    async with factory() as session, session.begin():
        rows = await ChunkRepo(job.ctx, session).list_for_doc(job.doc_id)
        return [view_of(r) for r in rows]


async def _stage(session: AsyncSession, job: DocJob, stage: ParseStage) -> None:
    await DocumentRepo(job.ctx, session).set_stage(job.doc_id, stage)


# --------------------------------------------------------------------- ① 解析 + 切片


async def stage_parse(
    factory: async_sessionmaker[AsyncSession] | None = None,
    doc_id: str = "",
    *,
    policy: ChunkPolicy | None = None,
) -> StageReport:
    factory = _factory(factory)
    job = await load_job(factory, doc_id)
    with use_context(job.ctx):
        async with factory() as session, session.begin():
            await _stage(session, job, ParseStage.PARSING)

        data = s3.get_bytes(job.source_uri)
        parsed = extract(data, job.mime, filename=job.filename)
        parts = split(parsed, doc_id=job.doc_id, policy=policy or ChunkPolicy.from_dict(job.chunk_policy))
        rows = rows_from_parts(
            job.ctx, parts, kb_id=job.kb_id, doc_id=job.doc_id, index_version=job.index_version
        )

        async with factory() as session, session.begin():
            repo = ChunkRepo(job.ctx, session)
            inserted = await repo.insert_many_ignoring(rows)
            removed = await repo.delete_keys_not_in(job.doc_id, {r["chunk_key"] for r in rows})
            await _stage(session, job, ParseStage.CHUNKED)
        return StageReport(
            "parse",
            {"blocks": len(parsed.blocks), "parts": len(rows), "inserted": inserted, "removed": removed},
            list(parsed.warnings),
        )


# --------------------------------------------------------------------- ② 视觉回填


async def stage_caption(
    factory: async_sessionmaker[AsyncSession] | None = None,
    doc_id: str = "",
    *,
    gateway: Gateway | None = None,
) -> StageReport:
    factory = _factory(factory)
    job = await load_job(factory, doc_id)
    views = [v for v in await _views(job, factory) if v.needs_caption]
    if not views:
        return StageReport("caption", {"chunks": 0, "filled": 0})

    gw = gateway or get_gateway()
    source = s3.get_bytes(job.source_uri) if job.is_pdf else b""
    filled: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    # 模型调用全程在事务外:一张图几十秒,连接被事务占着就是全池排队(§5.4)
    for view in views:
        captions = []
        for ref in view.image_refs:
            try:
                text = await _caption_one(job, ref, source, gw)
            except ModelUnavailable as exc:
                warnings.append(f"{ref}: {exc.detail}")
                continue
            if text:
                captions.append({"ref": ref, "text": text})
        if captions:
            filled[view.id] = captions

    if filled:
        async with factory() as session, session.begin():
            rows = await ChunkRepo(job.ctx, session).by_ids(list(filled))
            for row in rows:
                row.captions = filled[row.id]
    return StageReport("caption", {"chunks": len(views), "filled": len(filled)}, warnings)


async def _caption_one(job: DocJob, ref: str, source: bytes, gw: Gateway) -> str:
    if ref.startswith("page:"):
        if not source:
            return ""
        page_no = int(ref.split(":")[1])
        png = render_page_png(source, page_no)
        uri = s3.put_bytes(
            png,
            key=s3.page_image_key(job.ctx, job.doc_id, page_no),
            mime="image/png",
        )
        return gw.vision(s3.presign_get(uri), CAPTION_PROMPT).text.strip()

    candidate = _image_uri(job, ref)
    try:
        s3.get_bytes(candidate)
    except Exception as exc:  # noqa: BLE001  (对象不存在 = 这条引用不可读)
        raise ModelUnavailable("caption:image_missing", f"{ref} 无对应产物({type(exc).__name__})") from exc
    return gw.vision(s3.presign_get(candidate), CAPTION_PROMPT).text.strip()


def _image_uri(job: DocJob, ref: str) -> str:
    """文档里的图片引用是**作者写的字符串**,直接拼进 key 就等于让他试 `../../别的租户/…`。

    所以只允许相对路径形态:逐段过 `_SAFE_SEGMENT`,任何 `..`、绝对路径、查询串都当成不可读。
    """
    tail = ref.split("?", 1)[0].split("#", 1)[0].lstrip("./")
    segments = [seg for seg in tail.split("/") if seg]
    if not segments or any(not _SAFE_SEGMENT.fullmatch(seg) for seg in segments):
        raise ModelUnavailable("caption:image_ref_rejected", f"非法图片引用: {ref!r}")
    key = f"artifacts/{job.ctx.tenant_id}/{job.kb_id}/images/{'/'.join(segments)}"
    return f"s3://{get_settings().oss_bucket}/{key}"


# --------------------------------------------------------------------- ③ 嵌入 + 向量投影


async def stage_embed(
    factory: async_sessionmaker[AsyncSession] | None = None,
    doc_id: str = "",
    *,
    gateway: Gateway | None = None,
) -> StageReport:
    factory = _factory(factory)
    job = await load_job(factory, doc_id)
    views = await _views(job, factory)
    dim = get_settings().embedding_dim
    todo = [v for v in views if v.body() and (v.embedding is None or len(v.embedding or []) != dim)]

    embeddings = {
        v.id: list(v.embedding) for v in views if v.embedding is not None and len(v.embedding) == dim
    }
    if todo:
        gw = gateway or get_gateway()
        vectors, usage = gw.embed([v.body() for v in todo])  # 批量与维度校验在网关里
        if len(vectors) != len(todo):
            raise RuntimeError(f"嵌入返回 {len(vectors)} 条,与请求的 {len(todo)} 条不符")
        embeddings.update({v.id: list(vec) for v, vec in zip(todo, vectors, strict=True)})
        async with factory() as session, session.begin():
            rows = await ChunkRepo(job.ctx, session).by_ids([v.id for v in todo])
            for row in rows:
                row.embedding = embeddings[row.id]

    mv.ensure_collection()
    stats = index_vectors(job.ctx, job.doc_id, views, embeddings)
    async with factory() as session, session.begin():
        await _stage(session, job, ParseStage.EMBEDDED)
    return StageReport("embed", {**stats, "embedded": len(todo)})


# --------------------------------------------------------------------- ④ 三元组抽取


async def stage_extract(
    factory: async_sessionmaker[AsyncSession] | None = None,
    doc_id: str = "",
    *,
    gateway: Gateway | None = None,
    max_chunks: int = 200,
) -> StageReport:
    factory = _factory(factory)
    job = await load_job(factory, doc_id)
    views = [v for v in await _views(job, factory) if v.body()][:max_chunks]
    gw = gateway or get_gateway()

    stat = ge.DropStat()
    kept: list[ge.Triplet] = []
    for view in views:
        body = view.body()
        payload, _usage = gw.structured(ge.prompt_for(body), ge.TripletPayload, purpose="extract")
        # 第一道校验:抽取层。写入层(write_graph)还会再验一次,两道不许合并(FR-302)
        kept += ge.validate_triplets(payload, chunk_id=view.id, source_text=body, stat=stat)
    triplets = ge.dedupe(kept)

    entities: dict[str, dict[str, Any]] = {}
    for one in triplets:
        for ent in one.entities():
            entities[ent.entity_key] = {
                "entity_key": ent.entity_key,
                "etype": ent.etype,
                "label": ent.label,
                "aliases": [],
            }
    edge_rows = [
        {
            "src_key": t.src_key,
            "dst_key": t.dst_key,
            "rtype": t.rtype,
            "evidence": t.evidence,
            "chunk_id": t.chunk_id,
            "confidence": t.confidence,
        }
        for t in triplets
    ]
    async with factory() as session, session.begin():
        graph = GraphRepo(job.ctx, session)
        await graph.upsert_entities(job.kb_id, list(entities.values()))
        await graph.upsert_edges(job.kb_id, edge_rows)
        await _stage(session, job, ParseStage.GRAPH_QUEUED)
    return StageReport(
        "extract",
        {"chunks": len(views), "triplets": len(triplets), "entities": len(entities), "dropped": stat.total},
        [f"{reason}:{n}" for reason, n in sorted(stat.drops.items())],
    )


# --------------------------------------------------------------------- ⑤ 落图


async def stage_write_graph(
    factory: async_sessionmaker[AsyncSession] | None = None,
    doc_id: str = "",
    *,
    driver=None,
) -> StageReport:
    factory = _factory(factory)
    job = await load_job(factory, doc_id)
    views = await _views(job, factory)
    async with factory() as session, session.begin():
        entity_rows = list(
            (
                await session.execute(
                    select(Entity).where(Entity.tenant_id == job.ctx.tenant_id, Entity.kb_id == job.kb_id)
                )
            ).scalars()
        )
        edges = await _doc_edges(session, job)
        entities = [
            {
                "entity_key": e.entity_key,
                "etype": e.etype,
                "label": e.label,
                "aliases": list(e.aliases or []),
            }
            for e in entity_rows
        ]

    own = driver is None
    driver = driver or n4.get_driver()
    try:
        n4.ensure_schema(driver)
        made = {
            "entities": n4.merge_entities(driver, job.ctx, job.kb_id, entities),
            "chunkrefs": n4.merge_chunkrefs(driver, job.ctx, job.kb_id, _chunkref_rows(views)),
            "stale_chunkrefs": n4.prune_chunkrefs(driver, job.ctx, job.doc_id, {v.id for v in views}),
            "relations": n4.merge_relations(driver, job.ctx, job.kb_id, edges),
        }
    finally:
        if own:
            driver.close()
    return StageReport("graph", made)


async def _doc_edges(session: AsyncSession, job: DocJob) -> list[dict[str, Any]]:
    """按本文件的切片取边:边的来源切片是外键,租户 + kb 双谓词一起上。"""
    stmt = (
        select(RelationEdge)
        .join(Chunk, Chunk.id == RelationEdge.chunk_id)
        .where(
            RelationEdge.tenant_id == job.ctx.tenant_id,
            RelationEdge.kb_id == job.kb_id,
            Chunk.doc_id == job.doc_id,
        )
    )
    rows = list((await session.execute(stmt)).scalars())
    return [
        {
            "src_key": r.src_key,
            "dst_key": r.dst_key,
            "rtype": r.rtype,
            "evidence": r.evidence,
            "chunk_id": r.chunk_id,
            "confidence": float(r.confidence),
        }
        for r in rows
    ]


def _chunkref_rows(views: list[ChunkView]) -> list[dict[str, Any]]:
    return [{"chunk_id": v.id, "doc_id": v.doc_id, "kb_id": v.kb_id} for v in views]


# --------------------------------------------------------------------- ⑥ 桥接


async def stage_link(
    factory: async_sessionmaker[AsyncSession] | None = None,
    doc_id: str = "",
    *,
    driver=None,
) -> StageReport:
    factory = _factory(factory)
    job = await load_job(factory, doc_id)
    async with factory() as session, session.begin():
        edges = await _doc_edges(session, job)

    rows = [
        {"chunk_id": e["chunk_id"], "entity_key": key, "doc_id": job.doc_id, "kb_id": job.kb_id}
        for e in edges
        for key in (e["src_key"], e["dst_key"])
    ]

    own = driver is None
    driver = driver or n4.get_driver()
    try:
        n4.ensure_schema(driver)
        linked = n4.merge_mentions(driver, job.ctx, job.kb_id, rows)
    finally:
        if own:
            driver.close()

    async with factory() as session, session.begin():
        graph = GraphRepo(job.ctx, session)
        marked = await graph.mark_linked({r["chunk_id"] for r in rows})
        await _stage(session, job, ParseStage.GRAPH_DONE)
        await _stage(session, job, ParseStage.DONE)
    return StageReport("link", {"mentions": linked, "chunks_linked": marked})


STAGE_FUNCS = {
    "parse": stage_parse,
    "caption": stage_caption,
    "embed": stage_embed,
    "extract": stage_extract,
    "graph": stage_write_graph,
    "link": stage_link,
}


async def run_stage(
    stage: str,
    doc_id: str,
    *,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """单阶段执行 + 失败收口:异常一律落成 `FAILED(stage, error, retry_count)` 再抛。

    不吞异常是 §5.4 的后半句:吞掉后释放租约,下一个 worker 会看到「已成功」的假状态。
    """
    factory = _factory(session_factory)
    job = await load_job(factory, doc_id)
    func = STAGE_FUNCS[stage]
    with use_context(job.ctx):
        lease_key = lease.task_key(doc_id, job.index_version, stage)
        taken = lease.acquire(lease_key)
        if taken is None:
            done = lease.read_done(lease_key)
            if done:
                return done
            return {"stage": stage, "counts": {}, "warnings": [f"同阶段已在另一 worker 上运行: {doc_id}"]}
        try:
            report: StageReport = await func(factory, doc_id, **kwargs)
        except Exception as exc:  # noqa: BLE001
            await _mark_failed(factory, job, stage, exc)
            raise
        finally:
            lease.release(taken)
        payload = report.as_dict()
        lease.mark_done(lease_key, payload)
        return payload


async def _mark_failed(
    factory: async_sessionmaker[AsyncSession], job: DocJob, stage: str, exc: Exception
) -> None:
    async with factory() as session, session.begin():
        repo = DocumentRepo(job.ctx, session)
        row = await repo.get(job.doc_id)
        previous = (row.failure or {}) if row else {}
        await repo.mark_failed(
            job.doc_id,
            {
                "stage": stage,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "retry_count": int(previous.get("retry_count", 0)) + 1,
            },
        )


async def index_document(
    doc_id: str,
    *,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    gateway: Gateway | None = None,
    driver=None,
    policy: ChunkPolicy | None = None,
) -> list[dict[str, Any]]:
    """整条 DAG 顺序跑完(在线重索引与验收测试都走这里;Celery 上则由 chain 分阶段投递)。"""
    factory = _factory(session_factory)
    kwargs: dict[str, dict[str, Any]] = {
        "parse": {"policy": policy},
        "caption": {"gateway": gateway},
        "embed": {"gateway": gateway},
        "extract": {"gateway": gateway},
        "graph": {"driver": driver},
        "link": {"driver": driver},
    }
    reports = []
    for stage in PIPELINE_STAGES:
        extra = {k: v for k, v in kwargs[stage].items() if v is not None}
        reports.append(await run_stage(stage, doc_id, session_factory=factory, **extra))
    return reports
