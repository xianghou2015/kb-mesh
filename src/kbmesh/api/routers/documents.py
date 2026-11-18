"""文档入库与重索引的 HTTP 面(FR-106/107)。

入库三步:MinIO 写原件 → PG 建权威行 → Celery 投链,都成功才算入库。对象写入与任务投递
都经由 `storage` / `ingest_tasks` 的**模块符号**调用,测试可以整体替换;PG 行为照旧打真库,
租户隔离断言全部落在权威行上,不落在替身上。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from kbmesh.api.deps import get_ctx, get_db
from kbmesh.core.errors import IllegalTransition
from kbmesh.core.tenancy import TenantContext
from kbmesh.db.models import Document, ParseStage
from kbmesh.db.repositories import DocumentRepo
from kbmesh.ingest import tasks as ingest_tasks
from kbmesh.storage import s3 as storage

router = APIRouter(prefix="/api", tags=["documents"])


def _doc_payload(doc: Document, **extra: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": doc.id,
        "kb_id": doc.kb_id,
        "title": doc.title,
        "mime": doc.mime,
        "sha256": doc.sha256,
        "source_uri": doc.source_uri,
        "parse_status": doc.parse_status,
        "index_version": doc.index_version,
        "failure": doc.failure,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }
    data.update(extra)
    return data


@router.post("/kbs/{kb_id}/documents", status_code=202)
async def upload_document(
    kb_id: str,
    file: UploadFile,
    request: Request,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """上传即入库(FR-101):sha256 由服务端对字节流现算,对象键按 `artifact_key` 落位。"""
    await DocumentRepo(ctx, db).require_write(f"kb:{kb_id}")
    limit = max(1, int(request.app.state.graph_deps.settings.max_document_bytes))
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > limit:
                raise HTTPException(status_code=413, detail=f"上传超过上限 {limit} 字节")
        except ValueError:
            pass  # 伪造/损坏的 Content-Length 不当作可信长度,交给下面的分块计数

    # Content-Length 可伪造,真正的上限必须由实际读到的字节数兜底;分块读避免先落一个无界 bytes。
    buf = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(status_code=413, detail=f"上传超过上限 {limit} 字节")
    data = bytes(buf)
    if not data:
        raise IllegalTransition("上传内容为空")
    digest = storage.sha256_of(data)
    existing = await DocumentRepo(ctx, db).get_by_sha(kb_id, digest)
    if existing is not None:
        # 同内容幂等返回(FR-103 的精神):不建新行,也不再投一遍链。
        return _doc_payload(existing, duplicate=True, submitted=False)
    suffix = Path(file.filename or "").suffix or ""
    mime = file.content_type or "application/octet-stream"
    uri = storage.put_bytes(data, key=storage.artifact_key(ctx, kb_id, digest, suffix), mime=mime)
    title = Path(file.filename or "").stem or digest[:12]
    doc = await DocumentRepo(ctx, db).create(
        kb_id, source_uri=uri, sha256=digest, title=title, mime=mime
    )
    submission = ingest_tasks.submit_document(doc.id, doc.index_version)
    return _doc_payload(doc, duplicate=False, submission=submission)


@router.post("/documents/{doc_id}/reindex", status_code=202)
async def reindex_document(
    doc_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """重索引(FR-107):`index_version` +1 换一套任务 id,阶段从头跑。"""
    doc = await DocumentRepo(ctx, db).restart(doc_id)
    submission = ingest_tasks.submit_document(doc.id, doc.index_version)
    return _doc_payload(doc, submission=submission)


@router.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """轮询入库进度(FR-106):前端拿 `parse_status` 与 `failure` 渲染状态。"""
    return _doc_payload(await DocumentRepo(ctx, db).require(doc_id))


@router.get("/kbs/{kb_id}/documents")
async def list_documents(
    kb_id: str,
    status: ParseStage | None = None,
    ctx: TenantContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, object]]:
    rows = await DocumentRepo(ctx, db).list_for_kb(kb_id)
    if status is not None:
        rows = [row for row in rows if row.parse_status == status.value]
    return [_doc_payload(row) for row in rows]
