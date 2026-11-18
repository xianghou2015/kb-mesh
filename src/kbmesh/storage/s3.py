"""业务对象存储(MinIO,path-style)。

原始文件与整页转图都在这里,PG 只存 `source_uri` / `image_refs`:切片正文是从这些产物算出来的,
产物不可变 ⇒ 重索引才能拿到同一批切片(FR-102/103)。

path-style 是硬约定:MinIO 的 virtual-host 域名在这台机器上没有解析,`KBMESH_OSS_ADDRESSING`
留了开关但默认 path。
"""
from __future__ import annotations

import hashlib
import io

import boto3
from botocore.config import Config as BotoConfig

from kbmesh.core.config import get_settings
from kbmesh.core.tenancy import TenantContext

ARTIFACT_PREFIX = "artifacts"


def s3_client():
    settings = get_settings()
    if not settings.oss_endpoint:
        raise RuntimeError("KBMESH_OSS_ENDPOINT 未配置")
    return boto3.client(
        "s3",
        endpoint_url=settings.oss_endpoint,
        aws_access_key_id=settings.oss_access_key_id,
        aws_secret_access_key=settings.oss_secret_access_key,
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": settings.oss_addressing},
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def ensure_bucket(client=None) -> str:
    settings = get_settings()
    client = client or s3_client()
    try:
        client.head_bucket(Bucket=settings.oss_bucket)
    except Exception:  # noqa: BLE001  (404/403 都按「不存在」处理,由 create 决定成败)
        client.create_bucket(Bucket=settings.oss_bucket)
    return settings.oss_bucket


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_key(ctx: TenantContext, kb_id: str, sha256: str, suffix: str) -> str:
    """键里带租户与内容哈希:同内容同租户天然去重,跨租户物理分开。

    形参是 `ctx` 而不是 `tenant_id` —— §6.1 的红线对对象存储同样成立:能拿到上下文的调用点
    只有验签后的那一个,前缀里就不可能混进自报的租户。
    """
    return f"{ARTIFACT_PREFIX}/{ctx.tenant_id}/{kb_id}/{sha256}{suffix}"


def put_bytes(data: bytes, *, key: str, mime: str = "application/octet-stream", client=None) -> str:
    client = client or s3_client()
    client.put_object(
        Bucket=get_settings().oss_bucket,
        Key=key,
        Body=io.BytesIO(data),
        ContentType=mime,
    )
    return f"s3://{get_settings().oss_bucket}/{key}"


def get_bytes(uri: str, client=None) -> bytes:
    client = client or s3_client()
    buf = io.BytesIO()
    client.download_fileobj(_bucket(uri), _key(uri), buf)
    return buf.getvalue()


def presign_get(uri: str, expires_in: int = 600, client=None) -> str:
    """答案引用回传原图 URL(FR-104):给短期签名地址,不给公开读。"""
    client = client or s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(uri), "Key": _key(uri)},
        ExpiresIn=expires_in,
    )


def page_image_key(ctx: TenantContext, doc_id: str, page_no: int) -> str:
    return f"{ARTIFACT_PREFIX}/{ctx.tenant_id}/pages/{doc_id}/p{page_no:04d}.png"


def _bucket(uri: str) -> str:
    return uri.removeprefix("s3://").split("/", 1)[0]


def _key(uri: str) -> str:
    tail = uri.removeprefix("s3://").split("/", 1)
    return tail[1] if len(tail) > 1 else ""
