"""Milvus collection:双 BM25 + dense 三路,`tenant_id` 作 partition key。

字段与索引全部按 scripts/verify_milvus.py 的实测结果写,三条硬约束别改:

1. BM25 Function 的输出字段只能建 `metric_type=BM25` 的稀疏索引,写 IP 会被服务端拒绝;
2. 一次 RPC 拿三路结果必须走 `hybrid_search`(多向量 `search()` 不接受 anns_field 列表);
3. `search()` 把 output_fields 摊平在 hit 上,`hybrid_search()` 包在 `hit["entity"]` 里 —— 读回
   chunk_id 的两种形态都由 `hit_chunk_id()` 收口。

防线① 在这一层的样子是:租户条件写进每个 AnnSearchRequest 的 `expr`,而不是等调用方记得加。
"""
from __future__ import annotations

import re

from pymilvus import DataType, Function, FunctionType, MilvusClient

from kbmesh.core.config import get_settings
from kbmesh.core.tenancy import TenantContext

#: 表达式里出现的标识符只允许这种形态:`tnt-<hex>` / `kb-<hex>` / `chk-<hex>`,
#: 由服务端签发。带引号、反斜杠、空格的值一律拒 —— 表达式是字符串,不能靠调用方自觉。
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

BODY_TEXT_MAX = 8192
HEADING_TEXT_MAX = 2048


class UnsafeIdentifier(ValueError):
    """试图把可疑值拼进 Milvus 表达式。"""


def assert_safe(value: str) -> str:
    if not _SAFE_ID.match(value or ""):
        raise UnsafeIdentifier(f"非法标识符,拒绝拼进表达式: {value!r}")
    return value


def tenant_filter(ctx: TenantContext) -> str:
    """防线③ 的存储侧版本:每个检索请求都必须带上它,缺了就是跨租户。"""
    return f'tenant_id == "{assert_safe(ctx.tenant_id)}"'


def scope_filter(ctx: TenantContext) -> str | None:
    if not ctx.kb_scope:
        return None
    ids = ", ".join(f'"{assert_safe(kb)}"' for kb in sorted(ctx.kb_scope))
    return f"kb_id in [{ids}]"


def combined_filter(ctx: TenantContext) -> str:
    """租户 + 令牌 kb_scope;两条都在这,调用方没有「忘了加」的余地。"""
    scope = scope_filter(ctx)
    return tenant_filter(ctx) if scope is None else f"{tenant_filter(ctx)} and {scope}"


def collection_fields(dim: int) -> list[dict]:
    return [
        {"field": "pk", "type": DataType.INT64, "primary": True},
        {"field": "chunk_id", "type": DataType.VARCHAR, "max_len": 128},
        # partition key:同一租户的数据物理聚在一个分区,越权查询扫不到别人的分区
        {
            "field": "tenant_id",
            "type": DataType.VARCHAR,
            "max_len": 64,
            "partition_key": True,
        },
        {"field": "kb_id", "type": DataType.VARCHAR, "max_len": 64},
        {"field": "doc_id", "type": DataType.VARCHAR, "max_len": 64},
        {"field": "page_no", "type": DataType.INT64},
        {
            "field": "heading_path",
            "type": DataType.VARCHAR,
            "max_len": HEADING_TEXT_MAX,
            "analyzer": True,
        },
        {
            "field": "body_text",
            "type": DataType.VARCHAR,
            "max_len": BODY_TEXT_MAX,
            "analyzer": True,
        },
        {"field": "dense", "type": DataType.FLOAT_VECTOR, "dim": dim},
        {"field": "sparse_body", "type": DataType.SPARSE_FLOAT_VECTOR},
        {"field": "sparse_heading", "type": DataType.SPARSE_FLOAT_VECTOR},
    ]


def build_schema(client: MilvusClient, dim: int):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    for spec in collection_fields(dim):
        kwargs: dict = {}
        if spec.get("primary"):
            kwargs["is_primary"] = True
        if spec.get("max_len"):
            kwargs["max_length"] = spec["max_len"]
        if spec.get("dim"):
            kwargs["dim"] = spec["dim"]
        if spec.get("partition_key"):
            kwargs["is_partition_key"] = True
        if spec.get("analyzer"):
            # 实测:chinese analyzer 按词切分,逐字分词会让任何中文查询命中全部行
            kwargs["enable_analyzer"] = True
            kwargs["analyzer_params"] = {"type": "chinese"}
        schema.add_field(spec["field"], spec["type"], **kwargs)

    # 两路 BM25 必须是两个独立 Function:合并成一个就把章节检索并进了正文检索,
    # 标题路是否真生效再也没法单独验证(FR-205)。
    schema.add_function(
        Function(
            name="bm25_body",
            function_type=FunctionType.BM25,
            input_field_names=["body_text"],
            output_field_names=["sparse_body"],
        )
    )
    schema.add_function(
        Function(
            name="bm25_heading",
            function_type=FunctionType.BM25,
            input_field_names=["heading_path"],
            output_field_names=["sparse_heading"],
        )
    )
    return schema


def build_index_params(client: MilvusClient):
    ip = client.prepare_index_params()
    ip.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 64},
    )
    for field in ("sparse_body", "sparse_heading"):
        ip.add_index(field_name=field, index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    return ip


def get_client() -> MilvusClient:
    settings = get_settings()
    return MilvusClient(
        uri=settings.milvus_uri,
        token=settings.milvus_token or None,
        db_name=settings.milvus_db,
        timeout=60,
    )


def collection_ready(client: MilvusClient, name: str) -> bool:
    return name in client.list_collections()


def ensure_collection(client: MilvusClient | None = None, name: str | None = None) -> str:
    """建集合(幂等):上线顺序要求先建索引再放 worker 跑存量文档(§12)。"""
    settings = get_settings()
    client = client or get_client()
    name = name or settings.milvus_collection

    if collection_ready(client, name):
        verify_collection(client, name)
        return name

    client.create_collection(
        name,
        schema=build_schema(client, settings.embedding_dim),
        index_params=build_index_params(client),
    )
    verify_collection(client, name)
    return name


def drop_collection(client: MilvusClient, name: str) -> None:
    if collection_ready(client, name):
        client.drop_collection(name)


def verify_collection(client: MilvusClient, name: str) -> dict:
    """把「配置了但没生效」变成可执行断言:双 Function、partition key、analyzer 都在位。"""
    desc = client.describe_collection(name)
    fields = {f["name"]: f for f in desc["fields"]}
    functions = sorted(f.get("name", "") for f in desc.get("functions", []))

    problems: list[str] = []
    if functions != ["bm25_body", "bm25_heading"]:
        problems.append(f"BM25 Function 不对: {functions}")
    if [f for f, spec in fields.items() if spec.get("is_partition_key")] != ["tenant_id"]:
        problems.append("tenant_id 不是 partition key")
    for field in ("body_text", "heading_path"):
        params = fields.get(field, {}).get("params", {})
        if "chinese" not in str(params):
            problems.append(f"{field} 的中文分词未生效: {params}")

    if problems:
        raise RuntimeError("Milvus 集合不满足检索前提: " + "; ".join(problems))
    return {
        "collection": name,
        "fields": sorted(fields),
        "functions": functions,
        "dense_dim": fields["dense"].get("params", {}).get("dim"),
    }


def hit_chunk_id(hit) -> str:
    """两种检索形态一起收:search() 摊平,hybrid_search() 包在 entity 里。"""
    if "chunk_id" in hit:
        return hit["chunk_id"]
    return hit["entity"]["chunk_id"]


# --------------------------------------------------------------------- 写入侧(§5.3 幂等第二层)

#: 检索回表要的字段;`body_text` 带上是为了精排能拿到原文,不再回 PG 查一次
OUTPUT_FIELDS = ("chunk_id", "doc_id", "kb_id", "page_no", "heading_path", "body_text", "tenant_id")


def truncate_utf8(text: str, limit: int) -> str:
    """Milvus VARCHAR 的 max_length 按**字节**算,中文一字 3 字节:按字符截会在多字节边界上被拒。"""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    cut = encoded[:limit]
    while cut and (cut[-1] & 0xC0) == 0x80:  # 丢掉被劈开的半个多字节字符的前导续字节
        cut = cut[:-1]
    if cut and (cut[-1] & 0x80):  # 首字节指示的宽度没写满,这个字符也是残的
        cut = cut[:-1]
    return cut.decode("utf-8") + "…"


def to_row(
    ctx: TenantContext,
    *,
    pk: int,
    chunk_id: str,
    doc_id: str,
    kb_id: str,
    page_no: int | None,
    heading_path: str,
    body_text: str,
    dense: list[float],
) -> dict:
    """组装一行。稀疏字段是 BM25 Function 的输出,**不能**由客户端写入(实测会被拒)。"""
    settings = get_settings()
    if len(dense) != settings.embedding_dim:
        raise ValueError(f"向量维度 {len(dense)} 与集合 {settings.embedding_dim} 不符,写入会被拒")
    for value in (chunk_id, doc_id, kb_id):
        assert_safe(value)
    return {
        "pk": int(pk),
        "chunk_id": chunk_id,
        "tenant_id": ctx.tenant_id,
        "kb_id": kb_id,
        "doc_id": doc_id,
        "page_no": int(page_no or 0),
        "heading_path": truncate_utf8(heading_path, HEADING_TEXT_MAX),
        "body_text": truncate_utf8(body_text, BODY_TEXT_MAX),
        "dense": list(dense),
    }


def upsert_rows(rows: list[dict], *, client: MilvusClient | None = None, name: str | None = None) -> int:
    """同 pk 覆盖而非追加(实测 upsert 后 count 不变):重跑不产生重复向量行。"""
    if not rows:
        return 0
    settings = get_settings()
    client = client or get_client()
    client.upsert(name or settings.milvus_collection, rows)
    return len(rows)


def pks_for_doc(ctx: TenantContext, doc_id: str, *, client=None, name: str | None = None) -> set[int]:
    settings = get_settings()
    client = client or get_client()
    expr = f'{combined_filter(ctx)} and doc_id == "{assert_safe(doc_id)}"'
    rows = client.query(
        name or settings.milvus_collection,
        filter=expr,
        output_fields=["pk"],
        limit=16384,
        # 默认 Bounded 有 ~2s 的可见性延迟:embed 刚 upsert 完就来这里对账,读到的会是旧集合,
        # 孤儿判定随之漏删(或把刚写进去的行当孤儿)。这趟读的结果决定删什么,必须读到自己写的。
        consistency_level="Strong",
    )
    return {int(r["pk"]) for r in rows}


def delete_orphan_pks(pks: set[int], *, client=None, name: str | None = None) -> int:
    """收尾删除「本批次不在集合中的孤儿 pk」:切片策略变了,旧向量不能继续被召回。"""
    if not pks:
        return 0
    settings = get_settings()
    client = client or get_client()
    marks = ", ".join(str(int(p)) for p in sorted(pks))
    client.delete(name or settings.milvus_collection, filter=f"pk in [{marks}]")
    return len(pks)
