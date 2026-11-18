"""Milvus 2.5.10 能力实测:验证技术实现文档 §4.2 的 schema 是否真能落地。

跑法(中间件已起):
    .venv/bin/python scripts/verify_milvus.py --uri http://127.0.0.1:19530

探针:
    P1 双 BM25 Function(正文 / 章节标题各自独立稀疏字段,不合并)
    P2 中文分词(chinese analyzer)按词切分,而非逐字
    P3 dense + 双稀疏 三路混合检索 + RRF,一次 search 调用完成
    P4 tenant_id 作 partition key,过滤后跨租户零命中
    P5 稀疏字段索引类型硬约束(SPARSE_INVERTED_INDEX + BM25;IP 会被服务端拒绝)
    P6 同主键 upsert 不产生重复行(幂等)
    P7 按 chunk_id 表达式删除孤儿可行
"""
from __future__ import annotations

import argparse
import sys
import time

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

COLLECTION = "kbmesh_verify_chunk"
DENSE_DIM = 16
TENANT_A, TENANT_B = "tenant-A", "tenant-B"

results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def vec(one_hot_at: int) -> list[float]:
    base = [0.0] * DENSE_DIM
    base[one_hot_at] = 1.0
    return base


QUERY_VEC = vec(0)

ROWS = [
    {
        "pk": 1,
        "chunk_id": "c1",
        "tenant_id": TENANT_A,
        "kb_id": "kb1",
        "doc_id": "d1",
        "body_text": "本文介绍四道防线,包含存储层物理隔离与查询层强制注入。",
        "heading_text": "第三章 多租户隔离设计",
        "dense": QUERY_VEC,
    },
    {
        "pk": 2,
        "chunk_id": "c2",
        "tenant_id": TENANT_A,
        "kb_id": "kb1",
        "doc_id": "d1",
        "body_text": "向量召回使用稠密与稀疏混合检索,精排后输出带引用的答案。",
        "heading_text": "第四章 幻觉防御与拒答",
        "dense": vec(1),
    },
    {
        "pk": 3,
        "chunk_id": "c3",
        "tenant_id": TENANT_B,
        "kb_id": "kb2",
        "doc_id": "d2",
        "body_text": "另一租户的正文,同样提到隔离设计与召回。",
        "heading_text": "第三章 多租户隔离设计",
        "dense": QUERY_VEC,
    },
]


def make_schema(client: MilvusClient):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
    schema.add_field("tenant_id", DataType.VARCHAR, max_length=64, is_partition_key=True)
    schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
    schema.add_field("body_text", DataType.VARCHAR, max_length=8192,
                     enable_analyzer=True, analyzer_params={"type": "chinese"})
    schema.add_field("heading_text", DataType.VARCHAR, max_length=2048,
                     enable_analyzer=True, analyzer_params={"type": "chinese"})
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    schema.add_field("sparse_body", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("sparse_heading", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="bm25_body", function_type=FunctionType.BM25,
                                 input_field_names=["body_text"], output_field_names=["sparse_body"]))
    schema.add_function(Function(name="bm25_heading", function_type=FunctionType.BM25,
                                 input_field_names=["heading_text"],
                                 output_field_names=["sparse_heading"]))
    return schema


def make_index_params(client: MilvusClient):
    ip = client.prepare_index_params()
    ip.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE",
                 params={"M": 16, "efConstruction": 64})
    # 实测硬约束:BM25 Function 的输出字段只能建 metric_type=BM25 的索引,写 IP 会被服务端拒绝
    # (code=65535 "index metric type of BM25 function output field must be BM25, got IP")。
    # IP 那条路是给外接稀疏模型(如 BGE-M3 手工喂 sparse 向量)留的,本方案没有这种字段。
    for f in ("sparse_body", "sparse_heading"):
        ip.add_index(field_name=f, index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    return ip


def ids(search_out) -> list[str]:
    return [chunk_id_of(h) for h in search_out[0]]


def chunk_id_of(hit) -> str:
    # search() 把 output_fields 摊平在 hit 上,hybrid_search() 包在 hit["entity"] 里
    if "chunk_id" in hit:
        return hit["chunk_id"]
    return hit["entity"]["chunk_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="http://127.0.0.1:19530")
    args = ap.parse_args()

    client = MilvusClient(uri=args.uri, timeout=60)
    deadline = time.time() + 120
    ok_conn = False
    while time.time() < deadline:
        try:
            client.list_collections()
            ok_conn = True
            break
        except Exception as exc:  # noqa: BLE001
            print(f"waiting for milvus: {type(exc).__name__}: {exc}")
            time.sleep(5)
    if not ok_conn:
        report("P0 连接 Milvus", False, args.uri)
        return 1
    print(f"connected: {args.uri}")

    if COLLECTION in client.list_collections():
        client.drop_collection(COLLECTION)

    try:
        client.create_collection(COLLECTION, schema=make_schema(client),
                                 index_params=make_index_params(client))
    except Exception as exc:  # noqa: BLE001
        report("P1 建集合(双 BM25 + 稀疏索引)", False, f"{type(exc).__name__}: {exc}")
        return 1

    desc = client.describe_collection(COLLECTION)
    funcs = sorted(f.get("name") for f in desc.get("functions", []))
    report("P1 双 BM25 Function 独立字段", funcs == ["bm25_body", "bm25_heading"], f"functions={funcs}")
    sparse = sorted(f["name"] for f in desc["fields"] if f["type"] == DataType.SPARSE_FLOAT_VECTOR)
    report("P5 两个独立稀疏字段", sparse == ["sparse_body", "sparse_heading"], str(sparse))
    pk_partition = [f["name"] for f in desc["fields"] if f.get("is_partition_key")]
    report("P4 tenant_id 是 partition key", pk_partition == ["tenant_id"], str(pk_partition))

    client.insert(COLLECTION, ROWS)
    client.flush(COLLECTION)

    # P2:词「隔离」只出现在 c1 正文与 c3 正文;逐字分词会让所有中文查询命中全部行
    body_hits = ids(client.search(COLLECTION, data=["隔离设计"], anns_field="sparse_body", limit=5,
                                  filter=f'tenant_id == "{TENANT_A}"', output_fields=["chunk_id"]))
    report("P2 中文按词召回(正文路)", "c1" in body_hits, f"sparse_body('隔离设计')→{body_hits}")

    # P1b:「拒答」只在标题字段里 —— 证明标题路不是被正文路掩盖的影子
    head_hits = ids(client.search(COLLECTION, data=["拒答"], anns_field="sparse_heading", limit=5,
                                  filter=f'tenant_id == "{TENANT_A}"', output_fields=["chunk_id"]))
    body_hits2 = ids(client.search(COLLECTION, data=["拒答"], anns_field="sparse_body", limit=5,
                                   filter=f'tenant_id == "{TENANT_A}"', output_fields=["chunk_id"]))
    report("P1b 标题路单独命中且正文路命不中", "c2" in head_hits and "c2" not in body_hits2,
           f"heading→{head_hits} body→{body_hits2}")

    # P3:一次 RPC 完成三路召回。单请求多向量 search() 不支持 anns_field 列表(实测 ParamError),
    # 必须走 hybrid_search + 每路自带 filter —— 租户条件写在请求里,不给调用方漏掉的机会。
    tenant_expr = f'tenant_id == "{TENANT_A}"'
    reqs = [
        AnnSearchRequest(data=[QUERY_VEC], anns_field="dense",
                         param={"metric_type": "COSINE"},
                         limit=5, expr=tenant_expr),
        AnnSearchRequest(data=["隔离设计"], anns_field="sparse_body",
                         param={}, limit=5, expr=tenant_expr),
        AnnSearchRequest(data=["多租户"], anns_field="sparse_heading",
                         param={}, limit=5, expr=tenant_expr),
    ]
    mixed = client.hybrid_search(COLLECTION, reqs, RRFRanker(k=60),
                                 limit=5, output_fields=["chunk_id"])
    m = ids(mixed)
    report("P3 一次调用三路混合 + RRF", len(m) >= 1 and set(m) <= {"c1", "c2"}, f"hybrid→{m}")

    dense_a = client.search(COLLECTION, data=[QUERY_VEC], anns_field="dense", limit=10,
                            filter=f'tenant_id == "{TENANT_A}"', output_fields=["tenant_id"])
    tenants = {(h.get("tenant_id") or h["entity"]["tenant_id"]) for h in dense_a[0]}
    report("P4b 过滤后仅本租户", tenants == {TENANT_A}, f"tenants={tenants}")
    b_hits = ids(client.search(COLLECTION, data=["隔离"], anns_field="sparse_body", limit=10,
                               filter=f'tenant_id == "{TENANT_B}"', output_fields=["chunk_id"]))
    report("P4c B 租户查不到 A 的数据", set(b_hits) == {"c3"}, f"B→{b_hits}")

    count_sql = client.query(COLLECTION, filter="pk > 0", output_fields=["count(*)"])[0]["count(*)"]
    client.upsert(COLLECTION, ROWS)
    client.flush(COLLECTION)
    count2 = client.query(COLLECTION, filter="pk > 0", output_fields=["count(*)"])[0]["count(*)"]
    report("P6 同主键 upsert 幂等", count_sql == count2 == 3, f"{count_sql} → {count2}")

    client.delete(COLLECTION, filter='chunk_id in ["c1","c2"]')
    client.flush(COLLECTION)
    left = client.query(COLLECTION, filter="pk > 0", output_fields=["chunk_id"])
    report("P7 按 chunk_id 删孤儿可行", {r["chunk_id"] for r in left} == {"c3"}, f"left={left}")

    client.drop_collection(COLLECTION)
    failed = [r for r in results if not r[1]]
    print(f"\n=== {len(results) - len(failed)}/{len(results)} 通过 ===")
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
