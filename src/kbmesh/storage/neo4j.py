"""Neo4j schema + 注入防护白名单:图谱路的存储侧入口。

节点/关系形状照 §4.3,三条实测结论固化在这里:

- 关系类型与跳数**不能参数化**(解析期就 SyntaxError),所以遍历一律无类型 `-[r:REL]-` 再用
  `r.rtype IN $types` 收口,跳数从 `{1,2,3}` 白名单渲染成字面量;
- 只在起点绑租户会串数据:漏写 `tenant_id` 的一条边就能让遍历爬进别租户,所以谓词必须逐跳生效;
- 社区版全文索引不认识租户,`db.index.fulltext.queryNodes` 的命中集必须在服务端再收口;
- 社区版**不能**用 `REQUIRE ... IS NOT NULL`(实测 `ConstraintCreationFailed: requires Enterprise`),
  所以「节点必须带 tenant_id」这条只能在写入层拦(§5.1 `write_graph` 落图前校验),图上只留
  唯一约束 + tenant_id 普通索引。漏写属性的边靠遍历谓词逐跳过滤兜住(坑 4),不靠约束。

`entity` 的唯一键与 PG 的 `uq_entity_tenant_key` 对齐为 `(tenant_id, entity_key)` —— 加上 `kb_id`
会变成「PG 一行、图里两节点」,同一个实体在两个库里长分叉。
"""
from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

from neo4j import GraphDatabase

from kbmesh.core.config import get_settings
from kbmesh.core.tenancy import TenantContext

#: 图上承载业务关系;真实类型只作为属性 `rtype` 存在,值必须来自这张表。
REL_TYPES: frozenset[str] = frozenset(
    {
        "USES",
        "DEPENDS_ON",
        "PART_OF",
        "PRODUCES",
        "EVALUATED_BY",
        "IMPROVES",
        "LIMITED_BY",
        "RELATED_TO",
        "MENTIONED_IN",
        "HAS_CHUNK",
    }
)

#: 桥接边由管道自己维护(实体↔切片、文档↔切片),不该由模型产出,也不该出现在遍历的业务类型集合里。
BRIDGE_REL_TYPES: frozenset[str] = frozenset({"MENTIONED_IN", "HAS_CHUNK"})

#: 可遍历、可被抽取的业务关系 —— 与抽取层用的是同一张表,别在两个模块各写一份。
BUSINESS_REL_TYPES: frozenset[str] = REL_TYPES - BRIDGE_REL_TYPES

#: 变长路径不可参数化 → 跳数只能是这几个字面量(与 Settings 的校验同源)。
ALLOWED_HOPS: tuple[int, ...] = (1, 2, 3)

NODE_LABELS: frozenset[str] = frozenset({"Entity", "ChunkRef", "Document"})

CONSTRAINTS: tuple[tuple[str, str], ...] = (
    (
        "entity_tk_unique",
        "CREATE CONSTRAINT entity_tk_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE (e.tenant_id, e.entity_key) IS UNIQUE",
    ),
    (
        "chunkref_tk_unique",
        "CREATE CONSTRAINT chunkref_tk_unique IF NOT EXISTS "
        "FOR (c:ChunkRef) REQUIRE (c.tenant_id, c.chunk_id) IS UNIQUE",
    ),
    (
        "document_tk_unique",
        "CREATE CONSTRAINT document_tk_unique IF NOT EXISTS "
        "FOR (d:Document) REQUIRE (d.tenant_id, d.doc_id) IS UNIQUE",
    ),
)

#: REL 是唯一的业务关系载体,rtype 必须能查(否则按类型过滤要全图扫)。
INDEXES: tuple[tuple[str, str], ...] = (
    ("rel_rtype", "CREATE INDEX rel_rtype IF NOT EXISTS FOR ()-[r:REL]-() ON (r.rtype)"),
    (
        "entity_tenant",
        "CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id)",
    ),
    (
        "entity_label_ft",
        "CREATE FULLTEXT INDEX entity_label_ft IF NOT EXISTS "
        "FOR (e:Entity) ON EACH [e.label, e.aliases]",
    ),
)


class IllegalGraphIdentifier(ValueError):
    """关系类型 / label 不在白名单里 —— 拼进 Cypher 之前必须被这一步拦下。"""


def assert_rel_type(rtype: str) -> str:
    if rtype not in REL_TYPES:
        raise IllegalGraphIdentifier(f"关系类型不在白名单: {rtype!r}")
    return rtype


def assert_rel_types(rtypes: Iterable[str]) -> tuple[str, ...]:
    out = tuple(assert_rel_type(r) for r in rtypes)
    return out or tuple(sorted(REL_TYPES))


def assert_label(label: str) -> str:
    if label not in NODE_LABELS:
        raise IllegalGraphIdentifier(f"节点 label 不在白名单: {label!r}")
    return label


def hops_literal(value: int) -> int:
    """跳数进 Cypher 前唯一的过门:白名单外直接拒,绝不做字符串插值。"""
    if value not in ALLOWED_HOPS:
        raise IllegalGraphIdentifier(f"跳数必须落在 {ALLOWED_HOPS}: {value!r}")
    return value


def get_driver():
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        database=settings.neo4j_database,
    )


@contextmanager
def session(driver, *, tenant: TenantContext | None = None):
    """社区版单库:租户隔离靠属性,不靠 multi-database(企业版特性)。"""
    with driver.session(database=get_settings().neo4j_database) as s:
        yield s


def ensure_schema(driver=None) -> dict:
    """建约束/索引(幂等),并把「配置是否真生效」读回来 —— 与 Milvus 的 verify 同一姿态。"""
    own = driver is None
    driver = driver or get_driver()
    try:
        with session(driver) as s:
            for _, statement in CONSTRAINTS + INDEXES:
                s.run(statement).consume()
            constraints = {r["name"] for r in s.run("SHOW CONSTRAINTS YIELD name")}
            indexes = {r["name"] for r in s.run("SHOW INDEXES YIELD name")}
    finally:
        if own:
            driver.close()

    expected = {name for name, _ in CONSTRAINTS} | {name for name, _ in INDEXES}
    missing = expected - (constraints | indexes)
    if missing:
        raise RuntimeError(f"Neo4j 约束/索引未落地: {sorted(missing)}")
    return {"constraints": sorted(constraints), "indexes": sorted(indexes)}


def drop_schema(driver=None) -> None:
    own = driver is None
    driver = driver or get_driver()
    try:
        with session(driver) as s:
            for name, _ in CONSTRAINTS:
                s.run(f"DROP CONSTRAINT {name} IF EXISTS").consume()
            for name, _ in INDEXES:
                s.run(f"DROP INDEX {name} IF EXISTS").consume()
    finally:
        if own:
            driver.close()


def tenant_predicate(ctx: TenantContext) -> str:
    """逐跳租户过滤:写进每条遍历,而不是只绑起点。

    边上的 `tenant_id IS NOT NULL` 是刻意加的 —— 管道漏写属性时宁可让那条边查不到,
    也不能让遍历爬进别的租户(§4.3 坑 4)。
    """
    return (
        "all(n IN nodes(p) WHERE n.tenant_id = $tenant) "
        "AND all(e IN relationships(p) WHERE e.tenant_id = $tenant)"
    )


# --------------------------------------------------------------------- 写入侧(§5.3 幂等第三层)

_MERGE_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {tenant_id: $tenant, entity_key: row.entity_key})
ON CREATE SET e.etype = row.etype, e.label = row.label, e.aliases = row.aliases, e.kb_id = $kb
ON MATCH SET e.aliases = row.aliases
RETURN count(e) AS n
"""

#: 关系类型走 `[:REL]` 承载 + `rtype` 属性过滤:Cypher 里 `[r:$RT]` 会在解析期被拒(§4.3 坑 2)
_MERGE_RELATIONS = """
UNWIND $rows AS row
MATCH (a:Entity {tenant_id: $tenant, entity_key: row.src_key})
MATCH (b:Entity {tenant_id: $tenant, entity_key: row.dst_key})
MERGE (a)-[r:REL {rtype: row.rtype}]->(b)
ON CREATE SET r.tenant_id = $tenant, r.kb_id = $kb
SET r.evidence = row.evidence, r.chunk_id = row.chunk_id, r.confidence = row.confidence
RETURN count(r) AS n
"""

#: 桥接边:实体 → 原文切片。图谱路的每条命中都要能落回 chunk(FR-209 / FR-305)
_MERGE_MENTIONS = """
UNWIND $rows AS row
MERGE (c:ChunkRef {tenant_id: $tenant, chunk_id: row.chunk_id})
ON CREATE SET c.doc_id = row.doc_id, c.kb_id = row.kb_id
WITH c, row
MATCH (e:Entity {tenant_id: $tenant, entity_key: row.entity_key})
MERGE (e)-[m:REL {rtype: 'MENTIONED_IN'}]->(c)
ON CREATE SET m.tenant_id = $tenant, m.kb_id = row.kb_id
RETURN count(m) AS n
"""

_MERGE_CHUNKREFS = """
UNWIND $rows AS row
MERGE (c:ChunkRef {tenant_id: $tenant, chunk_id: row.chunk_id})
ON CREATE SET c.doc_id = row.doc_id, c.kb_id = row.kb_id
WITH c, row
MERGE (d:Document {tenant_id: $tenant, doc_id: row.doc_id})
ON CREATE SET d.kb_id = row.kb_id
MERGE (d)-[h:REL {rtype: 'HAS_CHUNK'}]->(c)
ON CREATE SET h.tenant_id = $tenant, h.kb_id = row.kb_id
RETURN count(c) AS n
"""


def _tenant_of(ctx: TenantContext) -> str:
    """防线① 在图上的落点:租户只能来自上下文,并且必须出现在每一条 MERGE 的匹配键里。"""
    if not ctx.tenant_id:
        raise IllegalGraphIdentifier("写入图谱缺少 tenant_id")
    return ctx.tenant_id


def merge_entities(driver, ctx: TenantContext, kb_id: str, rows: list[dict]) -> int:
    """幂等:同一 `entity_key` 三次重放只落一个节点(§4.3 实测)。"""
    if not rows:
        return 0
    with session(driver) as s:
        rec = s.run(_MERGE_ENTITIES, rows=rows, tenant=_tenant_of(ctx), kb=kb_id).single()
    return int(rec["n"]) if rec else 0


def merge_relations(driver, ctx: TenantContext, kb_id: str, rows: list[dict]) -> int:
    """写入层第二道校验(FR-302):这里再核一次白名单,不信任调用方是否跑过抽取层那道。"""
    if not rows:
        return 0
    for row in rows:
        assert_rel_type(row["rtype"])
    with session(driver) as s:
        rec = s.run(_MERGE_RELATIONS, rows=rows, tenant=_tenant_of(ctx), kb=kb_id).single()
    return int(rec["n"]) if rec else 0


def merge_chunkrefs(driver, ctx: TenantContext, kb_id: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    with session(driver) as s:
        rec = s.run(_MERGE_CHUNKREFS, rows=rows, tenant=_tenant_of(ctx), kb=kb_id).single()
    return int(rec["n"]) if rec else 0


_PRUNE_CHUNKREFS = """
MATCH (c:ChunkRef {tenant_id: $tenant, doc_id: $doc})
WHERE NOT c.chunk_id IN $keep
DETACH DELETE c
RETURN count(c) AS n
"""


def prune_chunkrefs(driver, ctx: TenantContext, doc_id: str, keep: Iterable[str]) -> int:
    """重切之后,权威表里已删的切片在图上不能留节点:留着就等于引用能指向一条不存在的切片。"""
    with session(driver) as s:
        rec = s.run(
            _PRUNE_CHUNKREFS, tenant=_tenant_of(ctx), doc=doc_id, keep=sorted(set(keep))
        ).single()
    return int(rec["n"]) if rec else 0


def merge_mentions(driver, ctx: TenantContext, kb_id: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    with session(driver) as s:
        rec = s.run(_MERGE_MENTIONS, rows=rows, tenant=_tenant_of(ctx), kb=kb_id).single()
    return int(rec["n"]) if rec else 0


def delete_tenant_subgraph(driver, ctx: TenantContext) -> int:
    """重索引/清理用:按租户整片删,跨租户删不掉(上下文里没有别人的 tenant_id)。"""
    with session(driver) as s:
        rec = s.run(
            "MATCH (r) WHERE r.tenant_id = $tenant DETACH DELETE r RETURN count(r) AS n",
            tenant=_tenant_of(ctx),
        ).single()
    return int(rec["n"]) if rec else 0


# --------------------------------------------------------------------- 读取侧(§6.3 图谱路五步)

#: 索引名与跳数一样不可参数化,只能取这张表里的字面量。
FT_INDEX = "entity_label_ft"

#: L1 严格匹配:entity_key / label / 别名逐个等值比,命中即高置信
_FIND_ENTITIES = """
MATCH (e:Entity)
WHERE e.tenant_id = $tenant AND ($kbs = [] OR e.kb_id IN $kbs)
  AND (e.entity_key IN $terms OR e.label IN $terms
       OR ANY(t IN $terms WHERE t IN coalesce(e.aliases, [])))
RETURN e.entity_key AS entity_key, e.label AS label, e.etype AS etype,
       e.aliases AS aliases, e.kb_id AS kb_id
"""

#: L2 只做**候选生成**:社区版全文索引默认按单字 AND、命中不认租户(§4.3 坑 5、6),
#: 所以租户与 kb_scope 的收口留在服务端,判定交给应用层做归一化 + 字符相似度。
_CANDIDATES = """
CALL db.index.fulltext.queryNodes($index, $query) YIELD node, score
WHERE node.tenant_id = $tenant AND ($kbs = [] OR node.kb_id IN $kbs)
RETURN node.entity_key AS entity_key, node.label AS label, node.etype AS etype,
       node.aliases AS aliases, node.kb_id AS kb_id, score
ORDER BY score DESC LIMIT $limit
"""

#: 变长路径的跳数渲染成字面量(坑 2),类型用 `rtype` 属性过滤。谓词逐跳生效:
#: 只绑起点挡不住一条漏写属性的边把遍历爬到别家(坑 4),所以每条边、每个节点都要核。
_EXPAND = """
MATCH (a:Entity)
WHERE a.tenant_id = $tenant AND a.entity_key IN $keys AND ($kbs = [] OR a.kb_id IN $kbs)
MATCH p = (a)-[r:REL*1__HOPS__]-(b:Entity)
WHERE all(e IN r WHERE e.tenant_id = $tenant AND e.rtype IN $rtypes
                     AND ($kbs = [] OR e.kb_id IN $kbs))
  AND all(n IN nodes(p) WHERE n.tenant_id = $tenant)
RETURN p LIMIT $limit
"""

#: 实体 → 原文切片的桥接边:图谱路的每条命中都要能落回 chunk(FR-209)
_MENTIONED_CHUNKS = """
UNWIND $keys AS key
MATCH (e:Entity {tenant_id: $tenant, entity_key: key})-[m:REL {rtype: 'MENTIONED_IN'}]->(c:ChunkRef)
WHERE c.tenant_id = $tenant AND ($kbs = [] OR c.kb_id IN $kbs)
RETURN c.chunk_id AS chunk_id, e.entity_key AS entity_key
LIMIT $limit
"""


def _kb_param(ctx: TenantContext) -> list[str]:
    """空 scope = 整租户;非空 = 令牌里指定的那几个 kb。两种都在 Cypher 里显式收口。"""
    return sorted(ctx.kb_scope) if ctx.kb_scope else []


def assert_index_name(name: str) -> str:
    if name not in {n for n, _ in INDEXES}:
        raise IllegalGraphIdentifier(f"索引名不在白名单: {name!r}")
    return name


def fulltext_query(term: str) -> str:
    """把用户给的实体名转成 Lucene 短语查询:剥掉操作符,整串加引号。

    全文查询语法自己也是一层注入面(`*`、`AND`、`TO` 会悄悄改变命中集),所以只留字词字符。
    """
    cleaned = "".join(ch for ch in (term or "") if ch.isalnum() or ch in " _-. ").strip()
    return f'"{cleaned}"' if cleaned else ""


def find_entities(driver, ctx: TenantContext, terms: Iterable[str]) -> list[dict[str, Any]]:
    seeds = sorted({t for t in terms if t})
    if not seeds:
        return []
    with session(driver) as s:
        return [
            dict(rec)
            for rec in s.run(
                _FIND_ENTITIES, tenant=_tenant_of(ctx), terms=seeds, kbs=_kb_param(ctx)
            )
        ]


def candidate_entities(
    driver, ctx: TenantContext, term: str, *, limit: int = 20
) -> list[dict[str, Any]]:
    assert_index_name(FT_INDEX)
    query = fulltext_query(term)
    if not query:
        return []
    with session(driver) as s:
        return [
            dict(rec)
            # 参数走 dict:`query` 是驱动 `run()` 的位置参数名,当 kwargs 传会撞车
            for rec in s.run(
                _CANDIDATES,
                {
                    "index": FT_INDEX,
                    "query": query,
                    "tenant": _tenant_of(ctx),
                    "kbs": _kb_param(ctx),
                    "limit": int(limit),
                },
            )
        ]


def _edge_from(rel, hop: int) -> dict[str, Any]:
    # 驱动 5.x 不暴露 `rel.properties`,关系属性按 Mapping 读
    start, end, props = rel.start_node, rel.end_node, dict(rel)
    return {
        "src_key": start.get("entity_key", ""),
        "src_label": start.get("label", ""),
        "rtype": props.get("rtype", ""),
        "dst_key": end.get("entity_key", ""),
        "dst_label": end.get("label", ""),
        "evidence": props.get("evidence", "") or "",
        "chunk_id": props.get("chunk_id", "") or "",
        "confidence": float(props.get("confidence", 1.0) or 0.0),
        "hop": hop,
    }


def expand_neighborhood(
    driver,
    ctx: TenantContext,
    keys: Iterable[str],
    *,
    hops: int = 2,
    rtypes: Iterable[str] | None = None,
    max_edges: int = 30,
) -> list[dict[str, Any]]:
    """邻域扩展(FR-208 / FR-212):跳数走白名单,遍历出的边按 (src,rtype,dst) 去重后截断。

    同一对实体被多条路径命中时留「跳数最浅、依据最长」的那条:浅的更可信,依据长的更好引。
    """
    seeds = sorted({k for k in keys if k})
    if not seeds:
        return []
    allowed = list(assert_rel_types(rtypes or BUSINESS_REL_TYPES))
    statement = _EXPAND.replace("__HOPS__", f"..{hops_literal(hops)}", 1)
    with session(driver) as s:
        records = s.run(
            statement,
            tenant=_tenant_of(ctx),
            keys=seeds,
            kbs=_kb_param(ctx),
            rtypes=allowed,
            limit=int(max_edges) * 4,
        )
        best: dict[tuple[str, str, str], dict[str, Any]] = {}
        for rec in records:
            for index, rel in enumerate(rec["p"].relationships, start=1):
                edge = _edge_from(rel, index)
                if not edge["src_key"] or edge["src_key"] == edge["dst_key"]:
                    continue
                mark = (edge["src_key"], edge["rtype"], edge["dst_key"])
                current = best.get(mark)
                rank = (edge["hop"], -edge["confidence"], -len(edge["evidence"]))
                if current is None or rank < (
                    current["hop"],
                    -current["confidence"],
                    -len(current["evidence"]),
                ):
                    best[mark] = edge
    ordered = sorted(
        best.values(),
        key=lambda e: (e["hop"], -e["confidence"], -len(e["evidence"]), e["src_key"], e["dst_key"]),
    )
    return ordered[: int(max_edges)]


def mentioned_chunks(
    driver, ctx: TenantContext, keys: Iterable[str], *, limit: int = 200
) -> list[dict[str, Any]]:
    seeds = sorted({k for k in keys if k})
    if not seeds:
        return []
    with session(driver) as s:
        return [
            dict(rec)
            for rec in s.run(
                _MENTIONED_CHUNKS,
                tenant=_tenant_of(ctx),
                keys=seeds,
                kbs=_kb_param(ctx),
                limit=int(limit),
            )
        ]
