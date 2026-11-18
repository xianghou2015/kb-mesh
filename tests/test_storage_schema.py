"""防线①在两个检索引擎里的落点:分区键/约束真生效,恶意标识符进不了 Cypher。

引擎连不上时 fail,不 skip:标题路配了但分词没生效、租户谓词漏了一路,都只有打真库才看得见。
"""
from __future__ import annotations

import pytest

from kbmesh.core.config import get_settings
from kbmesh.core.tenancy import ROLE_TENANT_ADMIN, TenantContext
from kbmesh.storage import milvus as mv
from kbmesh.storage import neo4j as n4

COLLECTION = get_settings().milvus_collection
DIM = get_settings().embedding_dim
CTX_A = TenantContext(tenant_id="tnt-schema-a", user_id="u", roles=(ROLE_TENANT_ADMIN,))
CTX_B = TenantContext(tenant_id="tnt-schema-b", user_id="u", roles=(ROLE_TENANT_ADMIN,))
COLLECTION = get_settings().milvus_collection
DIM = get_settings().embedding_dim


@pytest.fixture(scope="module")
def milvus_client():
    return mv.get_client()


@pytest.fixture(scope="module")
def neo4j_driver():
    driver = n4.get_driver()
    yield driver
    driver.close()


# ---------------------------------------------------------------- Milvus


def test_ensure_collection_is_idempotent_and_verifiable(milvus_client) -> None:
    name = COLLECTION
    first = mv.ensure_collection(milvus_client, name)
    again = mv.ensure_collection(milvus_client, name)
    assert first == again == name

    info = mv.verify_collection(milvus_client, name)
    assert info["functions"] == ["bm25_body", "bm25_heading"]
    assert int(info["dense_dim"]) == DIM
    assert "tenant_id" in info["fields"]


def test_heading_path_carries_the_chinese_analyzer(milvus_client) -> None:
    """标题路静默失效是本项目最贵的坑:这里直接读回 analyzer 配置。"""
    desc = milvus_client.describe_collection(COLLECTION)
    fields = {f["name"]: f for f in desc["fields"]}
    for field in ("body_text", "heading_path"):
        assert "chinese" in str(fields[field].get("params", {})), field


def test_partition_key_is_tenant(milvus_client) -> None:
    desc = milvus_client.describe_collection(COLLECTION)
    assert [f["name"] for f in desc["fields"] if f.get("is_partition_key")] == ["tenant_id"]


def test_tenant_filter_rejects_injection() -> None:
    for evil in (
        'a" or 1==1 or "b',
        "tnt-a; drop collection x",
        "tnt-a\x00",
        "",
        "a" * 65,
    ):
        with pytest.raises(mv.UnsafeIdentifier):
            mv.tenant_filter(TenantContext(tenant_id=evil, user_id="u"))


def test_combined_filter_adds_kb_scope() -> None:
    ctx = TenantContext(
        tenant_id="tnt-a", user_id="u", kb_scope=frozenset({"kb-2", "kb-1"})
    )
    expr = mv.combined_filter(ctx)
    assert expr == 'tenant_id == "tnt-a" and kb_id in ["kb-1", "kb-2"]'
    with pytest.raises(mv.UnsafeIdentifier):
        mv.combined_filter(
            TenantContext(tenant_id="tnt-a", user_id="u", kb_scope=frozenset({'k"b'}))
        )


# ---------------------------------------------------------------- Neo4j


def test_ensure_schema_creates_every_constraint(neo4j_driver) -> None:
    info = n4.ensure_schema(neo4j_driver)
    expected = {name for name, _ in n4.CONSTRAINTS} | {name for name, _ in n4.INDEXES}
    assert expected <= set(info["constraints"] + info["indexes"]), expected - set(
        info["constraints"] + info["indexes"]
    )


def test_ensure_schema_twice_is_harmless(neo4j_driver) -> None:
    assert n4.ensure_schema(neo4j_driver) == n4.ensure_schema(neo4j_driver)


@pytest.mark.parametrize(
    "payload",
    [
        "REL] MATCH (n) DETACH DELETE n; //",
        'MERGE (e:Entity {entity_key: "x"}) //',
        "USES) OR (1=1",
        "*1..10",
        "",
    ],
)
def test_malicious_rel_type_is_rejected_by_the_whitelist(payload: str) -> None:
    """§6.4:把「不可参数化所以只能白名单」变成可执行测试,而不是靠人记住。"""
    with pytest.raises(n4.IllegalGraphIdentifier):
        n4.assert_rel_type(payload)


def test_hop_count_is_whitelisted() -> None:
    assert n4.hops_literal(2) == 2
    for bad in (0, 4, 10, -1, "2"):
        with pytest.raises(n4.IllegalGraphIdentifier):
            n4.hops_literal(bad)  # type: ignore[arg-type]


def test_labels_are_whitelisted() -> None:
    assert n4.assert_label("Entity") == "Entity"
    with pytest.raises(n4.IllegalGraphIdentifier):
        n4.assert_label('Entity) DETACH DELETE n //')


def test_rel_type_set_falls_back_to_full_whitelist() -> None:
    assert set(n4.assert_rel_types(["USES", "PART_OF"])) == {"USES", "PART_OF"}
    assert set(n4.assert_rel_types([])) == set(n4.REL_TYPES)


def test_tenant_predicate_is_per_hop() -> None:
    """只在起点绑租户挡不住漏写属性的边(§4.3 坑 4)—— 谓词必须同时覆盖 nodes 与 rels。"""
    predicate = n4.tenant_predicate(CTX_A)
    assert "all(n IN nodes(p) WHERE n.tenant_id = $tenant)" in predicate
    assert "all(e IN relationships(p) WHERE e.tenant_id = $tenant)" in predicate


def test_cross_tenant_walk_returns_nothing(neo4j_driver) -> None:
    """§4.3 坑 4 的真实复现:一条漏写 tenant_id 的边就能让 2 跳遍历爬进别租户。

    图:甲(A) --带租户--> 乙(A) --漏属性--> 丙(B)。
    只绑起点的写法会返回丙;逐跳谓词留着乙、掐掉丙。
    """
    n4.ensure_schema(neo4j_driver)
    with n4.session(neo4j_driver) as s:
        try:
            s.run(
                """
                CREATE (:Entity {tenant_id: $a, kb_id: 'kb-1', entity_key: 'k:a1', label: '甲'})
                CREATE (:Entity {tenant_id: $a, kb_id: 'kb-1', entity_key: 'k:a2', label: '乙'})
                CREATE (:Entity {tenant_id: $b, kb_id: 'kb-2', entity_key: 'k:b1', label: '丙'})
                """,
                a=CTX_A.tenant_id,
                b=CTX_B.tenant_id,
            ).consume()
            by_key = {
                key: s.run(
                    "MATCH (e:Entity {tenant_id: $t, entity_key: $k}) RETURN e",
                    t=tenant,
                    k=key,
                ).single()["e"]
                for key, tenant in (
                    ("k:a1", CTX_A.tenant_id),
                    ("k:a2", CTX_A.tenant_id),
                    ("k:b1", CTX_B.tenant_id),
                )
            }
            # 合法边:甲→乙,带 tenant_id
            s.run(
                "MATCH (x), (y) WHERE elementId(x) = $x AND elementId(y) = $y "
                "CREATE (x)-[:REL {rtype: 'USES', tenant_id: $t}]->(y)",
                x=by_key["k:a1"].element_id,
                y=by_key["k:a2"].element_id,
                t=CTX_A.tenant_id,
            ).consume()
            # 管道 bug:乙→丙 漏写 tenant_id
            s.run(
                "MATCH (x), (y) WHERE elementId(x) = $x AND elementId(y) = $y "
                "CREATE (x)-[:REL {rtype: 'USES'}]->(y)",
                x=by_key["k:a2"].element_id,
                y=by_key["k:b1"].element_id,
            ).consume()

            head = (
                "MATCH p = (start {tenant_id: $tenant})-[:REL*1..2]-(other) "
                "WHERE start.entity_key = 'k:a1' "
            )
            loose = s.run(
                head + "RETURN collect(DISTINCT other.entity_key) AS keys",
                tenant=CTX_A.tenant_id,
            ).single()["keys"]
            assert "k:b1" in loose, "前提不成立:漏属性的边没把两个租户串起来"

            guarded = s.run(
                head
                + "AND "
                + n4.tenant_predicate(CTX_A)
                + " RETURN collect(DISTINCT other.entity_key) AS keys",
                tenant=CTX_A.tenant_id,
            ).single()["keys"]
            assert "k:b1" not in guarded, f"逐跳谓词没拦住跨租户: {guarded}"
            assert "k:a2" in guarded, f"合法的同租户跳不该被一起拦掉: {guarded}"
        finally:
            s.run(
                "MATCH (e:Entity) WHERE e.tenant_id IN $ts DETACH DELETE e",
                ts=[CTX_A.tenant_id, CTX_B.tenant_id],
            ).consume()
            left = s.run(
                "MATCH (e:Entity) WHERE e.tenant_id IN $ts RETURN count(e) AS c",
                ts=[CTX_A.tenant_id, CTX_B.tenant_id],
            ).single()["c"]
            assert left == 0
