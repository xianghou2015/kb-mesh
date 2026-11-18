"""Neo4j 5.26 能力实测:验证技术实现文档 §4.3 的图谱 schema 与遍历语义是否真能落地。

跑法(中间件已起):
    .venv/bin/python scripts/verify_neo4j.py          # 口令自动读 deploy/.env

探针:
    N1 连接与服务端版本
    N2 复合唯一约束 (tenant_id, entity_key) + MERGE 幂等(离线重跑的第三层保证)
    N3 关系类型不可参数化 → 必须在代码里走白名单
    N4 变长跳数不可参数化 → 必须把白名单值渲染成 Cypher 字面量
    N5 无向遍历仍保留每条边的真实方向(子图线性化的前提)
    N6 只在起点绑租户会串数据;逐跳过滤才挡得住漏写属性的边
    N7 实体链接两级:entity_key 精确匹配 + 全文索引候选生成
       (5.26 社区版无自定义分析器,默认按单字 AND,精度判定必须回到应用层)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

TENANT_A, TENANT_B = "verify-A", "verify-B"
LABEL = "KBEntity"

results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def read_password(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = Path(__file__).resolve().parents[1] / "deploy" / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("NEO4J_PASSWORD="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"NEO4J_PASSWORD not found in {env} and --password not given")


def raises_client_error(session, query: str, **params) -> tuple[bool, str]:
    """True when the server rejects the query outright — the reason a whitelist exists.

    The parameter is always *supplied*, so a ClientError can only mean the pattern itself is
    illegal; otherwise "expected parameter $N" would masquerade as a syntax rejection.
    """
    try:
        session.run(query, **params).consume()
    except Neo4jError as exc:
        return exc.code.startswith("Neo.ClientError"), f"{exc.code}: {exc.message}"
    return False, " unexpectedly succeeded"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="bolt://127.0.0.1:7687")
    ap.add_argument("--password", default=None)
    args = ap.parse_args()
    password = read_password(args.password)

    driver = GraphDatabase.driver(args.uri, auth=("neo4j", password))
    with driver.session() as s:
        comp = s.run("CALL dbms.components() YIELD name, versions, edition "
                     "RETURN name, versions[0] AS v, edition").single()
        report("N1 连接 Neo4j", comp["name"].startswith("Neo4j") and comp["v"].startswith("5."),
               f"{args.uri} / {comp['name']} {comp['v']} {comp['edition']}")

    with driver.session() as s:
        s.run(f"CREATE CONSTRAINT kbentity_tenant_key IF NOT EXISTS "
              f"FOR (e:{LABEL}) REQUIRE (e.tenant_id, e.entity_key) IS UNIQUE")
        cons = [r for r in s.run("SHOW CONSTRAINTS YIELD name, properties RETURN name, properties")]
        composite = any(c["name"] == "kbentity_tenant_key" and len(c["properties"]) == 2 for c in cons)
        report("N2a 复合唯一约束 (tenant_id, entity_key)", composite,
               str([c["properties"] for c in cons]))

        # N2b:同 key 重跑 MERGE 只有一行 —— 离线索引管道的第三层幂等就靠这个
        for _ in range(3):
            s.run(f"MERGE (e:{LABEL} {{tenant_id: $t, entity_key: $k}}) SET e.name = $n",
                  t=TENANT_A, k="ent:multi-tenant", n="多租户隔离")
        cnt = s.run(f"MATCH (e:{LABEL} {{tenant_id: $t, entity_key: $k}}) RETURN count(e) AS c",
                    t=TENANT_A, k="ent:multi-tenant").single()["c"]
        report("N2b MERGE 三重放幂等", cnt == 1, f"nodes={cnt}")

        # 约束真的拦得住不同 key 重名、同 key 跨租户的情况吗
        s.run(f"MERGE (e:{LABEL} {{tenant_id: $t, entity_key: $k}})", t=TENANT_B, k="ent:multi-tenant")
        cross = s.run(f"MATCH (e:{LABEL} {{entity_key: $k}}) RETURN count(e) AS c",
                      k="ent:multi-tenant").single()["c"]
        report("N2c 同 entity_key 允许跨租户并存(隔离靠 tenant_id)", cross == 2, f"rows={cross}")

    with driver.session() as s:
        s.run(f"MATCH (a:{LABEL} {{tenant_id: $t}}) DETACH DELETE a", t=TENANT_A)
        s.run(f"MATCH (a:{LABEL} {{tenant_id: $t}}) DETACH DELETE a", t=TENANT_B)
        # 链:A 的 概念 -[MENTIONS]-> A 的 文档 -[HAS_CHUNK]-> A 的 章节;
        #     再从 A 的 章节 拉一条「忘了写 tenant_id」的边到 B 的实体(模拟管道 bug)
        s.run(
            f"""
            CREATE (c1:{LABEL} {{tenant_id: $t, entity_key: 'ent:rlhf', name: '强化学习'}})
            CREATE (c2:{LABEL} {{tenant_id: $t, entity_key: 'ent:reward', name: '奖励模型'}})
            CREATE (d1:{LABEL} {{tenant_id: $t, entity_key: 'doc:101', name: '对齐综述'}})
            CREATE (b1:{LABEL} {{tenant_id: $o, entity_key: 'ent:other', name: '别租户实体'}})
            CREATE (c1)-[:MENTIONS {{tenant_id: $t}}]->(d1)
            CREATE (d1)-[:HAS_CHUNK {{tenant_id: $t}}]->(c2)
            CREATE (c2)-[:MENTIONS]->(b1)
            """,
            t=TENANT_A, o=TENANT_B)

        # N3:关系类型写在方括号里就不能是参数(这正是图谱白名单存在的原因)
        ok3, msg3 = raises_client_error(s, f"MATCH (a:{LABEL} {{tenant_id: $t}})-[r:$RT]->(b) RETURN r",
                                        t=TENANT_A, RT="MENTIONS")
        report("N3 关系类型不可参数化(须白名单渲染)", ok3, msg3[:120])
        # 反面:能跑通的写法 —— 无类型遍历 + type(r) IN $types,类型名从白名单来
        typed = sorted(r["k"] for r in s.run(
            f"MATCH (a:{LABEL} {{tenant_id: $t, entity_key: 'ent:rlhf'}})-[r*1..2]-(x:{LABEL}) "
            "WHERE all(e IN r WHERE e.tenant_id = $t AND type(e) IN $types) "
            "RETURN DISTINCT x.entity_key AS k", t=TENANT_A, types=["MENTIONS", "HAS_CHUNK"]).data())
        report("N3b type(r) IN $types 可参数化", typed == ["doc:101", "ent:reward"], str(typed))

        # N4:跳数同理,`*1..$n` 直接语法错误
        ok4, msg4 = raises_client_error(s, f"MATCH p=(a:{LABEL})-[*1..$N]->(b) RETURN p", N=2)
        report("N4 变长跳数不可参数化(须白名单渲染)", ok4, msg4[:120])
        hops = 2
        rendered = s.run(
            f"MATCH p=(a:{LABEL} {{tenant_id: $t, entity_key: 'ent:rlhf'}})-[*1..{hops}]-(x:{LABEL}) "
            "WHERE all(e IN relationships(p) WHERE e.tenant_id = $t) RETURN count(DISTINCT x) AS c",
            t=TENANT_A).single()["c"]
        report("N4b 白名单值渲染成字面量后正常", rendered == 2, f"reached={rendered}")

    with driver.session() as s:
        # N5:无向遍历拿到的每条边仍带原始端点 → 线性化给 LLM 时方向不会失真
        lin = s.run(
            f"""
            MATCH p=(a:{LABEL} {{tenant_id: $t, entity_key: 'ent:rlhf'}})-[*1..2]-(x:{LABEL})
            WHERE all(e IN relationships(p) WHERE e.tenant_id = $t)
            UNWIND relationships(p) AS e
            RETURN DISTINCT startNode(e).entity_key AS src, endNode(e).entity_key AS dst,
                   type(e) AS rel
            """, t=TENANT_A)
        edges = {(r["src"], r["dst"], r["rel"]) for r in lin}
        forward = ("ent:rlhf", "doc:101", "MENTIONS") in edges
        reversed_ok = ("doc:101", "ent:rlhf", "MENTIONS") not in edges
        report("N5 无向遍历保留边的真实方向", forward and reversed_ok, str(sorted(edges)))

        # N6:只在起点绑租户 → 经由漏写 tenant_id 的边爬进别租户
        naive = s.run(
            f"MATCH (a:{LABEL} {{tenant_id: $t, entity_key: 'ent:rlhf'}})-[*1..3]-(x:{LABEL}) "
            "RETURN DISTINCT x.tenant_id AS tenant", t=TENANT_A)
        naive_tenants = {r["tenant"] for r in naive}
        guarded = s.run(
            f"MATCH p=(a:{LABEL} {{tenant_id: $t, entity_key: 'ent:rlhf'}})-[*1..3]-(x:{LABEL}) "
            "WHERE all(e IN relationships(p) WHERE e.tenant_id = $t) "
            "RETURN DISTINCT x.tenant_id AS tenant", t=TENANT_A)
        guarded_tenants = {r["tenant"] for r in guarded}
        report("N6a 只绑起点确实会串租户", TENANT_B in naive_tenants, str(sorted(naive_tenants)))
        report("N6b 逐跳过滤后跨租户归零", guarded_tenants == {TENANT_A}, str(sorted(guarded_tenants)))

    with driver.session() as s:
        # N7:实体链接第一级靠精确 key,第二级靠全文索引做候选生成
        # 实测硬约束:5.26 社区版没有任何可选分析器(键 index.search.analyzer / analyzer /
        # index.analyzer 一律 "not recognized as an index setting"),自定义分词是企业版特性。
        # 默认分析器对中文按「单字 + AND」匹配,所以全文只能当候选生成器,判定必须回到应用层
        # 的归一化 + 编辑距离(技术文档 §6.3 第②级)。
        s.run("CREATE FULLTEXT INDEX kbentity_name_ft IF NOT EXISTS "
              f"FOR (e:{LABEL}) ON EACH [e.name, e.aliases]")
        s.run(f"MATCH (e:{LABEL} {{tenant_id: $t, entity_key: 'ent:rlhf'}}) "
              "SET e.aliases = ['RLHF', '人类反馈强化学习', 'PPO对齐']", t=TENANT_A)
        s.run("CALL db.awaitIndexes(300)")
        exact = s.run(f"MATCH (e:{LABEL} {{tenant_id: $t, entity_key: 'ent:reward'}}) "
                      "RETURN e.name AS n", t=TENANT_A).data()
        report("N7a 第一级:entity_key 精确链接", len(exact) == 1, str(exact))

        ft = s.run("CALL db.index.fulltext.queryNodes('kbentity_name_ft', '人类反馈') "
                   "YIELD node, score RETURN node.entity_key AS k, score").data()
        report("N7b 第二级:全文索引可召回中文别名", bool(ft) and ft[0]["k"] == "ent:rlhf", str(ft))

        sloppy = s.run("CALL db.index.fulltext.queryNodes('kbentity_name_ft', '型奖') "
                       "YIELD node RETURN node.name AS name").data()
        report("N7e 默认分析器按字符 AND(乱序两字也命中→判定须在应用层)",
               [r["name"] for r in sloppy] == ["奖励模型"], str(sloppy))

        # N7c:全文索引是全局的,不认识租户 —— 命中集必须在仓库层再按 tenant_id 收口
        unscoped = s.run("CALL db.index.fulltext.queryNodes('kbentity_name_ft', '实体') "
                         "YIELD node RETURN node.tenant_id AS tenant").data()
        scoped = s.run("CALL db.index.fulltext.queryNodes('kbentity_name_ft', '实体') YIELD node "
                       f"MATCH (n:{LABEL} {{tenant_id: $t}}) WHERE n = node RETURN n.entity_key AS k",
                       t=TENANT_A).data()
        report("N7c 全文索引本身不隔离(须服务端收口)",
               {r["tenant"] for r in unscoped} == {TENANT_B} and scoped == [],
               f"unscoped={sorted({r['tenant'] for r in unscoped})} scoped={scoped}")

    with driver.session() as s:
        s.run(f"MATCH (e:{LABEL}) WHERE e.tenant_id IN $ts DETACH DELETE e", ts=[TENANT_A, TENANT_B])
        s.run("DROP INDEX kbentity_name_ft IF EXISTS")
        s.run("DROP CONSTRAINT kbentity_tenant_key IF EXISTS")

    failed = [r for r in results if not r[1]]
    print(f"\n=== {len(results) - len(failed)}/{len(results)} 通过 ===")
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
