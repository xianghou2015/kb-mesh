"""§5.3 三层幂等 + §5.4 三段式的落地验收(重跑 5 次结果一分不差)。

真 PG、真 Milvus、真 Neo4j、真 MinIO:「重跑不翻倍」「孤儿向量被清掉」「租户是 MERGE 键的一部分」
只有打真库才看得见,mock 出来的相等是自证。

假的只有两处,各自有理由:
- **模型**:结果必须可复现,验收不能看运气;顺便用它数出「嵌入只烧了一次钱」。
- **Redis**:锁的过期语义要在可控时钟下测(见 `test_lease.py`)。
"""
from __future__ import annotations

import hashlib
import io
import random
import re
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import fakeredis
import pytest
from PIL import Image
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from conftest import database_url
from kbmesh.core.config import get_settings
from kbmesh.core.errors import IllegalTransition
from kbmesh.core.gateway import ChatResult, ModelUnavailable, Usage
from kbmesh.core.tenancy import worker_context
from kbmesh.db.base import make_engine
from kbmesh.db.models import Document, KnowledgeBase, ParseStage, Tenant
from kbmesh.db.repositories import ChunkRepo, DocumentRepo
from kbmesh.ingest import lease, pipeline
from kbmesh.ingest.chunking import ChunkPolicy
from kbmesh.storage import milvus as mv
from kbmesh.storage import neo4j as n4
from kbmesh.storage import s3

DIM = get_settings().embedding_dim
BUCKET = get_settings().oss_bucket
COLLECTION = get_settings().milvus_collection

TENANTS = (("tnt-pipe-a", "kb-pipe-a", "doc-pipe-a"), ("tnt-pipe-b", "kb-pipe-b", "doc-pipe-b"))
DOC_A, KB_A, TENANT_A = TENANTS[0][2], TENANTS[0][1], TENANTS[0][0]
DOC_B, KB_B, TENANT_B = TENANTS[1][2], TENANTS[1][1], TENANTS[1][0]
IMG_REF = "diagram.png"
CAPTION_TEXT = "图注:召回与图谱两路在同一层融合"

#: 三个小节各自成一片;「Milvus」出现在两节里,同一实体被两个切片提到才测得出合并与去重
DOC_MARKDOWN = f"""# 检索增强生成

## 召回
RAGFlow 用 Milvus 做向量检索,标题路与正文路各出一批候选,再由 RRF 融合成一份候选列表。
Milvus 的稀疏索引走中文分词,标题里的字不能整串匹配。
![架构图]({IMG_REF})

## 图谱
Neo4j 负责图谱召回,实体节点带租户属性,遍历的每一跳都要重新核对租户是否一致。
三元组置信度低于阈值的直接丢弃,丢弃统计里必须能看见一共丢掉了多少条。

## 回答
答案只能引用召回到的切片,而召回依赖 Milvus 做向量检索的结果与图谱那一路的融合。
引用回传原图时给短期签名地址,不给公开读。
"""

#: (触发词, src_type, dst_label, dst_type, rtype) —— 假抽取器按词命中,依据一律取原文子串
EXTRACT_RULES = (
    ("Milvus", "component", "向量检索", "method", "USES"),
    ("Neo4j", "component", "图谱召回", "method", "PART_OF"),
    ("RRF", "method", "融合排序", "concept", "IMPROVES"),
)

EXPECTED_ENTITIES = [
    "component:milvus",
    "component:neo4j",
    "concept:融合排序",
    "method:rrf",
    "method:向量检索",
    "method:图谱召回",
]

_SOURCE_IN_PROMPT = re.compile(r"<<<\n(.*?)\n>>>", re.S)


# --------------------------------------------------------------------- 看门狗(§5.4)


class Probe:
    """会话/事务还开着就打外部服务 ⇒ 当场报错。

    三段式是整条管道的正确性地基:模型一次几十秒,连接被事务占着就是全池排队。
    光靠 review 守不住,所以把它做成一个会咬人的断言。
    """

    def __init__(self) -> None:
        self.held: set[int] = set()
        self.calls: Counter[str] = Counter()

    def enter(self, key: int) -> None:
        self.held.add(key)

    def leave(self, key: int) -> None:
        self.held.discard(key)

    def call(self, label: str) -> None:
        self.calls[label] += 1
        if self.held:
            raise AssertionError(f"{label} 落在持事务期间(§5.4 三段式):{len(self.held)} 个会话未结束")


_ACTIVE: Probe | None = None


class _ProbedSession(AsyncSession):
    """只多做进出计数:业务代码看到的还是普通 session。"""

    async def __aenter__(self) -> Any:
        if _ACTIVE is not None:
            _ACTIVE.enter(id(self))
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            await super().__aexit__(exc_type, exc, tb)
        finally:
            if _ACTIVE is not None:
                _ACTIVE.leave(id(self))


class FakeGateway:
    """确定性模型替身:同一段文本永远给同一个向量与同一批三元组。"""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.fail_stage: str | None = None
        self.probe: Probe | None = None

    def _gate(self, kind: str) -> None:
        self.calls[kind] += 1
        if self.probe is not None:
            self.probe.call(f"model:{kind}")
        if self.fail_stage == kind:
            raise ModelUnavailable(f"{kind}:quota", "假网关被要求在这里失败")

    def embed(self, texts: Any, *, purpose: str = "embed") -> tuple[list[list[float]], Usage]:
        self._gate("embed")
        vectors = []
        for content in texts:
            rng = random.Random(hashlib.sha256(content.encode("utf-8")).digest())
            vectors.append([rng.random() for _ in range(DIM)])
        return vectors, Usage(token_in=sum(len(t) for t in texts), model="fake-embed")

    def structured(self, prompt: str, model: Any, *, purpose: str = "extract") -> tuple[Any, Usage]:
        self._gate("extract")
        found = _SOURCE_IN_PROMPT.search(prompt)
        body = found.group(1) if found else ""
        triplets = []
        for term, stype, dlabel, dtype, rtype in EXTRACT_RULES:
            if term not in body:
                continue
            at = body.index(term)
            triplets.append(
                {
                    "src_label": term,
                    "src_type": stype,
                    "rtype": rtype,
                    "dst_label": dlabel,
                    "dst_type": dtype,
                    # 依据必须是原文逐字子串:假网关也遵守,否则「通过」是骗出来的
                    "evidence": body[at : at + len(term) + 6],
                    "confidence": 0.9,
                }
            )
        return {"triplets": triplets}, Usage(token_out=len(triplets) * 20, model="fake-chat")

    def vision(self, image_url: str, prompt: str, *, purpose: str = "vision") -> ChatResult:
        self._gate("vision")
        return ChatResult(text=CAPTION_TEXT, usage=Usage(model="fake-vision"))


# --------------------------------------------------------------------- 环境


@dataclass
class Env:
    factory: async_sessionmaker[AsyncSession]
    driver: Any
    gateway: FakeGateway
    probe: Probe

    @property
    def ctx(self) -> Any:
        return worker_context(TENANT_A, KB_A)

    def ctx_for(self, tenant_id: str, kb_id: str) -> Any:
        return worker_context(tenant_id, kb_id)


def _png() -> bytes:
    handle = io.BytesIO()
    Image.new("RGB", (24, 16), (10, 80, 200)).save(handle, format="PNG")
    return handle.getvalue()


def _keys(tenant_id: str, kb_id: str) -> tuple[str, str]:
    return (
        f"artifacts/{tenant_id}/{kb_id}/source.md",
        f"artifacts/{tenant_id}/{kb_id}/images/{IMG_REF}",
    )


async def _purge_pg(factory: async_sessionmaker[AsyncSession]) -> None:
    ids = [t for t, _k, _d in TENANTS]
    statements = (
        "delete from platform.relation_edge where tenant_id = any(:ts)",
        "delete from platform.security_event where tenant_id = any(:ts)",
        "delete from platform.chunk where tenant_id = any(:ts)",
        "delete from platform.entity where tenant_id = any(:ts)",
        "delete from platform.document where tenant_id = any(:ts)",
        "delete from platform.kb where tenant_id = any(:ts)",
        "delete from platform.tenant where id = any(:ts)",
    )
    async with factory() as session, session.begin():
        for stmt in statements:
            await session.execute(sa_text(stmt), {"ts": ids})


async def _seed(factory: async_sessionmaker[AsyncSession]) -> None:
    data = DOC_MARKDOWN.encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    prepared = []
    # 对象存储写在事务之外:否则每条 `_seed` 自己就先违反了它要保护的那条规则
    for tenant_id, kb_id, doc_id in TENANTS:
        source_key, image_key = _keys(tenant_id, kb_id)
        uri = s3.put_bytes(data, key=source_key, mime="text/markdown")
        s3.put_bytes(_png(), key=image_key, mime="image/png")
        prepared.append((tenant_id, kb_id, doc_id, uri, sha))
    async with factory() as session, session.begin():
        for tenant_id, *_ in prepared:
            session.add(Tenant(id=tenant_id, name=f"fixture {tenant_id}"))
        await session.flush()
        for tenant_id, kb_id, *_ in prepared:
            session.add(
                KnowledgeBase(
                    id=kb_id,
                    tenant_id=tenant_id,
                    name="pipe-kb",
                    embedding_model="fake-embed",
                    chunk_policy={},
                )
            )
        await session.flush()
        for tenant_id, kb_id, doc_id, uri, digest in prepared:
            session.add(
                Document(
                    id=doc_id,
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    source_uri=uri,
                    mime="text/markdown",
                    sha256=digest,
                    title="fixture doc",
                )
            )


def _guard_network(probe: Probe, monkeypatch: pytest.MonkeyPatch) -> None:
    """给四个外部服务的入口套上看门狗:包裹而不是替换,真调用照常发生。"""
    targets = {
        s3: ("get_bytes", "put_bytes", "presign_get"),
        mv: ("ensure_collection", "upsert_rows", "pks_for_doc", "delete_orphan_pks"),
        n4: (
            "ensure_schema",
            "merge_entities",
            "merge_relations",
            "merge_chunkrefs",
            "prune_chunkrefs",
            "merge_mentions",
        ),
    }
    for module, names in targets.items():
        for name in names:
            original = getattr(module, name)
            label = f"{module.__name__.rsplit('.', 1)[-1]}:{name}"

            def wrapper(*args: Any, _fn: Any = original, _label: str = label, **kwargs: Any) -> Any:
                probe.call(_label)
                return _fn(*args, **kwargs)

            monkeypatch.setattr(module, name, wrapper)


@pytest.fixture(scope="module")
async def shared() -> AsyncIterator[tuple[Any, Any]]:
    """引擎与图驱动是进程级的;数据每个测试自己造自己清。"""
    engine = make_engine(database_url())
    driver = n4.get_driver()
    try:
        yield engine, driver
    finally:
        driver.close()
        await engine.dispose()


@pytest.fixture
async def env(shared: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Env]:
    engine, driver = shared
    probe = Probe()
    gateway = FakeGateway()
    gateway.probe = probe
    global _ACTIVE
    _ACTIVE = probe
    factory = async_sessionmaker(engine, class_=_ProbedSession, expire_on_commit=False)
    pks: set[int] = set()
    try:
        await _purge_pg(factory)
        for tenant_id, kb_id in ((TENANT_A, KB_A), (TENANT_B, KB_B)):
            n4.delete_tenant_subgraph(driver, worker_context(tenant_id, kb_id))
        await _seed(factory)
        lease.set_client(fakeredis.FakeStrictRedis(decode_responses=True))
        _guard_network(probe, monkeypatch)
        yield Env(factory=factory, driver=driver, gateway=gateway, probe=probe)
    finally:
        _ACTIVE = None
        async with factory() as session, session.begin():
            rows = await session.execute(
                sa_text("select milvus_pk from platform.chunk where tenant_id = any(:ts)"),
                {"ts": [t for t, _k, _d in TENANTS]},
            )
            pks = set(rows.scalars())
        await _purge_pg(factory)
        lease.set_client(None)
        # 清扫阶段的调用不该再被上一个测试的事务状态绊倒
        probe.held.clear()
        mv.delete_orphan_pks(pks)
        client = s3.s3_client()
        for tenant_id, kb_id in ((TENANT_A, KB_A), (TENANT_B, KB_B)):
            n4.delete_tenant_subgraph(driver, worker_context(tenant_id, kb_id))
            for key in _keys(tenant_id, kb_id):
                client.delete_object(Bucket=BUCKET, Key=key)


async def _run_all(env: Env, doc_id: str = DOC_A) -> list[dict[str, Any]]:
    return await pipeline.index_document(
        doc_id, session_factory=env.factory, gateway=env.gateway, driver=env.driver
    )


# --------------------------------------------------------------------- 快照


def _one(handle: Any, query: str, tenant_id: str) -> int:
    return int(handle.run(query, t=tenant_id).single()["c"])


async def _graph_counts(env: Env, tenant_id: str) -> dict[str, int]:
    with n4.session(env.driver) as handle:
        out = {
            "entity": _one(handle, "MATCH (e:Entity {tenant_id: $t}) RETURN count(e) AS c", tenant_id),
            "chunkref": _one(handle, "MATCH (c:ChunkRef {tenant_id: $t}) RETURN count(c) AS c", tenant_id),
            "document": _one(handle, "MATCH (d:Document {tenant_id: $t}) RETURN count(d) AS c", tenant_id),
        }
        for rec in handle.run(
            "MATCH ()-[r:REL]->() WHERE r.tenant_id = $t RETURN r.rtype AS rt, count(r) AS c", t=tenant_id
        ):
            out[f"rel:{rec['rt']}"] = int(rec["c"])
    return out


async def _snapshot(env: Env, doc_id: str = DOC_A, tenant_id: str = TENANT_A) -> dict[str, Any]:
    kb_id = KB_A if tenant_id == TENANT_A else KB_B
    ctx = env.ctx_for(tenant_id, kb_id)
    async with env.factory() as session, session.begin():
        chunks = await ChunkRepo(ctx, session).list_for_doc(doc_id)
        status = (
            await session.execute(
                sa_text("select parse_status from platform.document where id = :d"), {"d": doc_id}
            )
        ).scalar()
        snapshot = {
            "status": status,
            "chunks": [
                (
                    c.chunk_key,
                    c.position,
                    c.heading_path,
                    c.page_no,
                    c.token_len,
                    list(c.image_refs),
                    len(c.captions or []),
                    c.graph_linked,
                    c.index_version,
                    c.text,
                )
                for c in chunks
            ],
            "pg_pks": sorted({c.milvus_pk for c in chunks}),
            "entities": sorted(
                (
                    await session.execute(
                        sa_text("select entity_key from platform.entity where tenant_id = :t"),
                        {"t": tenant_id},
                    )
                ).scalars()
            ),
            "edges": sorted(
                tuple(row)
                for row in (
                    await session.execute(
                        sa_text(
                            "select src_key, dst_key, rtype from platform.relation_edge where tenant_id = :t"
                        ),
                        {"t": tenant_id},
                    )
                ).all()
            ),
        }
    snapshot["milvus_pks"] = sorted(mv.pks_for_doc(ctx, doc_id))
    snapshot["graph"] = await _graph_counts(env, tenant_id)
    return snapshot


# --------------------------------------------------------------------- 幂等验收


async def test_five_full_runs_change_nothing(env: Env) -> None:
    """§5.3 的验收句:重跑 5 次,chunk 数、Milvus pk 集合、Neo4j 节点数一律不变。"""
    snapshots = []
    for _ in range(5):
        reports = await _run_all(env)
        assert [r["stage"] for r in reports] == list(pipeline.PIPELINE_STAGES)
        snapshots.append(await _snapshot(env))

    first = snapshots[0]
    for run, snap in enumerate(snapshots[1:], start=2):
        assert snap == first, f"第 {run} 次重跑改变了落库结果"

    # 「相等」也可能是全 0 的空转:下面这几条证明东西确实落了,而且落对了
    assert first["status"] == ParseStage.DONE.value
    assert len(first["chunks"]) == 3
    assert [c[1] for c in first["chunks"]] == [0, 1, 2], "切片顺序要稳定"
    assert first["entities"] == EXPECTED_ENTITIES, first["entities"]
    assert len(first["edges"]) == 3, first["edges"]
    assert first["milvus_pks"] == first["pg_pks"], "向量投影与权威表不一致"
    assert first["graph"] == {
        "entity": 6,
        "chunkref": 3,
        "document": 1,
        "rel:USES": 1,
        "rel:PART_OF": 1,
        "rel:IMPROVES": 1,
        "rel:HAS_CHUNK": 3,
        "rel:MENTIONED_IN": 6,
    }, first["graph"]


async def test_second_parse_inserts_nothing(env: Env) -> None:
    """幂等第一层:同一个 `(tenant_id, chunk_key)` 冲突就跳过,而不是先查再插。"""
    first = await pipeline.run_stage("parse", DOC_A, session_factory=env.factory)
    assert first["counts"]["parts"] == 3
    assert first["counts"]["inserted"] == 3
    assert first["counts"]["removed"] == 0
    again = await pipeline.run_stage("parse", DOC_A, session_factory=env.factory)
    assert again["counts"]["inserted"] == 0
    assert again["counts"]["removed"] == 0
    assert again["counts"]["parts"] == 3
    assert env.probe.calls["s3:get_bytes"] == 2, "重跑要真的重新解析,不是被短路掉"


async def test_reruns_do_not_pay_the_model_twice(env: Env) -> None:
    """幂等不只是「数据没变」:向量已存权威表就不该再烧一次嵌入,视觉回填只做一次。"""
    for _ in range(3):
        await _run_all(env)
    assert env.gateway.calls["embed"] == 1, env.gateway.calls
    assert env.gateway.calls["vision"] == 1, env.gateway.calls
    assert env.gateway.calls["extract"] == 3 * 3, "抽取不缓存:键只由源文决定,重跑结果可复现"


async def test_caption_lands_in_the_indexed_body_not_in_the_key(env: Env) -> None:
    """FR-104 与 FR-103 的交点:图片内容参与检索,但模型输出绝不许进切片键。"""
    await _run_all(env)
    first = await _snapshot(env)
    assert first["chunks"][0][6] == 1, "带图的切片应有一条视觉回填"
    assert CAPTION_TEXT not in first["chunks"][0][9], "回填不该混进权威正文列"

    rows = mv.get_client().query(
        COLLECTION,
        filter=mv.combined_filter(env.ctx),
        output_fields=["body_text"],
        limit=16,
    )
    assert any(CAPTION_TEXT in str(r["body_text"]) for r in rows), "回填没进索引正文,图片内容检索不到"

    await _run_all(env)
    again = await _snapshot(env)
    assert [c[0] for c in again["chunks"]] == [c[0] for c in first["chunks"]], "切片键被模型输出污染了"


async def test_policy_change_leaves_no_orphan_vectors(env: Env) -> None:
    """切片策略变了:旧键的切片与旧 pk 的向量必须一起消失,否则老内容还能被召回。"""
    await _run_all(env)
    before = await _snapshot(env)
    old_pks = set(before["milvus_pks"])
    assert old_pks, "上一轮没留下向量,这一轮就什么也删不出来"

    async with env.factory() as session, session.begin():
        await session.execute(
            sa_text(
                "update platform.document set index_version = index_version + 1 where id = :d"
            ),
            {"d": DOC_A},
        )
    await pipeline.run_stage(
        "parse",
        DOC_A,
        session_factory=env.factory,
        policy=ChunkPolicy(max_tokens=40, overlap_tokens=0),
    )
    await pipeline.run_stage("embed", DOC_A, session_factory=env.factory, gateway=env.gateway)

    after = await _snapshot(env)
    assert len(after["chunks"]) > len(before["chunks"]), "切细了却没有更多切片"
    assert after["milvus_pks"] == after["pg_pks"], "Milvus 里留着权威表已删的孤儿"
    stale = (old_pks - set(after["pg_pks"])) & set(after["milvus_pks"])
    assert not stale, f"旧向量还在,会被继续召回: {sorted(stale)}"


# --------------------------------------------------------------------- 三段式


async def test_the_guard_really_bites(env: Env) -> None:
    """先看门狗能不能拦住「事务里调外部服务」,否则后面的干净结论是空转出来的。"""
    async with env.factory() as session, session.begin():
        for thunk in (
            lambda: s3.get_bytes("s3://bucket/whatever"),
            lambda: env.gateway.embed(["x"]),
            lambda: n4.merge_entities(env.driver, env.ctx, KB_A, []),
            lambda: mv.upsert_rows([]),
        ):
            with pytest.raises(AssertionError, match="三段式"):
                thunk()
    assert env.probe.held == set(), "事务块退出后计数没归还"


async def test_a_clean_run_touches_nothing_the_guard_objects_to(env: Env) -> None:
    await _run_all(env)
    assert {"s3:get_bytes", "milvus:upsert_rows", "neo4j:merge_relations"} <= set(env.probe.calls)
    assert env.probe.held == set(), "有会话没关"


# --------------------------------------------------------------------- 失败与闸门


async def test_stage_moves_beyond_one_step_are_rejected(env: Env) -> None:
    """§5.2 的「禁止跳跃」:±1 之外一律抛,重索引是唯一明写的例外。"""
    async with env.factory() as session, session.begin():
        repo = DocumentRepo(env.ctx, session)
        await repo.set_stage(DOC_A, ParseStage.PARSING)
        await repo.set_stage(DOC_A, ParseStage.CHUNKED)
        with pytest.raises(IllegalTransition, match="跳跃"):
            await repo.set_stage(DOC_A, ParseStage.DONE)  # +3
        with pytest.raises(IllegalTransition, match="跳跃"):
            await repo.set_stage(DOC_A, ParseStage.PENDING)  # -2
        await repo.set_stage(DOC_A, ParseStage.PARSING)  # -1:同一阶段原地重跑
        await repo.set_stage(DOC_A, ParseStage.CHUNKED)
        await repo.set_stage(DOC_A, ParseStage.EMBEDDED)
        await repo.set_stage(DOC_A, ParseStage.GRAPH_QUEUED)
        await repo.set_stage(DOC_A, ParseStage.GRAPH_DONE)
        await repo.set_stage(DOC_A, ParseStage.DONE)
        await repo.set_stage(DOC_A, ParseStage.PARSING)  # 重索引:只有 DONE 能回到头上


async def test_failed_stage_is_recorded_and_retried_in_place(env: Env) -> None:
    """§5.2 + §5.4 后半句:失败落成 FAILED(stage, error, retry_count) 并抛出去,不许静默。"""
    await pipeline.run_stage("parse", DOC_A, session_factory=env.factory)

    env.gateway.fail_stage = "embed"
    for attempt in (1, 2):
        with pytest.raises(ModelUnavailable):
            await pipeline.run_stage("embed", DOC_A, session_factory=env.factory, gateway=env.gateway)
        async with env.factory() as session, session.begin():
            row = await session.get(Document, DOC_A)
            assert row.parse_status == ParseStage.FAILED.value
            assert row.failure["stage"] == "embed"
            assert row.failure["retry_count"] == attempt
            assert "quota" in row.failure["error"]

    env.gateway.fail_stage = None
    report = await pipeline.run_stage("embed", DOC_A, session_factory=env.factory, gateway=env.gateway)
    assert report["stage"] == "embed"
    async with env.factory() as session, session.begin():
        row = await session.get(Document, DOC_A)
        assert row.parse_status == ParseStage.EMBEDDED.value
        assert row.failure["retry_count"] == 2, "原地重跑不该抹掉历史重试次数"


async def test_a_poisoned_edge_row_never_reaches_the_graph(env: Env) -> None:
    """第二道闸门:权威表被人写脏时,落图阶段照样拦 —— 抽取层那道守不住的这一半。"""
    await _run_all(env)
    before = (await _snapshot(env))["graph"]
    async with env.factory() as session, session.begin():
        chunk_id = (
            await session.execute(
                sa_text("select id from platform.chunk where tenant_id = :t order by position limit 1"),
                {"t": TENANT_A},
            )
        ).scalar()
        await session.execute(
            sa_text(
                "insert into platform.relation_edge"
                " (id, tenant_id, kb_id, src_key, dst_key, rtype, evidence, chunk_id, confidence)"
                " values ('ed-poison', :t, :k, 'component:milvus', 'component:neo4j',"
                " :rt, '脏数据', :c, 1.0)"
            ),
            {"t": TENANT_A, "k": KB_A, "c": chunk_id, "rt": 'REL] MATCH (n) DETACH DELETE n; //'},
        )

    with pytest.raises(n4.IllegalGraphIdentifier):
        await pipeline.run_stage("graph", DOC_A, session_factory=env.factory, driver=env.driver)

    with n4.session(env.driver) as handle:
        dirty = int(
            handle.run(
                "MATCH ()-[r:REL]->() WHERE r.tenant_id = $t AND r.rtype <> 'MENTIONED_IN' "
                "AND NOT r.rtype IN $ok RETURN count(r) AS c",
                t=TENANT_A,
                ok=sorted(n4.REL_TYPES),
            ).single()["c"]
        )
    assert dirty == 0, "白名单外的关系类型写进了图"
    assert (await _graph_counts(env, TENANT_A)) == before, "拦下脏边时把合法边也一起改了"
    async with env.factory() as session, session.begin():
        row = await session.get(Document, DOC_A)
        assert row.parse_status == ParseStage.FAILED.value
        assert row.failure["stage"] == "graph"


# --------------------------------------------------------------------- 租户与并发


async def test_identical_docs_from_two_tenants_stay_apart(env: Env) -> None:
    """同一份文档进两个租户:`entity_key` 一模一样,租户必须是 MERGE 键的一部分。"""
    await _run_all(env, DOC_A)
    await _run_all(env, DOC_B)

    a = await _snapshot(env, DOC_A, TENANT_A)
    b = await _snapshot(env, DOC_B, TENANT_B)
    assert a["entities"] == b["entities"] == EXPECTED_ENTITIES
    assert a["graph"] == b["graph"], "两个租户的图不一样"
    assert a["chunks"] != b["chunks"], "切片键含 doc_id,两租户不该相同"

    with n4.session(env.driver) as handle:
        total = int(
            handle.run(
                "MATCH (e:Entity) WHERE e.tenant_id IN $ts RETURN count(e) AS c",
                ts=[TENANT_A, TENANT_B],
            ).single()["c"]
        )
    assert total == 2 * a["graph"]["entity"], "两个租户共用了同一个实体节点"


async def test_deleting_one_tenant_leaves_the_other_intact(env: Env) -> None:
    await _run_all(env, DOC_A)
    await _run_all(env, DOC_B)
    assert n4.delete_tenant_subgraph(env.driver, env.ctx) > 0
    assert (await _graph_counts(env, TENANT_A))["entity"] == 0
    assert (await _graph_counts(env, TENANT_B))["entity"] == len(EXPECTED_ENTITIES)


async def test_a_second_worker_is_refused_and_writes_nothing(env: Env) -> None:
    """同一阶段已在别处跑:立刻返回并发提示,而不是把同一批切片再写一遍。"""
    taken = lease.acquire(lease.task_key(DOC_A, 1, "parse"))
    assert taken is not None
    report = await pipeline.run_stage("parse", DOC_A, session_factory=env.factory)
    assert report["warnings"], report
    assert env.probe.calls["s3:get_bytes"] == 0, "没抢到锁还去解析了"
    async with env.factory() as session, session.begin():
        assert await ChunkRepo(env.ctx, session).count_for_doc(DOC_A) == 0
    lease.release(taken)
    assert (await pipeline.run_stage("parse", DOC_A, session_factory=env.factory))["counts"]["parts"] == 3
