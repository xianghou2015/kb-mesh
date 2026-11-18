"""防线③ + 防线④:真实多租户数据上的越权用例(技术文档 §9 第 1 组)。

这一组必须跑真库,不接受 mock;「看函数签名对不对」那部分单独由源码扫描兜住(最后一节),
其余全部打到 PG 上验证结果集与审计流水。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kbmesh.core.tenancy import ROLE_PLATFORM_ADMIN, ROLE_READER, ROLE_TENANT_ADMIN, TenantContext
from kbmesh.db.models import Chunk, Document, KnowledgeBase, ParseStage, Tenant
from kbmesh.db.repositories import (
    ChunkRepo,
    DocumentRepo,
    KBRepo,
    TraceRepo,
    list_security_events,
)
from kbmesh.security.tokens import context_from_token, issue_access_token

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "kbmesh"

#: 租户身份的产生者:只能是从验签后的 claims 或从权威行读出来的那两处,别处一律不许收 tenant_id。
CONTEXT_PRODUCERS = frozenset({"worker_context"})


async def seed_tenant(session: AsyncSession, name: str, *, kb_name: str = "kb") -> dict[str, str]:
    """造一个真租户 + kb + 文档 + 两条切片,返回各层 id。"""
    tenant = Tenant(name=name)
    session.add(tenant)
    await session.flush()
    kb = KnowledgeBase(tenant_id=tenant.id, name=kb_name, embedding_model="m")
    session.add(kb)
    await session.flush()
    doc = Document(
        tenant_id=tenant.id,
        kb_id=kb.id,
        source_uri=f"s3://{name}/a.pdf",
        sha256=f"{name}-0".ljust(64, "0")[:64],
    )
    session.add(doc)
    await session.flush()
    session.add_all(
        [
            Chunk(
                tenant_id=tenant.id,
                kb_id=kb.id,
                doc_id=doc.id,
                chunk_key=f"{name}-0",
                text=f"{name} 的正文",
                milvus_pk=1,
            ),
            Chunk(
                tenant_id=tenant.id,
                kb_id=kb.id,
                doc_id=doc.id,
                chunk_key=f"{name}-1",
                text=f"{name} 的第二段",
                milvus_pk=2,
            ),
        ]
    )
    await session.flush()
    return {"tenant": tenant.id, "kb": kb.id, "doc": doc.id}


def ctx_for(
    seed: dict[str, str],
    *,
    roles: tuple[str, ...] = (ROLE_TENANT_ADMIN,),
    kb_scope: tuple[str, ...] = (),
) -> TenantContext:
    return TenantContext(
        tenant_id=seed["tenant"],
        user_id=f"u-{seed['tenant']}",
        roles=roles,
        kb_scope=frozenset(kb_scope),
    )


async def test_three_tenants_are_mutually_invisible(session: AsyncSession) -> None:
    a, b, c = (
        await seed_tenant(session, "A"),
        await seed_tenant(session, "B"),
        await seed_tenant(session, "C"),
    )
    for tag, seed, others in (("A", a, (b, c)), ("B", b, (a, c)), ("C", c, (a, b))):
        ctx = ctx_for(seed)
        assert [k.id for k in await KBRepo(ctx, session).list()] == [seed["kb"]]

        docs = await DocumentRepo(ctx, session).list_for_kb(seed["kb"])
        assert [d.id for d in docs] == [seed["doc"]]

        chunks = await ChunkRepo(ctx, session).list_for_doc(seed["doc"])
        assert {ch.text for ch in chunks} == {f"{tag} 的正文", f"{tag} 的第二段"}
        assert await ChunkRepo(ctx, session).count_for_doc(seed["doc"]) == 2

        for other in others:
            assert await KBRepo(ctx, session).get(other["kb"]) is None
            assert await DocumentRepo(ctx, session).get(other["doc"]) is None
            assert await ChunkRepo(ctx, session).by_ids([other["doc"]]) == []


async def test_cross_tenant_document_read_is_audited(session: AsyncSession) -> None:
    """§9:租户 A 的令牌查 B → 空结果 **且** 审计有记录。"""
    a, b = await seed_tenant(session, "A"), await seed_tenant(session, "B")
    ctx = ctx_for(a)

    with pytest.raises(Exception, match="cross_tenant_read"):
        await DocumentRepo(ctx, session).require(b["doc"])

    events = await list_security_events(ctx, session)
    assert [(e.event, e.target) for e in events] == [("cross_tenant_read", f"doc:{b['doc']}")]
    assert events[0].tenant_id == a["tenant"]

    assert await list_security_events(ctx_for(b), session) == []


async def test_kb_scope_narrows_within_the_same_tenant(session: AsyncSession) -> None:
    tenant = Tenant(name="scoped")
    session.add(tenant)
    await session.flush()
    whole = TenantContext(tenant_id=tenant.id, user_id="u", roles=(ROLE_TENANT_ADMIN,))
    kb1 = await KBRepo(whole, session).create("kb1", "m")
    kb2 = await KBRepo(whole, session).create("kb2", "m")

    narrow = TenantContext(
        tenant_id=tenant.id, user_id="u", roles=(ROLE_TENANT_ADMIN,), kb_scope=frozenset({kb1.id})
    )
    assert [k.id for k in await KBRepo(narrow, session).list()] == [kb1.id]
    assert await KBRepo(narrow, session).get(kb2.id) is None

    doc = await DocumentRepo(narrow, session).create(kb1.id, "s3://x", "0" * 64)
    assert doc.kb_id == kb1.id
    with pytest.raises(Exception, match="kb_scope_denied"):
        await DocumentRepo(narrow, session).list_for_kb(kb2.id)
    with pytest.raises(Exception, match="kb_scope_denied"):
        await DocumentRepo(narrow, session).create(kb2.id, "s3://y", "1" * 64)


async def test_reader_cannot_write_and_it_is_audited(session: AsyncSession) -> None:
    a = await seed_tenant(session, "A")
    reader = ctx_for(a, roles=(ROLE_READER,))
    with pytest.raises(Exception, match="role_denied"):
        await KBRepo(reader, session).create("new-kb", "m")
    with pytest.raises(Exception, match="role_denied"):
        await DocumentRepo(reader, session).create(a["kb"], "s3://x", "1" * 64)

    events = await list_security_events(reader, session)
    assert [e.event for e in events] == ["role_denied", "role_denied"]


async def test_stage_machine_rejects_jumps(session: AsyncSession) -> None:
    a = await seed_tenant(session, "A")
    ctx = ctx_for(a)
    repo = DocumentRepo(ctx, session)
    doc = await repo.require(a["doc"])
    assert doc.parse_status == "PENDING"

    await repo.set_stage(doc.id, ParseStage.PARSING)
    assert (await repo.require(doc.id)).parse_status == "PARSING"

    with pytest.raises(Exception, match="跳跃"):
        await repo.set_stage(doc.id, ParseStage.DONE)

    await repo.mark_failed(doc.id, {"reason": "boom"})
    assert (await repo.require(doc.id)).parse_status == "FAILED"
    await repo.set_stage(doc.id, ParseStage.PENDING)  # 失败后允许重跑


async def test_trace_run_lookup_is_tenant_bound(session: AsyncSession) -> None:
    a, b = await seed_tenant(session, "A"), await seed_tenant(session, "B")
    repo = TraceRepo(ctx_for(a), session)
    run = await repo.start_run(f"{a['tenant']}:t1", "问题")
    assert await repo.get_run(run.id) is run
    with pytest.raises(Exception, match="cross_tenant_read"):
        await TraceRepo(ctx_for(b), session).get_run(run.id)


async def test_finish_run_only_writes_declared_final_statuses(session: AsyncSession) -> None:
    """FR-406:拒答与答完在数据层可区分,靠的就是 `final_status` 只有 `FINAL_STATUSES` 那四个值。

    写进去一个没人认识的字符串,统计口径当天就会悄悄失真 —— 所以这里抛,而不是原样落库。
    """
    a = await seed_tenant(session, "A")
    repo = TraceRepo(ctx_for(a), session)
    run = await repo.start_run(f"{a['tenant']}:t9", "问题")
    assert run.final_status == "error", "没收尾的 run 默认就是失败,不许算进 answered"

    row = await repo.finish_run(run.id, final_status="refused", latency_ms=12, token_in=3, token_out=7)
    assert row.final_status == "refused"
    assert (row.token_in, row.token_out, row.latency_ms) == (3, 7, 12)
    assert row.degraded is False

    for bad in ("answered_but_refused", "", "REFUSED"):
        with pytest.raises(Exception, match="final_status"):
            await repo.finish_run(run.id, final_status=bad)

    expired = await repo.finish_run(run.id, final_status="expired", degraded=True)
    assert (expired.final_status, expired.degraded) == ("expired", True)


async def test_resume_assert_thread_blocks_foreign_and_forged_keys(session: AsyncSession) -> None:
    """跨租户续跑的第三道闸:前缀不对挡下,前缀换了也照样挡下(库里没有这条会话)。

    伪造 key 是最现实的一条:`thread_key = tenant:thread` 由客户端提交,把 A 的 thread 换个前缀
    就变成 B 自己的了 —— 只有库里的归属记录能证伪它。断言「拒绝 **且** 审计有记录」。
    """
    a, b = await seed_tenant(session, "A"), await seed_tenant(session, "B")
    repo_a = TraceRepo(ctx_for(a), session)
    run = await repo_a.start_run(f"{a['tenant']}:t1", "问题")
    assert run.thread_id == f"{a['tenant']}:t1"

    await TraceRepo(ctx_for(a), session).assert_thread(f"{a['tenant']}:t1")  # 自己的会话,应当不抛

    forged = (f"{b['tenant']}:t1", f"{a['tenant']}:", "t1", f"x:{a['tenant']}:t1")
    for key in forged:
        with pytest.raises(Exception, match="cross_tenant_read"):
            await TraceRepo(ctx_for(b), session).assert_thread(key)

    events = await list_security_events(ctx_for(b), session)
    assert {e.event for e in events} == {"cross_tenant_read"}
    assert {e.target for e in events} == {f"thread:{key}" for key in forged}


async def test_no_repo_takes_a_bare_tenant_id() -> None:
    """§6.1 红线:函数签名里出现裸 `tenant_id` 形参视为缺陷 —— 扫源码,不靠人记住。

    豁免只有一个:`worker_context()` 是离线身份的**产生者**,它的入参只能来自权威行
    (`document.tenant_id`),和在线侧的 `context_from_token()` 是同一地位。豁免写成函数名而不是
    目录白名单,新增一个越权入口立刻会被这条测试逮住。
    """
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            args = [a.arg for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs]
            if "tenant_id" in args and node.name not in CONTEXT_PRODUCERS:
                offenders.append(f"{path.relative_to(SRC_ROOT.parent)}:{node.lineno} {node.name}()")
    assert offenders == [], "这些函数暴露了裸 tenant_id 形参:\n" + "\n".join(offenders)


def test_repository_constructors_take_context_first() -> None:
    for cls in (KBRepo, DocumentRepo, ChunkRepo, TraceRepo):
        params = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
        assert params[0] == "ctx", f"{cls.__name__} 的第一个参数不是 ctx"
        # 模块带 `from __future__ import annotations`,签名里的注解是字符串
        annotation = inspect.signature(cls.__init__).parameters["ctx"].annotation
        assert annotation in ("TenantContext", TenantContext)
        for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
            assert "tenant_id" not in inspect.signature(fn).parameters, f"{cls.__name__}.{name}"


async def test_identity_comes_from_the_token_not_the_body(session: AsyncSession) -> None:
    """防线②:请求体里自报的 tenant_id 一律无效 —— 只有验签后的 claims 说话。"""
    a, b = await seed_tenant(session, "A"), await seed_tenant(session, "B")
    ctx = context_from_token(issue_access_token(ctx_for(a)))
    assert ctx.tenant_id == a["tenant"]

    # 客户端在 body 里自称 B:仓库层没有任何入口接收它,越权路径不存在
    assert await KBRepo(ctx, session).get(b["kb"]) is None
    with pytest.raises(Exception, match="cross_tenant_read"):
        await DocumentRepo(ctx, session).require(b["doc"])


async def test_audit_read_is_itself_tenant_filtered(session: AsyncSession) -> None:
    a, b = await seed_tenant(session, "A"), await seed_tenant(session, "B")
    for ctx, target in ((ctx_for(a), b["doc"]), (ctx_for(b), a["doc"])):
        with pytest.raises(Exception, match="cross_tenant_read"):
            await DocumentRepo(ctx, session).require(target)

    mine = await list_security_events(ctx_for(a), session)
    assert {e.tenant_id for e in mine} == {a["tenant"]}

    # 只数这两个租户名下的行:审计表是共享库里的活表,拿全表行数做断言必然随用例顺序漂移
    scoped = await session.execute(
        text(
            "SELECT count(*) FROM platform.security_event "
            "WHERE tenant_id IN (:a, :b) AND path = ''"
        ),
        {"a": a["tenant"], "b": b["tenant"]},
    )
    assert scoped.scalar_one() == 2, "两个租户各留一条"

    root = TenantContext(tenant_id=a["tenant"], user_id="root", roles=(ROLE_PLATFORM_ADMIN,))
    assert len(await list_security_events(root, session, cross_tenant=True)) >= 2
