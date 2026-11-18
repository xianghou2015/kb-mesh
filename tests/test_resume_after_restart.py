"""§13 M5 的验收:**重启后 resume 通过**(FR-403 / NFR-1)。

这一份是 #6 唯一打真检查点库、真两个进程的验收。和 `test_graph_orchestration.py` 的分工:那边
用 `InMemorySaver` 验「判定对不对」,这边验「进程死了还算不算数」。所以子进程里模型与检索都是
桩(`_restart_child.py`),而检查点库、`trace_run`、`trace_step` 全是真行。

**编排层的写入是一步一事务真实提交的**(见 `agents/trace.py`),`conftest.session` 的 SAVEPOINT
兜不住;检查点表更不归 Alembic 管。所以本文件自己 purge:`tnt-rs-*` 留下的 run 与
`lg.checkpoint*` 行不清掉,下一个用例读到的就是上一条问答的状态 —— 那种失败长得像代码坏了。

`SIGTERM` 由父进程发(`proc.terminate()`),对得 §9 原文「停在 await_confirm → SIGTERM 重启服务
→ Command(resume) 续跑成功」。不发 KILL:要模拟的是一次服务重启,不是一次内核 OOM。
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

import _restart_child as child
from conftest import database_url
from kbmesh.agents import nodes as nodes_mod
from kbmesh.agents.graph import compile_graph
from kbmesh.agents.nodes import REFUSAL_TEXT, GraphDeps
from kbmesh.agents.runtime import (
    Runtime,
    SaverHandle,
    assert_thread_shape,
    split_thread_key,
    sweep_stale,
    thread_key,
)
from kbmesh.agents.trace import TraceWriter
from kbmesh.core.config import get_settings
from kbmesh.core.errors import IllegalTransition, PermissionDenied
from kbmesh.core.tenancy import ROLE_READER, TenantContext, use_context
from kbmesh.db.base import make_engine
from kbmesh.db.models import Tenant, TraceRun
from kbmesh.retrieval.contracts import Citation, Hit
from kbmesh.retrieval.fusion import RetrievalResult

REPO = Path(__file__).resolve().parents[1]
CHILD = REPO / "tests" / "_restart_child.py"
TENANT_B = "tnt-rs-b"
#: mode → 子进程那行输出的前缀
PREFIX = {"a": "READY ", "b": "RESUMED "}
#: 子进程要自己建池、编译图、跑完五层,给两条都不小于半分钟的余量
LINE_BUDGET_S = 240.0
WAIT_BUDGET_S = 90.0


def _urls() -> tuple[str, str]:
    settings = get_settings()
    return (
        settings.test_db_url if settings.is_test else settings.db_url,
        settings.checkpoint_schema,
    )


async def _scalar(factory: async_sessionmaker[AsyncSession], sql: str, **params: Any) -> Any:
    async with factory() as session:
        return (await session.execute(sa_text(sql), params)).scalar()


async def _purge(factory: async_sessionmaker[AsyncSession]) -> None:
    """该前缀下的检查点行 + run / step / event / tenant。

    检查点按 `thread_id like 'tnt-rs-%'` 删,而不是逐条 `adelete_thread`:被杀掉的那个进程用的
    线程名带随机后缀,父进程只能按前缀扫。先删检查点后删 trace,反过来的话 tenant 会被 run 的
    外键拦住。
    """
    ids = [child.TENANT, TENANT_B]
    _, schema = _urls()
    ckpt = tuple(
        f'delete from "{schema}".{table} where thread_id like :pre'
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints")
    )
    trace = (
        "delete from platform.trace_step where run_id in "
        "(select id from platform.trace_run where tenant_id = any(:ts))",
        "delete from platform.trace_run where tenant_id = any(:ts)",
        "delete from platform.security_event where tenant_id = any(:ts)",
        "delete from platform.tenant where id = any(:ts)",
    )
    async with factory() as session, session.begin():
        for stmt in ckpt:
            await session.execute(sa_text(stmt), {"pre": f"{child.TENANT}-%"})
        for stmt in trace:
            await session.execute(sa_text(stmt), {"ts": ids})


@pytest.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = make_engine(database_url())
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def factory(engine: AsyncEngine) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    await _purge(maker)
    async with maker() as session, session.begin():
        for tenant_id in (child.TENANT, TENANT_B):
            session.add(Tenant(id=tenant_id, name=f"{tenant_id}-名字"))
    yield maker
    await _purge(maker)


@pytest.fixture
async def handle() -> AsyncIterator[SaverHandle]:
    """每个用例一条全新的检查点池 —— 「重启」在本文件里就是「换一个连接池」的别名。"""
    handler = SaverHandle.open_sync()
    await handler.open()
    try:
        yield handler
    finally:
        await handler.close()


# --------------------------------------------------------------------- 子进程
def _popen(mode: str, thread: str, extra: tuple[str, ...]) -> subprocess.Popen[str]:
    url, _ = _urls()
    return subprocess.Popen(  # noqa: S603 — 固定解释器 + 仓内脚本,参数里没有用户输入
        [sys.executable, str(CHILD), mode, thread, *extra],
        cwd=str(REPO),
        env={**os.environ, "KBMESH_ENV": "test", "KBMESH_TEST_DB_URL": url},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


async def _wait(proc: subprocess.Popen[str]) -> tuple[int, str]:
    loop = asyncio.get_running_loop()
    err = await loop.run_in_executor(None, proc.stderr.read) if proc.stderr else ""
    code = await asyncio.wait_for(loop.run_in_executor(None, proc.wait), WAIT_BUDGET_S)
    return code, err


async def _spawn(mode: str, thread: str, *extra: str) -> dict[str, Any]:
    """跑一段子进程,返回 `{payload, code, stderr, line}`。

    读首行丢进线程池而不是 `communicate()`:`mode=a` 是**故意**被杀的,它永远不会自己正常退出,
    等 EOF 只是把「子进程其实已经炸了」延迟到超时才报。payload 为 None 不在这里抛 —— 有的用例
    要的就是「它必须失败」,由用例自己判。

    mode=a 读到 READY 之后由这里发 SIGTERM:那是被测的动作品,不是收尾手段。
    """
    proc = _popen(mode, thread, extra)
    loop = asyncio.get_running_loop()
    try:
        line = await asyncio.wait_for(
            loop.run_in_executor(None, proc.stdout.readline), LINE_BUDGET_S
        )
    except TimeoutError:
        proc.kill()
        code, err = await _wait(proc)
        raise AssertionError(f"子进程 {mode} 没走到 {PREFIX[mode]!r} 行(rc={code})\n{err[-3000:]}") from None
    payload = json.loads(line[len(PREFIX[mode]) :]) if line.startswith(PREFIX[mode]) else None
    if mode == "a" and payload is not None:
        proc.terminate()
    code, err = await _wait(proc)
    return {"payload": payload, "code": code, "stderr": err, "line": line}


# --------------------------------------------------------------------- M5 主验收
async def test_sigterm_then_a_new_process_resumes(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    thread = f"rs-{os.getpid()}-main"
    stopped = await _spawn("a", thread)
    ready = stopped["payload"]
    assert ready is not None, stopped["line"]
    assert stopped["code"] == -signal.SIGTERM, f"不是被 SIGTERM 杀的:returncode={stopped['code']}"
    assert ready["kind"] == "await_confirm"
    assert ready["candidates"] == [child.CHOSEN], "确认框里的候选必须由中断前那一轮给出"
    run_id = ready["run_id"]

    # 进程已经死了。库里此刻必须留着两件「等确认」的事:run 在、检查点在
    steps_before = await _scalar(
        factory, "select count(*) from platform.trace_step where run_id = :i", i=run_id
    )
    assert steps_before >= 2, f"中断前只落了 {steps_before} 步:五层防御的留痕在重启现场是残缺的"
    _, schema = _urls()
    assert await _scalar(
        factory,
        f'select count(*) from "{schema}".checkpoints where thread_id = :k',
        k=f"{child.TENANT}:{thread}",
    ) >= 1, "SIGTERM 之前检查点没落盘 = FR-403 根本没实现"

    resumed = await _spawn("b", thread)
    assert resumed["code"] == 0, resumed["stderr"][-2000:]
    got = resumed["payload"]
    assert got["kind"] == "answered" and got["final_status"] == "answered"
    assert got["run_id"] == run_id, "换个进程就换个 run_id,一次问答会被统计成两次"
    assert got["answer"] == child.ANSWER
    assert got["confirmed_query"] == child.CHOSEN, "续跑用的是确认后的那句,不是原问"
    assert [one["chunk_id"] for one in got["citations"]] == ["chunk-rs-1"]
    assert await _scalar(
        factory,
        "select final_status || '|' || degraded::text from platform.trace_run where id = :i",
        i=run_id,
    ) == "answered|false"

    steps_after = await _scalar(
        factory, "select count(*) from platform.trace_step where run_id = :i", i=run_id
    )
    assert steps_after > steps_before, "续跑的那几步没留痕:重启之后这次问答在报告里是残缺的"
    # `confirm_interrupt` 的节点体在 resume 时会重跑一遍(实测 C4c),它写的那一步必须还是同一步
    seqs = await _scalar(
        factory,
        "select string_agg(seq::text, ',' order by seq) from platform.trace_step where run_id = :i",
        i=run_id,
    )
    assert seqs == ",".join(str(n) for n in range(1, steps_after + 1)), f"seq 有空洞或重复:{seqs}"


async def test_resume_without_any_ask_fails_and_creates_no_run(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """没问过就没有会话。

    实测 `adelete_thread` 之后 `Command(resume)` 会**静默开一个新 run**(C6),所以这条路必须在
    应用层就断掉:否则调用方拿到一个「看起来成功了」的空返回,库里多一条没人问过的 run。
    """
    result = await _spawn("b", f"rs-{os.getpid()}-ghost")
    assert result["payload"] is None
    assert result["code"] != 0
    assert "PermissionDenied" in result["stderr"], result["stderr"][-1500:]
    assert await _scalar(
        factory, "select count(*) from platform.trace_run where tenant_id = :t", t=child.TENANT
    ) == 0, "凭空多出一条 run:resume 的入口条件没挡住"


async def test_resume_after_finish_is_a_replay_not_a_second_run(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """答完之后再按一次确认键:回放既有答案,不重跑、不再花钱、不新增 step。"""
    thread = f"rs-{os.getpid()}-twice"
    assert (await _spawn("a", thread))["payload"] is not None
    first = await _spawn("b", thread)
    assert first["payload"]["kind"] == "answered"
    run_id = first["payload"]["run_id"]
    steps = await _scalar(
        factory, "select count(*) from platform.trace_step where run_id = :i", i=run_id
    )

    again = await _spawn("b", thread)
    assert again["code"] == 0, again["stderr"][-2000:]
    assert again["payload"]["kind"] == "replay"
    assert again["payload"]["run_id"] == run_id
    assert again["payload"]["answer"] == first["payload"]["answer"]
    assert await _scalar(
        factory, "select count(*) from platform.trace_step where run_id = :i", i=run_id
    ) == steps, "回放又写了 step:报表里这次问答的耗时开始说谎"


# --------------------------------------------------------------------- TTL 到期
def _stub_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    """编排只消费「候选集 + 逐路 stats + 降级清单」;恢复这条链跟引擎无关,桩越薄越确定是谁的锅。"""

    async def retrieve(query: str, ctx: TenantContext, **kwargs: Any) -> RetrievalResult:
        del ctx, kwargs
        hit = Hit(
            chunk_id="chunk-rs-1",
            score=0.5,
            source_routes={"body"},
            citation=Citation(doc_id="doc-rs", chunk_id="chunk-rs-1", heading_path="检索 > 隔离"),
            text=child.BODY,
            route_scores={"body": 0.5},
        )
        return RetrievalResult(
            query=query,
            hits=[hit],
            stats={
                route: {
                    "count": 1,
                    "fingerprint": "f",
                    "latency_ms": 1,
                    "error": None,
                    "degraded": False,
                }
                for route in ("body", "heading", "dense", "graph")
            },
            graph={"mentions": [], "lines": []},
            fused_count=1,
        )

    monkeypatch.setattr(nodes_mod, "retrieve", retrieve)  # type: ignore[arg-type]


def _runtime(
    factory: async_sessionmaker[AsyncSession], handle: SaverHandle, *, ttl_min: int | None = None
) -> Runtime:
    settings = get_settings()
    if ttl_min is not None:
        settings = settings.model_copy(update={"pending_confirm_ttl_min": ttl_min})
    deps = GraphDeps(
        factory=factory,  # type: ignore[arg-type]
        gateway=child.StubGateway(),  # type: ignore[arg-type]
        kb_id=child.KB,
        settings=settings,
        trace=TraceWriter(factory),
    )
    return Runtime(deps, compile_graph(deps, handle.saver), handle)


async def test_stale_confirm_expires_and_the_thread_is_dropped(
    factory: async_sessionmaker[AsyncSession],
    handle: SaverHandle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TTL 到期:`expired` 落 `trace_run`、检查点删掉、再 resume 必须报错。

    顺序在这里是能测出来的:`adelete_thread` 之后 `Command(resume)` 不报错而是从头跑一遍(实测
    C6)。把到期判断放到动图之后,这条用例会看到 `answered` —— 用户拿到一个没人确认过的答案,
    审计看到一条凭空多出来的 run。`ttl_min=-1` 就是「任何等待都算超时」。
    """
    _stub_retrieve(monkeypatch)
    runtime = _runtime(factory, handle, ttl_min=-1)
    thread = f"rs-{os.getpid()}-ttl"
    with use_context(child.CTX):
        stopped = await runtime.ask(child.QUERY, thread)
        assert stopped.kind == "await_confirm"
        expired = await runtime.resume(thread, {"choice": 1})
    assert expired.kind == "expired"
    assert expired.answer == "", "到期之后还给答案,FR-403 的 TTL 就成了装饰"
    assert await _scalar(
        factory, "select final_status from platform.trace_run where id = :i", i=stopped.run_id
    ) == "expired", "FR-406:到期与拒答是两个口径,必须都落得下来"
    _, schema = _urls()
    assert await _scalar(
        factory,
        f'select count(*) from "{schema}".checkpoints where thread_id = :k',
        k=f"{child.TENANT}:{thread}",
    ) == 0, "过期会话的检查点还留着 = 永久占位"

    with use_context(child.CTX), pytest.raises(IllegalTransition):
        await runtime.resume(thread, {"choice": 1})


async def test_expired_is_not_a_security_event(
    factory: async_sessionmaker[AsyncSession],
    handle: SaverHandle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """到期不是越权。把它写成审计事件,安全面板第一天就学会无视自己的告警。"""
    _stub_retrieve(monkeypatch)
    runtime = _runtime(factory, handle, ttl_min=-1)
    thread = f"rs-{os.getpid()}-ev"
    with use_context(child.CTX):
        await runtime.ask(child.QUERY, thread)
        await runtime.resume(thread, {"choice": 1})
    assert await _scalar(
        factory,
        "select count(*) from platform.security_event where tenant_id = :t",
        t=child.TENANT,
    ) == 0


async def test_sweep_stale_only_touches_its_own_tenant(
    factory: async_sessionmaker[AsyncSession],
    handle: SaverHandle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启动那一次的补漏扫描:本租户过期的会话落 `expired`,别人的一条都不碰。

    扫描点只有两个(#7 启动时 + resume 前的懒判),**不做常驻定时器**,所以它必须便宜:取的是
    「本租户最近 N 条 run」,不是全表扫。
    """
    _stub_retrieve(monkeypatch)
    ctx_b = TenantContext(
        tenant_id=TENANT_B,
        user_id="usr-rs-b",
        roles=(ROLE_READER,),
        kb_scope=frozenset({"kb-rs-b"}),
    )
    runtime = _runtime(factory, handle, ttl_min=-1)
    thread_mine, thread_theirs = f"rs-{os.getpid()}-sweep", f"rs-{os.getpid()}-other"
    with use_context(child.CTX):
        mine = await runtime.ask(child.QUERY, thread_mine)
    with use_context(ctx_b):
        theirs = await runtime.ask(child.QUERY, thread_theirs)

    with use_context(child.CTX):
        swept = await sweep_stale(runtime, limit=20)
    assert thread_key(child.CTX, thread_mine) in swept
    async with factory() as session:
        status_mine = (await session.get(TraceRun, mine.run_id)).final_status
        status_theirs = (await session.get(TraceRun, theirs.run_id)).final_status
    assert status_mine == "expired"
    assert status_theirs != "expired", "扫描越界改写了别人的 run:租户谓词必须落在键上"


async def test_cross_tenant_resume_is_denied_and_recorded(
    factory: async_sessionmaker[AsyncSession],
    handle: SaverHandle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """拿 B 的身份续跑 A 的会话:先留痕再拒。

    物理键里带租户前缀,所以「换个前缀重试」在检查点层就不成立;这条测的是应用层最后一道 ——
    库里查不到这条会话时,不许退化成「那就新开一条」。拒的形状是 `PermissionDenied` 而不是
    「找不到」:`deny()` 先写审计再抛,#7 才能既不给存在性泄露、又不放过探测。
    """
    _stub_retrieve(monkeypatch)
    ctx_b = TenantContext(
        tenant_id=TENANT_B,
        user_id="usr-rs-b",
        roles=(ROLE_READER,),
        kb_scope=frozenset({"kb-rs-b"}),
    )
    runtime = _runtime(factory, handle)
    thread = f"rs-{os.getpid()}-cross"
    with use_context(child.CTX):
        stopped = await runtime.ask(child.QUERY, thread)
    with use_context(ctx_b), pytest.raises(PermissionDenied):
        await runtime.resume(thread, {"choice": 1})

    event = await _scalar(
        factory,
        "select event || '@' || path from platform.security_event"
        " where tenant_id = :t order by created_at limit 1",
        t=TENANT_B,
    )
    assert event == "cross_tenant_read@resume"
    assert stopped.kind == "await_confirm", "被拒的那次不能把 A 的会话一起拖坏"


# --------------------------------------------------------------------- 线程键(纯检查,不起图)
def test_thread_key_is_the_tenant_prefixed_shape() -> None:
    key = thread_key(child.CTX, "th-1")
    assert key == f"{child.TENANT}:th-1"
    assert split_thread_key(key) == (child.TENANT, "th-1")


@pytest.mark.parametrize("bare", ["", "   ", "a:b", "x" * 97])
def test_thread_id_with_a_second_separator_or_wrong_length_is_rejected(bare: str) -> None:
    """冒号是租户前缀的分隔符,谁占它谁就能伪造边界;超长挡在写库之前。"""
    with pytest.raises(IllegalTransition):
        assert_thread_shape(bare)
    with pytest.raises(IllegalTransition):
        thread_key(child.CTX, bare)


def test_thread_key_never_leaks_the_query_or_bodies() -> None:
    """键里只有租户与线程 id:它随 URL 走,正文与答案一个字都不该在里面。"""
    key = thread_key(child.CTX, "th-2")
    for piece in (child.QUERY, child.BODY, child.ANSWER, REFUSAL_TEXT):
        assert piece not in key
