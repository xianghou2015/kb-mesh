"""LangGraph 检查点(PostgresSaver)能力实测:钉死技术实现文档 §7.2「中断可跨重启恢复」的落地前提。

跑法(需要 PG 可连):
    .venv/bin/python scripts/verify_checkpoint.py

为什么编排代码落笔前必须先跑这个:检查点表由 `saver.setup()` 自建、不归 Alembic,而它的
MIGRATIONS 里表名**不带 schema** —— 建到哪张 schema 完全由连接的 `search_path` 第一位决定。
落错地方(`platform`)会让 `alembic check` 要求 DROP 掉这三张表。中断态是否在进程被 SIGTERM
杀掉之前就已经落盘,决定了 FR-403 的验收能不能成立;这类事没法从函数签名读出来。

探针:
    C1 检查点四表落在 checkpoint_schema(默认 lg),platform 下不留同名
    C2 AsyncConnectionPool 复用下并发读不冲突(必需 kwargs 见 make_pool)
    C3 子进程被 SIGTERM 杀死后,新连接池仍能读回中断态(= 跨进程恢复的前提)
    C4 resume 时节点重跑范围(决定 trace_seq 怎么发号、confirm 节点必须零副作用)
       含 C4c:用 mark 文件数 `interrupt()` 之前的语句被执行了几次(顶点返回值只应用一次,看不出来)
    C5 停在 interrupt 时检查点是否已落盘(默认 durability 未显式指定)
    C6 delete_thread 语义(TTL 到期动作依赖它)
    C9 自定义 dataclass(Hit/Citation)进 checkpoint 的 serde 白名单与告警

C7(strict json_schema 带默认值的数组)属模型侧,并进 scripts/verify_models.py。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import operator
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kbmesh.core.config import get_settings  # noqa: E402
from kbmesh.retrieval.contracts import Citation, Hit  # noqa: E402

CKPT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations")
results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


class _LogCatcher(logging.Handler):
    """serde 的「unregistered type」提示走 logging,不是 warnings.warn —— 只能这么接。"""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def capture_logs() -> Iterator[list[str]]:
    root = logging.getLogger()
    catcher = _LogCatcher()
    root.addHandler(catcher)
    try:
        yield catcher.messages
    finally:
        root.removeHandler(catcher)


def psycopg_dsn() -> str:
    """asyncpg 串换成 psycopg 串,并且**剥掉 query 段**(里面的 search_path 会把表建错地方)。"""
    settings = get_settings()
    url = settings.db_url
    if not url:
        raise SystemExit("没有数据库 URL:设 KBMESH_DB_URL")
    return url.replace("postgresql+asyncpg://", "postgresql://", 1).split("?", 1)[0]


def make_serde(*, allow: bool = True) -> JsonPlusSerializer:
    if not allow:
        return JsonPlusSerializer()
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("kbmesh.retrieval.contracts", "Hit"),
            ("kbmesh.retrieval.contracts", "Citation"),
        ]
    )


def make_pool(dsn: str) -> AsyncConnectionPool:
    settings = get_settings()
    return AsyncConnectionPool(
        dsn,
        min_size=1,
        max_size=4,
        open=False,
        name="kbmesh-verify-checkpoint",
        kwargs={
            "autocommit": True,  # setup() 里有 CREATE INDEX CONCURRENTLY,不能在事务里跑
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": f"-c search_path={settings.checkpoint_schema}",
        },
    )


async def open_saver(dsn: str) -> tuple[AsyncConnectionPool, AsyncPostgresSaver]:
    pool = make_pool(dsn)
    await pool.open(wait=True)
    return pool, AsyncPostgresSaver(pool, serde=make_serde())


class ProbeState(TypedDict, total=False):
    executed: Annotated[list[str], operator.add]
    hits: list[Hit]
    resumed: Any


def build_probe_graph(mark: Path | None = None) -> StateGraph:
    """三节点最小图:干活 → interrupt → 收尾。节点写成 async,和编排层的执行栈保持一致。

    `mark` 文件用来观察 `interrupt()` 之前的代码在 resume 时到底重跑几次 —— 编排层的
    trace 发号就靠这个结论,光看 `executed` 列表看不出来(返回值只会被应用一次)。
    """

    async def work(state: ProbeState) -> dict:
        return {
            "executed": ["work"],
            "hits": [
                Hit(
                    chunk_id="chk-1",
                    score=0.9,
                    source_routes={"body"},
                    citation=Citation(doc_id="doc-1", chunk_id="chk-1", heading_path="标题"),
                )
            ],
        }

    async def confirm(state: ProbeState) -> dict:
        if mark is not None:
            mark.write_text(mark.read_text() + "E", encoding="utf-8")
        value = interrupt({"kind": "await_confirm", "candidates": ["改写甲", "改写乙"]})
        return {"executed": ["confirm"], "resumed": value}

    async def finish(state: ProbeState) -> dict:
        return {"executed": ["finish"]}

    builder = StateGraph(ProbeState)
    builder.add_node("work", work)
    builder.add_node("confirm", confirm)
    builder.add_node("finish", finish)
    builder.add_edge(START, "work")
    builder.add_edge("work", "confirm")
    builder.add_edge("confirm", "finish")
    builder.add_edge("finish", END)
    return builder


def spawn_interrupted_child(thread: str, mark: Path) -> tuple[int, dict[str, Any]]:
    """子进程跑到 interrupt → 打印一行 JSON → 自己 SIGTERM。

    故意不给它优雅关连接池的机会:这样父进程还能读到中断态,才是「进程死了但检查点活着」。
    """
    handle = subprocess.Popen(
        [sys.executable, __file__, "--child", thread, "--mark", str(mark)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env={**os.environ},
    )
    line = handle.stdout.readline() if handle.stdout else ""
    code = handle.wait(timeout=60)
    try:
        info = json.loads(line.partition(" ")[2] or "{}")
    except json.JSONDecodeError:
        info = {"raw": line[:200]}
    return code, info


async def run_child(thread: str, mark: Path) -> None:
    dsn = psycopg_dsn()
    pool, saver = await open_saver(dsn)
    await saver.setup()
    graph = build_probe_graph(mark).compile(checkpointer=saver)
    with capture_logs() as logs:
        state = await graph.ainvoke({"executed": []}, {"configurable": {"thread_id": thread}})
    print(
        "PROBE-READY "
        + json.dumps(
            {
                "keys": sorted(state),
                "interrupt": [getattr(one, "value", one) for one in state.get("__interrupt__", [])],
                "logs": [one[:160] for one in logs],
            },
            ensure_ascii=False,
        )
    )
    sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGTERM)


async def main(dsn: str) -> int:
    schema = get_settings().checkpoint_schema
    thread = f"verify-ckpt-{os.getpid()}"
    mark = Path(tempfile.gettempdir()) / f"kbmesh-ckpt-probe-{os.getpid()}.marks"
    mark.write_text("", encoding="utf-8")

    if not schema.isidentifier():
        raise SystemExit(f"checkpoint_schema 不是合法标识符: {schema!r}")

    pool, saver = await open_saver(dsn)
    try:
        # schema 必须先存在:`saver.setup()` 的建表语句不带 schema,全看连接的 search_path
        async with pool.connection() as conn:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        await saver.setup()
        async with pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "select table_schema, table_name from information_schema.tables"
                    " where table_name = any(%s)",
                    (list(CKPT_TABLES),),
                )
            ).fetchall()
        where: dict[str, list[str]] = {}
        for row in rows:
            where.setdefault(row["table_schema"], []).append(row["table_name"])
        in_schema = sorted(where.get(schema, []))
        leaked = sorted(where.get("platform", []))
        report("C1 检查点四表落在 checkpoint_schema", in_schema == sorted(CKPT_TABLES) and not leaked,
               f"{schema}={in_schema} platform={leaked}")

        absent = {"configurable": {"thread_id": f"{thread}-absent"}}
        outs = await asyncio.gather(*(saver.aget_tuple(absent) for _ in range(8)))
        report("C2 连接池并发复用不冲突", all(one is None for one in outs),
               f"pool min=1 max=4, {len(outs)} 次并发读全部无错")

        # C9:自定义 dataclass 进 checkpoint —— 默认 serde 现在放行但会 log 警告,白名单才是正解
        probe = Hit(chunk_id="chk-1", score=0.5, source_routes={"body"})
        bare = make_serde(allow=False)
        with capture_logs() as logs:
            back = bare.loads_typed(bare.dumps_typed(probe))
        report("C9a 默认 serde 反序列化 Hit(现状:放行 + log 警告)", back == probe,
               (next((one[:150] for one in logs if "unregistered" in one), "无日志")) or "无日志")
        listed = make_serde()
        with capture_logs() as logs:
            back = listed.loads_typed(listed.dumps_typed(probe))
        report("C9b allowed_msgpack_modules 收白名单后静默", back == probe and not logs,
               f"round_trip={back == probe} logs={[one[:60] for one in logs]}")

        # C3 + C5:子进程 SIGTERM 后,换一条全新连接池读回中断态
        code, info = await asyncio.to_thread(spawn_interrupted_child, thread, mark)
        enters_at_interrupt = mark.read_text(encoding="utf-8").count("E")
        cfg = {"configurable": {"thread_id": thread}}
        pool2, saver2 = await open_saver(dsn)
        graph = build_probe_graph(mark).compile(checkpointer=saver2)
        try:
            snap = await graph.aget_state(cfg)
            async with pool2.connection() as conn:
                persisted = await (
                    await conn.execute(
                        "select count(*) as c from checkpoints where thread_id = %s", (thread,)
                    )
                ).fetchone()
            payload = list(snap.interrupts) or [
                one for task in snap.tasks for one in task.interrupts
            ]
            report("C5 停在 interrupt 时检查点已落盘(默认 durability)",
                   code == -signal.SIGTERM and snap.next and persisted["c"] >= 1,
                   f"returncode={code} next={snap.next} checkpoints_rows={persisted['c']}")
            report("C3 跨进程读回中断态 + payload 落点", bool(snap.next) and bool(payload),
                   f"snap.interrupts={len(snap.interrupts)} tasks[].interrupts="
                   f"{sum(len(one.interrupts) for one in snap.tasks)} "
                   f"value={payload[0].value if payload else None}")
            report("C3b 子进程侧 ainvoke 返回值带 __interrupt__ 键", "__interrupt__" in info.get("keys", []),
                   f"keys={info.get('keys')} child_logs={info.get('logs')}")

            # C4:resume 看节点重跑范围
            with capture_logs() as logs4:
                state = await graph.ainvoke(Command(resume={"choice": 2, "edited": ""}), cfg)
            executed = state.get("executed", [])
            enters_after_resume = mark.read_text(encoding="utf-8").count("E")
            report("C4 resume 只重跑 confirm(work/finish 不重复)",
                   executed.count("work") == 1 and executed.count("finish") == 1,
                   f"executed={executed} confirm_updates={executed.count('confirm')} "
                   f"resumed_value={state.get('resumed')}")
            report("C4c confirm 节点体重跑两次(返回值只应用一次)",
                   enters_at_interrupt == 1 and enters_after_resume == 2,
                   f"interrupt 前={enters_at_interrupt} resume 后={enters_after_resume}"
                   " → interrupt() 之前的代码必须零副作用,trace 发号不能写在那儿")
            hits = state.get("hits") or []
            report("C4b 跨进程 state 里的 Hit 完整回来",
                   bool(hits) and hits[0].chunk_id == "chk-1" and hits[0].citation is not None,
                   f"logs={[one[:60] for one in logs4]}")

            # C6:delete_thread 之后必须读不到、且行数为 0;再 resume 的行为也记下来
            await saver2.adelete_thread(thread)
            gone = await graph.aget_state(cfg)
            async with pool2.connection() as conn:
                left = await (
                    await conn.execute(
                        "select count(*) as c from checkpoints where thread_id = %s", (thread,)
                    )
                ).fetchone()
            try:
                again = repr(await graph.ainvoke(Command(resume={"choice": 1}), cfg))[:80]
            except Exception as exc:  # noqa: BLE001 — 探针要的就是「到底是哪种失败」
                again = f"{type(exc).__name__}: {exc}"[:80]
            report("C6 delete_thread 清空该 thread 检查点", not gone.next and left["c"] == 0,
                   f"values={dict(gone.values)} rows={left['c']} next={gone.next} "
                   f"resume_after_delete={again}")
        finally:
            await pool2.close()
    finally:
        await pool.close()

    mark.unlink(missing_ok=True)
    failed = [one for one in results if not one[1]]
    print(f"\n=== {len(results) - len(failed)}/{len(results)} 通过 ===")
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    print("结论请回填技术文档 §7.2 与 §4.1(检查点表不归 Alembic,由 saver.setup() 自管)。")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", default=None, help="内部用:子进程跑到 interrupt 后自杀")
    parser.add_argument("--mark", default=None, help="内部用:子进程记录 confirm 重跑次数的文件")
    args = parser.parse_args()
    if args.child:
        asyncio.run(run_child(args.child, Path(args.mark)))
        raise SystemExit(0)
    raise SystemExit(asyncio.run(main(psycopg_dsn())))
