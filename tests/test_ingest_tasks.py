"""投递侧(§5.1 + §5.3 任务层):任务 id、chain 顺序、已完成短路、worker 被杀后的安全性。

不连 broker:`apply_async` 换成记录器,验的是**投递决策**(该不该投、投出去叫什么)。
执行侧的幂等在 `test_pipeline_idempotency.py` 里跑真库验,两边加起来才是完整的 §5.3。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import fakeredis
import pytest

from kbmesh.core.celery_app import make_celery
from kbmesh.core.config import get_settings
from kbmesh.ingest import lease, pipeline, tasks

DOC = "doc-task-1"
VERSION = 3
QUEUE = get_settings().celery_queue


@pytest.fixture(autouse=True)
def rdb():
    """整模块都不许碰真 Redis:短路读的是这把内存锁。"""
    lease.set_client(fakeredis.FakeStrictRedis(decode_responses=True))
    try:
        yield lease.client()
    finally:
        lease.set_client(None)


class FakeChain:
    """接住 chain 的签名列表,不碰 broker。"""

    last: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 1 and not isinstance(args[0], dict):
            self.sigs = list(args[0])  # 真实调用传的是生成器
        else:
            self.sigs = list(args)
        FakeChain.last = self

    def apply_async(self, *a: Any, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(id="group-1")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """把 `run_stage.apply_async` 换成记录器:投出去的 args / task_id / queue 都留痕。"""
    seen: list[dict[str, Any]] = []

    def fake_apply_async(args=None, kwargs=None, task_id=None, queue=None, **_ignored: Any):
        seen.append({"args": list(args or []), "task_id": task_id, "queue": queue})
        return SimpleNamespace(id=task_id)

    monkeypatch.setattr(tasks.run_stage, "apply_async", fake_apply_async)
    monkeypatch.setattr(tasks, "chain", FakeChain)
    return seen


def test_task_id_carries_doc_version_and_stage() -> None:
    assert tasks.task_id_for(DOC, VERSION, "parse") == f"{DOC}:{VERSION}:parse"
    assert tasks.task_id_for(DOC, VERSION + 1, "parse") != tasks.task_id_for(DOC, VERSION, "parse")


def test_submit_stage_uses_the_idempotent_task_id(recorder: list[dict[str, Any]]) -> None:
    out = tasks.submit_stage("embed", DOC, VERSION)
    assert out == {"submitted": True, "task_id": f"{DOC}:{VERSION}:embed"}
    assert recorder == [{"args": ["embed", DOC], "task_id": f"{DOC}:{VERSION}:embed", "queue": QUEUE}]


def test_submit_stage_short_circuits_on_the_done_mark(recorder: list[dict[str, Any]]) -> None:
    """同一 index_version 的同一阶段已成功:重投直接返回旧结果,一个字节都不再写。"""
    lease.mark_done(lease.task_key(DOC, VERSION, "embed"), {"stage": "embed", "counts": {"upserted": 3}})
    assert tasks.submit_stage("embed", DOC, VERSION) == {
        "submitted": False,
        "short_circuit": {"stage": "embed", "counts": {"upserted": 3}},
    }
    assert recorder == []


def test_document_chain_runs_the_dag_in_order(recorder: list[dict[str, Any]]) -> None:
    assert tasks.submit_document(DOC, VERSION)["submitted"] is True
    sigs = FakeChain.last.sigs
    assert [s["args"][0] for s in sigs] == list(pipeline.PIPELINE_STAGES)
    assert [s["options"]["task_id"] for s in sigs] == [
        tasks.task_id_for(DOC, VERSION, stage) for stage in pipeline.PIPELINE_STAGES
    ]
    assert {s["task"] for s in sigs} == {"kbmesh.ingest.stage"}
    assert recorder == [], "chain 整体投递,不该再单独投一遍首阶段"


def test_document_chain_skips_submission_when_parse_is_done(recorder: list[dict[str, Any]]) -> None:
    lease.mark_done(lease.task_key(DOC, VERSION, "parse"), {"stage": "parse"})
    assert tasks.submit_document(DOC, VERSION)["submitted"] is False
    assert recorder == []


def test_a_new_version_resubmits_everything(recorder: list[dict[str, Any]]) -> None:
    """重索引递增 `index_version`:旧版本的「已完成」不许把新一轮一起短路掉。"""
    lease.mark_done(lease.task_key(DOC, VERSION, "parse"), {"stage": "parse"})
    tasks.submit_document(DOC, VERSION + 1)
    assert [s["args"][0] for s in FakeChain.last.sigs] == list(pipeline.PIPELINE_STAGES)


def test_unknown_stage_is_rejected_before_any_io() -> None:
    with pytest.raises(ValueError, match="未知阶段"):
        tasks.run_stage("nope", DOC)


def test_worker_kill_safety_is_configured_not_assumed() -> None:
    """§5.4:worker 被杀 = 任务重投,所以 acks_late + reject_on_worker_lost 是正确性依赖。"""
    conf = tasks.celery_app.conf
    assert conf.task_acks_late is True
    assert conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1, "预取 >1 会让一个 worker 囤住整批长任务"


def test_visibility_timeout_exceeds_the_hard_time_limit() -> None:
    """默认 360s 的可见性超时会把跑 8 分钟的阶段当成丢掉而重投:数据靠幂等保命,钱白烧。"""
    conf = tasks.celery_app.conf
    visibility = conf.broker_transport_options["visibility_timeout"]
    assert visibility > conf.task_time_limit >= conf.task_soft_time_limit


def test_stage_retries_are_bounded_with_backoff() -> None:
    assert tasks.run_stage.max_retries == 2
    assert Exception in (tasks.run_stage.autoretry_for or ())
    assert tasks.run_stage.retry_backoff is True


def test_queue_name_comes_from_settings() -> None:
    assert tasks.celery_app.conf.task_default_queue == QUEUE


def test_celery_refuses_to_start_without_a_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    blank = SimpleNamespace(
        celery_broker_url="",
        celery_result_backend="",
        redis_url="",
        celery_queue=QUEUE,
    )
    monkeypatch.setattr("kbmesh.core.celery_app.get_settings", lambda: blank)
    with pytest.raises(RuntimeError, match="KBMESH_CELERY_BROKER_URL"):
        make_celery()
