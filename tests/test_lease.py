"""§5.3 任务层幂等锁 + §5.4「持锁区间内不许有网络调用」。

用 fakeredis(带 lupa,Lua 真跑):这把锁的语义全靠 `eval` 的比较-再删除,
mock 掉 eval 就等于把要验的东西假设成成立。
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import fakeredis
import pytest

from kbmesh.ingest import lease


@pytest.fixture
def rdb():
    """换成内存 Redis,测完清空全局:别的测试不许摸到这个假客户端。"""
    fake = fakeredis.FakeStrictRedis(decode_responses=True)
    lease.set_client(fake)
    try:
        yield fake
    finally:
        lease.set_client(None)


def test_acquire_is_exclusive(rdb) -> None:
    first = lease.acquire("k")
    assert first is not None and first.token
    assert lease.acquire("k") is None


def test_release_frees_the_lock(rdb) -> None:
    taken = lease.acquire("k")
    assert lease.release(taken) is True
    assert lease.acquire("k") is not None


def test_release_is_idempotent(rdb) -> None:
    taken = lease.acquire("k")
    assert lease.release(taken) is True
    assert lease.release(taken) is False


def test_a_stale_holder_cannot_delete_a_live_lock(rdb) -> None:
    """A 超时后 B 拿到同名锁;A 收尾时的 release 必须只删自己那把。

    没有 Lua 比较,这里就是「A 把 B 的锁删了」⇒ 同一文档两个 worker 并行写图。
    """
    stale = lease.acquire("k", ttl=60)
    rdb.delete("k")  # 模拟 A 的租约到期
    fresh = lease.acquire("k", ttl=60)
    assert fresh is not None and fresh.token != stale.token

    assert lease.release(stale) is False
    assert rdb.get("k") == fresh.token, "过期持有者把别人的锁删掉了"
    assert lease.release(fresh) is True
    assert rdb.get("k") is None


def test_lock_carries_a_ttl(rdb) -> None:
    """worker 被 OOM 杀掉时没人 release:TTL 是唯一让它重新可处理的东西。"""
    lease.acquire("k", ttl=123)
    ttl = rdb.ttl("k")
    assert 0 < ttl <= 123


def test_ttl_defaults_cover_a_whole_document(rdb) -> None:
    assert lease.acquire("k").ttl >= 300


def test_task_key_separates_stage_doc_and_version() -> None:
    base = lease.task_key("doc-1", 1, "parse")
    assert base == "kbmesh:lease:parse:doc-1:1"
    assert lease.task_key("doc-1", 1, "embed") != base
    assert lease.task_key("doc-1", 2, "parse") != base
    assert lease.task_key("doc-2", 1, "parse") != base


def test_version_bump_does_not_inherit_the_old_run_mark(rdb) -> None:
    """改切片策略会 bump `index_version`:旧版本的「已完成」不许让新索引整段跳过。"""
    old = lease.task_key("doc-1", 1, "parse")
    lease.mark_done(old, {"stage": "parse"})
    assert lease.read_done(lease.task_key("doc-1", 2, "parse")) is None
    assert lease.acquire(lease.task_key("doc-1", 2, "parse")) is not None


def test_done_roundtrip_keeps_chinese_and_non_json_values(rdb) -> None:
    key = lease.task_key("doc-1", 1, "extract")
    assert lease.read_done(key) is None
    report = {
        "stage": "extract",
        "counts": {"三元组": 3},
        "warnings": ["跳过"],
        "at": 1.5,
    }
    lease.mark_done(key, report)
    assert lease.read_done(key) == report
    assert 0 < rdb.ttl(f"{key}:done") <= 7 * 24 * 3600


def test_done_mark_expires(rdb) -> None:
    key = "kbmesh:lease:parse:doc-1:1"
    lease.mark_done(key, {"stage": "parse"}, ttl=60)
    assert rdb.ttl(f"{key}:done") == 60


def test_client_never_falls_back_to_a_quiet_default(monkeypatch) -> None:
    """没配 Redis 时报错,而不是「锁永远抢得到」:静默降级会同时废掉幂等与并发保护。"""
    lease.set_client(None)
    monkeypatch.setattr(lease, "get_settings", lambda: SimpleNamespace(redis_url=""))
    with pytest.raises(RuntimeError, match="KBMESH_REDIS_URL"):
        lease.client()


def test_module_exposes_no_lock_scope() -> None:
    """§5.4 的结构化写法:模块里故意没有 `with lease:` 那种作用域。

    有上下文管理器的地方,迟早有人把模型调用写进锁里。
    """
    source = Path(inspect.getfile(lease)).read_text(encoding="utf-8")
    assert "__enter__" not in source
    assert "contextmanager" not in source


def test_no_pipeline_stage_runs_inside_a_lock_scope() -> None:
    """业务调用必须落在 `acquire()` 与 `release()` 之间,而不是被包在某个 with 里。"""
    src = Path(__file__).resolve().parents[1] / "src" / "kbmesh" / "ingest"
    for path in sorted(src.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"with\s+lease\.", text), path.name
        assert not re.search(r"with\s+\w*lease\w*\(", text), path.name
