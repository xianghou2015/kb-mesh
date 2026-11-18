"""Redis 租约与任务幂等锁(§5.3 任务层 + §5.4 三段式)。

这里只放**短临界区**的构件:抢标记、比对 token 释放、写「已完成」标记。
规则是持锁区间内不允许任何网络调用,所以 `acquire()` 的返回值必须立刻交还出去 ——
模块里故意没有 `with lease:` 包住业务代码的用法,那种写法迟早会把 LLM 调用装进锁里。

Lua 比较再删除:否则 A 的租约过期后 B 拿走同名 key,A 收尾时会把 B 的锁删掉。
"""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass

import redis

from kbmesh.core.config import get_settings

#: 只删自己那把锁
_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

_client: redis.Redis | None = None


def client() -> redis.Redis:
    global _client
    if _client is None:
        url = get_settings().redis_url
        if not url:
            raise RuntimeError("KBMESH_REDIS_URL 未设置:离线索引的幂等锁没有落点")
        _client = redis.Redis.from_url(url, decode_responses=True)
    return _client


def set_client(new: redis.Redis | None) -> None:
    """测试/CLI 钩子:换成 fakeredis 或另一台库,别改动生产 Redis。"""
    global _client
    if new is not None:
        _client = new
    elif _client is not None:
        _client.close()
        _client = None


@dataclass(frozen=True)
class Lease:
    key: str
    token: str
    ttl: int


def acquire(key: str, *, ttl: int = 900) -> Lease | None:
    """SET NX EX:抢到返回 Lease,没抢到返回 None(说明同一条目正被别的 worker 处理)。"""
    token = secrets.token_hex(16)
    ok = client().set(key, token, nx=True, ex=ttl)
    return Lease(key, token, ttl) if ok else None


def release(lease: Lease) -> bool:
    return bool(client().eval(_RELEASE, 1, lease.key, lease.token))


def task_key(doc_id: str, index_version: int, stage: str) -> str:
    """§5.3 的任务幂等键:同一 index_version 的同一阶段只有一份锁。"""
    return f"kbmesh:lease:{stage}:{doc_id}:{index_version}"


def mark_done(key: str, result: dict, *, ttl: int = 7 * 24 * 3600) -> None:
    client().set(f"{key}:done", json.dumps(result, ensure_ascii=False, default=str), ex=ttl)


def read_done(key: str) -> dict | None:
    raw = client().get(f"{key}:done")
    return json.loads(raw) if raw else None
