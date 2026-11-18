"""SSE 通道(§7.4 / #7):九类事件,与观测分通道互不阻塞。

通道边界:
- 图执行与留痕(节点体内的 `TraceWriter`,一步一事务)在 **producer** 任务里;
- HTTP 消费端只从无界队列取帧。客户端再慢、再断,producer 的留痕提交都不会等它 ——
  这是「互不阻塞」的落点;反过来,SSE 帧从图的 updates 派生,不经过观测通道,
  观测关掉(`trace.enabled=False`)问答照常流出。
- producer 的异常收敛为最后一帧 `error`,不把栈曝给客户端;连接断开时 producer 被
  cancel,检查点停在当前节点 —— 恢复语义与 FR-403 相同。
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from kbmesh.agents.runtime import StreamEvent
from kbmesh.core.errors import KbmError


def format_frame(frame: StreamEvent) -> dict[str, str]:
    """sse-starlette 的帧形状:`event:` + `data:`(JSON,非 ASCII 不转义)。"""
    return {"event": frame.event, "data": json.dumps(frame.data, ensure_ascii=False, default=str)}


async def pump(
    agen: AsyncIterator[StreamEvent], first: StreamEvent | None = None
) -> AsyncIterator[dict[str, str]]:
    """producer(图)→ 无界队列 → consumer(SSE 响应)。

    `first` 是路由层预检时已经取出的那一帧(见 chat 路由):4xx 语义靠「流开始前抛」才成立。
    """
    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
    producer = asyncio.create_task(_produce(agen, queue, first))
    try:
        while (frame := await queue.get()) is not None:
            yield format_frame(frame)
    finally:
        if not producer.done():
            producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        aclose = getattr(agen, "aclose", None)
        if aclose is not None:
            await aclose()


async def _produce(
    agen: AsyncIterator[StreamEvent],
    queue: asyncio.Queue[StreamEvent | None],
    first: StreamEvent | None,
) -> None:
    try:
        if first is not None:
            await queue.put(first)
        async for frame in agen:
            await queue.put(frame)
    except KbmError as exc:
        await queue.put(StreamEvent("error", {"code": type(exc).__name__, "message": str(exc)}))
    except asyncio.CancelledError:
        raise
    except Exception:
        # 未知异常不给客户端看栈;排障走日志与 trace,不走响应体
        await queue.put(StreamEvent("error", {"code": "internal_error", "message": "问答执行失败"}))
    finally:
        await queue.put(None)


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """解析 SSE 响应体为 `(event, data)` 列表 —— 测试与排障共用一个口径。"""
    out: list[tuple[str, dict[str, Any]]] = []
    # sse-starlette 按 SSE 规范发 CRLF;只按 "\n\n" 切会把整段响应当成一帧(每行覆盖后
    # 只剩最后一个 event/data)。先归一化行尾,测试与排障才看到真实帧序列。
    for block in body.replace("\r\n", "\n").split("\n\n"):
        event = ""
        raw = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                raw = line.removeprefix("data: ").strip()
        if event:
            out.append((event, json.loads(raw) if raw else {}))
    return out
