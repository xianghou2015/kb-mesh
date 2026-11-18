"""Celery 任务薄壳(§5.1 的投递侧)。

这里刻意只做三件事:起一条自己的事件循环、调 `pipeline.run_stage`、把结果交给 backend。
业务逻辑一律不进 tasks:阶段函数在验收测试里能被直接 `await`,套上 Celery 只是换了个触发方式,
「测试跑的是同一条代码」这件事比省一层封装重要。
"""
from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from celery import chain

from kbmesh.core.celery_app import celery_app
from kbmesh.core.config import get_settings
from kbmesh.db.base import make_engine, set_engine
from kbmesh.ingest import lease, pipeline

settings = get_settings()


def run_coro(coro: Coroutine[Any, Any, Any]) -> Any:
    """一条任务一条事件循环 + 一个引擎,跑完即弃。

    asyncpg 的连接**绑死在创建它的事件循环上**:进程级缓存的引擎在第二个任务里会撞
    `AttachmentError: ... is attached to a different loop`。所以这里不用全局引擎,
    每次 `asyncio.run()` 前重建、结束后 dispose。
    """

    async def main() -> Any:
        engine = make_engine()
        set_engine(engine)
        try:
            return await coro
        finally:
            await engine.dispose()

    return asyncio.run(main())


@celery_app.task(
    name="kbmesh.ingest.stage",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=2,
)
def run_stage(self, stage: str, doc_id: str) -> dict[str, Any]:
    if stage not in pipeline.STAGE_FUNCS:
        raise ValueError(f"未知阶段: {stage!r}")
    return run_coro(pipeline.run_stage(stage, doc_id))


def task_id_for(doc_id: str, index_version: int, stage: str) -> str:
    """§5.3 的任务幂等键:重索引靠 `index_version` 递增换一套 id,不会和新旧两轮撞车。"""
    return f"{doc_id}:{index_version}:{stage}"


def submit_stage(stage: str, doc_id: str, index_version: int) -> dict[str, Any]:
    """投递单个阶段;同一 `index_version` 已完成过就直接短路返回已有结果(§5.3 任务层)。"""
    key = lease.task_key(doc_id, index_version, stage)
    done = lease.read_done(key)
    if done is not None:
        return {"submitted": False, "short_circuit": done}
    result = run_stage.apply_async(
        args=[stage, doc_id],
        task_id=task_id_for(doc_id, index_version, stage),
        queue=settings.celery_queue,
    )
    return {"submitted": True, "task_id": result.id}


def submit_document(doc_id: str, index_version: int) -> dict[str, Any]:
    """整篇文档:六个阶段按 §5.1 的依赖串成 chain,前一步失败就不投后一步。"""
    key = lease.task_key(doc_id, index_version, "parse")
    done = lease.read_done(key)
    if done is not None:
        return {"submitted": False, "short_circuit": done}
    workflow = chain(
        run_stage.si(stage, doc_id).set(
            task_id=task_id_for(doc_id, index_version, stage),
            queue=settings.celery_queue,
        )
        for stage in pipeline.PIPELINE_STAGES
    )
    return {"submitted": True, "group_id": workflow.apply_async().id}
