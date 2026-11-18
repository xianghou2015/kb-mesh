"""Celery 应用(§5 离线索引的运行时)。

broker 与 backend 都落在 Redis(独立 db,见 AGENTS「中间件落点」);
`task_acks_late` + 阶段幂等 = worker 被杀之后重投是安全的,这不是可选优化而是正确性依赖。

`visibility_timeout` 必须大于最慢一个阶段的耗时:默认 360s 时,一条跑 8 分钟的任务会被 broker
当成丢掉而重投一次,同一阶段并发改写同一篇文档 —— 幂等键能保住数据,但白烧一次模型。
"""
from __future__ import annotations

from celery import Celery

from kbmesh.core.config import get_settings


def make_celery(
    *, include: tuple[str, ...] = ("kbmesh.ingest.tasks", "kbmesh.eval.tasks")
) -> Celery:
    settings = get_settings()
    broker = settings.celery_broker_url or settings.redis_url
    backend = settings.celery_result_backend or settings.redis_url
    if not broker:
        raise RuntimeError("KBMESH_CELERY_BROKER_URL / KBMESH_REDIS_URL 未设置:worker 没有 broker 可连")

    app = Celery("kbmesh", broker=broker, backend=backend, include=list(include))
    app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        task_time_limit=1800,
        task_soft_time_limit=1500,
        task_default_queue=settings.celery_queue,
        broker_transport_options={"visibility_timeout": 2400},
        result_expires=7 * 24 * 3600,
        timezone="Asia/Shanghai",
        enable_utc=True,
    )
    return app


celery_app = make_celery()
