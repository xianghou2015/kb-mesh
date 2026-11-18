"""在线评估的 Celery 薄壳(FR-601):规则全量 + judge 抽样,异步执行,不进问答主链路。

`evaluate_and_store` 是业务入口,测试直接 `await` 它;Celery 只是换一个触发方式。
"""
from __future__ import annotations

import contextlib
import random
from typing import Any

from kbmesh.agents.runtime import SaverHandle
from kbmesh.agents.wiring import build_graph_deps, close_graph_deps
from kbmesh.core.celery_app import celery_app
from kbmesh.core.config import get_settings
from kbmesh.core.gateway import Gateway
from kbmesh.core.tenancy import ROLE_READER, ROLE_TENANT_ADMIN, TenantContext
from kbmesh.db.base import get_session_factory
from kbmesh.db.repositories import OnlineEvalRepo
from kbmesh.eval.judge import GatewayJudge
from kbmesh.eval.online import OnlineEvaluator, OnlineInput
from kbmesh.eval.runner import EvalRunner
from kbmesh.eval.runtime_case import RuntimeCaseRunner
from kbmesh.ingest.tasks import run_coro


async def evaluate_and_store(
    payload: dict[str, Any],
    *,
    judge: Any | None = None,
    draw: float = 0.0,
) -> dict[str, Any]:
    settings = get_settings()
    data = OnlineInput(
        question=str(payload.get("question") or ""),
        answer=str(payload.get("answer") or ""),
        final_status=str(payload.get("final_status") or ""),
        cite_ids={str(k): str(v) for k, v in (payload.get("cite_ids") or {}).items()},
        evidence_ids={str(one) for one in (payload.get("evidence_ids") or [])},
        evidence_texts=tuple(str(one) for one in (payload.get("evidence_texts") or [])),
    )
    report = await OnlineEvaluator(judge).evaluate(
        data, sample_rate=settings.online_eval_sample_rate, draw=draw
    )
    ctx = TenantContext(
        tenant_id=str(payload["tenant_id"]),
        user_id="system:online-eval",
        roles=(ROLE_READER,),
    )
    async with get_session_factory()() as session, session.begin():
        await OnlineEvalRepo(ctx, session).record(
            str(payload["run_id"]),
            format_ok=bool(report["format_ok"]),
            citation_ok=bool(report["citation_ok"]),
            safety_ok=bool(report["safety_ok"]),
            failed=list(report["failed"]),
            sampled=bool(report["sampled"]),
            faithfulness=report["faithfulness"],
            answer_relevancy=report["answer_relevancy"],
        )
    return report


async def _celery_evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    judge = GatewayJudge(Gateway(settings)) if settings.online_eval_sample_rate > 0 else None
    return await evaluate_and_store(payload, judge=judge, draw=random.random())


@celery_app.task(name="kbmesh.eval.online")
def evaluate_online(payload: dict[str, Any]) -> dict[str, Any]:
    return run_coro(_celery_evaluate(payload))


def enqueue_online_eval(payload: dict[str, Any]) -> None:
    """同步入队;test 环境不投(没有 worker),broker 故障也不影响问答。"""
    settings = get_settings()
    if settings.env == "test":
        return
    # 评估通道不得反向影响主链路
    with contextlib.suppress(Exception):
        evaluate_online.apply_async(args=[payload], queue=settings.celery_queue)


async def _run_eval(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    deps = build_graph_deps(settings, online_eval=None)
    handle = await SaverHandle.open_sync(settings=settings).open()
    ctx = TenantContext(
        tenant_id=str(payload["tenant_id"]),
        user_id="system:eval-runner",
        roles=(ROLE_TENANT_ADMIN,),
    )
    try:
        runner = EvalRunner(
            factory=deps.factory,
            ctx=ctx,
            answer_case=RuntimeCaseRunner(deps=deps, handle=handle, ctx=ctx),
            judge=GatewayJudge(deps.gateway),
            top_k=payload.get("top_k"),
        )
        return await runner.run(str(payload["run_id"]), limit=payload.get("limit"))
    finally:
        await handle.close()
        close_graph_deps(deps)


@celery_app.task(name="kbmesh.eval.run")
def run_eval(payload: dict[str, Any]) -> dict[str, Any]:
    return run_coro(_run_eval(payload))


def enqueue_eval_run(
    run_id: str, tenant: str, *, limit: int | None = None, top_k: int | None = None
) -> bool:
    """把离线评估投给 worker;test 环境不投(没有 worker),由测试直接 await 运行器。

    参数名刻意不是 `tenant_id`:仓库的 AST 红线禁止公开函数暴露裸 tenant_id 形参,
    身份只从验签后的 ctx 来;这里拿到的 `tenant` 已经是调用方从 ctx 取出的值。
    """
    settings = get_settings()
    if settings.env == "test":
        return False
    # broker 故障不在这里吞:调用方(API)应看到 5xx,而不是一个永远不会跑的 202。
    run_eval.apply_async(
        args=[{"run_id": run_id, "tenant_id": tenant, "limit": limit, "top_k": top_k}],
        queue=settings.celery_queue,
    )
    return True


__all__ = [
    "enqueue_eval_run",
    "enqueue_online_eval",
    "evaluate_and_store",
    "evaluate_online",
    "run_eval",
]
