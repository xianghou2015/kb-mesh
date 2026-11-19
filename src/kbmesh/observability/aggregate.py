"""按租户聚合观测(FR-704):量、P50/P95、token、成本、拒答率、降级率、路命中分布。

纯函数只吃 `trace_run` / `trace_step` 的形状,不连库 —— 租户谓词由 `TraceRepo` 下推,
聚合层再按 `ctx.tenant_id` 过滤一次都不做,避免两套口径。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ROUTES: tuple[str, ...] = ("body", "heading", "dense", "graph")


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值百分位;空集返回 0 —— 报表里「没有样本」不是 500。"""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (max(0.0, min(100.0, float(p))) / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _status_counts(runs: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {"answered": 0, "refused": 0, "error": 0, "expired": 0}
    for run in runs:
        status = str(_get(run, "final_status", "") or "")
        if status in out:
            out[status] += 1
    return out


def _route_hits(steps_by_run: Mapping[str, Sequence[Any]]) -> dict[str, int]:
    hits = {route: 0 for route in ROUTES}
    for steps in steps_by_run.values():
        seen: set[str] = set()
        for step in steps:
            stage = str(_get(step, "stage", "") or "")
            if not stage.startswith("retrieve_"):
                continue
            route = stage.removeprefix("retrieve_")
            if route not in hits:
                continue
            payload = _get(step, "payload", {}) or {}
            if int(payload.get("count") or 0) > 0:
                seen.add(route)
        for route in seen:
            hits[route] += 1
    return hits


def _token_breakdown(
    steps_by_run: Mapping[str, Sequence[Any]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    by_stage: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    for steps in steps_by_run.values():
        for step in steps:
            token_in = int(_get(step, "token_in", 0) or 0)
            token_out = int(_get(step, "token_out", 0) or 0)
            stage = str(_get(step, "stage", "") or "")
            if stage:
                bucket = by_stage.setdefault(stage, {"token_in": 0, "token_out": 0})
                bucket["token_in"] += token_in
                bucket["token_out"] += token_out
            model = str(_get(step, "model", "") or "")
            if model:
                bucket = by_model.setdefault(model, {"token_in": 0, "token_out": 0})
                bucket["token_in"] += token_in
                bucket["token_out"] += token_out
    return by_stage, by_model


def _stage_latency(steps_by_run: Mapping[str, Sequence[Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for steps in steps_by_run.values():
        for step in steps:
            stage = str(_get(step, "stage", "") or "")
            if not stage:
                continue
            buckets.setdefault(stage, []).append(float(_get(step, "latency_ms", 0) or 0))
    return {
        stage: {"p50": percentile(values, 50), "p95": percentile(values, 95)}
        for stage, values in sorted(buckets.items())
    }


def summarize_runs(
    runs: Sequence[Any],
    *,
    steps_by_run: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """一次窗口内的租户视图。`runs` 必须已经按租户过滤(调用方负责)。"""
    steps_by_run = steps_by_run or {}
    total = len(runs)
    statuses = _status_counts(runs)
    latencies = [float(_get(run, "latency_ms", 0) or 0) for run in runs]
    token_in = sum(int(_get(run, "token_in", 0) or 0) for run in runs)
    token_out = sum(int(_get(run, "token_out", 0) or 0) for run in runs)
    cost = sum(float(_get(run, "cost_usd", 0) or 0) for run in runs)
    degraded = sum(1 for run in runs if bool(_get(run, "degraded", False)))
    tokens_by_stage, tokens_by_model = _token_breakdown(steps_by_run)

    def rate(n: int) -> float:
        return (n / total) if total else 0.0

    route_hits = _route_hits(steps_by_run)
    route_alerts = [r for r in ROUTES if total > 0 and route_hits[r] == 0]

    return {
        "runs": total,
        **statuses,
        "refusal_rate": rate(statuses["refused"]),
        "error_rate": rate(statuses["error"]),
        "expired_rate": rate(statuses["expired"]),
        "degraded_rate": rate(degraded),
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "token_in": token_in,
        "token_out": token_out,
        "cost_usd": cost,
        "route_hits": route_hits,
        "route_alerts": route_alerts,
        "stage_latency_ms": _stage_latency(steps_by_run),
        "tokens_by_stage": tokens_by_stage,
        "tokens_by_model": tokens_by_model,
    }
