"""#9 可观测与成本:单价表版本化 + 按租户聚合口径。

聚合是纯函数,不连库;成本口径必须能单测到「哪一段 token × 哪一档单价」,
否则换模型时历史成本会跟着新单价一起漂。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kbmesh.observability.aggregate import percentile, summarize_runs
from kbmesh.observability.pricing import (
    DEFAULT_PRICE_TABLE,
    PRICE_TABLE_VERSION,
    ModelPrice,
    PriceTable,
    estimate_cost,
)


# --------------------------------------------------------------------- 成本
def test_estimate_cost_sums_token_and_rerank_prices() -> None:
    table = PriceTable(
        version="test-v1",
        prices={
            "chat-a": ModelPrice(input_per_1k=0.01, output_per_1k=0.02),
            "rerank-a": ModelPrice(rerank_per_call=0.003),
        },
    )
    usage = [
        {"model": "chat-a", "token_in": 1000, "token_out": 500},
        {"model": "chat-a", "token_in": 200, "token_out": 0},
    ]
    # 1000/1000*0.01 + 500/1000*0.02 + 200/1000*0.01 + 2*0.003
    assert estimate_cost(usage, rerank_calls=2, rerank_model="rerank-a", table=table) == pytest.approx(
        0.01 + 0.01 + 0.002 + 0.006
    )


def test_unknown_model_contributes_zero_but_does_not_crash() -> None:
    usage = [{"model": "not-in-table", "token_in": 1000, "token_out": 1000}]
    assert estimate_cost(usage, table=DEFAULT_PRICE_TABLE) == 0.0


def test_default_table_is_versioned() -> None:
    assert PRICE_TABLE_VERSION
    assert DEFAULT_PRICE_TABLE.version == PRICE_TABLE_VERSION


def test_rerank_calls_without_model_price_are_zero() -> None:
    table = PriceTable(version="v", prices={})
    assert estimate_cost([], rerank_calls=10, rerank_model="missing", table=table) == 0.0


# --------------------------------------------------------------------- 百分位
def test_percentile_empty_is_zero() -> None:
    assert percentile([], 95) == 0.0


def test_percentile_linear_interpolation() -> None:
    values = [10, 20, 30, 40]
    assert percentile(values, 50) == pytest.approx(25.0)
    assert percentile(values, 95) == pytest.approx(38.5)
    assert percentile([5], 95) == pytest.approx(5.0)


# --------------------------------------------------------------------- 聚合
def _run(**over: object) -> SimpleNamespace:
    base = {
        "id": "run",
        "final_status": "answered",
        "degraded": False,
        "latency_ms": 100,
        "token_in": 10,
        "token_out": 5,
        "cost_usd": 0.01,
    }
    base.update(over)
    return SimpleNamespace(**base)


def _step(
    seq: int,
    stage: str,
    *,
    latency_ms: int = 10,
    token_in: int = 0,
    token_out: int = 0,
    model: str = "",
    **payload: object,
) -> SimpleNamespace:
    return SimpleNamespace(
        seq=seq,
        stage=stage,
        payload=payload,
        latency_ms=latency_ms,
        token_in=token_in,
        token_out=token_out,
        model=model,
    )


def test_summarize_runs_covers_rates_latency_tokens_and_cost() -> None:
    runs = [
        _run(id="a", final_status="answered", latency_ms=100, token_in=10, token_out=5, cost_usd=0.01),
        _run(id="b", final_status="refused", latency_ms=200, token_in=20, token_out=0, cost_usd=0.02),
        _run(
            id="c",
            final_status="error",
            latency_ms=300,
            degraded=True,
            token_in=0,
            token_out=0,
            cost_usd=0.0,
        ),
        _run(id="d", final_status="expired", latency_ms=400, token_in=0, token_out=0, cost_usd=0.0),
    ]
    stats = summarize_runs(runs, steps_by_run={})
    assert stats["runs"] == 4
    assert stats["answered"] == 1
    assert stats["refused"] == 1
    assert stats["error"] == 1
    assert stats["expired"] == 1
    assert stats["refusal_rate"] == pytest.approx(0.25)
    assert stats["error_rate"] == pytest.approx(0.25)
    assert stats["expired_rate"] == pytest.approx(0.25)
    assert stats["degraded_rate"] == pytest.approx(0.25)
    assert stats["latency_ms"]["p50"] == pytest.approx(250.0)
    assert stats["latency_ms"]["p95"] == pytest.approx(385.0)
    assert stats["token_in"] == 30 and stats["token_out"] == 5
    assert stats["cost_usd"] == pytest.approx(0.03)


def test_summarize_runs_counts_route_hits_from_retrieve_steps() -> None:
    runs = [_run(id="a"), _run(id="b")]
    steps = {
        "a": [
            _step(1, "retrieve_body", count=5),
            _step(2, "retrieve_heading", count=0),
            _step(3, "retrieve_dense", count=8),
            _step(4, "retrieve_graph", count=2),
            _step(5, "rerank", latency_ms=40, enabled=True),
        ],
        "b": [
            _step(1, "retrieve_body", count=3),
            _step(2, "retrieve_heading", count=4),
            _step(3, "retrieve_dense", count=0),
            _step(4, "retrieve_graph", count=0),
        ],
    }
    stats = summarize_runs(runs, steps_by_run=steps)
    assert stats["route_hits"] == {"body": 2, "heading": 1, "dense": 1, "graph": 1}
    assert stats["stage_latency_ms"]["rerank"]["p95"] == pytest.approx(40.0)


def test_summarize_runs_breaks_tokens_down_by_stage_and_model() -> None:
    runs = [_run(id="a")]
    steps = {
        "a": [
            _step(1, "generate", token_in=10, token_out=5, model="chat-a"),
            _step(2, "assess", token_in=4, token_out=1, model="router-a"),
            _step(3, "generate", token_in=2, token_out=1, model="chat-a"),
        ]
    }
    stats = summarize_runs(runs, steps_by_run=steps)
    assert stats["tokens_by_stage"]["generate"] == {"token_in": 12, "token_out": 6}
    assert stats["tokens_by_model"]["chat-a"] == {"token_in": 12, "token_out": 6}
    assert stats["tokens_by_model"]["router-a"] == {"token_in": 4, "token_out": 1}


def test_summarize_runs_on_empty_is_all_zero() -> None:
    stats = summarize_runs([], steps_by_run={})
    assert stats["runs"] == 0
    assert stats["latency_ms"] == {"p50": 0.0, "p95": 0.0}
    assert stats["route_hits"] == {"body": 0, "heading": 0, "dense": 0, "graph": 0}
