"""Token / rerank 成本口径(FR-703,技术文档 §11)。

单价表按版本冻结:`trace_run.cost_usd` 是**落库当时**用哪一版单价算出来的,历史成本不因
后来调价而漂。单价本身不写死在调用点 —— 换模型/换档位只换 `PriceTable`,口径只有一份。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: 改价必须 bump 这一版号,并让 `trace_run.config_snapshot` 记住它(见 runtime._start_run)。
PRICE_TABLE_VERSION = "2026-11-06.1"


@dataclass(frozen=True)
class ModelPrice:
    """每 1K token 的单价;rerank 按次计费。单位与 `cost_usd` 列一致(USD)。"""

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    rerank_per_call: float = 0.0


@dataclass(frozen=True)
class PriceTable:
    version: str
    prices: Mapping[str, ModelPrice] = field(default_factory=dict)


#: 运营单价尚未确认的模型一律 0 —— 宁可成本先显示 0,也不在代码里编一个价。
#: 换价时改这里 + bump `PRICE_TABLE_VERSION`;已有 run 的 `cost_usd` 不受影响。
DEFAULT_PRICE_TABLE = PriceTable(
    version=PRICE_TABLE_VERSION,
    prices={
        "qwen3.8-flash": ModelPrice(),
        "qwen3.7-text-embedding-flash": ModelPrice(),
        "qwen3.7-text-rerank": ModelPrice(),
        "qwen3.8-omni-flash": ModelPrice(),
    },
)


def estimate_cost(
    usage: Iterable[Mapping[str, Any]],
    *,
    rerank_calls: int = 0,
    rerank_model: str = "",
    table: PriceTable = DEFAULT_PRICE_TABLE,
) -> float:
    """`cost = Σ(token_in×单价_in + token_out×单价_out) + rerank次数×单价`。

    未知模型按 0 计,不抛:一次新模型上线不该把整条问答打成 500;缺价的事实由
    `PRICE_TABLE_VERSION` 与聚合报表的 0 成本暴露出来。
    """
    total = 0.0
    for entry in usage:
        price = table.prices.get(str(entry.get("model") or ""))
        if price is None:
            continue
        token_in = float(entry.get("token_in") or 0)
        token_out = float(entry.get("token_out") or 0)
        total += token_in / 1000.0 * price.input_per_1k
        total += token_out / 1000.0 * price.output_per_1k
    if rerank_calls:
        price = table.prices.get(rerank_model)
        if price is not None:
            total += float(rerank_calls) * price.rerank_per_call
    return total
