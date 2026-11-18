"""§7.1 的拓扑连边,逐字照抄(除了一条文档没画、但实测必须有的回边)。

```
START → route_intent → rewrite_query → confirm_interrupt → retrieve_dual
      → dedupe → rerank_fused → assess_relevance ─┬→ retry_targeted ─┐(回 retrieve_dual,≤2 次)
                                                  ├→ refuse ─────────┤
                                                  └→ generate_cite → END
```

`after_retry` 是第三处条件边:§7.1 只画了 `assess_relevance` 的分叉,但「重试到底能不能跑」要
等改写模型给出新约束才知道,分叉点只能落在 `retry_targeted` 之后 —— 放在 assess 那里就会变成
「先决定重跑,再发现没有新约束」,而 FR-405 禁的正是这一句。
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from kbmesh.agents.nodes import (
    ALL_NODES,
    GraphDeps,
    after_assess,
    after_intent,
    after_retry,
    make_assess_relevance,
    make_confirm_interrupt,
    make_dedupe,
    make_generate_cite,
    make_refuse,
    make_rerank_fused,
    make_retrieve_dual,
    make_retry_targeted,
    make_rewrite_query,
    make_route_intent,
)
from kbmesh.agents.state import AgentState


def build_graph(deps: GraphDeps) -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("route_intent", make_route_intent(deps))
    graph.add_node("rewrite_query", make_rewrite_query(deps))
    graph.add_node("confirm_interrupt", make_confirm_interrupt(deps))
    graph.add_node("retrieve_dual", make_retrieve_dual(deps))
    graph.add_node("dedupe", make_dedupe(deps))
    graph.add_node("rerank_fused", make_rerank_fused(deps))
    graph.add_node("assess_relevance", make_assess_relevance(deps))
    graph.add_node("retry_targeted", make_retry_targeted(deps))
    graph.add_node("refuse", make_refuse(deps))
    graph.add_node("generate_cite", make_generate_cite(deps))

    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges(
        "route_intent", after_intent, {"rewrite_query": "rewrite_query", "refuse": "refuse"}
    )
    graph.add_edge("rewrite_query", "confirm_interrupt")
    graph.add_edge("confirm_interrupt", "retrieve_dual")
    graph.add_edge("retrieve_dual", "dedupe")
    graph.add_edge("dedupe", "rerank_fused")
    graph.add_edge("rerank_fused", "assess_relevance")
    graph.add_conditional_edges(
        "assess_relevance",
        after_assess,
        {"retry_targeted": "retry_targeted", "refuse": "refuse", "generate_cite": "generate_cite"},
    )
    graph.add_conditional_edges(
        "retry_targeted", after_retry, {"retrieve_dual": "retrieve_dual", "refuse": "refuse"}
    )
    graph.add_edge("generate_cite", END)
    graph.add_edge("refuse", END)
    return graph


def compile_graph(deps: GraphDeps, checkpointer: Any = None):
    """交出一张编译好的图。

    `checkpointer` 为空时图**不能**走到 `confirm_interrupt`:`interrupt()` 没有落盘的地方。
    单节点测试可以直接 await 工厂产物绕过这一点;整图跑通一律给真 saver(MemorySaver 或
    `AsyncPostgresSaver`)。#7 的 SSE 要 `astream`/`astream_events`,所以这里把图交出去。
    """
    return build_graph(deps).compile(checkpointer=checkpointer)


def mermaid(deps: GraphDeps) -> str:
    """FR-401「拓扑可视化」:与 §7.1 的连线同形,拿它做回归比对比看图说话可靠。"""
    return compile_graph(deps).get_graph().draw_mermaid()


def node_names(deps: GraphDeps) -> list[str]:
    """编译产物里的节点集合 —— 测试用它证明 §7.1 的十个节点一个都没少。"""
    return [str(name) for name in compile_graph(deps).get_graph().nodes]


def topology_ok(deps: GraphDeps) -> bool:
    return set(ALL_NODES) <= set(node_names(deps))


__all__ = ["build_graph", "compile_graph", "mermaid", "node_names", "topology_ok"]
