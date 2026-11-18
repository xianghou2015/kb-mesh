"""状态图的 State(技术文档 §7.1)。

会被 msgpack 写进检查点的字段只有三类:标量、list/dict,以及 `retrieval.contracts` 里那两个
dataclass(`Hit`/`Citation`,已在 runtime 的 serde 白名单里)。模型客户端、AsyncSession、Neo4j
驱动一律不许进 —— 它们恢复不出来,而 FR-403 要的正是「进程死了还能续」。

字段名沿用 §7.1 的十二项,不够用的部分每多一个都要写清为什么:文档的清单里没有重试计数,而
FR-405 的「禁止原样重跑」必须有个可判定的落点。
"""
from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from kbmesh.retrieval.contracts import Citation, Hit

#: §11 的 trace stage 白名单,12 个值。`retrieve_dual` 是一个节点、四个 stage(文档就这么拆的)。
STAGES: tuple[str, ...] = (
    "route_intent",
    "rewrite",
    "retrieve_body",
    "retrieve_heading",
    "retrieve_dense",
    "retrieve_graph",
    "merge",
    "rerank",
    "assess",
    "retry",
    "generate",
    "cite_check",
)

#: `trace_run.final_status ∈ FINAL_STATUSES`;`reject_code` 是它的下一级分类(FR-406
#: 「拒答样本不进入正常答案统计口径」在数据层要靠这两列,不靠读答案文本)。
REJECT_CODES: tuple[str, ...] = (
    "intent_forbidden",  # 层1:判为越权
    "out_of_scope",      # 层1:超出知识库范围
    "chitchat",          # 层1:闲聊 —— 同样终态拒答,不进检索、不调生成模型
    "intent_unknown",    # 层1:判不出来 → fail closed,绝不放行
    "no_evidence",       # 层4:过 θ_rel 的证据数不足
    "below_theta",       # 层4:最高分 < θ_answer
    "cite_unparsable",   # 层5:编号不可解析,或剥离后一条合法引用都不剩
)


class AgentState(TypedDict, total=False):
    # --- 身份与运行标识 -----------------------------------------------------------
    #: 用户原问,永不改写:定向重试和 trace 都要能对照原文
    query: str
    #: 会话 id(不带租户前缀;前缀规则只有 `runtime.thread_key()` 一处)
    thread_id: str
    #: 检查点主键 `tenant:thread`(§7.2)
    thread_key: str
    #: `trace_run.id`:一次问答一个,跨 resume 不变(FR-403 的「同一会话仍能续跑」)
    run_id: str
    #: `config_snapshot` 指纹,进 await_confirm payload,让「哪套配置在回答我」可回答
    config_hash: str
    #: §10 的开关矩阵,进图就定死,节点中途不改它
    route_flags: dict[str, Any]
    # --- 层1:意图与改写 -----------------------------------------------------------
    intent: dict[str, Any]
    rewrite_candidates: list[dict[str, str]]
    #: resume 原样落一份,trace 要记「用户选了哪条」而不是只记最终用的词
    user_choice: dict[str, Any]
    #: 检索用的是确认后的词;§7.1 只给了 choice 编号,不够下游节点用
    confirmed_query: str
    # --- 检索与融合 ----------------------------------------------------------------
    hits: list[Hit]
    reranked: list[Hit]
    route_stats: dict[str, Any]
    graph_trace: dict[str, Any]
    #: 被权威表挡下 + 去重丢弃 + 低于 θ_rel 的 chunk_id,累积成一份点名清单
    dropped: Annotated[list[str], operator.add]
    #: 跑过的所有 chunk_id:定向重试要靠它算出「这轮到底新增了几条证据」
    seen_chunk_ids: Annotated[list[str], operator.add]
    #: 最近一轮检索相对 `seen_chunk_ids` 新增的证据条数 —— 层3 的第四个条件(「换了个方向但
    #: 一条新东西都没捞到」等于这个方向是死的,再跑只是烧钱)。由 `retrieve_dual` 每轮覆写。
    evidence_gain: int
    # --- 层2/层3:相关性、缺失与定向重试 --------------------------------------------
    relevance_scores: dict[str, float]
    missing_info: list[str]
    #: 「原样重跑」的判据就是这个列表里有没有归一后相同的问题
    queries_tried: Annotated[list[str], operator.add]
    retry_constraints: list[str]
    retry_attempt: int
    # --- 层4/层5:生成与引用 --------------------------------------------------------
    answer: str
    citations: list[Citation]
    #: 编号→chunk_id 的冻结映射,层5 程序侧校验的唯一真源(不经模型,见 guards.check_citations)
    cite_ids: dict[str, str]
    # --- 终态与留痕 ----------------------------------------------------------------
    final_status: str
    reject_code: str
    degraded: Annotated[list[str], operator.add]
    #: ISO UTC。TTL 判据随检查点落盘,不依赖 saver 内部时钟(重启后才知道等了多久)
    pending_since: str
    #: 逐节点用量 {stage, model, token_in, token_out, latency_ms},FR-703 按阶段汇总靠它
    usage: Annotated[list[dict[str, Any]], operator.add]
    #: 已用掉的最大 trace_step.seq。`(run_id, seq)` 唯一,发号只能从这里走
    trace_seq: int
    error: str


def utc_now() -> datetime:
    return datetime.now(UTC)


def empty_state(
    *,
    query: str,
    thread_id: str,
    thread_key: str,
    run_id: str,
    route_flags: dict[str, Any] | None = None,
    config_hash: str = "",
) -> AgentState:
    """一次新问答的初始状态。

    列表字段全部显式给空:TypedDict 是 total=False,少写一个键,读侧就得到处 `.get()`;而
    `Annotated[..., operator.add]` 的字段只有在节点里第一次写时才存在,发号与 TTL 之类的
    「必须有值」的依赖不能建在这种不确定性上。
    """
    return {
        "query": query,
        "thread_id": thread_id,
        "thread_key": thread_key,
        "run_id": run_id,
        "route_flags": dict(route_flags or {}),
        "config_hash": config_hash,
        "intent": {},
        "rewrite_candidates": [],
        "user_choice": {},
        "confirmed_query": query,
        "hits": [],
        "reranked": [],
        "route_stats": {},
        "graph_trace": {},
        "dropped": [],
        "seen_chunk_ids": [],
        "evidence_gain": 0,
        "relevance_scores": {},
        "missing_info": [],
        "queries_tried": [query],
        "retry_constraints": [],
        "retry_attempt": 0,
        "citations": [],
        "cite_ids": {},
        "degraded": [],
        "pending_since": utc_now().isoformat(timespec="seconds"),
        "usage": [],
        "trace_seq": 0,
    }


def next_seq(st: AgentState) -> int:
    """下一条 `trace_step.seq`。

    节点里发号必须走这里:`interrupt()` 之前的代码在 resume 时会重跑一遍(实测
    `verify_checkpoint.py` C4c),写死数字的写法会把第二条 step 撞在唯一约束上。
    """
    return int(st.get("trace_seq", 0) or 0) + 1


def used_seq(st: AgentState, count: int = 1) -> int:
    """本节点用掉 count 条 step 之后要写回 state 的 `trace_seq`。"""
    return next_seq(st) + max(0, count - 1)
