"""§7.1 的十个节点:工厂产物可以直接 await(FR-401「每个节点可单独测试」)。

三条纪律,决定了这里为什么长成这个样子:

1. **身份不进 State**。`TenantContext` 一律在节点运行时从 ContextVar 取(`deps.ctx_provider`),
   绝不写进 state:检查点会落盘、会被另一个进程读回来,而「哪个租户」必须由那一刻的验签令牌
   决定,不能由一条三小时前写下的 checkpoint 决定。
2. **同步网关调用一律 `asyncio.to_thread`**。`Gateway` 是同步栈(它自己用 `BoundedSemaphore`
   收口并发),在异步节点里直接调就是把事件循环交给一次 60 秒的 HTTP。
3. **留痕先于返回**。每个节点用 `_Log` 从 `next_seq(st)` 起发号,一步一提交;`interrupt()` 之前
   的节点体在 resume 时会重跑,所以发号必须来自 state 而不是硬编码,写入必须幂等。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from langgraph.types import interrupt
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kbmesh.agents import guards
from kbmesh.agents.schemas import IntentOut, RelevanceOut, RewriteOut
from kbmesh.agents.state import AgentState, empty_state, next_seq, utc_now
from kbmesh.agents.trace import TraceWriter
from kbmesh.core.config import Settings, get_settings
from kbmesh.core.errors import IllegalTransition
from kbmesh.core.gateway import Gateway, ModelUnavailable, Usage
from kbmesh.core.tenancy import TenantContext, current_context
from kbmesh.retrieval.contracts import ROUTES, Citation, Hit, RetrievalConfig
from kbmesh.retrieval.fusion import apply_rerank, retrieve

#: 层4 的固定文案。措辞是产品口径,不许节点自己现编 —— 「证据不足」是这套防御对外的唯一拒答语。
REFUSAL_TEXT = "证据不足"

#: 进上下文的每条证据在提示词里最多带这么多个字符,再多就是把成本花在模型已经不会读的内容上
EVIDENCE_CHARS = 900

Node = Callable[[AgentState], Awaitable[dict[str, Any]]]
Router = Callable[[AgentState], str]


# ------------------------------------------------------------------------------
# 依赖
# ------------------------------------------------------------------------------
@dataclass
class GraphDeps:
    """一张图的全部外部依赖。节点工厂只认这一个对象,所以换一套引擎就是换一个 deps。"""

    factory: async_sessionmaker[AsyncSession]
    gateway: Gateway | Any | None = None
    driver: Any = None
    client: Any = None
    collection: str | None = None
    config: RetrievalConfig | None = None
    kb_id: str | None = None
    settings: Settings = field(default_factory=get_settings)
    trace: TraceWriter | None = None
    #: 在线评估投递钩子(FR-601):同步入队、失败不影响问答;None = 本次不投。
    online_eval: Callable[[dict[str, Any]], None] | None = None
    ctx_provider: Callable[[], TenantContext] = current_context

    # --- 阈值的唯一真值 = Settings(层2/层4 绝不读 RetrievalConfig.theta_rel)------------
    @property
    def theta_rel(self) -> float:
        return self.settings.theta_rel

    @property
    def theta_answer(self) -> float:
        return self.settings.theta_answer

    @property
    def max_retries(self) -> int:
        return self.settings.max_targeted_retry

    @property
    def top_k_fusion(self) -> int:
        return self.config.top_k_fusion if self.config is not None else self.settings.top_k_fusion

    @property
    def top_n_context(self) -> int:
        return self.config.top_n_context if self.config is not None else self.settings.top_n_context

    @property
    def use_rerank(self) -> bool:
        """§10:`Flags.rerank` 是总闸,请求级 `config.use_rerank` 只能再往下收,不能反过来打开。"""
        if not self.settings.flags.rerank:
            return False
        return True if self.config is None else self.config.use_rerank

    def effective_config(self) -> RetrievalConfig:
        """把 §10 的「增益可关」落到路集合上:关掉标题路 / 图谱路 = 从 routes 里摘掉那一路。

        `retrieve()` 自己也会按有无驱动决定图谱路,但拿 `driver is not None` 当开关会让
        `KBMESH_FEATURE_ROUTE_GRAPH=false` 变成一句空话。
        """
        cfg = self.config or RetrievalConfig(
            routes=ROUTES,
            top_k_route=self.settings.top_k_route,
            top_k_fusion=self.settings.top_k_fusion,
            top_n_context=self.settings.top_n_context,
            rrf_k=self.settings.rrf_k,
            use_rerank=self.settings.flags.rerank,
            graph_max_hops=self.settings.graph_max_hops,
            graph_max_edges=self.settings.graph_max_edges,
            theta_rel=self.settings.theta_rel,
        )
        routes: list[str] = list(cfg.routes)
        if not self.settings.flags.route_heading and "heading" in routes:
            routes.remove("heading")
        if not self.settings.flags.route_graph and "graph" in routes:
            routes.remove("graph")
        return cfg if tuple(routes) == cfg.routes else replace(cfg, routes=tuple(routes))

    def flags_snapshot(self) -> dict[str, Any]:
        f = self.settings.flags
        return {
            "route_heading": f.route_heading,
            "route_graph": f.route_graph,
            "rerank": f.rerank,
            "crosscheck": f.crosscheck,
            "online_eval": f.online_eval,
        }

    def config_hash(self) -> str:
        """进 `await_confirm` 与 `trace_run.config_snapshot`:让「哪套配置在回答我」可回答。"""
        blob = json.dumps(
            {**asdict(self.effective_config()), "flags": self.flags_snapshot()},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    async def emit(self, run_id: str, seq: int, stage: str, **kwargs: Any) -> None:
        if self.trace is not None:
            await self.trace.step(run_id, seq, stage, **kwargs)

    async def event(self, event: str, target: str, path: str = "", **detail: Any) -> None:
        if self.trace is not None:
            await self.trace.security_event(event, target, path, **detail)


# ------------------------------------------------------------------------------
# 模型调用接缝
# ------------------------------------------------------------------------------
async def _structured(deps: GraphDeps, prompt: str, schema: type, *, purpose: str):
    if deps.gateway is None:
        raise ModelUnavailable(f"structured:{purpose}", "未配置模型")
    return await asyncio.to_thread(deps.gateway.structured, prompt, schema, purpose=purpose)


async def _chat(deps: GraphDeps, prompt: str, *, purpose: str = "generate") -> Any:
    if deps.gateway is None:
        raise ModelUnavailable(f"chat:{purpose}", "未配置模型")
    return await asyncio.to_thread(deps.gateway.chat, prompt, purpose=purpose)


def _plain(value: Any) -> Any:
    """进 JSONB / msgpack 的东西必须是原生类型:pydantic、dataclass、set 都要在这里收敛。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(one) for one in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _plain(dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    return str(value)


def _ids(hits: Iterable[Hit]) -> list[str]:
    return [h.chunk_id for h in hits]


def _digest(hits: Sequence[Hit]) -> dict[str, Any]:
    """名次摘要只留头三条尾三条:FR-701 要能对账「精排把谁挪动了」,不需要整份名单进库。"""
    labels = _ids(hits)
    return {"n": len(labels), "head": labels[:3], "tail": labels[-3:] if len(labels) > 3 else []}


class _Log:
    """节点内的发号器:seq 从 `next_seq(st)` 起,写一条进一步,收尾时把最大值交回 state。"""

    def __init__(self, deps: GraphDeps, st: AgentState) -> None:
        self.deps = deps
        self.run_id = str(st.get("run_id") or "")
        self.base = int(st.get("trace_seq") or 0)
        self.seq = next_seq(st)
        self.usage: list[dict[str, Any]] = []

    async def step(
        self,
        stage: str,
        *,
        payload: dict[str, Any] | None = None,
        latency_ms: int = 0,
        usage: Usage | None = None,
    ) -> None:
        token_in = token_out = 0
        model = ""
        if usage is not None:
            token_in, token_out = usage.token_in, usage.token_out
            latency_ms = usage.latency_ms or latency_ms
            model = usage.model
            self.usage.append(
                {
                    "stage": stage,
                    "model": usage.model,
                    "token_in": usage.token_in,
                    "token_out": usage.token_out,
                    "latency_ms": usage.latency_ms,
                }
            )
        await self.deps.emit(
            self.run_id,
            self.seq,
            stage,
            payload=payload,
            latency_ms=latency_ms,
            token_in=token_in,
            token_out=token_out,
            model=model,
        )
        self.seq += 1

    def done(self) -> dict[str, Any]:
        out: dict[str, Any] = {"trace_seq": max(self.seq - 1, self.base)}
        if self.usage:
            out["usage"] = self.usage
        return out


# ------------------------------------------------------------------------------
# 提示词(FR-407:四类结构化输出的判定方向写在这一处,不散进各节点)
# ------------------------------------------------------------------------------
def intent_prompt(query: str) -> str:
    return (
        "判断这个问题该不该由企业知识库来回答,四选一:\n"
        "- answerable:指向库内资料可回答的事实\n"
        "- chitchat:寒暄或与知识库内容无关\n"
        "- out_of_privilege:问题在库内,但要求越出提问者的权限面"
        "(索要他租户/他人原始数据、要求绕过隔离、索要全量导出)\n"
        "- out_of_scope:与本平台知识域无关(实时行情、天气、别家公司的业务)\n"
        f"问题:{query}\n"
        "判不准就给一个不在上面四类里的值,系统会按 fail closed 处理。"
    )


def rewrite_prompt(query: str, *, extra: str = "") -> str:
    return (
        "把下面这个问题改写成最多 4 条更适合检索的表述,每条给出侧重与理由:\n"
        "- 保留原问题的实体,补出隐含的时间/主体/口径/文档类型限定\n"
        "- 不许改变提问的原意,也不许引入原问题里没有的事实前提\n"
        f"{extra}"
        f"问题:{query}"
    )


def assess_prompt(query: str, hits: Sequence[Hit]) -> str:
    blocks = "\n".join(
        f"- chunk_id={h.chunk_id}\n  正文:{(h.text or '')[:EVIDENCE_CHARS]}" for h in hits
    )
    return (
        "逐条评估下面这些候选证据对回答该问题的相关度(0 到 1,1 表示直接回答了问题),"
        "并列出回答仍然缺失的信息:\n"
        "- 只按正文内容打分,与出现顺序无关\n"
        "- chunk_id 必须原样抄写,不许编造、不许改写\n"
        "- 缺失项写「要回答这个问题还差什么」,证据够了就给空列表\n"
        f"问题:{query}\n"
        f"候选证据:\n{blocks}"
    )


def retry_prompt(*, current: str, missing: Sequence[str], tried: Sequence[str]) -> str:
    return rewrite_prompt(
        current,
        extra=(
            "上一轮检索的缺失项:" + " / ".join(str(m) for m in missing) + "\n"
            "已经试过的问法(禁止再给出归一化之后等价的问法,必须新增限定词):"
            + " || ".join(str(t) for t in tried)
            + "\n"
        ),
    )


def generate_prompt(query: str, hits: Sequence[Hit]) -> str:
    blocks = "\n".join(
        f"[{i + 1}] {(h.citation.label() if h.citation else h.chunk_id)}"
        f"\n{(h.text or '')[:EVIDENCE_CHARS]}"
        for i, h in enumerate(hits)
    )
    return (
        "只依据下面编号后的资料回答用户问题。规则:\n"
        "- 每个结论后面紧跟来源编号 [n],编号只能来自上面的列表\n"
        "- 资料里没有的不要写,也不要用自身知识补全\n"
        "- 资料不足以回答时,只输出「证据不足」四个字,不要编造\n"
        f"问题:{query}\n"
        f"资料:\n{blocks}"
    )


def await_confirm_payload(st: AgentState, deps: GraphDeps) -> dict[str, Any]:
    """`interrupt()` 的载荷。

    **不含任何正文**:确认发生在检索之前,把候选切片的内容放进来,等于让一个还没被授权的会话
    拿库里的文字当探针。这里只给「要问什么」和「什么时候过期」。
    """
    ttl = int(deps.settings.pending_confirm_ttl_min)
    since = str(st.get("pending_since") or utc_now().isoformat(timespec="seconds"))
    try:
        start = datetime.fromisoformat(since)
    except ValueError:
        start = utc_now()
    return {
        "kind": "await_confirm",
        "thread_id": st.get("thread_id", ""),
        "run_id": st.get("run_id", ""),
        "question": st.get("query", ""),
        "rewrite_candidates": list(st.get("rewrite_candidates") or []),
        "edited_allowed": True,
        "pending_ttl_min": ttl,
        "expires_at": (start + timedelta(minutes=ttl)).isoformat(timespec="seconds"),
        "config_hash": st.get("config_hash", ""),
    }


# ------------------------------------------------------------------------------
# 节点
# ------------------------------------------------------------------------------
def make_route_intent(deps: GraphDeps) -> Node:
    """层1:意图门禁。§10 把它列为**不可降级** —— 判不出来就不放行,宁要 error 不要答案。"""

    async def route_intent(st: AgentState) -> dict[str, Any]:
        log = _Log(deps, st)
        query = str(st.get("query") or "")
        try:
            out, usage = await _structured(deps, intent_prompt(query), IntentOut, purpose="route")
        except ModelUnavailable as exc:
            await log.step("route_intent", payload={"error": str(exc)[:300], "fail_closed": "error"})
            raise
        gate = guards.intent_gate(out)
        await log.step(
            "route_intent",
            payload={
                "verdict": gate.verdict,
                "allow": gate.allow,
                "confidence": round(gate.confidence, 4),
                "reject_code": gate.reject_code,
            },
            usage=usage,
        )
        if gate.security_event:
            # 越权是审计事件,拒答本身不是;detail 只记判定与置信度,不把问题原文抄进安全流水
            await deps.event(
                gate.security_event,
                f"thread:{st.get('thread_key') or st.get('run_id')}",
                "intent",
                verdict=gate.verdict,
                confidence=round(gate.confidence, 4),
            )
        patch: dict[str, Any] = {"intent": _plain(out), "reject_code": gate.reject_code}
        if not gate.allow:
            patch["final_status"] = gate.final_status
        return {**patch, **log.done()}

    return route_intent


def make_rewrite_query(deps: GraphDeps) -> Node:
    """FR-402:给出可确认的改写候选。改写挂掉不影响能不能答,所以这一层可以降级。"""

    async def rewrite_query(st: AgentState) -> dict[str, Any]:
        log = _Log(deps, st)
        query = str(st.get("query") or "")
        candidates: list[dict[str, Any]] = []
        degraded: list[str] = []
        usage: Usage | None = None
        try:
            out, usage = await _structured(deps, rewrite_prompt(query), RewriteOut, purpose="rewrite")
            candidates = [
                {
                    "index": i + 1,
                    "text": one.text,
                    "focus": one.focus,
                    "rationale": one.rationale,
                }
                for i, one in enumerate(out.candidates)
                if str(one.text).strip()
            ]
        except ModelUnavailable as exc:
            # §10 没列改写这一层:候选为空时确认框只剩「自己改写」,仍然是可回答的路
            degraded.append(f"改写候选降级:{exc}")
        await log.step(
            "rewrite",
            payload={"candidates": len(candidates), "degraded": bool(degraded)},
            usage=usage,
        )
        return {
            "rewrite_candidates": candidates,
            "pending_since": utc_now().isoformat(timespec="seconds"),
            "degraded": degraded,
            **log.done(),
        }

    return rewrite_query


def make_confirm_interrupt(deps: GraphDeps) -> Node:
    """FR-402 + FR-403:`interrupt()` 落检查点,等用户选一条改写或自己改写。

    节点体在 `interrupt()` **之前**只许做纯计算 —— resume 会把它整段重跑一遍(实测 C4c)。
    留痕因此放在 `interrupt()` 之后:那一半只在续跑的那条进程里执行一次。
    """

    async def confirm_interrupt(st: AgentState) -> dict[str, Any]:
        payload = await_confirm_payload(st, deps)
        decision = interrupt(payload)
        log = _Log(deps, st)
        try:
            confirmed = guards.validate_resume(st, decision)
        except IllegalTransition as exc:
            # 不静默回退原问:回退等于跳过确认,FR-402 白做。留痕后交给调用方(API 层是 409)。
            await log.step("rewrite", payload={"kind": "resume_rejected", "reason": str(exc)[:200]})
            raise
        record = _plain(decision)
        if not isinstance(record, dict):
            record = {"value": record}
        await log.step(
            "rewrite",
            payload={
                "kind": "await_confirmed",
                "choice": record.get("choice"),
                "edited": bool(str(record.get("edited") or "").strip()),
                "confirmed_query": confirmed,
            },
        )
        return {"user_choice": record, "confirmed_query": confirmed, **log.done()}

    return confirm_interrupt


def make_retrieve_dual(deps: GraphDeps) -> Node:
    """路与融合归 `retrieve`,编排层只搬运与留痕;精排延后(§7.1 给它单独一个节点)。"""

    async def retrieve_dual(st: AgentState) -> dict[str, Any]:
        log = _Log(deps, st)
        ctx = deps.ctx_provider()
        query = str(st.get("confirmed_query") or st.get("query") or "")
        cfg = deps.effective_config()
        result = await retrieve(
            query,
            ctx,
            factory=deps.factory,
            gateway=deps.gateway,
            driver=deps.driver if "graph" in cfg.routes else None,
            client=deps.client,
            collection=deps.collection,
            config=cfg,
            kb_id=deps.kb_id,
            defer_rerank=True,
        )
        for route in ROUTES:
            stat = result.stats.get(route)
            if stat is None:
                continue
            await log.step(
                f"retrieve_{route}",
                payload={
                    "count": stat.get("count"),
                    "fingerprint": stat.get("fingerprint"),
                    "degraded": bool(stat.get("degraded")),
                    "error": stat.get("error"),
                },
                latency_ms=int(stat.get("latency_ms") or 0),
            )
        ids = _ids(result.hits)
        seen = set(st.get("seen_chunk_ids") or [])
        return {
            "hits": list(result.hits),
            "route_stats": dict(result.stats),
            "graph_trace": _plain(result.graph or {}),
            "degraded": list(result.degraded),
            "dropped": list(result.dropped),
            "seen_chunk_ids": ids,
            "evidence_gain": sum(1 for one in ids if one not in seen),
            "queries_tried": [query],
            **log.done(),
        }

    return retrieve_dual


def make_dedupe(deps: GraphDeps) -> Node:
    """层2 前置:去重键只认 `chunk_id`,融合后的名次在这里定形(§11 的 merge 步)。"""

    async def dedupe(st: AgentState) -> dict[str, Any]:
        log = _Log(deps, st)
        hits = list(st.get("hits") or [])
        kept, dropped = guards.dedupe_hits(hits, limit=deps.top_k_fusion)
        await log.step(
            "merge",
            payload={
                "in": len(hits),
                "kept": len(kept),
                "limit": deps.top_k_fusion,
                "dropped": dropped,
                "dropped_by_authority": list(st.get("dropped") or []),
            },
        )
        return {"hits": kept, "dropped": dropped, **log.done()}

    return dedupe


def make_rerank_fused(deps: GraphDeps) -> Node:
    """FR-210 统一精排:与 `fusion` 内同一个函数,挂掉同样只按原序截断 + 一条 note。"""

    async def rerank_fused(st: AgentState) -> dict[str, Any]:
        log = _Log(deps, st)
        hits = list(st.get("hits") or [])
        query = str(st.get("confirmed_query") or st.get("query") or "")
        enabled = deps.use_rerank
        before = _digest(hits)
        ranked, note = await asyncio.to_thread(
            apply_rerank, query, hits, deps.gateway, top_n=deps.top_n_context, enabled=enabled
        )
        await log.step(
            "rerank",
            payload={"enabled": enabled, "before": before, "after": _digest(ranked), "note": note},
        )
        return {"reranked": ranked, "degraded": [note] if note else [], **log.done()}

    return rerank_fused


def make_assess_relevance(deps: GraphDeps) -> Node:
    """层2:θ_rel 逐条打分过滤 + 层4 的判据在这里产出。

    `Flags.crosscheck` 打开时跑两遍(第二遍把候选顺序倒过来),同一 chunk 取两遍的**较低分**——
    位置偏置是精排和判分共同的毛病,取 min 是让「换个顺序就涨分」的证据过不了关。

    判分模型挂掉**不降级放行**:这一层的产物是「这条证据算不算数」的依据,拿不到依据时唯一
    安全的方向是判成证据不足。但仍然在 `degraded` 里留一笔 —— 结论是对的,理由不是。
    """

    async def assess_relevance(st: AgentState) -> dict[str, Any]:
        log = _Log(deps, st)
        query = str(st.get("confirmed_query") or st.get("query") or "")
        hits = list(st.get("reranked") or st.get("hits") or [])
        if not hits:
            await log.step("assess", payload={"evidence": 0, "reason": "候选集为空"})
            return {"hits": [], "relevance_scores": {}, "missing_info": [], "reject_code": "no_evidence",
                    **log.done()}
        crosscheck = bool(deps.settings.flags.crosscheck) and len(hits) > 1
        try:
            out, usage = await _structured(deps, assess_prompt(query, hits), RelevanceOut, purpose="assess")
            scores = {str(one.chunk_id): float(one.score) for one in out.scores}
            missing = [str(m) for m in out.missing_info]
            if crosscheck:
                again, second = await _structured(
                    deps, assess_prompt(query, list(reversed(hits))), RelevanceOut, purpose="assess"
                )
                for one in again.scores:
                    cid = str(one.chunk_id)
                    scores[cid] = min(scores[cid], float(one.score)) if cid in scores else float(one.score)
                missing = missing or [str(m) for m in again.missing_info]
                usage = usage.merge(second)
        except ModelUnavailable as exc:
            await log.step("assess", payload={"error": str(exc)[:300], "refused": "no_evidence"})
            return {
                "hits": [],
                "relevance_scores": {},
                "missing_info": [],
                "reject_code": "no_evidence",
                "degraded": [f"相关性判定不可用,按证据不足拒答:{exc}"],
                **log.done(),
            }

        known = _ids(hits)
        invented = guards.unknown_ids(scores, known)
        for cid in invented:
            scores.pop(cid, None)
        kept, dropped = guards.filter_by_theta(hits, scores, theta_rel=deps.theta_rel)
        refusal = guards.decide_refusal(kept, scores, theta_answer=deps.theta_answer)
        await log.step(
            "assess",
            payload={
                "candidates": len(hits),
                "kept": len(kept),
                "theta_rel": deps.theta_rel,
                "best_score": round(refusal.best_score, 4),
                "dropped": dropped,
                "invented_ids": invented,
                "crosscheck": crosscheck,
                "missing_info": missing,
                "refuse": refusal.refuse,
                "reject_code": refusal.reject_code,
            },
            usage=usage,
        )
        return {
            "hits": kept,
            "relevance_scores": scores,
            "missing_info": missing,
            "dropped": [*dropped, *invented],
            "reject_code": refusal.reject_code,
            **log.done(),
        }

    return assess_relevance


def make_retry_targeted(deps: GraphDeps) -> Node:
    """层3:定向重试。FR-405 看的就是这一步 —— payload 里必须写清「补了什么」,而不是再跑一次。"""

    async def retry_targeted(st: AgentState) -> dict[str, Any]:
        log = _Log(deps, st)
        current = str(st.get("confirmed_query") or st.get("query") or "")
        missing = [m for m in (st.get("missing_info") or []) if str(m).strip()]
        tried = [str(t) for t in (st.get("queries_tried") or [])]
        attempt = int(st.get("retry_attempt") or 0)
        budget = deps.max_retries
        if not missing or attempt >= budget:
            await log.step(
                "retry",
                payload={
                    "run": False,
                    "reason": "no_missing" if not missing else "no_budget",
                    "attempt": attempt,
                    "max_attempts": budget,
                },
            )
            return {"retry_constraints": [], **log.done()}
        try:
            out, usage = await _structured(
                deps,
                retry_prompt(current=current, missing=missing, tried=tried),
                RewriteOut,
                purpose="rewrite",
            )
            fresh = next((str(one.text).strip() for one in out.candidates if str(one.text).strip()), "")
        except ModelUnavailable as exc:
            await log.step("retry", payload={"run": False, "error": str(exc)[:300]})
            return {"retry_constraints": [], "degraded": [f"定向重试改写降级:{exc}"], **log.done()}
        constraints = guards.constraints_of(current, fresh) if fresh else []
        plan = guards.plan_retry(
            query=fresh,
            missing=missing,
            queries_tried=tried,
            new_constraints=constraints,
            evidence_gain=int(st.get("evidence_gain") or 0),
            attempt=attempt,
            max_attempts=budget,
        )
        await log.step(
            "retry",
            payload={
                "run": plan is not None,
                "attempt": plan.attempt if plan else attempt,
                "max_attempts": budget,
                "missing": [str(m) for m in missing],
                "constraints": constraints,
                "evidence_gain": int(st.get("evidence_gain") or 0),
                "rejected_reason": "" if plan else _why_no_plan(fresh=fresh, constraints=constraints),
            },
            usage=usage,
        )
        if plan is None:
            return {"retry_constraints": [], **log.done()}
        return {
            "confirmed_query": plan.query,
            "retry_constraints": plan.constraints,
            "retry_attempt": plan.attempt,
            **log.done(),
        }

    return retry_targeted


def _why_no_plan(*, fresh: str, constraints: Sequence[str]) -> str:
    """拒了这次重试的原因,进 trace 让 FR-405 可审计(缺哪一条约束就是哪个词)。"""
    if not fresh:
        return "no_candidate"
    if not constraints:
        return "no_new_constraint"
    return "no_evidence_gain_or_reorder"


def make_refuse(deps: GraphDeps) -> Node:
    """层4:固定文案拒答。不调模型 —— 「答案必须有证据」在这里的体现是不产生任何文字。"""

    async def refuse(st: AgentState) -> dict[str, Any]:
        log = _Log(deps, st)
        code = str(st.get("reject_code") or "no_evidence")
        await log.step("generate", payload={"refused": True, "reject_code": code, "text": REFUSAL_TEXT})
        return {
            "answer": REFUSAL_TEXT,
            "final_status": "refused",
            "reject_code": code,
            "citations": [],
            "cite_ids": {},
            **log.done(),
        }

    return refuse


def make_generate_cite(deps: GraphDeps) -> Node:
    """层5:生成 + 程序侧硬校验。

    编号在调用生成模型**之前**冻结:过了 θ_rel 的证据按精排后的顺序 1-based 编号,`cite_ids`
    就是这份冻结映射。事后按模型实际引用的样子重新编号,等于让模型自己决定哪些证据算数 ——
    那正是这套防御要防的事。越界编号只从正文里剥离,不重编号也不改写其余正文。
    """

    async def generate_cite(st: AgentState) -> dict[str, Any]:
        log = _Log(deps, st)
        query = str(st.get("confirmed_query") or st.get("query") or "")
        numbered: list[Hit] = [h for h in (st.get("hits") or []) if h.citation is not None]
        citations: list[Citation] = [h.citation for h in numbered if h.citation is not None]
        cite_ids = {str(i + 1): h.chunk_id for i, h in enumerate(numbered)}
        try:
            res = await _chat(deps, generate_prompt(query, numbered), purpose="generate")
        except ModelUnavailable as exc:
            await log.step("generate", payload={"error": str(exc)[:300], "fail_closed": "error"})
            raise
        await log.step(
            "generate", payload={"chars": len(res.text), "evidence": len(numbered)}, usage=res.usage
        )
        check = guards.check_citations(res.text, cite_ids, evidence_ids=_ids(numbered))
        await log.step(
            "cite_check",
            payload={
                "verdict": check.verdict,
                "candidates": len(cite_ids),
                "cited_ids": check.cited_ids,
                "stripped": check.stripped,
            },
        )
        if check.refused:
            return {
                "answer": REFUSAL_TEXT,
                "final_status": "refused",
                "reject_code": "cite_unparsable",
                "citations": [],
                "cite_ids": {},
                **log.done(),
            }
        return {
            "answer": check.text,
            "final_status": "answered",
            "citations": citations,
            "cite_ids": cite_ids,
            **log.done(),
        }

    return generate_cite


# ------------------------------------------------------------------------------
# 条件边路由(§7.1 的两处,加一处实测必要的)
# ------------------------------------------------------------------------------
def after_intent(st: AgentState) -> str:
    """层1 判死的问题不进检索:让越权问题查一轮再拒,等于已经查了库。"""
    return "refuse" if str(st.get("reject_code") or "") else "rewrite_query"


def after_assess(st: AgentState) -> str:
    """证据够了就去生成;不够但还有一次像样的重试机会,先补约束再跑。"""
    if not str(st.get("reject_code") or ""):
        return "generate_cite"
    missing = [m for m in (st.get("missing_info") or []) if str(m).strip()]
    if missing and int(st.get("retry_attempt") or 0) < get_settings().max_targeted_retry:
        # 预算与缺失项在这里挡:省掉一次注定给不出计划的模型调用
        return "retry_targeted"
    return "refuse"


def after_retry(st: AgentState) -> str:
    """`plan_retry` 给不出计划 → `retry_constraints` 为空 → 转拒答,绝不原样重跑。"""
    return "retrieve_dual" if (st.get("retry_constraints") or []) else "refuse"


#: §7.1 拓扑里的十个节点,顺序即数据流顺序(测试拿它和 mermaid 对账)
ALL_NODES: tuple[str, ...] = (
    "route_intent",
    "rewrite_query",
    "confirm_interrupt",
    "retrieve_dual",
    "dedupe",
    "rerank_fused",
    "assess_relevance",
    "retry_targeted",
    "refuse",
    "generate_cite",
)

#: 每个节点都必须在 `STAGES` 里对得上至少一个 stage,否则 FR-701 的白名单对账过不去
NODE_STAGES: dict[str, tuple[str, ...]] = {
    "route_intent": ("route_intent",),
    "rewrite_query": ("rewrite",),
    "confirm_interrupt": ("rewrite",),
    "retrieve_dual": ("retrieve_body", "retrieve_heading", "retrieve_dense", "retrieve_graph"),
    "dedupe": ("merge",),
    "rerank_fused": ("rerank",),
    "assess_relevance": ("assess",),
    "retry_targeted": ("retry",),
    "refuse": ("generate",),
    "generate_cite": ("generate", "cite_check"),
}


def seed_state(
    *, query: str, thread_id: str, thread_key: str, run_id: str, deps: GraphDeps
) -> AgentState:
    """`empty_state` 加上编排层才算得出的两样东西,免得每个调用点自己拼。"""
    return empty_state(
        query=query,
        thread_id=thread_id,
        thread_key=thread_key,
        run_id=run_id,
        route_flags=deps.flags_snapshot(),
        config_hash=deps.config_hash(),
    )
