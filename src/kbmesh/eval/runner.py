"""#8 离线评估运行器:逐 case 落库、cursor 续跑、分组报告、基线回归与 CSV(FR-603~608)。

业务逻辑与「怎么跑一次问答」解耦:`answer_case` 是注入的可调用对象,生产用
`runtime_case_runner`(见 `eval/runtime_case.py`),测试用纯替身。这样中断/续跑/分组这些
最容易错的账,可以在不连模型、不起图的情况下逐条钉住。
"""
from __future__ import annotations

import csv
import io
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kbmesh.core.config import get_settings
from kbmesh.core.tenancy import TenantContext
from kbmesh.db.models import EvalCase
from kbmesh.db.repositories import EvalRepo
from kbmesh.eval.metrics import (
    AgentCase,
    agent_case_metrics,
    citation_validity,
    retrieval_metrics,
    system_metrics,
)


@dataclass(frozen=True)
class CaseOutcome:
    """一次问答在评估侧需要的最少切片(不含模型客户端等不可序列化对象)。"""

    answer: str
    final_status: str = "answered"
    reject_code: str = ""
    retrieved_ids: Sequence[str] = field(default_factory=tuple)
    cite_ids: Mapping[str, str] = field(default_factory=dict)
    evidence_ids: Sequence[str] = field(default_factory=tuple)
    evidence_texts: Sequence[str] = field(default_factory=tuple)
    rewrite_candidates: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    user_choice: Mapping[str, Any] = field(default_factory=dict)
    retry_attempt: int = 0
    latency_ms: int = 0
    token_in: int = 0
    token_out: int = 0
    cost_usd: float = 0.0
    degraded: bool = False


class Judge(Protocol):
    """LLM judge 只暴露两个指标;不可用/两次采样不一致时返回 None(该样本不计入)。"""

    async def faithfulness(
        self, question: str, answer: str, evidence: Sequence[str]
    ) -> float | None: ...

    async def answer_relevancy(self, question: str, answer: str) -> float | None: ...


CaseAnswerer = Callable[[EvalCase], Awaitable[CaseOutcome]]


class EvalRunner:
    def __init__(
        self,
        *,
        factory: async_sessionmaker[AsyncSession],
        ctx: TenantContext,
        answer_case: CaseAnswerer,
        judge: Judge | None = None,
        top_k: int | None = None,
        gate: float | None = None,
    ) -> None:
        self.factory = factory
        self.ctx = ctx
        self.answer_case = answer_case
        self.judge = judge
        self.top_k = int(top_k if top_k is not None else get_settings().top_n_context)
        self.gate = float(gate if gate is not None else get_settings().regression_gate)

    # ------------------------------------------------------------------ 执行
    async def run(self, run_id: str, *, limit: int | None = None) -> dict[str, Any]:
        """跑完（或跑一批）case;每完成一个 case 就提交 cursor,进程被杀也不丢已完成结果。"""
        async with self.factory() as session:
            repo = EvalRepo(self.ctx, session)
            run = await repo.get_run(run_id)
            cases = await repo.cases_for_set(run.set_id)

        if run.status == "done":
            return await self.report(run_id)

        start = 0
        if run.cursor_case_id:
            ids = [case.id for case in cases]
            if run.cursor_case_id in ids:
                start = ids.index(run.cursor_case_id) + 1
        pending = cases[start : start + limit] if limit is not None else cases[start:]

        for case in pending:
            outcome = await self.answer_case(case)
            rows = self._score_case(case, outcome)
            if self.judge is not None:
                faithful = await self.judge.faithfulness(
                    case.question, outcome.answer, outcome.evidence_texts
                )
                if faithful is not None:
                    rows.append(("faithfulness", faithful, self._detail(case, outcome)))
                relevant = await self.judge.answer_relevancy(case.question, outcome.answer)
                if relevant is not None:
                    rows.append(("answer_relevancy", relevant, self._detail(case, outcome)))
            async with self.factory() as session, session.begin():
                repo = EvalRepo(self.ctx, session)
                await repo.record_case_results(run_id, case.id, rows)
                await repo.advance_run(run_id, cursor_case_id=case.id)

        if start + len(pending) >= len(cases):
            async with self.factory() as session, session.begin():
                await EvalRepo(self.ctx, session).complete_run(run_id)
        return await self.report(run_id)

    def _detail(self, case: EvalCase, outcome: CaseOutcome) -> dict[str, Any]:
        return {
            "question": case.question,
            "group_key": case.group_key,
            "expected_refusal": case.expected_refusal,
            "answer": outcome.answer,
            "final_status": outcome.final_status,
            "reject_code": outcome.reject_code,
            "degraded": outcome.degraded,
            "latency_ms": outcome.latency_ms,
            "token_in": outcome.token_in,
            "token_out": outcome.token_out,
            "cost_usd": outcome.cost_usd,
        }

    def _score_case(self, case: EvalCase, outcome: CaseOutcome) -> list[tuple[str, float, dict[str, Any]]]:
        detail = self._detail(case, outcome)
        rows: list[tuple[str, float, dict[str, Any]]] = []
        retrieval = retrieval_metrics(case.gold_refs, outcome.retrieved_ids, k=self.top_k)
        if retrieval is not None:
            rows.append((f"recall@{self.top_k}", retrieval.recall_at_k, detail))
            rows.append(("mrr", retrieval.mrr, detail))
            rows.append(("ndcg@10", retrieval.ndcg_at_10, detail))
        rows.append(
            (
                "citation_validity",
                citation_validity(
                    outcome.answer, outcome.cite_ids, evidence_ids=outcome.evidence_ids
                ),
                detail,
            )
        )
        signals = AgentCase(
            expected_refusal=case.expected_refusal,
            final_status=outcome.final_status,
            reject_code=outcome.reject_code,
            rewrite_candidates=outcome.rewrite_candidates,
            user_choice=outcome.user_choice,
            retry_attempt=outcome.retry_attempt,
        )
        metric_names = {
            "rewrite_accept": "rewrite_accept_rate",
            "retry_success": "retry_success_rate",
            "refuse_correct": "refuse_correctness",
        }
        for metric, value in agent_case_metrics(signals).items():
            if value is not None:
                rows.append((metric_names[metric], 1.0 if value else 0.0, detail))
        return rows

    # ------------------------------------------------------------------ 报告
    async def report(self, run_id: str) -> dict[str, Any]:
        async with self.factory() as session:
            repo = EvalRepo(self.ctx, session)
            run = await repo.get_run(run_id)
            pairs = await repo.results_with_cases(run_id)
            cases_total = await repo.case_count(run.set_id)
        overall = _means(pairs)
        system = _system_summary(pairs)
        groups: dict[str, dict[str, list[float]]] = {}
        for result, case in pairs:
            groups.setdefault(case.group_key, {}).setdefault(result.metric, []).append(
                float(result.value)
            )
        return {
            "run_id": run_id,
            "status": run.status,
            "baseline_run_id": run.baseline_run_id,
            "cases_total": cases_total,
            "cases_done": len({case.id for _, case in pairs}),
            "metrics": overall,
            "system": system,
            "groups": {
                group: {metric: sum(values) / len(values) for metric, values in metrics.items()}
                for group, metrics in groups.items()
            },
        }

    async def compare(
        self, current_run_id: str, baseline_run_id: str, *, gate: float | None = None
    ) -> dict[str, Any]:
        threshold = self.gate if gate is None else float(gate)
        current = await self._pair_results(current_run_id)
        baseline = await self._pair_results(baseline_run_id)
        current_group = _group_means(current)
        baseline_group = _group_means(baseline)

        groups: dict[str, dict[str, dict[str, float]]] = {}
        regressions: list[dict[str, Any]] = []
        for group, metrics in current_group.items():
            for metric, value in metrics.items():
                base = baseline_group.get(group, {}).get(metric)
                if base is None:
                    continue
                delta = value - base
                groups.setdefault(group, {})[metric] = {
                    "baseline": base,
                    "current": value,
                    "delta": delta,
                }
                if delta < threshold:
                    regressions.append(
                        {
                            "group_key": group,
                            "metric": metric,
                            "baseline": base,
                            "current": value,
                            "delta": delta,
                        }
                    )

        regressed = {(row["group_key"], row["metric"]) for row in regressions}
        baseline_results = {(case.id, result.metric): result for result, case in baseline}
        samples: list[dict[str, Any]] = []
        for result, case in current:
            key = (case.group_key, result.metric)
            if key not in regressed:
                continue
            base_result = baseline_results.get((case.id, result.metric))
            if base_result is None:
                continue
            samples.append(
                {
                    "case_id": case.id,
                    "question": case.question,
                    "group_key": case.group_key,
                    "metric": result.metric,
                    "baseline": float(base_result.value),
                    "current": float(result.value),
                    "delta": float(result.value) - float(base_result.value),
                    "baseline_answer": str((base_result.detail or {}).get("answer") or ""),
                    "current_answer": str((result.detail or {}).get("answer") or ""),
                }
            )
        return {
            "current_run_id": current_run_id,
            "baseline_run_id": baseline_run_id,
            "gate": threshold,
            "groups": groups,
            "regressions": regressions,
            "samples": samples,
        }

    async def regression_csv(
        self, current_run_id: str, baseline_run_id: str, *, gate: float | None = None
    ) -> str:
        report = await self.compare(current_run_id, baseline_run_id, gate=gate)
        out = io.StringIO()
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "case_id",
                "question",
                "group_key",
                "metric",
                "baseline",
                "current",
                "delta",
                "baseline_answer",
                "current_answer",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {key: _csv_safe(value) for key, value in row.items()} for row in report["samples"]
        )
        return out.getvalue()

    async def export_csv(self, current_run_id: str, baseline_run_id: str, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(
            await self.regression_csv(current_run_id, baseline_run_id), encoding="utf-8"
        )
        return target

    async def _pair_results(self, run_id: str) -> list[tuple[Any, EvalCase]]:
        async with self.factory() as session:
            return await EvalRepo(self.ctx, session).results_with_cases(run_id)


_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: Any) -> Any:
    """Excel/Sheets 公式注入防护:危险首字符(含前导空白后的首字符)前加单引号。"""
    if not isinstance(value, str):
        return value
    if value.lstrip().startswith(_CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


def _system_summary(pairs: Sequence[tuple[Any, EvalCase]]) -> dict[str, Any]:
    """系统层按 case 去重取一次 detail —— 同一 case 的每个指标行都带同一份 system 字段。"""
    seen: dict[str, dict[str, Any]] = {}
    for result, case in pairs:
        seen.setdefault(case.id, dict(result.detail or {}))
    details = list(seen.values())
    return system_metrics(
        latencies=[float(one.get("latency_ms") or 0) for one in details],
        token_in=[int(one.get("token_in") or 0) for one in details],
        token_out=[int(one.get("token_out") or 0) for one in details],
        cost_usd=[float(one.get("cost_usd") or 0) for one in details],
        statuses=[str(one.get("final_status") or "") for one in details],
        degraded=[bool(one.get("degraded") or False) for one in details],
    )


def _means(pairs: Sequence[tuple[Any, EvalCase]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for result, _ in pairs:
        buckets.setdefault(result.metric, []).append(float(result.value))
    return {metric: sum(values) / len(values) for metric, values in buckets.items()}


def _group_means(pairs: Sequence[tuple[Any, EvalCase]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = {}
    for result, case in pairs:
        buckets.setdefault(case.group_key, {}).setdefault(result.metric, []).append(
            float(result.value)
        )
    return {
        group: {metric: sum(values) / len(values) for metric, values in metrics.items()}
        for group, metrics in buckets.items()
    }
