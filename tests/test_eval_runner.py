"""#8 离线评估运行器:逐 case 落库、cursor 续跑、分组报告、基线回退与 CSV。

运行器只认注入的 `answer_case`,不在这里起图/连模型 —— 那是 `runtime_case_runner` 的事;
这个文件验的是「中断不丢已完成结果」与「报告口径」。
"""
from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from conftest import database_url, purge_tenant_data
from kbmesh.core.tenancy import ROLE_TENANT_ADMIN, TenantContext
from kbmesh.db.base import make_engine, set_engine
from kbmesh.db.models import EvalCase, EvalResult, EvalRun, EvalSet, OnlineEval, Tenant, TraceRun
from kbmesh.eval.runner import CaseOutcome, EvalRunner

TENANT = "tnt-eval-run"
CTX = TenantContext(tenant_id=TENANT, user_id="eval-user", roles=(ROLE_TENANT_ADMIN,))


async def _purge(engine: AsyncEngine) -> None:
    await purge_tenant_data(engine, prefix="tnt-eval-")


async def _seed(factory: async_sessionmaker[AsyncSession], *, groups: dict[str, str]) -> tuple[str, str]:
    async with factory() as session, session.begin():
        session.add(Tenant(id=TENANT, name="eval-run"))
        await session.flush()  # eval_set 的 FK 指向 tenant;没有 relationship 可让 UoW 推导顺序
        evset = EvalSet(tenant_id=TENANT, name="set", version=1)
        evset.cases = [
            EvalCase(
                id=f"case-{idx:02d}", question=f"q-{idx}", gold_refs=[f"c{idx}"], group_key=group
            )
            for idx, group in enumerate(groups.values(), start=1)
        ]
        session.add(evset)
        await session.flush()
        run = EvalRun(tenant_id=TENANT, set_id=evset.id, status="pending")
        session.add(run)
        await session.flush()
        return evset.id, run.id


@pytest.fixture
async def rig(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KBMESH_ENV", "test")
    eng = make_engine(database_url())
    set_engine(eng)
    factory = async_sessionmaker(eng, expire_on_commit=False)
    await _purge(eng)
    try:
        yield factory
    finally:
        await eng.dispose()
        await _purge(eng)


async def _results(factory: async_sessionmaker[AsyncSession], run_id: str) -> list[EvalResult]:
    async with factory() as session:
        return list(
            (
                await session.execute(
                    sa.select(EvalResult).where(EvalResult.eval_run_id == run_id).order_by(EvalResult.metric)
                )
            )
            .scalars()
            .all()
        )


async def test_runner_is_resumable_without_duplicating_results(rig) -> None:
    factory = rig
    groups = {
        "a": "factual_single",
        "b": "multi_hop",
        "c": "aggregation",
    }
    _, run_id = await _seed(factory, groups=groups)
    calls: list[str] = []

    async def answer_case(case: EvalCase) -> CaseOutcome:
        calls.append(case.question)
        idx = int(case.question.rsplit("-", 1)[1])
        return CaseOutcome(
            answer="答案 [1]。",
            retrieved_ids=[f"c{idx}"],
            cite_ids={"1": f"c{idx}"},
            evidence_ids=[f"c{idx}"],
            user_choice={"choice": 1},
            rewrite_candidates=[{"index": 1, "text": case.question}],
        )

    runner = EvalRunner(factory=factory, ctx=CTX, answer_case=answer_case, top_k=5)
    first = await runner.run(run_id, limit=2)
    assert first["status"] == "running"
    assert len(calls) == 2
    # 每 case:recall/mrr/ndcg/citation_validity + 适用的 rewrite_accept
    assert len(await _results(factory, run_id)) == 2 * 5
    async with factory() as session:
        run = await session.get(EvalRun, run_id)
        assert run.cursor_case_id, "每完成一个 case 必须推进 cursor"

    second = await runner.run(run_id)
    assert second["status"] == "done"
    assert calls == ["q-1", "q-2", "q-3"]
    assert len(await _results(factory, run_id)) == 3 * 5, "续跑不许把已完成的 case 再写一遍"

    report = await runner.report(run_id)
    assert report["cases_total"] == 3 and report["cases_done"] == 3
    assert report["metrics"]["recall@5"] == pytest.approx(1.0)
    assert report["groups"]["multi_hop"]["recall@5"] == pytest.approx(1.0)
    assert report["system"]["runs"] == 3 and report["system"]["error_rate"] == 0.0


async def test_runner_can_resume_after_a_case_fails(rig) -> None:
    factory = rig
    _, run_id = await _seed(factory, groups={"factual_single": "factual_single", "multi": "multi_hop"})
    attempts: list[str] = []

    async def flaky(case: EvalCase) -> CaseOutcome:
        attempts.append(case.question)
        if case.question == "q-2" and attempts.count("q-2") == 1:
            raise RuntimeError("模型临时不可用")
        return CaseOutcome(
            answer="答案 [1]。",
            retrieved_ids=["c1"],
            cite_ids={"1": "c1"},
            evidence_ids={"c1"},
        )

    runner = EvalRunner(factory=factory, ctx=CTX, answer_case=flaky, top_k=5)
    with pytest.raises(RuntimeError):
        await runner.run(run_id)
    async with factory() as session:
        run = await session.get(EvalRun, run_id)
        assert run.cursor_case_id, "失败前完成的 case 必须已经落 cursor"
    await runner.run(run_id)
    assert attempts.count("q-2") == 2
    assert (await runner.report(run_id))["status"] == "done"


async def test_regression_csv_neutralizes_formula_cells(rig) -> None:
    factory = rig
    set_id, baseline_id = await _seed(factory, groups={"a": "factual_single"})
    async with factory() as session, session.begin():
        case = (
            await session.execute(sa.select(EvalCase).where(EvalCase.set_id == set_id))
        ).scalars().one()
        case.question = '=HYPERLINK("http://evil","x")'
        session.add(
            EvalResult(
                eval_run_id=baseline_id,
                case_id=case.id,
                metric="recall@5",
                value=1.0,
                detail={"answer": "基线", "question": case.question},
            )
        )
        case_id = case.id
    async with factory() as session, session.begin():
        current = EvalRun(
            tenant_id=TENANT, set_id=set_id, baseline_run_id=baseline_id, status="done"
        )
        session.add(current)
        await session.flush()
        session.add(
            EvalResult(
                eval_run_id=current.id,
                case_id=case_id,
                metric="recall@5",
                value=0.0,
                detail={"answer": "本次", "question": "=HYPERLINK(...)"},
            )
        )
        current_id = current.id

    async def unused(case: EvalCase) -> CaseOutcome:
        raise AssertionError("这个用例只导出 CSV")

    runner = EvalRunner(factory=factory, ctx=CTX, answer_case=unused, top_k=5)
    text = await runner.regression_csv(current_id, baseline_id)
    assert "'=HYPERLINK" in text, "以 = + - @ 开头的单元格必须加前缀,防止表格公式执行"


async def test_online_eval_service_stores_rule_results(rig) -> None:
    from kbmesh.eval.tasks import evaluate_and_store

    factory = rig
    await _seed(factory, groups={"factual_single": "factual_single"})
    async with factory() as session, session.begin():
        session.add(
            TraceRun(id="run-online-1", tenant_id=TENANT, thread_id=f"{TENANT}:th-1", query="q")
        )
        session.add(
            TraceRun(id="run-online-2", tenant_id=TENANT, thread_id=f"{TENANT}:th-2", query="q")
        )
    good = await evaluate_and_store(
        {
            "run_id": "run-online-1",
            "tenant_id": TENANT,
            "question": "隔离怎么做?",
            "answer": "隔离靠分区键 [1]。",
            "final_status": "answered",
            "reject_code": "",
            "cite_ids": {"1": "c1"},
            "evidence_ids": ["c1"],
            "citations": [],
            "usage": [],
        },
        judge=None,
        draw=0.0,
    )
    assert good["failed"] == []
    bad = await evaluate_and_store(
        {
            "run_id": "run-online-2",
            "tenant_id": TENANT,
            "question": "隔离怎么做?",
            "answer": "没有引用的答案。",
            "final_status": "answered",
            "reject_code": "",
            "cite_ids": {"1": "c1"},
            "evidence_ids": ["c1"],
            "citations": [],
            "usage": [],
        },
        judge=None,
        draw=0.0,
    )
    assert bad["failed"] == ["citation"]
    async with factory() as session:
        rows = list((await session.execute(sa.select(OnlineEval))).scalars())
    assert {row.run_id for row in rows} == {"run-online-1", "run-online-2"}
    assert [row.failed for row in rows if row.run_id == "run-online-2"] == [["citation"]]


async def test_regression_report_and_csv_list_worse_samples(rig) -> None:
    factory = rig
    set_id, baseline_id = await _seed(factory, groups={"factual_single": "factual_single"})
    # 基线:两个 case 都满分
    async with factory() as session, session.begin():
        cases = list(
            (await session.execute(sa.select(EvalCase).where(EvalCase.set_id == set_id))).scalars()
        )
        for case in cases:
            session.add(
                EvalResult(
                    eval_run_id=baseline_id,
                    case_id=case.id,
                    metric="recall@5",
                    value=1.0,
                    detail={"answer": "基线答案", "question": case.question},
                )
            )
        baseline = await session.get(EvalRun, baseline_id)
        baseline.status = "done"

    async with factory() as session, session.begin():
        current = EvalRun(
            tenant_id=TENANT, set_id=set_id, baseline_run_id=baseline_id, status="pending"
        )
        session.add(current)
        await session.flush()
        current_id = current.id

    async def worse(case: EvalCase) -> CaseOutcome:
        return CaseOutcome(answer="答案 [1]。", retrieved_ids=[], cite_ids={"1": "c1"}, evidence_ids={"c1"})

    runner = EvalRunner(factory=factory, ctx=CTX, answer_case=worse, top_k=5)
    await runner.run(current_id)
    report = await runner.compare(current_id, baseline_id, gate=-0.02)
    assert report["regressions"], "recall 从 1.0 掉到 0.0 必须被拦下"
    row = report["regressions"][0]
    assert row["metric"] == "recall@5" and row["delta"] == pytest.approx(-1.0)
    assert report["samples"], "回退样本要能定位到具体 query"
    csv_text = await runner.regression_csv(current_id, baseline_id, gate=-0.02)
    assert "question" in csv_text and "q-1" in csv_text
