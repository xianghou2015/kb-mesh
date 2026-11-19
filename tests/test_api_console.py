"""#7 控制台面验收:登录/反馈/评估端点/Trace 查询/就绪探针。

种子行必须**提交**——API 的请求走另一条连接,SAVEPOINT 里未提交的行对它不可见。
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from conftest import database_url, purge_tenant_data
from kbmesh.agents.nodes import GraphDeps
from kbmesh.api.app import create_app
from kbmesh.core.config import clear_settings_cache
from kbmesh.core.tenancy import ROLE_READER, ROLE_TENANT_ADMIN, TenantContext
from kbmesh.db.base import make_engine, set_engine
from kbmesh.db.models import (
    AppUser,
    EvalCase,
    EvalResult,
    EvalRun,
    EvalSet,
    Feedback,
    OnlineEval,
    SecurityEvent,
    Tenant,
    TraceRun,
    TraceStep,
)
from kbmesh.db.repositories import record_security_event
from kbmesh.security.passwords import hash_password
from kbmesh.security.tokens import issue_access_token

TENANT_A, TENANT_B = "tnt-api-a", "tnt-api-b"
CTX_INDEX = TenantContext(tenant_id=TENANT_A, user_id="usr-api-i", roles=(ROLE_TENANT_ADMIN,))
CTX_READ = TenantContext(tenant_id=TENANT_A, user_id="usr-api-r", roles=(ROLE_READER,))
CTX_B = TenantContext(tenant_id=TENANT_B, user_id="usr-api-b", roles=(ROLE_TENANT_ADMIN,))


def bearer(ctx: TenantContext) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_access_token(ctx)}"}


async def _purge(engine: AsyncEngine) -> None:
    await purge_tenant_data(engine)


@pytest.fixture
async def rig(engine, monkeypatch: pytest.MonkeyPatch):
    clear_settings_cache()
    eng = make_engine(database_url())
    set_engine(eng)
    factory = async_sessionmaker(eng, expire_on_commit=False)
    await _purge(eng)
    async with factory() as session, session.begin():
        session.add(Tenant(id=TENANT_A, name="api-a"))
        session.add(Tenant(id=TENANT_B, name="api-b"))
    deps = GraphDeps(factory=factory, trace=None)
    app = create_app(deps=deps, saver=None)
    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                yield SimpleNamespace(client=client, factory=factory)
    finally:
        await eng.dispose()
        await _purge(eng)
        clear_settings_cache()


async def _seed_run(factory: async_sessionmaker[AsyncSession], *, tenant: str, thread: str) -> str:
    async with factory() as session, session.begin():
        session.add(TraceRun(id=f"run-{thread}", tenant_id=tenant, thread_id=thread, query="q"))
        session.add_all(
            [
                TraceStep(run_id=f"run-{thread}", seq=1, stage="route_intent", payload={"s": 1}),
                TraceStep(run_id=f"run-{thread}", seq=2, stage="retrieve_dual", payload={"s": 2}),
            ]
        )
    return f"run-{thread}"


# --------------------------------------------------------------------- 登录
async def test_login_issues_working_token(rig):
    async with rig.factory() as session, session.begin():
        session.add(
            AppUser(
                tenant_id=TENANT_A,
                username="alice",
                password_hash=hash_password("test-only-password"),
                role=ROLE_READER,
            )
        )
    resp = await rig.client.post(
        "/api/auth/login",
        json={"tenant_id": TENANT_A, "username": "alice", "password": "test-only-password"},
    )
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    assert resp.json()["role"] == ROLE_READER
    # 这张令牌能过中间件
    resp = await rig.client.get("/api/threads", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    # /api/auth/token 是同一凭据的语义化别名
    resp = await rig.client.post(
        "/api/auth/token",
        json={"tenant_id": TENANT_A, "username": "alice", "password": "test-only-password"},
    )
    assert resp.status_code == 200 and resp.json()["access_token"]


async def test_login_failure_is_401_and_audited(rig):
    async with rig.factory() as session, session.begin():
        session.add(
            AppUser(
                tenant_id=TENANT_A,
                username="alice",
                password_hash=hash_password("test-only-password"),
                role=ROLE_READER,
            )
        )
    resp = await rig.client.post(
        "/api/auth/login",
        json={"tenant_id": TENANT_A, "username": "alice", "password": "definitely-wrong-password"},
    )
    assert resp.status_code == 401
    async with rig.factory() as session:
        rows = (
            (
                await session.execute(
                    sa.select(SecurityEvent).where(SecurityEvent.event == "login_failed")
                )
            )
            .scalars()
            .all()
        )
    assert rows, "失败的登录必须留痕"


async def test_login_is_rate_limited_after_repeated_failures(rig):
    """同一账号/IP 连续失败达到阈值后应回 429,而不是无限次免费猜密码。"""
    async with rig.factory() as session, session.begin():
        session.add(
            AppUser(
                tenant_id=TENANT_A,
                username="alice",
                password_hash=hash_password("test-only-password"),
                role=ROLE_READER,
            )
        )
    for _ in range(5):
        resp = await rig.client.post(
            "/api/auth/login",
            json={"tenant_id": TENANT_A, "username": "alice", "password": "definitely-wrong-password"},
        )
        assert resp.status_code == 401
    resp = await rig.client.post(
        "/api/auth/login",
        json={"tenant_id": TENANT_A, "username": "alice", "password": "definitely-wrong-password"},
    )
    assert resp.status_code == 429
    assert resp.headers.get("retry-after")


async def test_valid_password_succeeds_even_after_threshold(rig):
    """节流只影响失败尝试:合法口令不能被攻击者刷错密码锁死。"""
    async with rig.factory() as session, session.begin():
        session.add(
            AppUser(
                tenant_id=TENANT_A,
                username="alice",
                password_hash=hash_password("test-only-password"),
                role=ROLE_READER,
            )
        )
    for _ in range(5):
        resp = await rig.client.post(
            "/api/auth/login",
            json={"tenant_id": TENANT_A, "username": "alice", "password": "definitely-wrong-password"},
        )
        assert resp.status_code == 401
    resp = await rig.client.post(
        "/api/auth/login",
        json={"tenant_id": TENANT_A, "username": "alice", "password": "test-only-password"},
    )
    assert resp.status_code == 200, resp.text


async def test_concurrent_failed_logins_do_not_overshoot_audit_cap(rig):
    """check+insert 必须在同一把锁/事务里;并发 10 次失败只能落 5 条 login_failed。"""
    import asyncio

    async with rig.factory() as session, session.begin():
        session.add(
            AppUser(
                tenant_id=TENANT_A,
                username="alice",
                password_hash=hash_password("test-only-password"),
                role=ROLE_READER,
            )
        )
    responses = await asyncio.gather(
        *[
            rig.client.post(
                "/api/auth/login",
                json={"tenant_id": TENANT_A, "username": "alice", "password": "definitely-wrong-password"},
            )
            for _ in range(10)
        ]
    )
    assert sum(1 for one in responses if one.status_code == 429) >= 1
    async with rig.factory() as session:
        failed = (
            await session.execute(
                sa.select(sa.func.count())
                .select_from(SecurityEvent)
                .where(SecurityEvent.event == "login_failed")
            )
        ).scalar_one()
        throttled = (
            await session.execute(
                sa.select(sa.func.count())
                .select_from(SecurityEvent)
                .where(SecurityEvent.event == "login_throttled")
            )
        ).scalar_one()
    assert failed == 5, "并发下失败审计不能超过阈值"
    assert throttled == 1, "进入节流只记一次"


async def test_per_ip_throttle_stops_single_source_spray(rig, monkeypatch):
    from kbmesh.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "login_ip_max_failures", 3)
    monkeypatch.setattr(settings, "login_max_failures", 99)
    for idx in range(3):
        resp = await rig.client.post(
            "/api/auth/login",
            json={"tenant_id": TENANT_A, "username": f"user-{idx}", "password": "definitely-wrong-password"},
        )
        assert resp.status_code == 401
    resp = await rig.client.post(
        "/api/auth/login",
        json={"tenant_id": TENANT_A, "username": "user-x", "password": "definitely-wrong-password"},
    )
    assert resp.status_code == 429


def test_forwarded_for_is_walked_right_to_left(monkeypatch):
    from starlette.requests import Request

    from kbmesh.api.routers.auth import _client_ip
    from kbmesh.core.config import get_settings

    monkeypatch.setattr(get_settings(), "trusted_proxies", "10.0.0.0/8,192.168.0.1")

    def req(peer: str, xff: str) -> Request:
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/auth/login",
                "query_string": b"",
                "headers": [(b"x-forwarded-for", xff.encode())] if xff else [],
                "client": (peer, 1234),
            }
        )

    assert _client_ip(req("10.0.0.5", "203.0.113.9, 10.0.0.9")) == "203.0.113.9"
    assert _client_ip(req("10.0.0.5", "10.0.0.9, 10.0.0.8")) == "10.0.0.5"
    assert _client_ip(req("203.0.113.7", "1.2.3.4")) == "203.0.113.7"
    assert _client_ip(req("10.0.0.5", "not-an-ip")) == "10.0.0.5"


# --------------------------------------------------------------------- 反馈
async def test_feedback_roundtrip_and_cross_tenant_deny(rig):
    run_id = await _seed_run(rig.factory, tenant=TENANT_A, thread=f"{TENANT_A}:th-fb-1")
    resp = await rig.client.post(
        f"/api/runs/{run_id}/feedback",
        json={"score": 1, "corrected_answer": "隔离靠分区键。"},
        headers=bearer(CTX_READ),
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["score"] == 1
    resp = await rig.client.get(f"/api/runs/{run_id}/feedback", headers=bearer(CTX_READ))
    assert resp.status_code == 200 and len(resp.json()) == 1
    # 别人的 run 一分都打不上,还要留痕
    resp = await rig.client.post(
        f"/api/runs/{run_id}/feedback", json={"score": -1}, headers=bearer(CTX_B)
    )
    assert resp.status_code == 403
    async with rig.factory() as session:
        rows = (
            (
                await session.execute(
                    sa.select(SecurityEvent).where(
                        SecurityEvent.tenant_id == TENANT_B,
                        SecurityEvent.event == "cross_tenant_read",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert rows


# --------------------------------------------------------------------- 评估端点
async def test_eval_set_lifecycle_with_freeze_guard(rig):
    resp = await rig.client.post(
        "/api/eval-sets",
        json={
            "name": "隔离验收集",
            "cases": [{"question": "隔离怎么做?", "gold_refs": ["chunk-1"], "group_key": "iso"}],
        },
        headers=bearer(CTX_INDEX),
    )
    assert resp.status_code == 201, resp.text
    set_id = resp.json()["id"]
    resp = await rig.client.get(f"/api/eval-sets/{set_id}", headers=bearer(CTX_READ))
    assert resp.status_code == 200 and resp.json()["cases_total"] == 1
    resp = await rig.client.post(f"/api/eval-sets/{set_id}/freeze", headers=bearer(CTX_INDEX))
    assert resp.status_code == 200 and resp.json()["frozen_at"]
    # 重复冻结按 409 处理,不让触发器在数据库层才说话
    resp = await rig.client.post(f"/api/eval-sets/{set_id}/freeze", headers=bearer(CTX_INDEX))
    assert resp.status_code == 409
    # 评估集是租户资产:B 摸不到
    resp = await rig.client.get(f"/api/eval-sets/{set_id}", headers=bearer(CTX_B))
    assert resp.status_code == 403


async def test_eval_run_lifecycle_and_results(rig):
    resp = await rig.client.post(
        "/api/eval-sets",
        json={"name": "集合B", "cases": [{"question": "q1"}, {"question": "q2"}]},
        headers=bearer(CTX_INDEX),
    )
    set_id = resp.json()["id"]
    resp = await rig.client.post(
        "/api/eval-runs",
        json={"set_id": set_id, "config_snapshot": {"theta": 0.35}},
        headers=bearer(CTX_INDEX),
    )
    assert resp.status_code == 201, resp.text
    run_id = resp.json()["id"]
    assert resp.json()["status"] == "pending"  # 执行器是 #8 的活,端点先开账
    assert resp.json()["cases_total"] == 2
    resp = await rig.client.get(f"/api/eval-runs/{run_id}", headers=bearer(CTX_READ))
    assert resp.status_code == 200 and resp.json()["config_snapshot"] == {"theta": 0.35}
    resp = await rig.client.get(f"/api/eval-runs/{run_id}/results", headers=bearer(CTX_READ))
    assert resp.status_code == 200 and resp.json() == []
    resp = await rig.client.get(f"/api/eval-runs/{run_id}", headers=bearer(CTX_B))
    assert resp.status_code == 403


# --------------------------------------------------------------------- Trace 查询
async def test_trace_query_scoped_with_steps(rig):
    run_id = await _seed_run(rig.factory, tenant=TENANT_A, thread=f"{TENANT_A}:th-tr-1")
    resp = await rig.client.get("/api/traces/runs", headers=bearer(CTX_READ))
    assert resp.status_code == 200
    assert [row["id"] for row in resp.json()] == [run_id]
    resp = await rig.client.get(f"/api/traces/runs/{run_id}", headers=bearer(CTX_READ))
    assert resp.status_code == 200
    steps = resp.json()["steps"]
    assert [s["seq"] for s in steps] == [1, 2], "steps 按 seq 有序返回"
    assert [s["stage"] for s in steps] == ["route_intent", "retrieve_dual"]
    resp = await rig.client.get(f"/api/traces/runs/{run_id}", headers=bearer(CTX_B))
    assert resp.status_code == 403


# --------------------------------------------------------------------- 健康
async def test_audit_text_is_bounded_to_column_widths(rig):
    """审计字段不能因为调用方传了超长 target/path 就 DataError 或吞掉整条记录。"""
    async with rig.factory() as session, session.begin():
        await record_security_event(
            session,
            actor=CTX_READ,
            event="e" * 100,
            target="t" * 1000,
            path="p" * 1000,
        )
    async with rig.factory() as session:
        row = (
            await session.execute(
                sa.select(SecurityEvent).where(SecurityEvent.event.like("e%"))
            )
        ).scalars().one()
    assert len(row.event) == 40 and len(row.target) == 300 and len(row.path) == 300


async def test_online_eval_query_and_failure_list(rig):
    run_id = await _seed_run(rig.factory, tenant=TENANT_A, thread=f"{TENANT_A}:th-oe-1")
    async with rig.factory() as session, session.begin():
        session.add(
            OnlineEval(
                tenant_id=TENANT_A,
                run_id=run_id,
                format_ok=True,
                citation_ok=False,
                safety_ok=True,
                failed=["citation"],
                sampled=True,
                faithfulness=0.8,
                answer_relevancy=0.7,
            )
        )
    resp = await rig.client.get(
        f"/api/traces/runs/{run_id}/online-eval", headers=bearer(CTX_READ)
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["failed"] == ["citation"]
    assert body["faithfulness"] == pytest.approx(0.8)
    resp = await rig.client.get("/api/online-evals/failures", headers=bearer(CTX_READ))
    assert resp.status_code == 200 and len(resp.json()) == 1
    resp = await rig.client.get("/api/online-evals/failures", headers=bearer(CTX_B))
    assert resp.status_code == 200 and resp.json() == []


async def _seed_eval_report(rig, *, name: str, recall: float) -> tuple[str, str, str]:
    async with rig.factory() as session, session.begin():
        evset = EvalSet(tenant_id=TENANT_A, name=name, version=1)
        evset.cases = [
            EvalCase(
                id=f"case-{name}",
                question=f"q-{name}",
                gold_refs=["c1"],
                group_key="factual_single",
            )
        ]
        session.add(evset)
        await session.flush()
        run = EvalRun(tenant_id=TENANT_A, set_id=evset.id, status="done")
        session.add(run)
        await session.flush()
        session.add(
            EvalResult(
                eval_run_id=run.id,
                case_id=evset.cases[0].id,
                metric="recall@5",
                value=recall,
                detail={"answer": "答案", "question": f"q-{name}"},
            )
        )
        return evset.id, run.id, evset.cases[0].id


async def test_execute_eval_run_requires_admin(rig):
    _, run_id, _ = await _seed_eval_report(rig, name="exec", recall=1.0)
    resp = await rig.client.post(f"/api/eval-runs/{run_id}/execute", headers=bearer(CTX_READ))
    assert resp.status_code == 403
    async with rig.factory() as session:
        audits = list(
            (
                await session.execute(
                    sa.select(SecurityEvent).where(SecurityEvent.event == "role_denied")
                )
            )
            .scalars()
            .all()
        )
    assert audits, "越权执行评估必须留痕,与其它 role_denied 门禁同口径"
    resp = await rig.client.post(f"/api/eval-runs/{run_id}/execute", headers=bearer(CTX_INDEX))
    assert resp.status_code == 202, resp.text
    assert resp.json()["run_id"] == run_id
    assert resp.json()["submitted"] is True
    # 超长/非法 run_id 在路由层就被 422 挡下,不会带着任意长度写审计
    resp = await rig.client.post(
        f"/api/eval-runs/{'x' * 100}/execute", headers=bearer(CTX_INDEX)
    )
    assert resp.status_code == 422


async def test_eval_report_regression_and_csv_endpoints(rig):
    set_id, baseline_id, case_id = await _seed_eval_report(rig, name="base", recall=1.0)
    async with rig.factory() as session, session.begin():
        current = EvalRun(
            tenant_id=TENANT_A, set_id=set_id, baseline_run_id=baseline_id, status="done"
        )
        session.add(current)
        await session.flush()
        session.add(
            EvalResult(
                eval_run_id=current.id,
                case_id=case_id,
                metric="recall@5",
                value=0.0,
                detail={"answer": "变差的答案", "question": "q-base"},
            )
        )
        current_id = current.id

    resp = await rig.client.get(f"/api/eval-runs/{baseline_id}/report", headers=bearer(CTX_READ))
    assert resp.status_code == 200, resp.text
    assert resp.json()["metrics"]["recall@5"] == pytest.approx(1.0)

    resp = await rig.client.get(
        f"/api/eval-runs/{current_id}/regressions",
        params={"baseline_run_id": baseline_id},
        headers=bearer(CTX_READ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["regressions"][0]["delta"] == pytest.approx(-1.0)
    assert body["samples"][0]["question"] == "q-base"

    resp = await rig.client.get(
        f"/api/eval-runs/{current_id}/export.csv",
        params={"baseline_run_id": baseline_id},
        headers=bearer(CTX_READ),
    )
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    assert "q-base" in resp.text and "recall@5" in resp.text


async def test_import_eval_set_from_negative_feedback(rig):
    run_a = await _seed_run(rig.factory, tenant=TENANT_A, thread=f"{TENANT_A}:th-fb-import-a")
    run_b = await _seed_run(rig.factory, tenant=TENANT_B, thread=f"{TENANT_B}:th-fb-import-b")
    async with rig.factory() as session, session.begin():
        session.add(
            Feedback(
                tenant_id=TENANT_A,
                run_id=run_a,
                user_id="u",
                score=-1,
                corrected_answer="隔离靠分区键 [1]。",
            )
        )
        session.add(
            Feedback(
                tenant_id=TENANT_B,
                run_id=run_b,
                user_id="u",
                score=-1,
                corrected_answer="别人租户的答案",
            )
        )
    resp = await rig.client.post(
        "/api/eval-sets/from-feedback",
        json={"name": "反馈导入集", "max_score": -1},
        headers=bearer(CTX_INDEX),
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["cases_total"] == 1, "只能导入本租户的反馈"
    set_id = resp.json()["id"]
    async with rig.factory() as session:
        cases = list(
            (await session.execute(sa.select(EvalCase).where(EvalCase.set_id == set_id))).scalars()
        )
    assert len(cases) == 1
    assert cases[0].question == "q", "题目来自对应 trace_run 的原始 query"
    assert cases[0].reference_answer == "隔离靠分区键 [1]。"


async def test_trace_stats_aggregate_and_stay_in_tenant(rig):
    async with rig.factory() as session, session.begin():
        session.add(
            TraceRun(
                id="run-stats-a",
                tenant_id=TENANT_A,
                thread_id=f"{TENANT_A}:th-stats",
                query="q",
                final_status="answered",
                latency_ms=100,
                token_in=10,
                token_out=5,
                cost_usd=0.01,
            )
        )
        session.add_all(
            [
                TraceStep(
                    run_id="run-stats-a", seq=1, stage="retrieve_body", payload={"count": 3}, latency_ms=20
                ),
                TraceStep(
                    run_id="run-stats-a", seq=2, stage="retrieve_dense", payload={"count": 0}, latency_ms=30
                ),
                TraceStep(run_id="run-stats-a", seq=3, stage="rerank", payload={}, latency_ms=40),
            ]
        )
        session.add(
            TraceRun(
                id="run-stats-b",
                tenant_id=TENANT_B,
                thread_id=f"{TENANT_B}:th-stats",
                query="q",
                final_status="error",
            )
        )
    resp = await rig.client.get("/api/traces/stats", headers=bearer(CTX_READ))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["runs"] == 1
    assert body["answered"] == 1
    assert body["latency_ms"]["p95"] == pytest.approx(100.0)
    assert body["token_in"] == 10 and body["token_out"] == 5
    assert body["route_hits"] == {"body": 1, "heading": 0, "dense": 0, "graph": 0}
    assert body["stage_latency_ms"]["rerank"]["p95"] == pytest.approx(40.0)
    # 另一个租户只看得到自己那条 error,看不到 A 的 answered 与 token
    resp = await rig.client.get("/api/traces/stats", headers=bearer(CTX_B))
    assert resp.status_code == 200
    other = resp.json()
    assert other["runs"] == 1 and other["error"] == 1 and other["answered"] == 0
    assert other["token_in"] == 0 and other["token_out"] == 0


async def test_readyz_reports_components(rig):
    resp = await rig.client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["components"] == {"db": "ok", "checkpoint": "ok"}
    resp = await rig.client.get("/healthz")
    assert resp.status_code == 200


async def test_unauthenticated_request_is_401(rig):
    resp = await rig.client.get("/api/threads")
    assert resp.status_code == 401
    resp = await rig.client.get("/api/threads", headers=bearer(CTX_READ))
    assert resp.status_code == 200  # 对照:带令牌就能进
