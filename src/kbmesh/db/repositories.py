"""防线③:仓储层只接收 `TenantContext`,不接收裸 `tenant_id`。

技术文档 §6.1 把这条写成审查红线 —— 「任何函数签名里出现裸 `tenant_id: str` 参数都视为缺陷」。
这里的做法是让越权**在类型上写不出来**:仓库构造时就要 ctx,每条语句自动挂租户谓词,
调用方连「传别人的租户」这个参数位置都找不到。

`kb_scope` 与 `tenant_id` 是两回事:前者是令牌里带的知识库子集(可以缺省=整租户),后者永远来自
验签。所以两者都要进谓词,缺一个都是漏。
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from kbmesh.core.errors import IllegalTransition, PermissionDenied
from kbmesh.core.tenancy import TenantContext
from kbmesh.db.models import (
    FINAL_STATUSES,
    STAGE_ORDER,
    AppUser,
    Chunk,
    Document,
    Entity,
    EvalCase,
    EvalResult,
    EvalRun,
    EvalSet,
    Feedback,
    KnowledgeBase,
    OnlineEval,
    ParseStage,
    RelationEdge,
    SecurityEvent,
    Tenant,
    TraceRun,
    TraceStep,
)


def _audit_text(value: Any, limit: int) -> str:
    """审计列有 varchar 上限;超长输入截断而不是让整条审计因 DataError 丢失。"""
    text = str(value if value is not None else "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class _Scoped:
    """租户谓词的唯一下发点:任何绕过 `_scope()` 的语句都算缺陷。"""

    def __init__(self, ctx: TenantContext, session: AsyncSession) -> None:
        self.ctx = ctx
        self.session = session

    def tenant_only(self, model: Any) -> Select:
        return select(model).where(model.tenant_id == self.ctx.tenant_id)

    def scoped(self, model: Any, *, kb_column: str = "kb_id") -> Select:
        """租户 + 令牌 kb_scope 双重过滤。

        `kb_column` 指到「这张表上的 kb 归属列」:业务表是 `kb_id`,`KnowledgeBase` 自身是 `id`。
        默认跳过过滤是错的 —— scope 非空时它会把整个租户都放出来。
        """
        stmt = self.tenant_only(model)
        if self.ctx.kb_scope:
            stmt = stmt.where(getattr(model, kb_column).in_(set(self.ctx.kb_scope)))
        return stmt

    async def deny(self, event: str, target: str, path: str = "", **detail: Any) -> None:
        """越权/拒绝一律留痕后再抛,不静默返回空。"""
        self.session.add(
            SecurityEvent(
                tenant_id=self.ctx.tenant_id,
                user_id=self.ctx.user_id,
                event=_audit_text(event, 40),
                target=_audit_text(target, 300),
                path=_audit_text(path, 300),
                detail=detail or {},
            )
        )
        await self.session.flush()
        raise PermissionDenied(f"{event}: {target}")

    async def require_write(self, target: str, path: str = "") -> None:
        """写权限也走 deny():拒绝必须留痕,不能只抛一个异常就完事。"""
        if not self.ctx.can_write_kb():
            await self.deny("role_denied", target, path)


class KBRepo(_Scoped):
    async def list(self) -> list[KnowledgeBase]:
        return list((await self.session.execute(self.scoped(KnowledgeBase, kb_column="id"))).scalars())

    async def get(self, kb_id: str) -> KnowledgeBase | None:
        """跨租户或超出 kb_scope 都返回 None —— 不区分「不存在」与「不归你」,避免存在性泄露。"""
        stmt = self.scoped(KnowledgeBase, kb_column="id").where(KnowledgeBase.id == kb_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def create(
        self, name: str, embedding_model: str, chunk_policy: dict[str, Any] | None = None
    ) -> KnowledgeBase:
        await self.require_write(f"kb:{name}")
        row = KnowledgeBase(
            tenant_id=self.ctx.tenant_id,
            name=name,
            embedding_model=embedding_model,
            chunk_policy=chunk_policy or {},
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def assert_readable(self, kb_id: str) -> KnowledgeBase:
        row = await self.get(kb_id)
        if row is None:
            await self.deny("kb_scope_denied", f"kb:{kb_id}")
        return row  # type: ignore[return-value]

    async def delete(self, kb_id: str) -> None:
        await self.require_write(f"kb:{kb_id}")
        row = await self.get(kb_id)
        if row is None:
            await self.deny("kb_scope_denied", f"kb:{kb_id}")
        await self.session.delete(row)
        await self.session.flush()


class DocumentRepo(_Scoped):
    async def list_for_kb(self, kb_id: str) -> list[Document]:
        await KBRepo(self.ctx, self.session).assert_readable(kb_id)
        stmt = self.scoped(Document).where(Document.kb_id == kb_id)
        return list((await self.session.execute(stmt)).scalars())

    async def get(self, doc_id: str) -> Document | None:
        stmt = self.scoped(Document).where(Document.id == doc_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def require(self, doc_id: str) -> Document:
        row = await self.get(doc_id)
        if row is None:
            await self.deny("cross_tenant_read", f"doc:{doc_id}")
        return row  # type: ignore[return-value]

    async def create(
        self, kb_id: str, source_uri: str, sha256: str, title: str = "", mime: str = ""
    ) -> Document:
        await self.require_write(f"kb:{kb_id}")
        await KBRepo(self.ctx, self.session).assert_readable(kb_id)
        row = Document(
            tenant_id=self.ctx.tenant_id,
            kb_id=kb_id,
            source_uri=source_uri,
            sha256=sha256,
            title=title,
            mime=mime,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get_by_sha(self, kb_id: str, sha256: str) -> Document | None:
        """同内容幂等的查询侧:上传入口拿它决定「建新行」还是「返回已有的」。"""
        stmt = self.scoped(Document).where(Document.kb_id == kb_id, Document.sha256 == sha256)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def restart(self, doc_id: str) -> Document:
        """重索引(§5.3):`index_version` 递增换一套任务 id,阶段从头跑。

        状态机的「禁止跳跃」管的是**管道内**的推进(§5.2);重索引是新一轮运行,回到 PENDING
        不算跳跃 —— 任务幂等靠 `index_version` 隔开,不靠状态机拦(与 ChunkRepo 的同一注释)。
        旧一轮的向量/切片由管道的孤儿清理与 Milvus upsert 收敛,这里只负责开新一轮的账。
        """
        row = await self.require(doc_id)
        await self.require_write(f"doc:{doc_id}")
        row.index_version = int(row.index_version or 1) + 1
        row.parse_status = ParseStage.PENDING.value
        row.failure = None
        await self.session.flush()
        # `updated_at` 由数据库的 onupdate 生成,flush 后该列被标记过期;HTTP 层紧接着要序列化
        # 它,异步 session 里的懒加载会抛 MissingGreenlet。这里主动取回,别把 IO 留到序列化。
        await self.session.refresh(row, attribute_names=["updated_at"])
        return row

    async def set_stage(self, doc_id: str, new_stage: ParseStage) -> Document:
        """§5.2 状态机:只能前进一步、原地重试,或从 FAILED 重启;跳跃直接抛。"""
        row = await self.require(doc_id)
        await self.require_write(f"doc:{doc_id}")
        _assert_stage_move(row.parse_status, new_stage)
        row.parse_status = new_stage.value
        await self.session.flush()
        return row

    async def mark_failed(self, doc_id: str, failure: dict[str, Any]) -> Document:
        row = await self.require(doc_id)
        await self.require_write(f"doc:{doc_id}")
        row.parse_status = ParseStage.FAILED.value
        row.failure = failure
        await self.session.flush()
        return row


def _assert_stage_move(current: str, new_stage: ParseStage) -> None:
    if current == ParseStage.FAILED.value:
        return  # 失败重试:允许重跑任意前置阶段
    try:
        now_idx = STAGE_ORDER.index(ParseStage(current))
    except ValueError as exc:  # 库里出现了枚举外的状态 = 数据被手改过
        raise IllegalTransition(f"unknown parse_status {current!r}") from exc
    target_idx = STAGE_ORDER.index(new_stage)
    # -1 是「原地重跑」:每阶段完成时状态已经前进一格,再跑同一阶段得先退回自己的运行中标记(§5.3)
    if target_idx in (now_idx - 1, now_idx, now_idx + 1):
        return
    if current == ParseStage.DONE.value and target_idx <= 1:
        # 重索引:换了切片策略或嵌入模型,已完成的文档要从解析重来。
        # 与旧一轮的冲突靠 `index_version` 递增隔开,不靠状态机拦。
        return
    raise IllegalTransition(f"{current} → {new_stage.value} 是跳跃")


class ChunkRepo(_Scoped):
    async def list_for_doc(self, doc_id: str) -> list[Chunk]:
        stmt = self.scoped(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.position)
        return list((await self.session.execute(stmt)).scalars())

    async def by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """检索融合后的回表:跨租户的 id 会被谓词吃掉,不会带出别人的正文。"""
        if not chunk_ids:
            return []
        stmt = self.scoped(Chunk).where(Chunk.id.in_(set(chunk_ids)))
        return list((await self.session.execute(stmt)).scalars())

    async def count_for_doc(self, doc_id: str) -> int:
        stmt = self.tenant_only(Chunk).where(Chunk.doc_id == doc_id)
        count = await self.session.scalar(
            select(func.count()).select_from(stmt.subquery())
        )
        return int(count or 0)

    async def insert_many_ignoring(self, rows: list[dict[str, Any]]) -> int:
        """§5.3 幂等第一层:`(tenant_id, chunk_key)` 冲突就跳过,重跑不翻倍。"""
        if not rows:
            return 0
        stmt = (
            pg_insert(Chunk)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["tenant_id", "chunk_key"])
            .returning(Chunk.id)
        )
        inserted = (await self.session.execute(stmt)).scalars().all()
        return len(inserted)

    async def delete_for_doc(self, doc_id: str) -> int:
        await self.require_write(f"doc:{doc_id}")
        stmt = self.scoped(Chunk).where(Chunk.doc_id == doc_id)
        rows = list((await self.session.execute(stmt)).scalars())
        for row in rows:
            await self.session.delete(row)
        await self.session.flush()
        return len(rows)

    async def delete_keys_not_in(self, doc_id: str, keys: set[str]) -> int:
        """重切后的孤儿清理:键不在本轮结果里的旧切片必须消失,否则旧向量还能被召回。"""
        await self.require_write(f"doc:{doc_id}")
        stmt = self.scoped(Chunk).where(Chunk.doc_id == doc_id)
        if keys:
            stmt = stmt.where(Chunk.chunk_key.notin_(keys))
        rows = list((await self.session.execute(stmt)).scalars())
        ids = [r.id for r in rows]
        if ids:
            # 边是抽取的产物:切片没了,挂在它上面的证据边就同时失效(外键也会挡住删除)。
            # 本轮的 extract 阶段会按新切片重新导出这些边,所以这里删掉不是丢数据。
            await self.session.execute(
                delete(RelationEdge).where(
                    RelationEdge.tenant_id == self.ctx.tenant_id,
                    RelationEdge.chunk_id.in_(ids),
                )
            )
        for row in rows:
            await self.session.delete(row)
        await self.session.flush()
        return len(rows)


class GraphRepo(_Scoped):
    """实体/关系的权威记录(Neo4j 只是可重建的投影)。"""

    async def upsert_entities(self, kb_id: str, items: list[dict[str, Any]]) -> int:
        if not self.ctx.allows_kb(kb_id):
            await self.deny("kb_scope_denied", f"kb:{kb_id}")
        if not items:
            return 0
        stmt = pg_insert(Entity).values(
            [{**one, "tenant_id": self.ctx.tenant_id, "kb_id": kb_id} for one in items]
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["tenant_id", "entity_key"],
            # label 取首次写入的那一条:实体一经入图就该有稳定名字,重跑不能把它改来改去
            set_={"aliases": stmt.excluded.aliases},
        )
        await self.session.execute(stmt)
        return len(items)

    async def upsert_edges(self, kb_id: str, items: list[dict[str, Any]]) -> int:
        if not self.ctx.allows_kb(kb_id):
            await self.deny("kb_scope_denied", f"kb:{kb_id}")
        if not items:
            return 0
        stmt = pg_insert(RelationEdge).values(
            [{**one, "tenant_id": self.ctx.tenant_id, "kb_id": kb_id} for one in items]
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["tenant_id", "src_key", "dst_key", "rtype"],
            set_={
                "evidence": stmt.excluded.evidence,
                "chunk_id": stmt.excluded.chunk_id,
                "confidence": stmt.excluded.confidence,
            },
        )
        await self.session.execute(stmt)
        return len(items)

    async def mark_linked(self, chunk_ids: set[str]) -> int:
        if not chunk_ids:
            return 0
        stmt = (
            update(Chunk)
            .where(Chunk.tenant_id == self.ctx.tenant_id, Chunk.id.in_(chunk_ids))
            .values(graph_linked=True)
        )
        result = await self.session.execute(stmt)
        return int(result.rowcount or 0)


class TraceRepo(_Scoped):
    async def start_run(
        self,
        thread_id: str,
        query: str,
        *,
        route_flags: dict[str, Any] | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> TraceRun:
        row = TraceRun(
            tenant_id=self.ctx.tenant_id,
            thread_id=thread_id,
            user_id=self.ctx.user_id,
            query=query,
            route_flags=route_flags or {},
            config_snapshot=config_snapshot or {},
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def add_step(
        self,
        run_id: str,
        seq: int,
        stage: str,
        *,
        payload: dict[str, Any] | None = None,
        latency_ms: int = 0,
        token_in: int = 0,
        token_out: int = 0,
        model: str = "",
    ) -> TraceStep:
        run = await self.get_run(run_id)
        step = TraceStep(
            run_id=run.id,
            seq=seq,
            stage=stage,
            payload=payload or {},
            latency_ms=latency_ms,
            token_in=token_in,
            token_out=token_out,
            model=model,
        )
        self.session.add(step)
        await self.session.flush()
        return step

    async def step_once(
        self,
        run_id: str,
        seq: int,
        stage: str,
        *,
        payload: dict[str, Any] | None = None,
        latency_ms: int = 0,
        token_in: int = 0,
        token_out: int = 0,
        model: str = "",
    ) -> bool:
        """和 `add_step` 同一件事,但 `(run_id, seq)` 已存在就跳过,返回是否真的写入。

        编排层的 trace 是一步一事务真实提交的(`agents/trace.py`):进程崩在「step 已提交、
        检查点里的 `trace_seq` 还没前进」中间时,resume 会拿同一个 seq 再写一次。这里必须幂等,
        否则 FR-403 的恢复路径开口第一句就是唯一约束报错。
        """
        await self.get_run(run_id)  # 租户校验:别人的 run 一步都不许写
        stmt = (
            pg_insert(TraceStep)
            .values(
                run_id=run_id,
                seq=seq,
                stage=stage,
                payload=payload or {},
                latency_ms=latency_ms,
                token_in=token_in,
                token_out=token_out,
                model=model,
            )
            .on_conflict_do_nothing(index_elements=["run_id", "seq"])
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        return (result.rowcount or 0) > 0

    async def finish_run(
        self,
        run_id: str,
        *,
        final_status: str,
        degraded: bool = False,
        latency_ms: int = 0,
        token_in: int = 0,
        token_out: int = 0,
        cost_usd: float = 0,
    ) -> TraceRun:
        """收尾一次问答:终态只有 `FINAL_STATUSES` 四个值,写别的直接抛。

        这个常量在这里第一次被引用不是巧合:FR-406 要「拒答样本不进入正常答案统计口径」,
        靠的就是这一列的取值范围 —— 让编排层随便写一个字符串,统计口径当天就会失真。
        `expired`(TTL 到期)也是在这里落,不是靠额外加的标记列。
        """
        if final_status not in FINAL_STATUSES:
            raise IllegalTransition(f"final_status 必须是 {FINAL_STATUSES} 之一: {final_status!r}")
        row = await self.get_run(run_id)
        row.final_status = final_status
        row.degraded = degraded
        row.latency_ms = latency_ms
        row.token_in = token_in
        row.token_out = token_out
        row.cost_usd = cost_usd
        await self.session.flush()
        return row

    async def assert_thread(self, thread_key: str) -> None:
        """跨租户续跑的第三道闸:前缀必须是自己租户,且库里查得到这条会话。

        单独看 `thread_key` 的租户前缀是可信的(由 `runtime.thread_key()` 从验签后的 ctx 拼),
        但 resume 的入参来自客户端 —— 拿 B 租户的 key 换个前缀就变成 A 的了。这里两道都要过,
        并且**先 deny 留痕再抛**:「有人试图续跑别人的会话」是审计事件,不是 404。
        """
        prefix, sep, bare = thread_key.partition(":")
        if not sep or prefix != self.ctx.tenant_id or not bare:
            await self.deny("cross_tenant_read", f"thread:{thread_key}", path="resume")
        stmt = self.tenant_only(TraceRun).where(TraceRun.thread_id == thread_key)
        if (await self.session.execute(stmt.limit(1))).scalar_one_or_none() is None:
            await self.deny("cross_tenant_read", f"thread:{thread_key}", path="resume",
                           reason="该会话在本租户下没有运行记录")

    async def get_run(self, run_id: str) -> TraceRun:
        stmt = self.tenant_only(TraceRun).where(TraceRun.id == run_id)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is None:
            await self.deny("cross_tenant_read", f"run:{run_id}")
        return row  # type: ignore[return-value]

    async def list_runs(self, limit: int = 20) -> list[TraceRun]:
        stmt = self.tenant_only(TraceRun).order_by(TraceRun.created_at.desc()).limit(limit)
        return list((await self.session.execute(stmt)).scalars())

    async def list_steps(self, run_id: str) -> list[TraceStep]:
        """`trace_step` 没有 `tenant_id` 列:租户谓词必须借 `trace_run` 下推,直接裸查就是跨租户读。"""
        await self.get_run(run_id)
        stmt = (
            select(TraceStep)
            .join(TraceRun, TraceStep.run_id == TraceRun.id)
            .where(TraceStep.run_id == run_id, TraceRun.tenant_id == self.ctx.tenant_id)
            .order_by(TraceStep.seq)
        )
        return list((await self.session.execute(stmt)).scalars())


class FeedbackRepo(_Scoped):
    """FR-609 反馈回流:评分 + 修正答案进反馈池,#8 的评估集从这里导入。"""

    async def add(self, run_id: str, *, score: int, corrected_answer: str | None = None) -> Feedback:
        run = await TraceRepo(self.ctx, self.session).get_run(run_id)  # 别人的 run 一分都打不上
        row = Feedback(
            tenant_id=self.ctx.tenant_id,
            run_id=run.id,
            user_id=self.ctx.user_id,
            score=score,
            corrected_answer=corrected_answer,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_for_run(self, run_id: str) -> list[Feedback]:
        await TraceRepo(self.ctx, self.session).get_run(run_id)
        stmt = self.tenant_only(Feedback).where(Feedback.run_id == run_id)
        return list((await self.session.execute(stmt)).scalars())


class EvalRepo(_Scoped):
    """评估集与评估运行(#7 只接端点;指标逻辑归 #8)。

    `eval_set` 冻结后有触发器拒 UPDATE/DELETE —— 这里同样先挡一道:对冻结集的改动请求根本
    不该走到数据库报错那一步,那等于把 DBA 的告警当成接口文档。
    """

    async def list_sets(self) -> list[EvalSet]:
        stmt = self.tenant_only(EvalSet).order_by(EvalSet.created_at.desc())
        return list((await self.session.execute(stmt)).scalars())

    async def get_set(self, set_id: str) -> EvalSet | None:
        stmt = self.tenant_only(EvalSet).where(EvalSet.id == set_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def require_set(self, set_id: str) -> EvalSet:
        row = await self.get_set(set_id)
        if row is None:
            await self.deny("cross_tenant_read", f"eval_set:{set_id}")
        return row  # type: ignore[return-value]

    async def create_set(self, name: str, *, cases: list[dict[str, Any]] | None = None) -> EvalSet:
        await self.require_write(f"eval_set:{name}")
        row = EvalSet(tenant_id=self.ctx.tenant_id, name=name, version=1)
        row.cases = [
            EvalCase(
                question=str(one.get("question") or ""),
                reference_answer=(
                    str(one["reference_answer"]) if one.get("reference_answer") else None
                ),
                gold_refs=list(one.get("gold_refs") or []),
                group_key=str(one.get("group_key") or "default"),
                expected_refusal=bool(one.get("expected_refusal") or False),
            )
            for one in (cases or [])
        ]
        self.session.add(row)
        await self.session.flush()
        return row

    async def import_feedback(
        self, name: str, *, max_score: int | None = None, limit: int = 200
    ) -> EvalSet:
        """FR-609:把本租户的反馈（评分 + 修正答案）导入成一个候选评估集。

        `max_score` 是分层口径：默认导入全部；传 -1 只导入踩。题目取对应 run 的原问，
        修正答案进 `reference_answer`；`gold_refs` 留空等人工确认，不拿文本冒充 chunk id。
        """
        await self.require_write(f"eval_set:{name}")
        stmt = (
            select(Feedback, TraceRun)
            .join(TraceRun, Feedback.run_id == TraceRun.id)
            .where(
                Feedback.tenant_id == self.ctx.tenant_id,
                TraceRun.tenant_id == self.ctx.tenant_id,
            )
            .order_by(Feedback.created_at.desc())
            .limit(max(1, min(limit, 500)))
        )
        if max_score is not None:
            stmt = stmt.where(Feedback.score <= max_score)

        existing = await self.session.execute(
            select(func.max(EvalSet.version)).where(
                EvalSet.tenant_id == self.ctx.tenant_id, EvalSet.name == name
            )
        )
        version = int(existing.scalar_one_or_none() or 0) + 1

        cases: list[EvalCase] = []
        seen: set[str] = set()
        for feedback, run in (await self.session.execute(stmt)).all():
            question = str(run.query or "").strip()
            if not question or question in seen:
                continue
            seen.add(question)
            cases.append(
                EvalCase(
                    question=question,
                    reference_answer=feedback.corrected_answer,
                    gold_refs=[],
                    group_key="feedback",
                )
            )
        row = EvalSet(tenant_id=self.ctx.tenant_id, name=name, version=version)
        row.cases = cases
        self.session.add(row)
        await self.session.flush()
        return row

    async def freeze_set(self, set_id: str) -> EvalSet:
        """FR-606:冻结之后内容不可原地变更,评估才可复现。重复冻结按 409 处理。"""
        row = await self.require_set(set_id)
        await self.require_write(f"eval_set:{row.name}")
        if row.frozen_at is not None:
            raise IllegalTransition(f"评估集已冻结({row.frozen_at}),内容不可再改")
        row.frozen_at = datetime.now(UTC)
        await self.session.flush()
        return row

    async def case_count(self, set_id: str) -> int:
        stmt = select(func.count()).select_from(EvalCase).where(EvalCase.set_id == set_id)
        return int((await self.session.execute(stmt)).scalar_one())

    async def create_run(
        self,
        set_id: str,
        *,
        config_snapshot: dict[str, Any] | None = None,
        baseline_run_id: str | None = None,
    ) -> EvalRun:
        row = await self.require_set(set_id)
        await self.require_write(f"eval_set:{row.name}")
        run = EvalRun(
            tenant_id=self.ctx.tenant_id,
            set_id=row.id,
            config_snapshot=config_snapshot or {},
            baseline_run_id=baseline_run_id,
            status="pending",
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def get_run(self, run_id: str) -> EvalRun:
        stmt = self.tenant_only(EvalRun).where(EvalRun.id == run_id)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is None:
            await self.deny("cross_tenant_read", f"eval_run:{run_id}")
        return row  # type: ignore[return-value]

    async def list_runs(self, *, set_id: str | None = None, limit: int = 20) -> list[EvalRun]:
        stmt = self.tenant_only(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)
        if set_id is not None:
            stmt = stmt.where(EvalRun.set_id == set_id)
        return list((await self.session.execute(stmt)).scalars())

    async def results(self, run_id: str) -> list[EvalResult]:
        # `eval_result` 没有 tenant_id:租户归属由 `eval_run` 决定。先 get_run 完成租户校验与
        # 越权留痕,再按 run_id 取结果,不再套一层 tenant_only(那张表没有这个列)。
        await self.get_run(run_id)
        stmt = select(EvalResult).where(EvalResult.eval_run_id == run_id)
        return list((await self.session.execute(stmt)).scalars())

    async def cases_for_set(self, set_id: str) -> list[EvalCase]:
        """一次评估的题目按创建顺序取;`require_set` 先把租户校验做掉(case 表没有 tenant_id)。"""
        await self.require_set(set_id)
        stmt = (
            select(EvalCase)
            .where(EvalCase.set_id == set_id)
            .order_by(EvalCase.created_at, EvalCase.id)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def record_case_results(
        self, run_id: str, case_id: str, rows: list[tuple[str, float, dict[str, Any]]]
    ) -> None:
        """一个 case 的全部指标一次写入;(run,case) 先删后插,续跑重算不会撞唯一约束。"""
        await self.get_run(run_id)
        await self.session.execute(
            delete(EvalResult).where(
                EvalResult.eval_run_id == run_id, EvalResult.case_id == case_id
            )
        )
        for metric, value, detail in rows:
            self.session.add(
                EvalResult(
                    eval_run_id=run_id,
                    case_id=case_id,
                    metric=metric,
                    value=float(value),
                    detail=detail,
                )
            )
        await self.session.flush()

    async def advance_run(
        self, run_id: str, *, cursor_case_id: str, status: str = "running"
    ) -> EvalRun:
        row = await self.get_run(run_id)
        row.cursor_case_id = cursor_case_id
        row.status = status
        await self.session.flush()
        return row

    async def complete_run(self, run_id: str) -> EvalRun:
        row = await self.get_run(run_id)
        row.status = "done"
        row.finished_at = datetime.now(UTC)
        await self.session.flush()
        return row

    async def results_with_cases(self, run_id: str) -> list[tuple[EvalResult, EvalCase]]:
        """报告要按 group 切片,所以结果必须带上 case 的 group/question。"""
        await self.get_run(run_id)
        stmt = (
            select(EvalResult, EvalCase)
            .join(EvalCase, EvalResult.case_id == EvalCase.id)
            .where(EvalResult.eval_run_id == run_id)
            .order_by(EvalCase.id, EvalResult.metric)
        )
        return list((await self.session.execute(stmt)).all())


class OnlineEvalRepo(_Scoped):
    """在线评估结果(FR-601):一次 run 一行;失败样本用 `failed` 非空筛。"""

    async def record(
        self,
        run_id: str,
        *,
        format_ok: bool,
        citation_ok: bool,
        safety_ok: bool,
        failed: list[str],
        sampled: bool,
        faithfulness: float | None = None,
        answer_relevancy: float | None = None,
    ) -> OnlineEval:
        await TraceRepo(self.ctx, self.session).get_run(run_id)  # 租户校验 + 越权留痕
        stmt = select(OnlineEval).where(OnlineEval.run_id == run_id)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is None:
            row = OnlineEval(tenant_id=self.ctx.tenant_id, run_id=run_id)
            self.session.add(row)
        row.format_ok = format_ok
        row.citation_ok = citation_ok
        row.safety_ok = safety_ok
        row.failed = list(failed)
        row.sampled = sampled
        row.faithfulness = faithfulness
        row.answer_relevancy = answer_relevancy
        await self.session.flush()
        return row

    async def get_for_run(self, run_id: str) -> OnlineEval | None:
        await TraceRepo(self.ctx, self.session).get_run(run_id)
        stmt = self.tenant_only(OnlineEval).where(OnlineEval.run_id == run_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_failures(self, limit: int = 100) -> list[OnlineEval]:
        stmt = (
            self.tenant_only(OnlineEval)
            .order_by(OnlineEval.created_at.desc())
            .limit(max(1, min(limit, 500)))
        )
        rows = list((await self.session.execute(stmt)).scalars())
        return [row for row in rows if row.failed]


async def record_security_event(
    session: AsyncSession,
    *,
    actor: TenantContext | None,
    event: str,
    target: str = "",
    path: str = "",
    detail: dict[str, Any] | None = None,
) -> None:
    """无令牌/坏令牌也留痕:`actor=None` 时 `tenant_id` 为空,这是这张表唯一允许空租户的地方。"""
    session.add(
        SecurityEvent(
            tenant_id=actor.tenant_id if actor else None,
            user_id=_audit_text(actor.user_id, 64) if actor else "",
            event=_audit_text(event, 40),
            target=_audit_text(target, 300),
            path=_audit_text(path, 300),
            detail=detail or {},
        )
    )
    await session.flush()


async def list_security_events(
    ctx: TenantContext, session: AsyncSession, *, cross_tenant: bool = False, limit: int = 100
) -> list[SecurityEvent]:
    stmt = select(SecurityEvent)
    if not (cross_tenant and ctx.can_write_kb()):
        stmt = stmt.where(SecurityEvent.tenant_id == ctx.tenant_id)
    stmt = stmt.order_by(SecurityEvent.created_at.desc()).limit(limit)
    return list((await session.execute(stmt)).scalars())


class UserRepo(_Scoped):
    async def list(self) -> list[AppUser]:
        return list((await self.session.execute(self.tenant_only(AppUser))).scalars())

    async def create(self, username: str, password_hash: str, role: str) -> AppUser:
        await self.require_write(f"user:{username}")
        row = AppUser(
            tenant_id=self.ctx.tenant_id, username=username, password_hash=password_hash, role=role
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def delete(self, user_id: str) -> None:
        await self.require_write(f"user:{user_id}")
        stmt = self.tenant_only(AppUser).where(AppUser.id == user_id)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is None:
            await self.deny("kb_scope_denied", f"user:{user_id}")
        await self.session.delete(row)
        await self.session.flush()


class TenantAdminRepo:
    """跨租户管理,仅 platform_admin 可用。ctx 仍必传:审计留痕需要身份。"""

    def __init__(self, ctx: TenantContext, session: AsyncSession) -> None:
        self.ctx = ctx
        self.session = session

    async def list_tenants(self) -> list[Tenant]:
        return list((await self.session.execute(select(Tenant).order_by(Tenant.created_at))).scalars())

    async def create_tenant(self, name: str) -> Tenant:
        row = Tenant(name=name)
        self.session.add(row)
        await self.session.flush()
        return row

    async def delete_tenant(self, target_id: str) -> None:
        row = (await self.session.execute(select(Tenant).where(Tenant.id == target_id))).scalar_one_or_none()
        if row is None:
            raise IllegalTransition(f"租户不存在: {target_id}")
        await self.session.delete(row)
        await self.session.flush()
