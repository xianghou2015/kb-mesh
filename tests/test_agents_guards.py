"""五层防御的判定验收(FR-404 / FR-405 / FR-408)—— 只跑纯函数,不起图、不连模型。

这些用例存在的理由全部是「方向」:0.35 这条线上「低于」才丢,0.45 等于算达标;`[9]` 是剥离
还是拒答;重试到底算不算「原样重跑」。每一条都是读代码看不出来、错了也不会报错的那种,所以
全部钉成断言,而不靠节点里的注释。

阈值在这里一律写成显式入参(`theta_rel=0.35` 而非读 Settings),这样改配置和改判定在测试里
是两个动作 —— 只有后者会让这批用例响。
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import pytest

from kbmesh.agents import guards
from kbmesh.agents.schemas import IntentOut, ResumeChoice, RewriteCandidate
from kbmesh.agents.state import REJECT_CODES, STAGES, empty_state, next_seq, used_seq
from kbmesh.core.errors import IllegalTransition
from kbmesh.retrieval.contracts import Hit


def _hit(chunk_id: str, score: float = 0.5) -> Hit:
    return Hit(chunk_id=chunk_id, score=score, source_routes={"body"}, text=f"正文 {chunk_id}")


def _hits(*chunk_ids: str) -> list[Hit]:
    return [_hit(cid) for cid in chunk_ids]


def _intent(verdict: str, confidence: float = 0.9) -> IntentOut:
    return IntentOut(verdict=verdict, confidence=confidence, reason="fixture")  # type: ignore[arg-type]


# ---------------------------------------------------------------- 层1:意图门禁
@pytest.mark.parametrize(
    ("verdict", "allow", "final_status", "reject_code", "event"),
    [
        ("answerable", True, "", "", None),
        ("out_of_privilege", False, "refused", "intent_forbidden", "intent_out_of_privilege"),
        ("out_of_scope", False, "refused", "out_of_scope", None),
        ("chitchat", False, "refused", "chitchat", None),
    ],
)
def test_intent_gate_table(verdict, allow, final_status, reject_code, event) -> None:
    """四档判定逐条钉死。闲聊也是终态:让闲聊去检索一轮,答案就必须有证据这条前提就破了。"""
    gate = guards.intent_gate(_intent(verdict))
    assert (gate.allow, gate.final_status, gate.reject_code, gate.security_event) == (
        allow,
        final_status,
        reject_code,
        event,
    )


def test_intent_gate_fails_closed_on_an_unknown_verdict() -> None:
    """判不出 → 不放行。误拒会进拒答口径被评估抓到,误放连审计都没有。"""
    for bad in ("", "maybe", None, 0):
        gate = guards.intent_gate({"verdict": bad, "confidence": 0.99})
        assert gate.allow is False and gate.reject_code == "intent_unknown", bad


def test_only_privilege_violation_produces_a_security_event() -> None:
    """拒答不是越权:超范围/闲聊/判不出都只留 trace,否则审计会被噪声淹掉。"""
    events = [guards.intent_gate(_intent(v)).security_event for v in
              ("answerable", "out_of_scope", "chitchat")]
    events.append(guards.intent_gate({"verdict": "not-a-verdict"}).security_event)
    assert events == [None, None, None, None]
    assert guards.intent_gate(_intent("out_of_privilege")).security_event is not None


def test_every_reject_code_the_guards_emit_is_declared() -> None:
    """层1/层4 产出的分类必须都在 §11 声明的白名单里 —— 漏一个就是数据层多了一个没人认识的口径。"""
    emitted = [guards.intent_gate(_intent(v)).reject_code for v in
               ("answerable", "out_of_privilege", "out_of_scope", "chitchat")]
    emitted.append(guards.intent_gate({"verdict": "?"}).reject_code)
    emitted += [
        guards.decide_refusal([], {}, theta_answer=0.45).reject_code,
        guards.decide_refusal(_hits("c1"), {"c1": 0.10}, theta_answer=0.45).reject_code,
    ]
    assert {one for one in emitted if one} <= set(REJECT_CODES)


# ---------------------------------------------------------------- 层2:去重与 θ_rel
def test_dedupe_keys_on_chunk_id_not_on_text() -> None:
    """两条同 chunk_id、不同渲染 → 必须塌成一条;按正文去重会让同一证据占两个编号。"""
    a, b = _hit("c1"), Hit(chunk_id="c1", score=0.9, text="另一路的渲染")
    kept, dropped = guards.dedupe_hits([a, b, _hit("c2")], limit=30)
    assert [h.chunk_id for h in kept] == ["c1", "c2"] and dropped == ["c1"]


def test_dedupe_reports_the_over_budget_tail_as_dropped() -> None:
    kept, dropped = guards.dedupe_hits(_hits("c1", "c2", "c3"), limit=2)
    assert [h.chunk_id for h in kept] == ["c1", "c2"] and dropped == ["c3"]


def test_theta_rel_boundary_keeps_exactly_035_and_drops_0349() -> None:
    """§7.3 写的是「低于 θ_rel 丢弃」,所以 0.35 本身要留下。"""
    hits = _hits("keep", "cut", "tie")
    kept, dropped = guards.filter_by_theta(
        hits, {"keep": 0.349, "cut": 0.10, "tie": 0.35}, theta_rel=0.35
    )
    assert [h.chunk_id for h in kept] == ["tie"] and dropped == ["keep", "cut"]


def test_an_unscored_hit_is_dropped_because_there_is_no_evidence_for_it() -> None:
    """模型漏打分 = 这条没被评过 = 没有依据的引用,层2 不能替它脑补一个高分。"""
    kept, dropped = guards.filter_by_theta(_hits("c1", "c2"), {"c1": 0.9}, theta_rel=0.35)
    assert [h.chunk_id for h in kept] == ["c1"] and dropped == ["c2"]


def test_unknown_ids_catches_fabricated_chunk_ids() -> None:
    """评分表本身也会 hallucinate:不在证据集里的 id 点名给 trace,不是静默忽略。"""
    scores = {"c1": 0.9, "chk-does-not-exist": 0.8, "c2": 0.7}
    assert guards.unknown_ids(scores, ["c1", "c2"]) == ["chk-does-not-exist"]


# ---------------------------------------------------------------- 归一分词
def test_norm_tokens_splits_ascii_words_and_cjk_bigrams() -> None:
    assert guards.norm_tokens("2024年 RLHF 营收") == ["2024", "年", "rlhf", "营收"]


def test_constraints_of_sees_only_what_the_rewrite_added() -> None:
    before, after = "营收是多少", "2024年的营收是多少"
    #: 双字滑窗会把「的营」这种跨界组合也带出来 —— 它是噪声但不是错误:判定只看「有没有新词」。
    assert guards.constraints_of(before, after) == ["2024", "年的", "的营"]


def test_reordering_a_query_is_not_a_new_constraint() -> None:
    """FR-405 的「禁止原样重跑」包含换语序。

    双字滑窗会因为数字打断中文连写而把重排误报成新词(「2024年营收 / 营收2024年」差一个「年」),
    所以判据落在整句归一后的 token **集合**上:集合相同就是同一条查询,新增约束为空也拦得住一次
    真的没换词的写法,但挡不住这种切分噪声 —— 两条都要有,才是 FR-405 要的判定。
    """
    before, after = "2024年 营收", "营收 2024年"
    assert guards.constraints_of(before, after) == []
    assert _plan(query=after, queries_tried=[before], new_constraints=["年"]) is None
    #: 没有集合相同这一条,切分噪声就能骗过一次重跑
    assert _plan(query=after, queries_tried=["别的"], new_constraints=["年"]) is not None


# ---------------------------------------------------------------- 层3:定向重试
_ARGS = {
    "query": "2024年的营收是多少",
    "missing": ["缺 2024 年"],
    "queries_tried": ["营收是多少"],
    "new_constraints": ["2024"],
    "evidence_gain": 3,
    "attempt": 0,
    "max_attempts": 2,
}


def _plan(**overrides):
    args = {**_ARGS, **overrides}
    return guards.plan_retry(**args)


def test_a_retry_with_budget_missing_info_new_constraint_and_new_evidence_is_planned() -> None:
    plan = _plan()
    assert plan is not None and plan.attempt == 1 and plan.constraints == ["2024"]
    assert plan.missing == ["缺 2024 年"], "FR-405:重试请求里必须看得见本轮缺失项"


@pytest.mark.parametrize(
    "overrides",
    [
        {"attempt": 2, "max_attempts": 2},          # 预算用尽
        {"missing": []},                             # 没有方向 → 随机重跑
        {"missing": ["   "]},                        # 有方向但全是空白,同上
        {"new_constraints": []},                     # 没带新增约束
        {"new_constraints": ["营收"]},               # 「新」词其实早跑过
        {"queries_tried": ["营收是多少", "2024年的营收是多少"]},  # 整句已经跑过
        {"evidence_gain": 0},                        # 上一轮零新增证据,这个方向是死的
        {"evidence_gain": -1},
    ],
    ids=[
        "budget", "no-missing", "blank-missing", "no-constraint", "stale-constraint",
        "query-already-tried", "no-gain", "negative-gain",
    ],
)
def test_retry_refuses_to_run_without_every_condition(overrides) -> None:
    """四个条件逐个打破,每打破一个都不许重跑。测试逐条破坏是 FR-405 唯一的可证形式。"""
    assert _plan(**overrides) is None


def test_retry_attempt_counter_is_what_the_budget_reads() -> None:
    """第二次重试要拿 `attempt=1` 去比预算,否则 `max_targeted_retry` 永远挡不住第三次。"""
    assert _plan(attempt=1, max_attempts=2).attempt == 2
    assert _plan(attempt=2, max_attempts=2) is None


# ---------------------------------------------------------------- 层4:拒答
def test_no_evidence_refuses_before_any_score_is_consulted() -> None:
    out = guards.decide_refusal([], {"c1": 0.99}, theta_answer=0.45)
    assert (out.refuse, out.reject_code) == (True, "no_evidence")


def test_theta_answer_boundary_keeps_exactly_045() -> None:
    hits = _hits("c1")
    assert guards.decide_refusal(hits, {"c1": 0.45}, theta_answer=0.45).refuse is False
    out = guards.decide_refusal(hits, {"c1": 0.449}, theta_answer=0.45)
    assert (out.refuse, out.reject_code) == (True, "below_theta")


def test_min_evidence_can_ask_for_more_than_one_hit() -> None:
    out = guards.decide_refusal(_hits("c1"), {"c1": 0.9}, theta_answer=0.45, min_evidence=2)
    assert (out.refuse, out.reject_code, out.evidence) == (True, "no_evidence", 1)


def test_low_score_evidence_does_not_lift_the_best_score() -> None:
    """最高分是「最好那条」的分数,不是平均值:一条 0.9 就够格,别让平均把它拉下去。"""
    out = guards.decide_refusal(_hits("a", "b"), {"a": 0.9, "b": 0.1}, theta_answer=0.45)
    assert out.refuse is False and out.best_score == 0.9


# ---------------------------------------------------------------- 层5:引用编号
_CITES = {"1": "c1", "2": "c2", "3": "c3"}


def test_legal_numbers_map_back_to_chunk_ids_in_appearance_order() -> None:
    out = guards.check_citations("结论是甲[1], Also 乙[2]。", _CITES)
    assert out.verdict == "ok" and out.cited_ids == ["c1", "c2"] and out.stripped == []
    assert out.text == "结论是甲[1], Also 乙[2]。", "合法编号一个字都不许改写"


def test_out_of_range_numbers_are_stripped_and_recorded() -> None:
    """`[1][2][9][0]`:越界与 0 号剥离记录,剩余正文保留 → warn,不是整条拒答。"""
    out = guards.check_citations("甲[1]乙[2]丙[9]丁[0]。", _CITES)
    assert out.verdict == "warn" and out.stripped == [9, 0]
    assert out.cited_ids == ["c1", "c2"] and out.text == "甲[1]乙[2]丙丁。"


def test_numbering_is_never_renumbered_after_a_strip() -> None:
    """剥离 [1] 之后剩下的 `[3]` 仍然是 3 号:重编号会让用户对着引用列表找不到原文。"""
    out = guards.check_citations("甲[1]乙[3]", {"1": "ghost", "3": "c3"}, evidence_ids=["c3"])
    assert out.text == "甲乙[3]" and out.cited_ids == ["c3"] and out.verdict == "warn"


def test_a_answer_without_any_citation_refuses() -> None:
    """文档没写「缺失」算不算,保守按算:没有证据指针的答案正是这套防御要拦的东西。"""
    out = guards.check_citations("根据我的知识,答案是甲。", _CITES)
    assert out.verdict == "refuse" and out.cited_ids == []


def test_all_numbers_illegal_collapses_to_refuse() -> None:
    out = guards.check_citations("见[7]与[8]。", _CITES)
    assert out.verdict == "refuse" and out.stripped == [7, 8] and out.text == "见与。"


@pytest.mark.parametrize("form", ["答案[１]。", "答案【1】。", "答案［１］。"])
def test_fullwidth_brackets_and_digits_still_count(form: str) -> None:
    """模型爱输出全角:归一后照旧校验,不是当成「没有引用」而误拒。"""
    out = guards.check_citations(form, _CITES)
    assert out.verdict == "ok" and out.cited_ids == ["c1"], form


def test_a_multi_number_bracket_keeps_only_the_legal_members() -> None:
    out = guards.check_citations("见[1,9]。", _CITES)
    assert out.text == "见[1]。" and out.stripped == [9] and out.cited_ids == ["c1"]


def test_prose_brackets_are_not_citations_and_do_not_trigger_refusal() -> None:
    """`[图1]`、`[详见第3章]`、markdown 链接是正文不是引用:不剥离,也不因为它们没有编号就拒答。"""
    with_figure = guards.check_citations("见[图1]的说明[1]。", _CITES)
    assert with_figure.verdict == "ok" and with_figure.cited_ids == ["c1"]
    assert with_figure.text == "见[图1]的说明[1]。", "正文原样保留"

    chapter = guards.check_citations("[详见第3章][2]", _CITES)
    assert (chapter.verdict, chapter.stripped, chapter.cited_ids) == ("ok", [], ["c2"])

    markdown = guards.check_citations("[链接](http://x) [3]", _CITES)
    assert markdown.verdict == "ok" and markdown.cited_ids == ["c3"]

    #: 反过来,一篇只有这类括号的回答仍然是「零个可解析编号」→ 拒答
    assert guards.check_citations("见[图1]与[详见第3章]。", _CITES).verdict == "refuse"


def test_a_legal_index_pointing_outside_the_evidence_set_is_still_stripped() -> None:
    """FR-408 的后半句「该 citation 真在证据集内」:编号在表里但证据已被 θ_rel 丢掉,一样算伪造。"""
    out = guards.check_citations("甲[1]乙[2]", _CITES, evidence_ids=["c2"])
    assert out.verdict == "warn" and out.stripped == [1] and out.text == "甲乙[2]"


def test_empty_citation_table_refuses_anything_with_numbers() -> None:
    out = guards.check_citations("答案[1]。", {})
    assert out.verdict == "refuse" and out.stripped == [1]


def test_dedupe_of_cited_ids_preserves_first_appearance() -> None:
    out = guards.check_citations("甲[2]乙[1]丙[2]", _CITES)
    assert out.cited_ids == ["c2", "c1"]


# ---------------------------------------------------------------- resume 入参
_CANDIDATES = [{"text": "2024年的营收", "focus": "时间"}]


def _state(candidates: Iterable[dict] | None = None) -> dict:
    st = empty_state(query="营收是多少", thread_id="th-1", thread_key="tnt-a:th-1", run_id="run-1")
    st["rewrite_candidates"] = list(candidates if candidates is not None else _CANDIDATES)
    return st


def test_edited_wins_over_choice() -> None:
    assert guards.validate_resume(_state(), {"choice": 1, "edited": "换成季度口径"}) == "换成季度口径"


def test_blank_edited_falls_back_to_the_chosen_candidate() -> None:
    """前端「不改写」通常就是提交一个空串,这不是越界输入。"""
    assert guards.validate_resume(_state(), {"choice": 1, "edited": "   "}) == "2024年的营收"


@pytest.mark.parametrize(
    "payload",
    [
        {"choice": 0},                 # 没选
        {"choice": 2},                 # 选到不存在的候选
        {"choice": None, "edited": ""},
        {"edited": "短"},              # 少于 EDITED_MIN
        {"edited": "甲" * 600},        # 超过 EDITED_MAX
        {"edited": "正常问题\n第二段"},  # 控制字符
        {"edited": "前缀\x00后缀"},
    ],
)
def test_unusable_resume_input_raises_instead_of_quietly_reusing_the_original_query(payload) -> None:
    """静默回退原问 = 跳过确认环节,FR-402 白做。这条断言就是钉住「不许有回退」。"""
    with pytest.raises(IllegalTransition):
        guards.validate_resume(_state(), payload)


def test_an_int_like_choice_string_is_accepted_because_json_carries_it_that_way() -> None:
    """HTTP JSON 里 `choice` 常常是 "2";形状可容忍,语义不可容忍。"""
    assert guards.validate_resume(_state(), {"choice": "1"}) == "2024年的营收"


def test_a_candidate_with_blank_text_cannot_be_confirmed() -> None:
    st = _state([{"text": "   ", "focus": ""}])
    with pytest.raises(IllegalTransition, match="空串"):
        guards.validate_resume(st, {"choice": 1})


def test_resume_accepts_the_pydantic_model_as_well_as_a_mapping() -> None:
    assert guards.validate_resume(_state(), ResumeChoice(choice=1)) == "2024年的营收"


# ---------------------------------------------------------------- state 契约
def test_stages_are_exactly_the_twelve_in_the_doc() -> None:
    """§11 的白名单:`retrieve_dual` 是一个节点四个 stage,少一个就是 FR-701 还原不出决策链。"""
    assert len(STAGES) == 12
    assert {"route_intent", "rewrite", "merge", "rerank", "assess", "retry",
            "generate", "cite_check"} <= set(STAGES)
    assert {"retrieve_body", "retrieve_heading", "retrieve_dense", "retrieve_graph"} <= set(STAGES)


def test_seq_numbering_starts_at_one_and_advances_by_the_count_used() -> None:
    st = _state()
    assert next_seq(st) == 1
    st["trace_seq"] = used_seq(st, 3)
    assert st["trace_seq"] == 3 and next_seq(st) == 4


def test_empty_state_seeds_the_retry_prohibitions_and_the_ttl_clock() -> None:
    """`queries_tried` 一开头就得有原问,否则第一次定向重试的「原样重跑」判不出来。"""
    st = _state()
    assert st["queries_tried"] == ["营收是多少"] and st["retry_attempt"] == 0
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$", st["pending_since"])
    assert st["confirmed_query"] == st["query"], "确认前检索用的词还没有,不能凭空造"


def _bounds(model, name: str) -> tuple[int | None, int | None]:
    md = model.model_fields[name].metadata
    low = next((m.min_length for m in md if hasattr(m, "min_length")), None)
    high = next((m.max_length for m in md if hasattr(m, "max_length")), None)
    return low, high


def test_schema_bounds_and_resume_bounds_are_one_number() -> None:
    """候选的长度上下限就是 resume `edited` 的上下限:两处各写一个值,迟早漂成「合法候选被拒」。"""
    assert _bounds(RewriteCandidate, "text") == (guards.EDITED_MIN, guards.EDITED_MAX)
    assert _bounds(ResumeChoice, "edited")[1] == guards.EDITED_MAX
