"""RRF 融合的定义级验收(FR-204 / FR-210)。

融合公式写错不会报错,只会悄悄把「名次」变成「距离」,而这两者的量纲差着两个数量级。
所以这里把 §6.2 的那行算式直接展开成断言,而不是只看「返回了若干条」。
"""
from __future__ import annotations

from kbmesh.retrieval.contracts import Hit, RouteOutcome
from kbmesh.retrieval.rrf import RRF_K, fuse, rrf_score


def _outcome(route: str, chunk_ids: list[str]) -> RouteOutcome:
    hits = [
        Hit(
            chunk_id=cid,
            score=1.0 / (rank + 1),
            source_routes={route},
            text=f"{route}:{cid}",
            route_scores={route: 1.0 / (rank + 1)},
        )
        for rank, cid in enumerate(chunk_ids, start=1)
    ]
    return RouteOutcome(route=route, hits=hits)


def test_score_matches_the_formula_in_the_doc() -> None:
    ranks = {"body": 1, "heading": 3}
    assert rrf_score(ranks, k=60) == 1 / 61 + 1 / 63
    assert RRF_K == 60, "k 是 §6.2 写死的默认值,改动要过评估"


def test_weights_scale_each_route_contribution() -> None:
    ranks = {"body": 1}
    assert rrf_score(ranks, k=60, weights={"body": 0.5}) == 0.5 / 61
    #: 没给权重的路按 1.0,不给就是「这一路不存在」的那种默认最容易被误读
    assert rrf_score({"body": 1, "dense": 1}, k=60, weights={"dense": 2.0}) == 1 / 61 + 2 / 61


def test_two_routes_hitting_the_same_chunk_collapse_into_one_hit() -> None:
    """FR-210:去重基于稳定主键,同一 chunk 被两路命中只出现一次。"""
    fused = fuse([_outcome("body", ["chk-a", "chk-b"]), _outcome("heading", ["chk-b", "chk-c"])])
    ids = [h.chunk_id for h in fused]
    assert ids.count("chk-b") == 1
    assert sorted(ids) == ["chk-a", "chk-b", "chk-c"]
    b = next(h for h in fused if h.chunk_id == "chk-b")
    assert b.source_routes == {"body", "heading"}
    assert b.ranks == {"body": 2, "heading": 1}


def test_body_and_heading_ranks_sum_in_fusion_score() -> None:
    fused = fuse([_outcome("body", ["chk-a"]), _outcome("heading", ["chk-a"])])
    assert fused[0].score == 1 / 61 + 1 / 61
    assert fused[0].route_scores["body"] == 0.5, "路内原始分要留着,精排之前还要看"


def test_an_empty_route_is_not_an_error() -> None:
    """FR-204:任一路为空不报错 —— 标题路对短查询经常是空的。"""
    fused = fuse([_outcome("body", ["chk-a"]), _outcome("heading", []), _outcome("dense", [])])
    assert [h.chunk_id for h in fused] == ["chk-a"]
    assert fused[0].ranks == {"body": 1}
    assert fuse([]) == []


def test_single_route_mode_preserves_that_routes_order() -> None:
    """FR-205 的单路验证形态:只开一路时融合不得改变那一路的名次。"""
    outcome = _outcome("body", ["chk-1", "chk-2", "chk-3"])
    assert [h.chunk_id for h in fuse([outcome])] == ["chk-1", "chk-2", "chk-3"]


def test_a_chunk_hit_by_two_routes_outranks_a_single_route_first_place() -> None:
    """融合的意义:两路都靠前 > 单路第一。"""
    fused = fuse([_outcome("body", ["both", "only"]), _outcome("dense", ["only", "both"])])
    assert [h.chunk_id for h in fused] == ["both", "only"]


def test_fusion_is_deterministic_when_scores_tie() -> None:
    """同分按 chunk_id 定序:评估跑两次得到不同上下文,指标就没有可比性。

    两条路各把两个 chunk 排在相反的名次上,RRF 分完全相等,顺序只能由 chunk_id 决定。
    """
    left = fuse([_outcome("body", ["chk-z", "chk-a"]), _outcome("dense", ["chk-a", "chk-z"])])
    right = fuse([_outcome("dense", ["chk-a", "chk-z"]), _outcome("body", ["chk-z", "chk-a"])])
    assert [h.chunk_id for h in left] == [h.chunk_id for h in right] == ["chk-a", "chk-z"]
    assert left[0].score == left[1].score == 1 / 61 + 1 / 62


def test_limit_truncates_after_sorting_not_before() -> None:
    """FR-211 的前半段:截断发生在相关性排序之后。"""
    fused = fuse([_outcome("body", ["chk-a", "chk-b", "chk-c"])], limit=2)
    assert [h.chunk_id for h in fused] == ["chk-a", "chk-b"]


def test_text_and_citation_survive_the_merge() -> None:
    """合并去重时正文与引用不能被后来的路覆盖成空。"""
    body = _outcome("body", ["chk-a"])
    graph = RouteOutcome(
        route="graph",
        hits=[Hit(chunk_id="chk-a", score=0.5, source_routes={"graph"}, evidence="依据在此")],
    )
    fused = fuse([body, graph])
    assert len(fused) == 1
    assert fused[0].text == "body:chk-a"
    assert fused[0].evidence == "依据在此"
    assert fused[0].ranks == {"body": 1, "graph": 1}
