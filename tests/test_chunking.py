"""标题感知切片(FR-102)与切片幂等(FR-103)。

这里的断言全部是「重跑必相等」这类可判定断言 —— 幂等不是跑一次看起来对,而是第 N 次与第 1 次
逐字段相同,所以键的构成、位置编号、合并规则都必须与随机/时间无关。
"""
from __future__ import annotations

import hashlib

from kbmesh.ingest.chunking import ChunkPolicy, chunk_key, estimate_tokens, normalize_text, sentences, split
from kbmesh.ingest.extract import Block, ParsedDoc

LONG = "存储层做物理隔离,查询层强制注入过滤条件,身份只从令牌取,验收用真实数据断言。"


def _doc(*blocks: Block) -> ParsedDoc:
    return ParsedDoc(title="测试文档", blocks=list(blocks))


def _h(level: int, text: str) -> Block:
    return Block("heading", text, level=level)


def _p(text: str, page_no: int | None = None) -> Block:
    return Block("para", text, page_no=page_no)


def test_chunks_carry_the_heading_breadcrumb() -> None:
    parts = split(
        _doc(_h(1, "第一章"), _h(2, "1.1 隔离"), _p(LONG), _h(2, "1.2 检索"), _p("向量与稀疏混合召回。")),
        doc_id="doc-1",
    )
    assert [p.heading_path for p in parts] == ["第一章 > 1.1 隔离", "第一章 > 1.2 检索"]
    assert [p.level for p in parts] == [2, 2]
    assert [p.position for p in parts] == [0, 1], "切片可追溯到文档偏移"


def test_breadcrumb_depth_is_capped() -> None:
    parts = split(
        _doc(_h(1, "一"), _h(2, "二"), _h(3, "三"), _h(4, "四"), _p("正文。")),
        doc_id="doc-1",
        policy=ChunkPolicy(max_heading_depth=2),
    )
    assert parts[0].heading_path == "三 > 四"


def test_a_chunk_never_mixes_two_sections() -> None:
    parts = split(
        _doc(_h(1, "甲节"), _p(LONG), _h(1, "乙节"), _p(LONG)),
        doc_id="doc-1",
    )
    assert len(parts) == 2
    assert {p.heading_path for p in parts} == {"甲节", "乙节"}


def test_split_is_reproducible_for_the_same_document() -> None:
    doc = _doc(_h(1, "章"), *[_p(f"第{i}句。" + LONG) for i in range(40)])
    first = split(doc, doc_id="doc-9")
    for _ in range(4):
        again = split(doc, doc_id="doc-9")
        assert [p.chunk_key for p in again] == [p.chunk_key for p in first]
        assert [p.text for p in again] == [p.text for p in first]
    # 换 doc_id 必须换键:同一句话在两份文档里是两个切片
    assert {p.chunk_key for p in split(doc, doc_id="doc-10")}.isdisjoint({p.chunk_key for p in first})


def test_chunk_key_follows_the_documented_formula() -> None:
    parts = split(_doc(_h(1, "章"), _p("  空白   会被归一。 ")), doc_id="doc-k")
    raw = "doc-k\x1f章\x1f" + normalize_text("  空白   会被归一。 ")
    assert parts[0].chunk_key == hashlib.sha256(raw.encode()).hexdigest()
    assert parts[0].chunk_key == chunk_key("doc-k", "章", "空白  会被归一。")
    assert len(parts[0].chunk_key) == 64, "必须塞得进 chunk.chunk_key 的 String(64)"


def test_long_section_overflows_with_sentence_level_overlap() -> None:
    doc = _doc(_h(1, "长章"), *[_p(f"第{i}句{LONG}") for i in range(60)])
    pol = ChunkPolicy(max_tokens=200, overlap_tokens=60)
    parts = split(doc, doc_id="doc-o", policy=pol)
    assert len(parts) > 3
    for part in parts:
        assert part.token_len <= pol.max_tokens + pol.overlap_tokens, part.text[:60]
    shared = [
        set(sentences(a.text)) & set(sentences(b.text)) for a, b in zip(parts, parts[1:], strict=False)
    ]
    assert any(shared), "相邻切片必须重叠,否则跨边界的那句话谁都召回不到"


def test_one_oversized_sentence_is_hard_split() -> None:
    run_on = "隔离" * 600
    parts = split(_doc(_h(1, "无标点"), _p(run_on)), doc_id="doc-hard", policy=ChunkPolicy(max_tokens=300))
    assert len(parts) > 1
    assert all(p.token_len <= 300 for p in parts)
    assert "".join(p.text for p in parts) == run_on, "硬切不能吞字或改序"


def test_undersized_tail_merges_into_its_section() -> None:
    parts = split(
        _doc(_h(1, "章"), _p(LONG * 12), _p("残片。")),
        doc_id="doc-m",
        policy=ChunkPolicy(max_tokens=120, min_tokens=24),
    )
    assert parts[-1].token_len >= 24, "碎到给不出上下文的残片要并进前一片"
    assert all(p.text.strip() for p in parts)


def test_image_only_block_stays_a_chunk_for_caption_backfill() -> None:
    parts = split(
        _doc(_h(1, "图版"), Block("image", "", page_no=3, image_refs=["page:3:img:10-20"])),
        doc_id="doc-img",
    )
    assert len(parts) == 1
    assert parts[0].text == "" and parts[0].image_refs == ["page:3:img:10-20"]
    assert parts[0].page_no == 3, "caption_images 要靠页码把整页转图后回填正文"


def test_page_no_and_image_refs_attach_to_the_covering_chunk() -> None:
    parts = split(
        _doc(_h(1, "章"), _p("第一页正文。", page_no=1), Block("image", "", page_no=2, image_refs=["a.png"])),
        doc_id="doc-p",
    )
    assert len(parts) == 1
    assert parts[0].page_no == 1 and parts[0].image_refs == ["a.png"]


def test_tables_stay_row_oriented() -> None:
    rows = [Block("table", "防线 | 落点"), Block("table", "一 | 存储"), Block("table", "二 | 身份")]
    parts = split(_doc(_h(1, "附录"), *rows), doc_id="doc-t")
    assert parts[0].text.splitlines() == ["防线 | 落点", "一 | 存储", "二 | 身份"], "表格行不能被拼成一行"


def test_policy_from_dict_ignores_unknown_keys_but_keeps_zero() -> None:
    pol = ChunkPolicy.from_dict({"max_tokens": 500, "nonsense": 1, "overlap_tokens": 0})
    assert (pol.max_tokens, pol.overlap_tokens) == (500, 0), "重叠为 0 是合法配置,不能被真值筛掉"
    assert ChunkPolicy.from_dict(None).max_tokens == ChunkPolicy().max_tokens


def test_token_estimate_bounds_the_budget_without_a_downloaded_bpe() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("四个汉字") == 4
    assert estimate_tokens("twenty characters here") == -(-len("twenty characters here") // 4)
