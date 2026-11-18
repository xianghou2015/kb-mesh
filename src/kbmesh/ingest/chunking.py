"""标题感知的结构化切片(FR-102 / FR-103)。

两条硬约束决定了这个模块的写法:

1. **可追溯**:每个切片携带章节面包屑(`heading_path`)与页码,答案引用能落回原文位置。
2. **可复现**:同一文档重跑 N 次得到同一批切片。所以切片键只由
   `doc_id + heading_path + 归一化正文` 决定(§5.3),不掺时间戳、不掺随机数、不掺模型输出,
   合并与排序全部走确定性的后处理。

token 长度用启发式估算而不是 tiktoken:实测 tiktoken 首次取 `cl100k_base` 要联网下载 BPE
文件,离线环境直接挂住;这里的 token_len 只用来卡切片预算,不需要和计费口径一致。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from kbmesh.ingest.extract import Block, ParsedDoc

#: 句子单位的默认预算;实际取 min(该值, policy.max_tokens),单个单位不得大过一片的容量
MAX_UNIT_TOKENS = 300

#: 中日韩字符 + 全角标点:基本一字一 token
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
#: 句子边界:零宽切在 CJK 句末之后,ASCII 句末后吃掉空白(空白由 _join 决定是否补回)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；!?;])|(?<=[.!?])\s+")
_WS = re.compile(r"\s+")


@dataclass
class ChunkPolicy:
    """切片预算;KB 上的 `chunk_policy` JSONB 覆盖默认值,未知键忽略。"""

    max_tokens: int = 400
    overlap_tokens: int = 80
    #: 段末残片小于这个规模就并进上一片:碎切片既占召回位又给不出上下文
    min_tokens: int = 24
    #: 面包屑保留的层级数;越深的章节对检索没区分度,只加长字符串
    max_heading_depth: int = 3

    @classmethod
    def from_dict(cls, raw: dict | None) -> ChunkPolicy:
        if not raw:
            return cls()
        known = set(cls.__dataclass_fields__)
        #: `overlap_tokens: 0` 是合法配置(不要重叠),所以只跳 None,不按真值筛
        return cls(**{k: v for k, v in raw.items() if k in known and v is not None})


@dataclass
class ChunkPart:
    """切片对外形态(落库前的中间产物)。"""

    text: str
    heading_path: str
    level: int
    page_no: int | None
    token_len: int
    image_refs: list[str] = field(default_factory=list)
    chunk_key: str = ""
    position: int = 0


def estimate_tokens(text: str) -> int:
    """CJK 按字计,其余按 4 字符 ≈ 1 token。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return cjk + -(-(len(text) - cjk) // 4)


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def chunk_key(doc_id: str, heading_path: str, text: str) -> str:
    raw = "\x1f".join((doc_id, heading_path, normalize_text(text)))
    return hashlib.sha256(raw.encode()).hexdigest()


def sentences(text: str, unit_tokens: int = MAX_UNIT_TOKENS) -> list[str]:
    """切成句子;句子仍超长就按字符窗口硬切,保证预算有上界。"""
    out: list[str] = []
    for piece in (p.strip() for p in _SENTENCE_BOUNDARY.split(text) if p.strip()):
        out.extend(_hard_split(piece, unit_tokens))
    return out


def _hard_split(piece: str, unit_tokens: int) -> list[str]:
    cost = estimate_tokens(piece)
    if cost <= unit_tokens:
        return [piece]
    #: 按本段自身的「字符/元」折算窗口:纯中文一段 ≈ unit_tokens 字,纯英文 ≈ 4×
    width = max(20, int(len(piece) * unit_tokens / cost))
    return [piece[i : i + width] for i in range(0, len(piece), width)]


def _join(units: list[tuple[str, str]]) -> str:
    """把 (文本, 前置分隔) 拼回正文。

    CJK 句子自带句末标点,直接相接;块边界换行(保留版式);ASCII 句边界被正则吃掉的空白补回来。
    """
    out = ""
    for text, sep in units:
        if not text:
            continue
        if out:
            if sep:
                out += "\n"
            elif out[-1].isascii() and out[-1] in ".!?" and text[0].isascii() and text[0].isalnum():
                out += " "
        out += text
    return out.strip()


def _push_heading(stack: list[tuple[int, str]], block: Block) -> int:
    level = block.level or 1
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, block.text.strip()))
    return level


def _path_of(stack: list[tuple[int, str]], depth: int) -> str:
    return " > ".join(t for _, t in stack[-depth:] if t)


class _Buffer:
    """当前章节内尚未落盘的句子累积器;单元是 (文本, 前置分隔)。"""

    def __init__(self) -> None:
        self.units: list[tuple[str, str]] = []
        self.tokens = 0
        self.level = 0
        self.page_no: int | None = None
        self.image_refs: list[str] = []

    @property
    def busy(self) -> bool:
        return bool(self.units or self.image_refs)

    def add_text(self, block: Block, path: str, level: int, pol: ChunkPolicy) -> list[ChunkPart]:
        """一个块 = 一段/一行/一行表格;跨块换行,块内按句切。"""
        emitted: list[ChunkPart] = []
        if self.page_no is None:
            self.page_no = block.page_no
        self.level = self.level or level
        for index, unit in enumerate(sentences(block.text, pol.max_tokens)):
            cost = estimate_tokens(unit)
            if self.tokens and self.tokens + cost > pol.max_tokens:
                emitted.append(self.overflow(path, level, pol))
            self.units.append((unit, "\n" if index == 0 or block.kind == "table" else ""))
            self.tokens += cost
        return emitted

    def add_image(self, block: Block) -> None:
        self.image_refs.extend(r for r in block.image_refs if r not in self.image_refs)
        if block.page_no is not None and self.page_no is None:
            self.page_no = block.page_no

    def overflow(self, path: str, level: int, pol: ChunkPolicy) -> ChunkPart:
        """切出一片,新缓冲从旧缓冲的尾部句子起:句子级重叠保住跨片上下文。"""
        carried: list[tuple[str, str]] = []
        spent = 0
        for text, _sep in reversed(self.units):
            cost = estimate_tokens(text)
            if spent + cost > pol.overlap_tokens:
                break
            carried.insert(0, (text, ""))
            spent += cost
        part = self.take(path)
        self.units, self.tokens = carried, spent
        self.level = level or part.level
        return part

    def take(self, heading_path: str) -> ChunkPart:
        text = _join(self.units)
        part = ChunkPart(
            text=text,
            heading_path=heading_path,
            level=self.level,
            page_no=self.page_no,
            token_len=estimate_tokens(text),
            image_refs=list(self.image_refs),
        )
        self.reset()
        return part

    def reset(self) -> None:
        self.units, self.tokens, self.level, self.page_no, self.image_refs = [], 0, 0, None, []


def split(parsed: ParsedDoc, *, doc_id: str, policy: ChunkPolicy | None = None) -> list[ChunkPart]:
    """块序列 → 切片序列(按文档顺序,position 从 0 递增)。"""
    pol = policy or ChunkPolicy()
    parts: list[ChunkPart] = []
    stack: list[tuple[int, str]] = []
    level = 0
    buf = _Buffer()

    for block in parsed.blocks:
        if block.kind == "heading":
            if buf.busy:
                parts.append(buf.take(_path_of(stack, pol.max_heading_depth)))
            level = _push_heading(stack, block)
            continue
        if block.kind == "image":
            buf.add_image(block)
            continue
        parts.extend(buf.add_text(block, _path_of(stack, pol.max_heading_depth), level, pol))
    if buf.busy:
        parts.append(buf.take(_path_of(stack, pol.max_heading_depth)))

    _merge_undersized(parts, pol)
    for index, part in enumerate(parts):
        part.position = index
        part.chunk_key = chunk_key(doc_id, part.heading_path, part.text)
    return [p for p in parts if p.text or p.image_refs]


def _merge_undersized(parts: list[ChunkPart], pol: ChunkPolicy) -> None:
    """同章节的过小残片并进前一片(确定性后处理,原地写回保持调用方引用)。"""
    merged: list[ChunkPart] = []
    for part in parts:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.heading_path == part.heading_path
            and part.token_len < pol.min_tokens
            and prev.token_len + part.token_len <= int(pol.max_tokens * 1.5)
        ):
            prev.text = (prev.text + "\n" + part.text).strip()
            prev.token_len = estimate_tokens(prev.text)
            for ref in part.image_refs:
                if ref not in prev.image_refs:
                    prev.image_refs.append(ref)
            if prev.page_no is None:
                prev.page_no = part.page_no
            continue
        merged.append(part)
    parts[:] = merged
