"""多格式解析:字节 → 有序块(block)。

这里只做「读文件」这一件事:不联网、不调模型、不碰数据库 —— 解析产物一旦不可复现,
FR-102「同一文档重复解析得到同一批切片」就无从谈起。

块类型固定四种:heading / para / table / image(整页转图留给管道,不在这里做 OCR)。
"""
from __future__ import annotations

import io
import re
import statistics
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["heading", "para", "table", "image"]

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
#: markdown 表格的对齐行(`| --- | :--: |`):不携带内容,进索引只会污染 BM25 词频
_TABLE_DIVIDER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_SOFT_LIMIT = 200  # 无层级标记时,超过这个长度的行不当标题


@dataclass
class Block:
    kind: BlockKind
    text: str
    level: int = 0
    page_no: int | None = None
    image_refs: list[str] = field(default_factory=list)


@dataclass
class ParsedDoc:
    title: str
    blocks: list[Block]
    warnings: list[str] = field(default_factory=list)

    def headings(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == "heading"]


def extract(data: bytes, mime: str, *, filename: str = "") -> ParsedDoc:
    mime = (mime or "").lower()
    name = filename.lower()
    if "pdf" in mime or name.endswith(".pdf"):
        return _from_pdf(data, filename)
    if "wordprocessingml" in mime or name.endswith(".docx"):
        return _from_docx(data, filename)
    if "spreadsheet" in mime or name.endswith((".xlsx", ".xlsm")):
        return _from_xlsx(data, filename)
    if "markdown" in mime or name.endswith((".md", ".markdown")):
        return _from_markdown(data.decode("utf-8", "replace"), filename)
    if "json" in mime:
        raise ValueError("JSON 语料不在本项目范围(FR-101 排除项)")
    return _from_plain(data.decode("utf-8", "replace"), filename)


# --------------------------------------------------------------------- markdown


def _from_markdown(text: str, filename: str) -> ParsedDoc:
    blocks: list[Block] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            body = " ".join(x.strip() for x in buf).strip()
            if body:
                blocks.append(Block("para", body))
            buf.clear()

    for line in text.splitlines():
        heading = _MD_HEADING.match(line)
        if heading:
            flush()
            blocks.append(Block("heading", heading.group(2).strip(), level=len(heading.group(1))))
            continue
        if _TABLE_DIVIDER.match(line):
            continue
        if _TABLE_ROW.match(line):
            flush()
            blocks.append(Block("table", line.strip()))
            continue
        image = _MD_IMAGE.fullmatch(line.strip())
        if image:
            flush()
            blocks.append(Block("image", "", image_refs=[image.group(1)]))
            continue
        if line.strip() == "":
            flush()
        else:
            buf.append(line)
    flush()
    return ParsedDoc(title=_fallback_title(filename, blocks), blocks=blocks)


def _from_plain(text: str, filename: str) -> ParsedDoc:
    blocks = [Block("para", line.strip()) for line in text.splitlines() if line.strip()]
    return ParsedDoc(title=_fallback_title(filename, blocks), blocks=blocks)


# --------------------------------------------------------------------- docx


def _from_docx(data: bytes, filename: str) -> ParsedDoc:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(io.BytesIO(data))
    blocks: list[Block] = []
    for element in document.element.body.iterchildren():
        tag = element.tag.split("}")[-1]
        if tag == "p":
            para = Paragraph(element, document)
            text = para.text.strip()
            if not text:
                continue
            level = _heading_level(para.style.name if para.style else "")
            blocks.append(Block("heading" if level else "para", text, level=level))
        elif tag == "tbl":
            table = Table(element, document)
            blocks.extend(Block("table", row) for row in _render_table(_table_rows(table)))
    return ParsedDoc(title=_fallback_title(filename, blocks), blocks=blocks)


def _heading_level(style_name: str) -> int:
    m = re.match(r"(?:Heading|标题)\s*(\d)", style_name or "")
    return int(m.group(1)) if m else 0


def _table_rows(table) -> list[list[str]]:
    return [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]


# --------------------------------------------------------------------- xlsx


def _from_xlsx(data: bytes, filename: str) -> ParsedDoc:
    from openpyxl import load_workbook

    book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    blocks: list[Block] = []
    for sheet in book.worksheets:
        blocks.append(Block("heading", sheet.title, level=1))
        rows = [[("" if c is None else str(c)) for c in row] for row in sheet.iter_rows(values_only=True)]
        blocks.extend(Block("table", row) for row in _render_table(rows))
    book.close()
    return ParsedDoc(title=_fallback_title(filename, blocks), blocks=blocks)


# --------------------------------------------------------------------- pdf


def _from_pdf(data: bytes, filename: str) -> ParsedDoc:
    import pdfplumber

    blocks: list[Block] = []
    warnings: list[str] = []
    sizes: list[float] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        # 第一遍只收集字号分布:标题判定必须相对全文,而不是相对单页
        for page in pdf.pages:
            sizes.extend(c["size"] for c in page.chars)
        body_size = statistics.median(sizes) if sizes else 0.0
        title_from_meta = _real_title((pdf.metadata or {}).get("Title"))

        for page_no, page in enumerate(pdf.pages, start=1):
            lines = []
            for row in page.extract_text_lines():
                text = row["text"].strip()
                if not text:
                    continue
                # 中位数而不是首字符:混排一行里首字符的字号可能来自上标或链接
                lines.append((text, statistics.median(c["size"] for c in row["chars"])))
            for text, size in lines:
                level = _pdf_heading_level(size, body_size, text)
                blocks.append(Block("heading" if level else "para", text, level=level, page_no=page_no))
            for table in page.extract_tables() or []:
                blocks.extend(
                    Block("table", row, page_no=page_no) for row in _render_table(_clean_table(table))
                )
            for image in page.images:
                spot = f"{int(image.get('x0', 0))}-{int(image.get('top', 0))}"
                blocks.append(Block("image", "", page_no=page_no, image_refs=[f"page:{page_no}:img:{spot}"]))
            if len("".join(t for t, _ in lines)) < 4 and page.images:
                warnings.append(f"page {page_no} 文字层几乎为空,按扫描页整页转图处理")
        total_pages = len(pdf.pages)

    if total_pages and not any(b.page_no for b in blocks if b.kind != "image"):
        warnings.append("PDF 无文字层:整库将依赖视觉回填,检索质量以图片描述为限")
    return ParsedDoc(
        title=title_from_meta or _fallback_title(filename, blocks),
        blocks=blocks,
        warnings=warnings,
    )


def _real_title(raw: str | None) -> str:
    """PDF 元数据里的 untitled/unspecified 是生成器的占位值,当标题用会把文档全染成一个名。"""
    value = (raw or "").strip()
    return "" if value.lower() in {"", "untitled", "unspecified", "no title"} else value


def _pdf_heading_level(size: float, body: float, line: str) -> int:
    if not body or size <= body * 1.05 or len(line) > _SOFT_LIMIT:
        return 0
    if size >= body * 1.6:
        return 1
    if size >= body * 1.3:
        return 2
    return 3


def _clean_table(table: list[list[str | None]]) -> list[list[str]]:
    return [[("" if cell is None else str(cell).replace("\n", " ").strip()) for cell in row] for row in table]


def _render_table(rows: Iterator[list[str]] | list[list[str]]) -> list[str]:
    return [" | ".join(cell for cell in row) for row in rows if any(row)]


def _fallback_title(filename: str, blocks: list[Block]) -> str:
    if filename:
        return re.sub(r"\.[a-z0-9]+$", "", filename, flags=re.I)
    for block in blocks:
        if block.kind == "heading":
            return block.text
    return ""


def render_page_png(data: bytes, page_no: int, *, scale: float = 2.0) -> bytes:
    """整页转图(FR-104/105):扫描页没有文字层,只能渲成位图交给视觉档。

    用 pypdfium2(pdfplumber 的传递依赖)而不是 pdfplumber 的 `to_image()`:后者返回的
    ImageWriter 只给了 PIL/jpg 路径,管道要的是确定的 PNG 字节。
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        return _pil_png(pdf[page_no - 1].render(scale=scale).to_pil())
    finally:
        pdf.close()


def _pil_png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
