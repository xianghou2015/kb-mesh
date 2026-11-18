"""多格式解析(FR-101/102)与整页转图(FR-104/105)的实测。

fixture 全部现场生成:仓库里不放二进制文档,否则「同一文档重复解析得到同一批切片」
这条验收断言会退化成「同一个二进制文件重复读取」。
"""
from __future__ import annotations

import io

import pytest

from kbmesh.ingest.extract import Block, extract, render_page_png

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _pdf(pages: list[list[tuple[float, str]]], *, images: dict[int, bytes] | None = None) -> bytes:
    """每页给一组 (字号, 文本);`images` 指定「页码 → PNG」,用于造无文字层的扫描页。"""
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(420, 600))
    for page_no, lines in enumerate(pages, start=1):
        y = 560.0
        for size, text in lines:
            c.setFont("STSong-Light", size)
            c.drawString(36, y, text)
            y -= size + 14
        for png in (images or {}).get(page_no, []):
            c.drawImage(ImageReader(io.BytesIO(png)), 36, 60, width=300, height=120)
        c.showPage()
    c.save()
    return buf.getvalue()


def _png(color: str = "#222222") -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (120, 60), color).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------- 分发与拒绝


def test_json_is_out_of_scope() -> None:
    with pytest.raises(ValueError, match="FR-101"):
        extract(b'[{"a": 1}]', "application/json", filename="data.json")


def test_unknown_mime_falls_back_to_plain_text() -> None:
    doc = extract("第一行\n第二行\n\n".encode(), "application/octet-stream")
    assert [b.text for b in doc.blocks] == ["第一行", "第二行"]
    assert all(b.kind == "para" for b in doc.blocks)


# --------------------------------------------------------------------- markdown


def test_markdown_headings_tables_and_images() -> None:
    text = (
        "# 一级\n"
        "## 二级小节\n"
        "正文句子一。正文句子二。\n"
        "\n"
        "| 防线 | 落点 |\n"
        "| --- | --- |\n"
        "| 一 | 存储 |\n"
        "\n"
        "![架构图](images/arch.png)\n"
    )
    doc = extract(text.encode(), "text/markdown", filename="防线.md")
    kinds = [(b.kind, b.level, b.text) for b in doc.blocks]
    assert kinds[0] == ("heading", 1, "一级")
    assert kinds[1] == ("heading", 2, "二级小节")
    assert kinds[2] == ("para", 0, "正文句子一。正文句子二。")
    assert [k for k, _, _ in kinds if k == "table"] == ["table", "table"], "表头与数据行保留、对齐行丢弃"
    assert kinds[-1] == ("image", 0, "")
    assert doc.blocks[-1].image_refs == ["images/arch.png"]
    assert doc.title == "防线"


# --------------------------------------------------------------------- docx / xlsx


def test_docx_keeps_body_order_and_heading_styles() -> None:
    import docx

    document = docx.Document()
    document.add_heading("目标与范围", level=1)
    document.add_paragraph("第一段正文。")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "引擎", "职责"
    table.cell(1, 0).text, table.cell(1, 1).text = "Milvus", "向量与稀疏"
    document.add_paragraph("第二段正文。")
    buf = io.BytesIO()
    document.save(buf)

    doc = extract(buf.getvalue(), _DOCX_MIME)
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["heading", "para", "table", "table", "para"], "表格必须落在它原文所在的位置"
    assert doc.blocks[0].level == 1
    assert doc.blocks[2].text == "引擎 | 职责"
    assert doc.blocks[3].text == "Milvus | 向量与稀疏"


def test_xlsx_sheet_becomes_heading_then_tables() -> None:
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.title = "指标口径"
    sheet.append(["名称", "定义"])
    sheet.append(["命中率", "前K含gold"])
    buf = io.BytesIO()
    book.save(buf)

    doc = extract(buf.getvalue(), _XLSX_MIME, filename="指标.xlsx")
    assert doc.blocks[0] == Block("heading", "指标口径", level=1)
    assert [b.kind for b in doc.blocks[1:]] == ["table", "table"]
    assert doc.blocks[1].text == "名称 | 定义"
    assert doc.blocks[2].text == "命中率 | 前K含gold"


# --------------------------------------------------------------------- pdf


def test_pdf_heading_levels_come_from_font_size_not_position() -> None:
    doc = extract(
        _pdf(
            [
                [
                    (22, "多租户隔离设计"),
                    (14, "3.1 存储层"),
                    (10, "本文介绍四道防线。"),
                    (10, "存储层做物理隔离,查询层强制注入。"),
                ]
            ]
        ),
        "application/pdf",
        filename="隔离.pdf",
    )
    got = [(b.kind, b.level, b.text) for b in doc.blocks]
    assert got[0] == ("heading", 1, "多租户隔离设计")
    assert got[1] == ("heading", 2, "3.1 存储层")
    assert [k for k, _, _ in got[2:]] == ["para", "para"], "正文字号的行不能被当成标题"
    assert all(b.page_no == 1 for b in doc.blocks)
    assert doc.title == "隔离", "untitled 这类占位元数据不能当标题"


def test_pdf_page_numbers_track_the_source_pages() -> None:
    doc = extract(
        _pdf([[(12, "第一章"), (10, "第一页正文。")], [(10, "第二页正文。")]]),
        "application/pdf",
    )
    assert [(b.page_no, b.text) for b in doc.blocks] == [
        (1, "第一章"),
        (1, "第一页正文。"),
        (2, "第二页正文。"),
    ]


def test_scanned_page_warns_and_yields_only_image_blocks() -> None:
    doc = extract(_pdf([[(10, "文字层正常的一页。")], []], images={2: [_png()]}), "application/pdf")
    assert any("扫描页" in w for w in doc.warnings), doc.warnings
    page2 = [b for b in doc.blocks if b.page_no == 2]
    assert [b.kind for b in page2] == ["image"]


def test_extract_is_reproducible() -> None:
    data = _pdf([[(20, "标题"), (10, "正文一。"), (10, "正文二。")]])
    first, second = extract(data, "application/pdf"), extract(data, "application/pdf")
    assert [(b.kind, b.level, b.page_no, b.text) for b in first.blocks] == [
        (b.kind, b.level, b.page_no, b.text) for b in second.blocks
    ]


def test_render_page_png_produces_png_bytes() -> None:
    data = _pdf([[(14, "标题"), (10, "正文。")]])
    png = render_page_png(data, 1)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert render_page_png(data, 1) == png, "整页转图必须逐字节可复现,否则幂等键会漂"
