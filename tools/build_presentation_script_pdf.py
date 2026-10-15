"""Render the Khmer ASR speaking notes as a print-friendly rehearsal PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "slides" / "presentation_script_2026-10-16.md"
OUTPUT = ROOT / "output" / "pdf" / "Khmer_ASR_presentation_script_2026-10-16.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

NAVY = colors.HexColor("#142B44")
TEAL = colors.HexColor("#078F88")
GRAY = colors.HexColor("#516170")
PALE = colors.HexColor("#E8F3F3")

title_style = ParagraphStyle(
    "title", fontName="Arial-Bold", fontSize=22, leading=27,
    textColor=NAVY, spaceAfter=8,
)
sub_style = ParagraphStyle(
    "sub", fontName="Arial", fontSize=9.5, leading=14,
    textColor=GRAY, spaceAfter=15,
)
heading_style = ParagraphStyle(
    "heading", fontName="Arial-Bold", fontSize=11.5, leading=15,
    textColor=TEAL, spaceBefore=12, spaceAfter=5,
)
body_style = ParagraphStyle(
    "body", fontName="Arial", fontSize=10.2, leading=15.3,
    textColor=NAVY, alignment=TA_LEFT, spaceAfter=6,
)
qa_style = ParagraphStyle(
    "qa", parent=body_style, leftIndent=10, firstLineIndent=-10,
    spaceAfter=9,
)


class AccentLine(Flowable):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.height = 5

    def draw(self):
        self.canv.setFillColor(TEAL)
        self.canv.roundRect(0, 0, self.width, self.height, 2.5, fill=1, stroke=0)


def markup(value: str) -> str:
    escaped = html.escape(value)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(TEAL)
    canvas.rect(0, height - 9 * mm, width, 9 * mm, fill=1, stroke=0)
    canvas.setFont("Arial", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(19 * mm, 13 * mm, "Khmer ASR  |  Presentation rehearsal script")
    canvas.drawRightString(width - 19 * mm, 13 * mm, str(doc.page))
    canvas.restoreState()


raw = SOURCE.read_text(encoding="utf-8")
sections: list[tuple[str, list[str]]] = []
current = None
for line in raw.splitlines():
    if line.startswith("## "):
        current = (line[3:], [])
        sections.append(current)
    elif current is not None and line.strip():
        current[1].append(line.strip())

doc = BaseDocTemplate(
    str(OUTPUT), pagesize=A4,
    leftMargin=19 * mm, rightMargin=19 * mm,
    topMargin=21 * mm, bottomMargin=22 * mm,
    title="Khmer ASR presentation script",
    author="Khemrak Pasey",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
doc.addPageTemplates(PageTemplate(id="default", frames=[frame], onPage=on_page))

story = [
    Paragraph("Khmer automatic speech recognition", title_style),
    Paragraph("Speaking notes  •  16 October 2026  •  About 10–12 minutes plus demo", sub_style),
    AccentLine(doc.width),
    Spacer(1, 5),
]

for heading, paragraphs in sections:
    if heading == "Likely lecturer questions":
        story.append(PageBreak())
        story.append(Paragraph("Likely lecturer questions", title_style))
        story.append(AccentLine(doc.width))
        story.append(Spacer(1, 11))
        for answer in paragraphs:
            story.append(Paragraph(markup(answer), qa_style))
        continue
    group = [Paragraph(markup(heading), heading_style)]
    group.extend(Paragraph(markup(p), body_style) for p in paragraphs)
    story.append(KeepTogether(group))

doc.build(story)
print(OUTPUT, OUTPUT.stat().st_size)
