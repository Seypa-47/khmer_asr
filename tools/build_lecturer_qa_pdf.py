"""Build a readable viva rehearsal guide from the verified Markdown source."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
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
    PageTemplate,
    Paragraph,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "slides" / "Khmer_ASR_lecturer_QA_2026-10-16.md"
OUTPUT = ROOT / "output" / "pdf" / "Khmer_ASR_lecturer_QA_2026-10-16.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

NAVY = colors.HexColor("#142B44")
TEAL = colors.HexColor("#078F88")
MUTED = colors.HexColor("#526373")

TITLE = ParagraphStyle("title", fontName="Arial-Bold", fontSize=21, leading=25,
                       textColor=NAVY, spaceAfter=7)
SUB = ParagraphStyle("sub", fontName="Arial", fontSize=9.4, leading=13.5,
                     textColor=MUTED, spaceAfter=11)
H2 = ParagraphStyle("h2", fontName="Arial-Bold", fontSize=12.2, leading=16,
                    textColor=TEAL, spaceBefore=13, spaceAfter=6, keepWithNext=True)
H3 = ParagraphStyle("h3", fontName="Arial-Bold", fontSize=10.5, leading=14,
                    textColor=NAVY, spaceBefore=11, spaceAfter=4, keepWithNext=True)
BODY = ParagraphStyle("body", fontName="Arial", fontSize=9.55, leading=13.6,
                      textColor=NAVY, spaceAfter=7, splitLongWords=True)
QA = ParagraphStyle("qa", parent=BODY, leftIndent=0, spaceAfter=8)
LIST = ParagraphStyle("list", parent=BODY, leftIndent=14, firstLineIndent=-11,
                      spaceAfter=5)
EVIDENCE = ParagraphStyle("evidence", parent=LIST, fontSize=8.75,
                          leading=12.7, splitLongWords=True)


class AccentLine(Flowable):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.height = 4

    def draw(self):
        self.canv.setFillColor(TEAL)
        self.canv.roundRect(0, 0, self.width, self.height, 2, fill=1, stroke=0)


def markup(value: str) -> str:
    value = value.replace("—", "-").replace("–", "-").replace("×", "x")
    value = html.escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", value)
    return value


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(TEAL)
    canvas.rect(0, height - 8 * mm, width, 8 * mm, fill=1, stroke=0)
    canvas.setFont("Arial", 7.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 12 * mm, "Khmer ASR  |  Lecturer Q&A rehearsal")
    canvas.drawRightString(width - 17 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUTPUT), pagesize=A4,
    leftMargin=17 * mm, rightMargin=17 * mm,
    topMargin=20 * mm, bottomMargin=20 * mm,
    title="Khmer ASR lecturer questions and model answers",
    author="Khemrak Pasey",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
doc.addPageTemplates(PageTemplate(id="default", frames=[frame], onPage=on_page))

lines = SOURCE.read_text(encoding="utf-8").splitlines()
story = []
buffer = []
section = ""


def flush():
    global buffer
    if not buffer:
        return
    value = " ".join(buffer)
    style = QA if value.startswith("**") else BODY
    story.append(Paragraph(markup(value), style))
    buffer = []


for line in lines:
    if not line.strip():
        flush()
        continue
    if line.startswith("# "):
        flush()
        story.append(Paragraph(markup(line[2:]), TITLE))
        story.append(AccentLine(doc.width))
        story.append(Spacer(1, 8))
    elif line.startswith("## "):
        flush()
        section = line[3:]
        story.append(Paragraph(markup(section), H2))
    elif line.startswith("### "):
        flush()
        story.append(Paragraph(markup(line[4:]), H3))
    elif re.match(r"^\d+\. ", line):
        flush()
        number, body = line.split(". ", 1)
        story.append(KeepTogether([Paragraph(f"<b>{number}.</b> {markup(body)}", LIST)]))
    elif line.startswith("- "):
        flush()
        style = EVIDENCE if section == "Evidence to open if asked" else LIST
        story.append(KeepTogether([Paragraph("- " + markup(line[2:]), style)]))
    else:
        buffer.append(line.strip())
flush()

doc.build(story)
print(OUTPUT, OUTPUT.stat().st_size)
