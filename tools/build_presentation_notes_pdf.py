from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

NAVY = colors.HexColor("#172d49")
BLUE = colors.HexColor("#0b5271")
GRAY = colors.HexColor("#586a7d")

styles = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=19, leading=24, textColor=NAVY, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=12.5, leading=17, textColor=BLUE, spaceBefore=12, spaceAfter=6, keepWithNext=True),
    "h3": ParagraphStyle("h3", fontName="Arial-Bold", fontSize=10.5, leading=15, textColor=NAVY, spaceBefore=9, spaceAfter=4, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="Arial", fontSize=9.7, leading=14.3, textColor=NAVY, spaceAfter=7),
    "list": ParagraphStyle("list", fontName="Arial", fontSize=9.5, leading=13.7, textColor=NAVY, leftIndent=14, firstLineIndent=-10, spaceAfter=5),
    "intro": ParagraphStyle("intro", fontName="Arial", fontSize=9.5, leading=14, textColor=GRAY, spaceAfter=12),
}


def inline_markup(value: str) -> str:
    value = value.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", " - ")
    value = html.escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`(.+?)`", r'<font color="#0b5271">\1</font>', value)
    return value


def elements(markdown: str):
    story = []
    first_body = True
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), styles["title"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), styles["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), styles["h3"]))
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + inline_markup(line[2:]), styles["list"]))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(inline_markup(line), styles["list"]))
        else:
            story.append(Paragraph(inline_markup(line), styles["intro"] if first_body else styles["body"]))
            first_body = False
    return story


def footer(canvas, doc):
    canvas.saveState()
    w, _ = A4
    canvas.setStrokeColor(colors.HexColor("#d4dde4"))
    canvas.line(45, 40, w - 45, 40)
    canvas.setFont("Arial", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(45, 27, "Khmer ASR | Kirirom Institute of Technology")
    canvas.drawRightString(w - 45, 27, f"Page {doc.page}")
    canvas.restoreState()


def build(source: str, output: str, stop_at: str | None = None):
    inp = ROOT / source
    out = ROOT / output
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=54,
                            title=inp.stem.replace("_", " "), author="Khemrak Pasey")
    markdown = inp.read_text(encoding="utf-8")
    if stop_at:
        markdown = markdown.split(stop_at, 1)[0]
    doc.build(elements(markdown), onFirstPage=footer, onLaterPages=footer)
    print(out)


if __name__ == "__main__":
    build("slides/presentation_script_training_process_2026-10-20.md", "slides/Khmer_ASR_presentation_script_2026-10-22.pdf", "## Likely lecturer questions")
    build("slides/Khmer_ASR_lecturer_QA_training_process_2026-10-22.md", "slides/Khmer_ASR_lecturer_questions_2026-10-22.pdf", "## Evidence to open if asked")
