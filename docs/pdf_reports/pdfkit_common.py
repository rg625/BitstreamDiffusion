"""Shared layout for the three October 2026 PDF reports (CPU-only, no GPU).

Run with reportlab on PYTHONPATH; fonts are matplotlib's bundled DejaVu, so
Greek letters, arrows and math symbols render without a LaTeX install.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)

_TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, f in (("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                ("Sans-Italic", "DejaVuSans-Oblique.ttf"),
                ("Sans-BoldItalic", "DejaVuSans-BoldOblique.ttf"),
                ("Mono", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(_TTF / f)))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold",
                              italic="Sans-Italic", boldItalic="Sans-BoldItalic")

# Reference categorical palette (dataviz skill, light mode), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3"
GOOD, BAD, NEUTRAL = "#1e7a46", "#b3261e", "#52514e"
SURFACE_ALT = colors.HexColor("#f4f3ef")

S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=20, leading=25,
                            textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=10.5, leading=14,
                               textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=13.5, leading=17,
                         textColor=INK, spaceBefore=12, spaceAfter=5, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11, leading=14,
                         textColor=INK, spaceBefore=8, spaceAfter=3, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9.3, leading=13.8,
                           textColor=INK, alignment=TA_LEFT, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Sans", fontSize=7.8, leading=10.2,
                            textColor=INK2, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8, leading=11.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8, leading=10.2, textColor=INK),
    "callout": ParagraphStyle("callout", fontName="Sans", fontSize=9.3, leading=13.2, textColor=INK),
    "caption": ParagraphStyle("caption", fontName="Sans", fontSize=7.8, leading=10.2,
                              textColor=INK2, spaceBefore=2, spaceAfter=8),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def H1(text):
    return P(text, "h1")


def H2(text):
    return P(text, "h2")


def bullets(items, style="body"):
    return ListFlowable([ListItem(P(t, style), leftIndent=10, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontName="Sans",
                        bulletFontSize=8, spaceBefore=0, spaceAfter=4)


def table(rows, col_widths=None, header_rows=1, zebra=True, highlight_rows=()):
    """rows: list of lists of str (Paragraph-markup allowed)."""
    data = [[c if not isinstance(c, str) else Paragraph(c, S["cellb" if r < header_rows else "cell"])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=header_rows, hAlign="LEFT")
    st = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, colors.HexColor(INK2)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor(RULE)),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
    ]
    if zebra:
        for r in range(header_rows, len(rows)):
            if (r - header_rows) % 2 == 1:
                st.append(("BACKGROUND", (0, r), (-1, r), SURFACE_ALT))
    for r in highlight_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e7f0fb")))
    t.setStyle(TableStyle(st))
    return t


def callout(text, kind="key"):
    """A boxed takeaway. kind: key (blue rule), warn (red rule), keep (green rule)."""
    edge = {"key": SERIES[0], "warn": BAD, "keep": GOOD}[kind]
    t = Table([[Paragraph(text, S["callout"])]], colWidths=[170 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE_ALT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(edge)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return KeepTogether([t, Spacer(1, 6)])


def figure(fig, name, width_mm=165, caption=None):
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    w, h = fig.get_size_inches()
    width = width_mm * mm
    img = Image(str(path), width=width, height=width * h / w)
    out = [img]
    if caption:
        out.append(P(caption, "caption"))
    return KeepTogether(out)


def mpl_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#ecebe7", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    })


def build(path, story, title, subject):
    def on_page(c, doc):
        c.saveState()
        c.setFont("Sans", 7.2)
        c.setFillColor(colors.HexColor(MUTED))
        c.drawString(20 * mm, 10 * mm, title)
        c.drawRightString(190 * mm, 10 * mm, f"{doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title=title,
                            author="BitstreamDiffusion (CoBit) research notes", subject=subject)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return path


__all__ = ["P", "H1", "H2", "bullets", "table", "callout", "figure", "mpl_style", "build",
           "Spacer", "PageBreak", "KeepTogether", "SERIES", "INK", "INK2", "MUTED", "GOOD", "BAD",
           "NEUTRAL", "mm", "plt", "HERE"]
