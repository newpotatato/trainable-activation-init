"""Export the built article .docx to Markdown (full text, tables, figure links).

    <python with python-docx> docx_to_md.py

Writes ../article/статья_полный_текст.md next to the .docx; figures are linked to
results/figures in the order they appear in the article.
"""
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.table import Table
from docx.text.paragraph import Paragraph

from build_article import OUT

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(os.path.dirname(OUT), "статья_полный_текст.md")
# figure files in the order the article shows them (Рис. 1 ... Рис. 7)
FIG_ORDER = ["fig1_siglin_shape.png", "fig2_improvement.png", "fig4_curves.png", "fig3_param_drift.png",
             "fig6_smooth.png", "fig7_pilot_length.png", "fig5_deep.png", "fig8_vs_norm.png"]


def _inline(p):
    out = []
    for r in p.runs:
        t = r.text
        if not t:
            continue
        out.append(f"**{t.strip()}** " if r.bold and t.strip() else t)
    return "".join(out).replace("** **", " ").strip()


def _is_heading(p):
    runs = [r for r in p.runs if r.text.strip()]
    return (p.alignment == WD_ALIGN_PARAGRAPH.CENTER and runs and all(r.bold for r in runs)
            and len(p.text) < 120)


def main():
    doc = Document(OUT)
    lines, fig_i, title_done = [], 0, False
    for el in doc.element.body.iterchildren():
        tag = el.tag.split("}")[1]
        if tag == "tbl":
            t = Table(el, doc)
            rows = [[c.text.replace("|", "/").strip() for c in r.cells] for r in t.rows]
            lines += ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
            lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
            lines.append("")
            continue
        if tag != "p":
            continue
        p = Paragraph(el, doc)
        if el.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"):
            name = FIG_ORDER[fig_i] if fig_i < len(FIG_ORDER) else ""
            fig_i += 1
            lines += [f"![Рис. {fig_i}](../exp/results/figures/{name})", ""]
            continue
        text = p.text.strip()
        if not text:
            continue
        if not title_done and p.alignment == WD_ALIGN_PARAGRAPH.CENTER and all(r.bold for r in p.runs if r.text.strip()):
            lines += ["# " + text, ""]
            title_done = True
        elif _is_heading(p):
            lines += ["## " + text, ""]
        elif p.alignment == WD_ALIGN_PARAGRAPH.RIGHT and text.startswith("Таблица"):
            lines += [f"*{text}*", ""]
        elif text.startswith("Рис. "):
            lines += [f"*{text}*", ""]
        else:
            lines += [_inline(p), ""]
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("saved", MD, f"({fig_i} figures)")


if __name__ == "__main__":
    main()
