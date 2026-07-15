#!/usr/bin/env python3
"""Render an experiment's results/summary.md into an easy-to-read, self-contained PDF.

Reads results/summary.md (the machine-generated write-up) and the curated figures in
figures/, converts the markdown to styled HTML (tables and all), embeds every figure
inline as a base64 gallery so the PDF stands alone, and renders it with WeasyPrint.

Uses the SYSTEM python (/usr/bin/python3), which has `markdown` and `weasyprint`
installed — NOT the CUDA venv.

    /usr/bin/python3 scripts/make_results_pdf.py experiments/grounded_sam_local
    /usr/bin/python3 scripts/make_results_pdf.py experiments/grounded_sam2_local
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

CSS = """
@page {
    size: A4;
    margin: 16mm 15mm 18mm 15mm;
    @bottom-center { content: counter(page) " / " counter(pages);
                     font: 8pt "DejaVu Sans"; color: #9aa0a6; }
    @bottom-right  { content: string(doctitle);
                     font: 8pt "DejaVu Sans"; color: #9aa0a6; }
}
* { box-sizing: border-box; }
html { -weasy-hyphens: none; }
body {
    font-family: "DejaVu Sans", "Helvetica", "Arial", sans-serif;
    font-size: 10pt; line-height: 1.5; color: #24292f;
}
.report-meta {
    font-size: 8.5pt; color: #6a737d; letter-spacing: .04em;
    text-transform: uppercase; margin-bottom: 2mm;
}
h1 {
    string-set: doctitle content();
    font-size: 20pt; line-height: 1.2; color: #0b3d5c; margin: 0 0 3mm 0;
    padding-bottom: 2.5mm; border-bottom: 2.5pt solid #0b3d5c;
}
h2 {
    font-size: 13pt; color: #0b3d5c; margin: 8mm 0 2mm 0;
    padding-top: 2mm; border-top: 0.7pt solid #d7dde3;
    break-after: avoid;
}
h3 { font-size: 11pt; color: #1f5c86; margin: 5mm 0 1.5mm 0; break-after: avoid; }
h1 + p, h2 + p, h3 + p { margin-top: 0; }
p { margin: 0 0 2.6mm 0; }
a { color: #1f5c86; text-decoration: none; }
strong { color: #16324a; }
hr { border: none; border-top: 0.7pt solid #d7dde3; margin: 6mm 0; }

/* the classification callout */
p.callout {
    background: #eef5fb; border: 0.8pt solid #bcd6ea; border-left: 4pt solid #1f7ac0;
    border-radius: 3pt; padding: 3.5mm 4mm; margin: 3mm 0 4mm 0; font-size: 10.5pt;
}
p.callout strong { color: #0b3d5c; }

ul, ol { margin: 0 0 2.6mm 0; padding-left: 6mm; }
li { margin: 0 0 1.2mm 0; }

table {
    border-collapse: collapse; width: 100%; margin: 2mm 0 4mm 0;
    font-size: 9pt; break-inside: avoid;
}
th, td {
    border: 0.6pt solid #cfd6dd; padding: 1.6mm 2.2mm; text-align: left;
    vertical-align: top;
}
thead th { background: #0b3d5c; color: #fff; font-weight: bold; border-color: #0b3d5c; }
tbody tr:nth-child(even) { background: #f4f7fa; }
td strong { color: #0b3d5c; }

code {
    font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt;
    background: #f2f4f6; padding: 0.3mm 1mm; border-radius: 2pt;
}
pre {
    background: #f6f8fa; border: 0.6pt solid #e1e4e8; border-radius: 3pt;
    padding: 3mm; font-size: 8pt; line-height: 1.4; overflow-x: auto;
    white-space: pre-wrap; word-wrap: break-word; break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8pt; }

/* figure gallery */
.gallery-sep { break-before: page; }
figure {
    margin: 0 0 7mm 0; break-inside: avoid; text-align: center;
}
figure img {
    max-width: 100%; max-height: 115mm; border: 0.8pt solid #cfd6dd; border-radius: 2pt;
}
figcaption {
    font-size: 8.7pt; color: #57606a; text-align: left; margin-top: 1.6mm;
    line-height: 1.4;
}
figcaption .figlabel { color: #0b3d5c; font-weight: bold; }
em { color: #444; }
"""

FIG_HEADING_RE = re.compile(r"^###\s+(?P<file>\S+\.jpg)\s*$")


def parse_figures(fig_readme: Path):
    """Extract (filename, caption) pairs from a figures/README.md."""
    if not fig_readme.exists():
        return []
    lines = fig_readme.read_text().splitlines()
    figs, i = [], 0
    while i < len(lines):
        m = FIG_HEADING_RE.match(lines[i].strip())
        if m:
            fname = m.group("file")
            # caption = first non-empty, non-image line after the heading
            cap, j = "", i + 1
            while j < len(lines):
                s = lines[j].strip()
                if s and not s.startswith("!["):
                    cap = s
                    break
                j += 1
            figs.append((fname, cap))
        i += 1
    return figs


def data_uri(path: Path) -> str:
    b = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b}"


def build_gallery(fig_dir: Path) -> str:
    figs = parse_figures(fig_dir / "README.md")
    if not figs:
        return ""
    parts = ['<h2 class="gallery-sep">Figure gallery</h2>']
    for fname, cap in figs:
        fp = fig_dir / fname
        if not fp.exists():
            continue
        num = fname.split("_", 1)[0]
        label = f"Figure {num}"
        parts.append(
            "<figure>"
            f'<img src="{data_uri(fp)}" />'
            f'<figcaption><span class="figlabel">{label}.</span> {cap}</figcaption>'
            "</figure>"
        )
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=Path,
                    help="experiment dir, e.g. experiments/grounded_sam_local")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PDF path (default: <exp_dir>/results/<name>_report.pdf)")
    args = ap.parse_args()

    exp = args.exp_dir.resolve()
    summary = exp / "results" / "summary.md"
    if not summary.exists():
        sys.exit(f"missing {summary}")

    md_text = summary.read_text()
    # drop the trailing "*Generated by ...*" italic footer; we add our own meta line
    md_text = re.sub(r"\n---\n\n\*Generated by.*\Z", "\n", md_text, flags=re.S)

    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    # highlight the classification / bottom-line paragraph
    body = body.replace("<p><strong>Bottom line",
                        '<p class="callout"><strong>Bottom line')

    gallery = build_gallery(exp / "figures")

    today = dt.date.today().isoformat()
    meta = (f'<div class="report-meta">Results report · {exp.name} · '
            f'generated {today}</div>')

    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{meta}{body}{gallery}</body></html>")

    out = args.out or (exp / "results" / f"{exp.name}_report.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(exp)).write_pdf(str(out))
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
