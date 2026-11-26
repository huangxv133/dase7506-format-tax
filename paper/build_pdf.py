"""Build an IEEE-style two-column PDF from paper.md.

Pipeline: markdown -> HTML (IEEE conference styling) -> Chrome headless print-to-pdf.
No external Python deps: the markdown subset used by the paper is parsed by hand.
"""

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = ROOT / "figs"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

FIG_BLOCKS = {
    "**A. The anchor behaves as required.**": (
        "fig1_heatmap.png",
        "Figure 1. Lenient accuracy (%) for the six format conditions, by model. "
        "Rows are verbosity, columns are answer format.",
        100,
    ),
    "**E. Brevity and boxed format interact super-additively.**": (
        "fig2_pareto.png",
        "Figure 2. Accuracy against average output tokens; lower-right is better. "
        "C0 is omitted (3-4 tokens, 5-19% accuracy).",
        72,
    ),
    "**TABLE I \u2014 Accuracy (%) and cost by condition.**": (
        "fig3_tax.png",
        "Figure 3. Capability tax relative to C1, in percentage points. Only the 7B model "
        "shows a detectable penalty, and it is largest when brevity and the boxed format "
        "are combined.",
        82,
    ),
}


def inline(text: str) -> str:
    """Escape HTML then apply the inline markup used in the paper."""
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", out)
    return out


def parse_table(lines):
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    head, body = rows[0], rows[1:]
    out = ["<table>"]
    out.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead>")
    out.append("<tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def parse_markdown(md: str):
    lines = md.split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("# "):
            blocks.append(("h1", s[2:].strip()))
            i += 1
        elif s.startswith("## "):
            blocks.append(("h2", s[3:].strip()))
            i += 1
        elif s.startswith("|"):
            j = i
            while j < n and lines[j].strip().startswith("|"):
                j += 1
            blocks.append(("table", lines[i:j]))
            i = j
        elif s.startswith("- "):
            items = []
            while i < n and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append(("ul", items))
        else:
            j = i
            para = []
            while j < n and lines[j].strip() and not lines[j].strip().startswith(("#", "|", "- ")):
                para.append(lines[j].strip())
                j += 1
            blocks.append(("p", " ".join(para)))
            i = j
    return blocks


def build_html(md: str, title_meta: dict) -> str:
    blocks = parse_markdown(md)
    body = []
    in_abstract = False
    in_refs = False
    pending_caption = [None]

    for idx, (kind, val) in enumerate(blocks):
        if kind == "h1":
            body.append(
                '<div class="titleblock">'
                f'<h1 class="title">{inline(val)}</h1>'
                f'<p class="subtitle">{inline(title_meta["subtitle"])}</p>'
                f'<p class="author">{inline(title_meta["author"])}<br>'
                f'<span class="affil">{inline(title_meta["affil"])}</span></p>'
                "</div>"
            )
            continue

        if kind == "h2":
            if val.lower() == "abstract":
                in_abstract = True
                continue
            in_abstract = False
            low = val.lower()
            if low.startswith("reference"):
                in_refs = True
                body.append('<h2 class="refs-h">References</h2>')
                continue
            if low.startswith("appendix"):
                body.append('<div class="pagebreak"></div>')
                body.append('<h2 class="appx-h">Appendix</h2>')
                continue
            body.append(f"<h2>{inline(val)}</h2>")
            continue

        if kind == "p":
            if in_abstract:
                body.append(f'<div class="abstract"><span class="abs-lab">Abstract</span>\u2014{inline(val)}</div>')
                in_abstract = False
                continue
            if in_refs:
                body.append(f'<p class="ref">{inline(val)}</p>')
                continue
            if val.strip() == title_meta["subtitle"]:
                continue
            # figures are injected immediately before the anchored paragraph
            for key, (fn, cap, w) in FIG_BLOCKS.items():
                if val.startswith(key):
                    src = (FIGS / fn).as_uri()
                    body.append(
                        f'<figure class="fig"><img src="{src}" style="width:{w}%">'
                        f'<figcaption>{inline(cap)}</figcaption></figure>'
                    )
                    break
            if val.startswith("**TABLE"):
                pending_caption[0] = val
            else:
                body.append(f"<p>{inline(val)}</p>")
            continue

        if kind == "ul":
            body.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in val) + "</ul>")
            continue

        if kind == "table":
            cap = pending_caption[0]
            pending_caption[0] = None
            cap_html = f'<p class="tblcap">{inline(cap)}</p>' if cap else ""
            body.append(f'<div class="tblwrap">{cap_html}{parse_table(val)}</div>')
            continue

    css = """@page { size: letter; margin: 0.85in 0.95in 0.85in 0.95in; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font-family: "Times New Roman", Times, serif; font-size: 10.2pt; line-height: 1.30;
       color: #000; text-align: justify; -webkit-print-color-adjust: exact; }
.titleblock { text-align: center; margin: 0 0 12pt 0; }
h1.title { font-size: 18.5pt; line-height: 1.15; font-weight: bold;
           text-align: center; margin: 0 0 8pt 0; letter-spacing: 0.1pt; }
p.subtitle { text-align: center; font-size: 10pt; font-style: italic;
             margin: 0 0 4pt 0; text-indent: 0; }
p.author { text-align: center; font-size: 11pt; margin: 0; text-indent: 0; }
p.author .affil { font-size: 9.5pt; font-style: italic; }
.abstract { font-size: 9.7pt; font-weight: normal; margin: 0 0.28in 12pt 0.28in;
            text-align: justify; line-height: 1.26; }
.abs-lab { font-style: italic; font-weight: bold; }
h2 { font-size: 10.6pt; font-weight: normal; font-variant: small-caps; letter-spacing: 0.4pt;
     text-align: center; margin: 13pt 0 5pt 0; break-after: avoid; }
p { margin: 0 0 0 0; text-indent: 0.20in; orphans: 2; widows: 2; }
h2 + p { text-indent: 0; }
p.ref { font-size: 8.9pt; text-indent: -0.20in; margin-left: 0.20in; margin-bottom: 3.4pt;
        line-height: 1.18; text-align: left; }
ul { margin: 3pt 0 6pt 0; padding-left: 0.30in; }
li { margin-bottom: 3pt; }
code { font-family: "Consolas", "Courier New", monospace; font-size: 9.2pt; }
.tblcap { font-size: 8.6pt; text-align: justify; margin: 10pt 0 3.5pt 0; text-indent: 0; }
.tblwrap { margin: 0 0 10pt 0; break-inside: avoid; }
.tblwrap table { width: 100%; border-collapse: collapse; font-size: 8.5pt; table-layout: auto; }
.tblwrap th:first-child, .tblwrap td:first-child { text-align: left; padding-right: 8pt; }
.tblwrap td, .tblwrap th { white-space: nowrap; }
.tblwrap th, .tblwrap td { border-top: 0.5pt solid #000; border-bottom: 0.5pt solid #000;
                           padding: 2.4pt 3pt; text-align: center; line-height: 1.16; }
.tblwrap thead th { border-top: 1.2pt solid #000; }
.tblwrap tbody tr:last-child td { border-bottom: 1.2pt solid #000; }
figure.fig { margin: 11pt 0 10pt 0; break-inside: avoid; text-align: center; }
figure.fig img { display: block; margin: 0 auto; }
figure.fig figcaption { font-size: 8.6pt; text-align: justify; margin-top: 4pt; line-height: 1.2; }
.pagebreak { break-before: page; }"""
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title_meta['title'])}</title><style>{css}</style></head>"
        f'<body><div class="flow">{"".join(body)}</div></body></html>'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(HERE / "paper.md"))
    ap.add_argument("--appendix", default=str(HERE / "appendix.md"))
    ap.add_argument("--out", default=str(HERE / "paper.pdf"))
    ap.add_argument("--author", default="[Student Name] \u00b7 [Student ID]")
    ap.add_argument("--affil", default="The University of Hong Kong")
    ap.add_argument("--repo", default="[REPOSITORY URL]")
    args = ap.parse_args()

    md = Path(args.md).read_text(encoding="utf-8")
    ap_path = Path(args.appendix)
    if ap_path.exists():
        md = md.rstrip() + "\n\n" + ap_path.read_text(encoding="utf-8")

    if args.repo and args.repo != "[REPOSITORY URL]":
        md = md.replace("[REPOSITORY URL]", args.repo)

    title = md.split("\n", 1)[0].lstrip("# ").strip()
    meta = {
        "title": title,
        "subtitle": "DASE7506 Advanced Machine Learning Methods \u2014 Final Project",
        "author": args.author,
        "affil": args.affil,
    }
    html_str = build_html(md, meta)
    html_path = HERE / "paper_print.html"
    html_path.write_text(html_str, encoding="utf-8")

    out = Path(args.out)
    cmd = [
        CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", "--allow-file-access-from-files",
        "--virtual-time-budget=8000",
        f"--print-to-pdf={out}",
        html_path.as_uri(),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not out.exists():
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit(1)
    print(f"PDF {out} {out.stat().st_size} bytes")

    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(out))
        print(f"pages {len(doc)}")
        for i, page in enumerate(doc):
            txt = page.get_textpage().get_text_range()
            first = " ".join(txt.split())[:70]
            print(f"  p{i+1}: {first}")
    except Exception as e:  # noqa: BLE001
        print("page introspection unavailable:", e)


if __name__ == "__main__":
    main()
