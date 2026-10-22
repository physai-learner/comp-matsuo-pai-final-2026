"""docs/report.md を整形済み HTML に変換し、ブラウザのヘッドレス印刷で PDF（A4）を書き出す。

uv run python scripts/make_pdf.py                 # outputs/report.pdf
uv run python scripts/make_pdf.py --out report.pdf
"""

import argparse
import html
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MD = ROOT / "docs" / "report.md"
DEFAULT_OUT = ROOT / "outputs" / "report.pdf"

BROWSERS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]

CSS = """\
@page {
  size: A4;
  margin: 14mm 16mm 14mm 16mm;
}
body {
  font-family: "BIZ UDPGothic", "Yu Gothic Medium", "Yu Gothic", "Meiryo", sans-serif;
  font-size: 9.8pt;
  line-height: 1.55;
  color: #1a1a1a;
  margin: 0;
}
h1 {
  font-size: 16pt;
  border-bottom: 2px solid #222;
  padding-bottom: 4px;
  margin: 0 0 10px 0;
}
h2 {
  font-size: 12pt;
  border-bottom: 1px solid #bbb;
  padding-bottom: 2px;
  margin: 14px 0 6px 0;
  page-break-after: avoid;
}
h3 {
  font-size: 10.5pt;
  margin: 10px 0 4px 0;
  page-break-after: avoid;
}
p {
  margin: 5px 0;
  text-align: justify;
}
ul, ol {
  margin: 5px 0;
  padding-left: 20px;
}
li {
  margin: 2.5px 0;
}
pre {
  background: #f5f6f8;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 7px 10px;
  font-family: "BIZ UDGothic", "Consolas", monospace;
  font-size: 9pt;
  line-height: 1.4;
  margin: 6px 0;
  page-break-inside: avoid;
}
code {
  font-family: "Consolas", "BIZ UDGothic", monospace;
  font-size: 9pt;
  background: #f2f3f5;
  padding: 1px 4px;
  border-radius: 3px;
}
pre code {
  background: none;
  padding: 0;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 6px 0 8px 0;
  font-size: 9.2pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #ccc;
  padding: 4px 8px;
  text-align: left;
  vertical-align: top;
}
th:first-child, td:first-child {
  white-space: nowrap;
}
th {
  background: #f0f2f5;
  font-weight: bold;
}
a {
  color: #0b57d0;
  text-decoration: none;
}
"""


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(https?://[^\s<）)]+)", r'<a href="\1">\1</a>', text)
    return text


def md_to_html(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(html.escape(lines[i], quote=False))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            continue
        if line.startswith("# "):
            out.append(f"<h1>{inline(line[2:].strip())}</h1>")
            i += 1
            continue
        if line.startswith("## "):
            out.append(f"<h2>{inline(line[3:].strip())}</h2>")
            i += 1
            continue
        if line.startswith("### "):
            out.append(f"<h3>{inline(line[4:].strip())}</h3>")
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            out.append("<table><thead><tr>")
            for c in rows[0]:
                out.append(f"<th>{inline(c)}</th>")
            out.append("</tr></thead><tbody>")
            for r in rows[2:]:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            continue
        if line.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{inline(lines[i][2:].strip())}</li>")
                i += 1
            out.append("</ul>")
            continue
        if re.match(r"^\d+\.\s", line):
            out.append("<ol>")
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i]):
                item = re.sub(r"^\d+\.\s+", "", lines[i]).strip()
                out.append(f"<li>{inline(item)}</li>")
                i += 1
            out.append("</ol>")
            continue
        para = []
        while (
            i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "```", "|", "- "))
        ):
            para.append(inline(lines[i].strip()))
            i += 1
        out.append("<p>" + "".join(para) + "</p>")
    body = "\n".join(out)
    return (
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def find_browser() -> Path:
    for p in BROWSERS:
        if p.exists():
            return p
    raise SystemExit("Edge または Chrome が見つからない")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_MD)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    html_path = args.out.with_suffix(".html")
    html_path.write_text(md_to_html(args.src.read_text(encoding="utf-8")), encoding="utf-8")

    browser = find_browser()
    try:
        subprocess.run(
            [
                str(browser),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={args.out.resolve()}",
                html_path.resolve().as_uri(),
            ],
            check=True,
        )
    finally:
        html_path.unlink(missing_ok=True)
    data = args.out.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page\b", data))
    print(f"{args.out} ({pages}ページ, {len(data) // 1024}KB)")


if __name__ == "__main__":
    main()
