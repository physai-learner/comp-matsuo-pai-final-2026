"""docs/report.md を tool-docx-kit の体裁で docx と PDF（A4）に書き出す。

uv run --project C:/dev/tool-docx-kit python scripts/make_pdf.py
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MD = ROOT / "docs" / "report.md"
DEFAULT_OUT = ROOT / "outputs" / "report.pdf"
DOCX_KIT = Path("C:/dev/tool-docx-kit")
SUBTITLE = (
    "LLMの2次元画素推論と逆運動学の分担により、28指示×5配置で96%（135/140）の把持・運搬を達成"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_MD)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--kit", type=Path, default=DOCX_KIT)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    docx_path = args.out.with_suffix(".docx").resolve()
    pdf_path = args.out.resolve()

    sys.path.insert(0, str(args.kit))
    import win32com.client as win32  # type: ignore[import-not-found]
    from md2docx import build_from_md  # type: ignore[import-not-found]
    from page_footer import apply_page_number_rule  # type: ignore[import-not-found]
    from qa_preview import docx_to_pdf  # type: ignore[import-not-found]

    build_from_md(
        args.src,
        docx_path,
        header="event",
        preset="compact",
        kind="Physical AI 講座 最終課題",
        subtitle=SUBTITLE,
    )
    pages, _ = apply_page_number_rule(str(docx_path), default=True)
    subprocess.run(
        [sys.executable, str(args.kit / "verify_docx.py"), str(docx_path)],
        check=True,
    )

    app = win32.DispatchEx("Word.Application")
    app.Visible = False
    app.DisplayAlerts = 0
    try:
        docx_to_pdf(app, docx_path, pdf_path)
    finally:
        app.Quit()

    size_kb = pdf_path.stat().st_size // 1024
    print(f"{pdf_path} ({pages}ページ, {size_kb}KB)")
    desk = Path.home() / "Desktop"
    for candidate in (Path(r"G:\マイドライブ\Desktop"), desk):
        if candidate.exists():
            shutil.copy2(pdf_path, candidate / pdf_path.name)
            print(f"copied: {candidate / pdf_path.name}")
            break


if __name__ == "__main__":
    main()
