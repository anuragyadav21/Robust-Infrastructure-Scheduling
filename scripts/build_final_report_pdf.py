#!/usr/bin/env python3
"""Rebuild Final_Report.pdf from REPORT.md (Markdown → HTML → PDF).

Requires: pip install markdown xhtml2pdf

Embeds local PNGs (paths under the repo root), adds page numbers, and enables
raw HTML (figure montage) via the ``markdown`` *extra* extension.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _link_callback(uri: str, rel: str) -> str:
    """Resolve relative image paths for xhtml2pdf (absolute filesystem path)."""
    if uri.startswith(("http://", "https://", "data:")):
        return uri
    path = (ROOT / uri).resolve()
    if path.is_file():
        return str(path)
    return uri


def main() -> None:
    try:
        import markdown
        from xhtml2pdf import pisa
    except ImportError as e:
        raise SystemExit(
            "Missing dependency. Run:\n"
            "  pip install markdown xhtml2pdf\n"
            f"({e})"
        ) from e

    md_path = ROOT / "REPORT.md"
    pdf_path = ROOT / "Final_Report.pdf"
    md = md_path.read_text(encoding="utf-8")

    html_body = markdown.markdown(
        md,
        extensions=["tables", "fenced_code", "nl2br", "extra"],
    )

    footer = """
<div id="pager" style="position: fixed; bottom: 0; left: 0; right: 0; text-align: center;
            font-size: 8pt; padding: 4px 0 2px 0; border-top: 1px solid #ccc; background: #fff;">
  Page <pdf:pagenumber example="%" /> of <pdf:pagecount />
</div>
"""

    html_simple = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@page {{ size: a4; margin: 1.4cm 1.3cm 1.6cm 1.3cm; }}
body {{
  font-family: Helvetica, Arial, sans-serif;
  line-height: 1.35;
  font-size: 9pt;
  padding-bottom: 18pt;
}}
h1 {{ font-size: 16pt; page-break-after: avoid; }}
h2 {{ font-size: 12pt; margin-top: 1em; page-break-after: avoid; }}
h3 {{ font-size: 10pt; page-break-after: avoid; }}
h4 {{ font-size: 9.5pt; page-break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.4em 0; font-size: 8pt; }}
th, td {{ border: 1px solid #999; padding: 3px 4px; vertical-align: top; }}
code {{ font-size: 8pt; background: #f0f0f0; }}
pre {{ background: #f0f0f0; padding: 6px; font-size: 7.5pt; white-space: pre-wrap; }}
figure {{ margin: 0.6em 0; text-align: center; page-break-inside: avoid; }}
img {{ max-width: 100%; height: auto; }}
</style></head><body>
{html_body}
{footer}
</body></html>"""

    with pdf_path.open("wb") as f:
        status = pisa.CreatePDF(
            html_simple,
            dest=f,
            encoding="utf-8",
            link_callback=_link_callback,
        )
    if status.err:
        raise SystemExit("xhtml2pdf reported errors; PDF may be incomplete.")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
