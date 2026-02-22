"""PDF extraction module using PyMuPDF.

Handles text extraction and page-to-image rendering for vision model input.
Strips configurable junk strings (from strip_strings.txt) before returning text.
"""

import re
from pathlib import Path
from typing import cast

import fitz  # PyMuPDF

_strip_strings: list[str] | None = None


def _load_strip_strings(directory: str | Path | None = None) -> list[str]:
    """Load strings to strip from strip_strings.txt.

    Looks in the given directory, then falls back to cwd.
    Lines starting with # and blank lines are ignored.
    """
    global _strip_strings
    if _strip_strings is not None:
        return _strip_strings

    candidates = []
    if directory:
        candidates.append(Path(directory) / "strip_strings.txt")
    candidates.append(Path.cwd() / "strip_strings.txt")

    for path in candidates:
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            _strip_strings = [
                line
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            ]
            return _strip_strings

    _strip_strings = []
    return _strip_strings


def clean_text(text: str, directory: str | Path | None = None) -> str:
    """Remove all configured strip strings from extracted text."""
    strips = _load_strip_strings(directory)
    for s in strips:
        text = text.replace(s, "")
    # Collapse multiple blank lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_pdf_files(directory: str | Path) -> list[Path]:
    """Scan a directory for PDF files, sorted alphabetically."""
    directory = Path(directory)
    pdfs = sorted(directory.glob("*.pdf"))
    return pdfs


def get_pdf_name(pdf_path: str | Path) -> str:
    """Extract a clean book name from the PDF filename.

    e.g. 'SEC560 - Book 1_3472340-1.pdf' → 'SEC560 - Book 1'
    """
    stem = Path(pdf_path).stem
    # Remove trailing ID patterns like _3472340-1 or _3472340
    clean = re.sub(r"_\d+(-\d+)?$", "", stem)
    return clean.strip()


def get_page_count(pdf_path: str | Path) -> int:
    """Return the total number of pages in a PDF."""
    with fitz.open(str(pdf_path)) as doc:
        return len(doc)


def extract_page(
    pdf_path: str | Path, page_num: int, dpi: int = 200
) -> tuple[str, bytes]:
    """Extract text and render an image for a single PDF page.

    Args:
        pdf_path: Path to the PDF file.
        page_num: 1-indexed page number.
        dpi: Resolution for the rendered page image.

    Returns:
        Tuple of (cleaned_text, png_image_bytes).
    """
    pdf_path = Path(pdf_path)
    with fitz.open(str(pdf_path)) as doc:
        # fitz uses 0-indexed pages
        page = doc[page_num - 1]

        # Extract and clean text
        raw_text = cast(str, page.get_text("text"))
        text = clean_text(raw_text, directory=pdf_path.parent)

        # Render page to high-res PNG
        zoom = dpi / 72  # 72 is the default DPI
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_bytes = pixmap.tobytes("png")

    return text, image_bytes
