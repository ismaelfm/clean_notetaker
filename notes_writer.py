"""Markdown notes file writer.

Handles creating, appending, and deduplicating page entries
in the per-book markdown notes files.
"""

import re
from datetime import datetime
from pathlib import Path


NOTES_DIR = "notes"


def _get_notes_path(base_dir: str | Path, book_name: str) -> Path:
    """Get the path to the notes file for a given book."""
    notes_dir = Path(base_dir) / NOTES_DIR
    notes_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", book_name)
    return notes_dir / f"{safe_name}_Notes.md"


def _get_existing_pages(notes_path: Path) -> set[int]:
    """Parse an existing notes file and return page numbers already present."""
    if not notes_path.exists():
        return set()

    content = notes_path.read_text(encoding="utf-8")
    pattern = r"^## .+ - Page (\d+):"
    pages = set()
    for match in re.finditer(pattern, content, re.MULTILINE):
        pages.add(int(match.group(1)))
    return pages


def _create_header(book_name: str, course_id: str, cert_name: str) -> str:
    """Create the document header for a new notes file."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"# {book_name} — Study Notes\n\n"
        f"**Course:** {course_id} ({cert_name})  \n"
        f"**Generated:** {now}  \n"
        f"**Tool:** PDF Study Notes Extractor (OpenRouter Vision API)  \n\n"
        f"---\n\n"
    )


def append_page_notes(
    base_dir: str | Path,
    book_name: str,
    page_number: int,
    notes_content: str,
    course_id: str = "SEC560",
    cert_name: str = "GPEN",
) -> tuple[Path, bool]:
    """Append page notes to the book's markdown file.

    Returns:
        Tuple of (notes_file_path, was_skipped).
        was_skipped is True if the page already existed.
    """
    notes_path = _get_notes_path(base_dir, book_name)
    existing_pages = _get_existing_pages(notes_path)

    if page_number in existing_pages:
        return notes_path, True

    if not notes_path.exists():
        notes_path.write_text(
            _create_header(book_name, course_id, cert_name),
            encoding="utf-8",
        )

    with open(notes_path, "a", encoding="utf-8") as f:
        f.write(notes_content.strip())
        f.write("\n\n---\n\n")

    return notes_path, False


def get_notes_path(base_dir: str | Path, book_name: str) -> Path:
    """Public accessor for the notes file path."""
    return _get_notes_path(base_dir, book_name)
