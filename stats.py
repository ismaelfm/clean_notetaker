"""Token statistics for PDFs and generated notes.

Compares the token size of source PDF text against extracted notes
to give a sense of compression ratio and coverage.
"""

from pathlib import Path

import fitz  # PyMuPDF
import tiktoken

from notes_writer import get_notes_path

# cl100k_base is a good general-purpose tokenizer approximation
_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a string using cl100k_base encoding."""
    return len(_encoder.encode(text))


def get_pdf_text_tokens(pdf_path: str | Path) -> tuple[int, int]:
    """Extract all text from a PDF and return (char_count, token_count).

    Returns:
        Tuple of (total_characters, total_tokens).
    """
    total_text = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            total_text.append(page.get_text("text"))

    full_text = "\n".join(total_text)
    chars = len(full_text)
    tokens = count_tokens(full_text)
    return chars, tokens


def get_notes_tokens(base_dir: str | Path, book_name: str) -> tuple[int, int] | None:
    """Read the notes file for a book and return (char_count, token_count).

    Returns None if no notes file exists.
    """
    notes_path = get_notes_path(base_dir, book_name)
    if not notes_path.exists():
        return None

    content = notes_path.read_text(encoding="utf-8")
    chars = len(content)
    tokens = count_tokens(content)
    return chars, tokens


def get_notes_page_count(base_dir: str | Path, book_name: str) -> int:
    """Count how many pages have been extracted in the notes file."""
    import re

    notes_path = get_notes_path(base_dir, book_name)
    if not notes_path.exists():
        return 0

    content = notes_path.read_text(encoding="utf-8")
    return len(re.findall(r"^## .+ - Page \d+:", content, re.MULTILINE))
