"""Interactive PDF study notes extractor.

Scans current directory for PDFs, lets you pick a book and page range,
then extracts page-by-page notes using a vision AI model via OpenRouter.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

from ai_client import analyze_page
from extractor import extract_page, get_page_count, get_pdf_files, get_pdf_name
from notes_writer import append_page_notes, get_notes_path
from prompts import PAGE_PROMPT, SYSTEM_PROMPT
from stats import get_notes_page_count, get_notes_tokens, get_pdf_text_tokens

load_dotenv()

# ── Theme ──────────────────────────────────────────────────────────────
custom_theme = Theme(
    {
        "title": "bold cyan",
        "info": "dim white",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "highlight": "bold magenta",
    }
)

console = Console(theme=custom_theme)


def get_course_config() -> tuple[str, str]:
    """Load course configuration from environment."""
    course_id = os.getenv("COURSE_ID", "COURSE")
    cert_name = os.getenv("CERT_NAME", "CERT")
    return course_id, cert_name


def display_banner():
    """Show the application banner."""
    banner = (
        "[title]📚 PDF Study Notes Extractor[/title]\n"
        "[info]Page-by-page note extraction using Vision AI[/info]"
    )
    console.print(
        Panel(banner, box=box.DOUBLE_EDGE, border_style="cyan", padding=(1, 2))
    )


def _build_pdf_table(pdfs: list[Path], token_cache: dict[str, int]) -> Table:
    """Build the PDF table, using cached tokens or a loading placeholder."""
    table = Table(
        title="Available PDFs",
        box=box.ROUNDED,
        border_style="cyan",
        title_style="title",
        show_lines=True,
    )
    table.add_column("#", style="highlight", justify="right", width=4)
    table.add_column("Book Name", style="bold white")
    table.add_column("Pages", style="info", justify="center")
    table.add_column("Tokens", style="bold yellow", justify="right")
    table.add_column("File", style="dim")

    for i, pdf in enumerate(pdfs, 1):
        name = get_pdf_name(pdf)
        try:
            pages = str(get_page_count(pdf))
        except Exception:
            pages = "?"

        pdf_key = str(pdf)
        if pdf_key in token_cache:
            token_str = f"{token_cache[pdf_key]:,}" if token_cache[pdf_key] else "?"
        else:
            token_str = "[dim italic]⏳ loading…[/dim italic]"

        table.add_row(str(i), name, pages, token_str, pdf.name)

    return table


def display_pdf_list(pdfs: list[Path], token_cache: dict[str, int]) -> None:
    """Display PDFs with token counts loading asynchronously."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.live import Live

    # Find which PDFs still need token counting
    to_compute = [(str(pdf), pdf) for pdf in pdfs if str(pdf) not in token_cache]

    if not to_compute:
        # All cached — just print the table directly
        console.print(_build_pdf_table(pdfs, token_cache))
        return

    # Show table with loading placeholders, update live as tokens come in
    with Live(
        _build_pdf_table(pdfs, token_cache), console=console, refresh_per_second=4
    ) as live:

        def _count_tokens(pdf_path: Path) -> tuple[str, int]:
            try:
                _, tokens = get_pdf_text_tokens(pdf_path)
                return str(pdf_path), tokens
            except Exception:
                return str(pdf_path), 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_count_tokens, pdf): pdf for _, pdf in to_compute
            }
            for future in as_completed(futures):
                pdf_key, tokens = future.result()
                token_cache[pdf_key] = tokens
                live.update(_build_pdf_table(pdfs, token_cache))


def parse_page_range(range_str: str, max_pages: int) -> list[int] | None:
    """Parse a page range string like '1-20' or '5' into a list of page numbers.

    Supports: '1-20', '5', '1,5,10', '1-5,10,15-20'
    """
    pages = []
    try:
        for part in range_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                start, end = int(start.strip()), int(end.strip())
                if start < 1 or end > max_pages or start > end:
                    return None
                pages.extend(range(start, end + 1))
            else:
                p = int(part)
                if p < 1 or p > max_pages:
                    return None
                pages.append(p)
    except ValueError:
        return None

    return sorted(set(pages))


def process_pages(
    pdf_path: Path,
    pages: list[int],
    course_id: str,
    cert_name: str,
    send_image: bool = True,
) -> None:
    """Process a list of pages: extract → analyze → write notes."""
    book_name = get_pdf_name(pdf_path)
    base_dir = pdf_path.parent

    system_prompt = SYSTEM_PROMPT.format(
        course_id=course_id,
        cert_name=cert_name,
        book_name=book_name,
    )

    notes_path = get_notes_path(base_dir, book_name)
    console.print(f"\n[info]Notes file:[/info] [highlight]{notes_path}[/highlight]")
    console.print()

    skipped = 0
    errors = 0
    processed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[info]{task.fields[status]}[/info]"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Extracting {book_name}",
            total=len(pages),
            status="Starting...",
        )

        for page_num in pages:
            progress.update(task, status=f"Page {page_num}")

            try:
                # Step 1: Extract text + image from PDF
                text, image_bytes = extract_page(pdf_path, page_num)

                # Step 2: Build the per-page prompt
                user_prompt = PAGE_PROMPT.format(
                    book_name=book_name,
                    page_number=page_num,
                    page_text=text
                    if text.strip()
                    else "(No extractable text — rely on the image.)",
                )

                # Step 3: Send to model (with or without image)
                response = analyze_page(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_bytes=image_bytes if send_image else None,
                    send_image=send_image,
                )

                # Step 4: Write to notes file
                _, was_skipped = append_page_notes(
                    base_dir=base_dir,
                    book_name=book_name,
                    page_number=page_num,
                    notes_content=response,
                    course_id=course_id,
                    cert_name=cert_name,
                )

                if was_skipped:
                    skipped += 1
                    progress.update(
                        task, status=f"Page {page_num} (skipped — already exists)"
                    )
                else:
                    processed += 1

            except Exception as e:
                errors += 1
                console.print(f"\n[error]  ✗ Page {page_num} failed: {e}[/error]")

            progress.advance(task)

    # Summary
    console.print()
    summary = Table(box=box.SIMPLE, show_header=False, border_style="dim")
    summary.add_column("Label", style="info")
    summary.add_column("Value", style="bold")
    summary.add_row("✓ Processed", f"[success]{processed}[/success]")
    summary.add_row("⟳ Skipped (duplicates)", f"[warning]{skipped}[/warning]")
    summary.add_row("✗ Errors", f"[error]{errors}[/error]")
    summary.add_row("📄 Notes file", str(notes_path))
    console.print(
        Panel(
            summary,
            title="[title]Extraction Complete[/title]",
            box=box.ROUNDED,
            border_style="green",
        )
    )


def display_token_stats(pdfs: list[Path]) -> None:
    """Show a comparison table of PDF text tokens vs notes tokens."""
    table = Table(
        title="Token Statistics",
        box=box.ROUNDED,
        border_style="cyan",
        title_style="title",
        show_lines=True,
    )
    table.add_column("Book", style="bold white")
    table.add_column("PDF Pages", style="info", justify="center")
    table.add_column("PDF Tokens", style="bold", justify="right")
    table.add_column("Notes Pages", style="info", justify="center")
    table.add_column("Notes Tokens", style="bold", justify="right")
    table.add_column("Ratio", style="highlight", justify="center")

    total_pdf_tokens = 0
    total_notes_tokens = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Counting tokens...", total=len(pdfs))

        for pdf in pdfs:
            name = get_pdf_name(pdf)
            base_dir = pdf.parent

            # PDF tokens
            try:
                pdf_pages = get_page_count(pdf)
                _, pdf_tokens = get_pdf_text_tokens(pdf)
                total_pdf_tokens += pdf_tokens
            except Exception:
                pdf_pages = 0
                pdf_tokens = 0

            # Notes tokens
            notes_result = get_notes_tokens(base_dir, name)
            notes_extracted = get_notes_page_count(base_dir, name)
            if notes_result:
                _, notes_tokens = notes_result
                total_notes_tokens += notes_tokens
                ratio = f"{notes_tokens / pdf_tokens:.1%}" if pdf_tokens > 0 else "—"
            else:
                notes_tokens = 0
                ratio = "—"

            table.add_row(
                name,
                str(pdf_pages),
                f"{pdf_tokens:,}",
                f"{notes_extracted}/{pdf_pages}",
                f"{notes_tokens:,}" if notes_tokens else "[dim]—[/dim]",
                ratio,
            )
            progress.advance(task)

    # Totals row
    table.add_row(
        "[bold cyan]TOTAL[/bold cyan]",
        "",
        f"[bold]{total_pdf_tokens:,}[/bold]",
        "",
        f"[bold]{total_notes_tokens:,}[/bold]"
        if total_notes_tokens
        else "[dim]—[/dim]",
        f"{total_notes_tokens / total_pdf_tokens:.1%}"
        if total_pdf_tokens > 0 and total_notes_tokens > 0
        else "—",
    )

    console.print()
    console.print(table)
    console.print(
        "[info]Tokens counted using cl100k_base encoding (GPT-4 / general approximation)[/info]"
    )
    console.print()


def main():
    """Main interactive loop."""
    display_banner()

    course_id, cert_name = get_course_config()
    model = os.getenv("OPENROUTER_MODEL", "")
    send_image = True  # Toggleable image mode
    token_cache: dict[str, int] = {}  # Cache token counts across loop iterations

    console.print(f"[info]Course:[/info] [bold]{course_id} ({cert_name})[/bold]")
    console.print(f"[info]Model:[/info]  [bold]{model}[/bold]")
    console.print()

    scan_dir = Path.cwd() / "pdf"

    while True:
        # Scan for PDFs
        pdfs = get_pdf_files(scan_dir)

        if not pdfs:
            console.print(f"[error]No PDF files found in {scan_dir}[/error]")
            sys.exit(1)

        display_pdf_list(pdfs, token_cache)

        # Show current mode
        mode_label = (
            "[success]ON[/success] (text + image)"
            if send_image
            else "[warning]OFF[/warning] (text only)"
        )
        console.print(f"[info]Image mode:[/info] {mode_label}")

        # Pick a book
        console.print()
        choice = Prompt.ask(
            "[title]Select a book[/title] (number, [bold]i[/bold] = toggle image, [bold]s[/bold] = stats, [bold]q[/bold] = quit)",
            default="q",
        )

        if choice.lower() == "q":
            console.print("[info]Goodbye! 👋[/info]")
            break

        if choice.lower() == "i":
            send_image = not send_image
            state = (
                "[success]ON[/success] — sending text + image"
                if send_image
                else "[warning]OFF[/warning] — sending text only"
            )
            console.print(f"[title]Image mode:[/title] {state}\n")
            continue

        if choice.lower() == "s":
            display_token_stats(pdfs)
            continue

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(pdfs):
                raise ValueError
        except ValueError:
            console.print("[error]Invalid selection. Try again.[/error]\n")
            continue

        selected_pdf = pdfs[idx]
        book_name = get_pdf_name(selected_pdf)
        total_pages = get_page_count(selected_pdf)

        console.print(
            f"\n[success]Selected:[/success] [bold]{book_name}[/bold] ({total_pages} pages)"
        )

        # Enter page range
        range_str = Prompt.ask(
            f"[title]Page range[/title] (1-{total_pages}, e.g. [bold]1-20[/bold] or [bold]5,10,15-20[/bold])",
        )

        pages = parse_page_range(range_str, total_pages)
        if pages is None:
            console.print(
                f"[error]Invalid range. Must be between 1 and {total_pages}.[/error]\n"
            )
            continue

        console.print(
            f"[info]Will process {len(pages)} page(s): {pages[0]}–{pages[-1]}[/info]"
        )

        # Confirm
        confirm = Prompt.ask(
            "[warning]Start extraction?[/warning]", choices=["y", "n"], default="y"
        )
        if confirm.lower() != "y":
            console.print("[info]Cancelled.[/info]\n")
            continue

        # Process
        process_pages(selected_pdf, pages, course_id, cert_name, send_image=send_image)
        console.print()


if __name__ == "__main__":
    main()
