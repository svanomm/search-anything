"""Cost estimation for vlmembed — counts PDF pages and estimates embedding cost."""

from __future__ import annotations

from pathlib import Path

import fitz

from vlmembed.contract import DEFAULT_DOCS_DIR, DEFAULT_DPI

# ---------------------------------------------------------------------------
# Pricing constants
# ---------------------------------------------------------------------------

# Approximate cost per million image tokens (OpenRouter / Gemini Embedding 2).
# This is an estimate; actual pricing may differ.
_PRICE_PER_M_TOKENS: float = 0.2

# Tokens per page — based on testing with default model.
TOKENS_PER_PAGE: int = 258

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def count_pdf_pages(docs_dir: Path = DEFAULT_DOCS_DIR) -> dict[str, int]:
    """Return a mapping of ``{filename: page_count}`` for all PDFs in *docs_dir*.

    Args:
        docs_dir: Directory to scan for ``*.pdf`` files.

    Returns:
        Dict mapping filename (not full path) to page count.  Empty when no
        PDFs are found or the directory does not exist.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return {}
    result: dict[str, int] = {}
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        with fitz.open(str(pdf_path)) as doc:
            result[pdf_path.name] = doc.page_count
    return result


def estimate_tokens_per_page(dpi: int = DEFAULT_DPI) -> int:
    """Estimate the number of image tokens for one page rendered at *dpi*.

    Based on empirical testing, a typical page contains approximately
    258 image tokens when rendered at standard DPI.

    Args:
        dpi: Render resolution in dots per inch (currently unused).

    Returns:
        Approximate token count per page.
    """
    return TOKENS_PER_PAGE


def estimate_cost_from_page_counts(
    per_file: dict[str, int],
    *,
    dpi: int = DEFAULT_DPI,
) -> dict:
    """Estimate embedding cost from a ``{filename: page_count}`` mapping.

    Args:
        per_file: Mapping of filename to page count.
        dpi: Render resolution used when calculating token estimates.

    Returns:
        Dict with the same shape as :func:`estimate_cost`.
    """
    total_pages = sum(per_file.values())
    tokens_per_page = estimate_tokens_per_page(dpi)
    total_tokens = total_pages * tokens_per_page
    estimated_usd = total_tokens * _PRICE_PER_M_TOKENS / 1_000_000
    return {
        "per_file": per_file,
        "pages": total_pages,
        "tokens_per_page": tokens_per_page,
        "total_tokens": total_tokens,
        "estimated_usd": estimated_usd,
    }


def estimate_cost(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    dpi: int = DEFAULT_DPI,
) -> dict:
    """Estimate the total embedding cost for all PDFs in *docs_dir*.

    Args:
        docs_dir: Directory containing source PDFs.
        dpi: Render resolution used when calculating pixel dimensions.

    Returns:
        Dict with keys:

        * ``per_file`` — ``{filename: page_count}`` for each PDF found.
        * ``pages`` — total page count across all PDFs.
        * ``tokens_per_page`` — estimated tokens per page at *dpi*.
        * ``total_tokens`` — ``pages × tokens_per_page``.
        * ``estimated_usd`` — cost estimate in US dollars.
    """
    per_file = count_pdf_pages(docs_dir)
    return estimate_cost_from_page_counts(per_file, dpi=dpi)
