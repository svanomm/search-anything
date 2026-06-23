"""Cost estimation utilities for recursive multimodal ingestion."""

from __future__ import annotations

import math
from pathlib import Path

import fitz

from search_anything.contract import DEFAULT_DOCS_DIR, DEFAULT_DPI

# ---------------------------------------------------------------------------
# Pricing constants
# ---------------------------------------------------------------------------

# Approximate cost per million token-equivalents.
# This is an estimate; actual pricing may differ.
_PRICE_PER_M_TOKENS: float = 0.2

# Tokens per page — based on testing with default model.
TOKENS_PER_PAGE: int = 258
TOKENS_PER_IMAGE: int = 258
TOKENS_PER_AUDIO_SECOND: int = 25
TOKENS_PER_VIDEO_FRAME: int = 66

_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".heif", ".avif"}
_AUDIO_SUFFIXES = {".mp3", ".wav"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

_TEXT_CHARS_PER_TOKEN = 4
_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 200

_ASSUMED_AUDIO_BYTES_PER_SECOND = 20_000
_AUDIO_SEGMENT_COUNT = 3
_AUDIO_MAX_SECONDS_PER_SEGMENT = 180

_VIDEO_SEGMENT_COUNT = 3
_VIDEO_FRAMES_PER_SEGMENT = 40

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def count_pdf_pages(docs_dir: Path = DEFAULT_DOCS_DIR) -> dict[str, int]:
    """Return a mapping of ``{filename: page_count}`` for all PDFs in *docs_dir*.

    Args:
        docs_dir: Directory to scan recursively for ``*.pdf`` files.

    Returns:
        Dict mapping filename (not full path) to page count.  Empty when no
        PDFs are found or the directory does not exist.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return {}
    result: dict[str, int] = {}
    for pdf_path in sorted(docs_dir.rglob("*.pdf")):
        with fitz.open(str(pdf_path)) as doc:
            result[pdf_path.name] = doc.page_count
    return result


def _estimate_text_tokens_from_content(text: str) -> int:
    """Estimate chunked text token count using character-based assumptions."""
    clean = text.strip()
    if not clean:
        return 0

    chunks = max(1, math.ceil(len(clean) / _TEXT_CHUNK_SIZE))
    overlap_chars = max(chunks - 1, 0) * _TEXT_CHUNK_OVERLAP
    effective_chars = len(clean) + overlap_chars
    return max(1, math.ceil(effective_chars / _TEXT_CHARS_PER_TOKEN))


def count_multimodal_inputs(docs_dir: Path = DEFAULT_DOCS_DIR) -> dict:
    """Count multimodal ingestion units under *docs_dir* recursively.

    Returned units mirror the ingestion strategy:
    - PDF pages (image-equivalent)
    - Image files
    - Text token estimate (chunking-aware approximation)
    - Audio seconds (size-based approximation, capped to 3×180s)
    - Video frames (3 segments × 40 frames at default 1 fps)
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return {
            "pdf_pages": 0,
            "images": 0,
            "text_tokens": 0,
            "audio_seconds": 0.0,
            "video_frames": 0,
        }

    pdf_pages = sum(count_pdf_pages(docs_dir).values())

    image_count = 0
    text_tokens = 0
    audio_seconds = 0.0
    video_files = 0

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            image_count += 1
            continue

        if suffix in _TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")
            text_tokens += _estimate_text_tokens_from_content(text)
            continue

        if suffix in _AUDIO_SUFFIXES:
            approx_seconds = path.stat().st_size / _ASSUMED_AUDIO_BYTES_PER_SECOND
            max_embedded = _AUDIO_SEGMENT_COUNT * _AUDIO_MAX_SECONDS_PER_SEGMENT
            audio_seconds += min(approx_seconds, max_embedded)
            continue

        if suffix in _VIDEO_SUFFIXES:
            video_files += 1

    video_frames = video_files * _VIDEO_SEGMENT_COUNT * _VIDEO_FRAMES_PER_SEGMENT

    return {
        "pdf_pages": pdf_pages,
        "images": image_count,
        "text_tokens": int(text_tokens),
        "audio_seconds": round(audio_seconds, 2),
        "video_frames": int(video_frames),
    }


def estimate_multimodal_cost_from_counts(counts: dict) -> dict:
    """Estimate multimodal embedding cost from modality unit counts."""
    token_breakdown = {
        "pdf": int(counts.get("pdf_pages", 0) * TOKENS_PER_PAGE),
        "image": int(counts.get("images", 0) * TOKENS_PER_IMAGE),
        "text": int(counts.get("text_tokens", 0)),
        "audio": int(round(counts.get("audio_seconds", 0.0) * TOKENS_PER_AUDIO_SECOND)),
        "video": int(counts.get("video_frames", 0) * TOKENS_PER_VIDEO_FRAME),
    }

    per_modality_usd = {
        key: value * _PRICE_PER_M_TOKENS / 1_000_000
        for key, value in token_breakdown.items()
    }

    total_tokens = sum(token_breakdown.values())
    estimated_usd = sum(per_modality_usd.values())
    return {
        "token_breakdown": token_breakdown,
        "per_modality_usd": per_modality_usd,
        "total_tokens": total_tokens,
        "estimated_usd": estimated_usd,
    }


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
    """Estimate total embedding cost for all supported files in *docs_dir*.

    Args:
        docs_dir: Directory containing source files.
        dpi: Render resolution used for PDF token estimates.

    Returns:
        Dict with keys:

        * ``per_file`` — ``{filename: page_count}`` for PDFs found.
        * ``pages`` — total PDF page count.
        * ``tokens_per_page`` — estimated tokens per page at *dpi*.
        * ``total_tokens`` — token-equivalent total across all modalities.
        * ``estimated_usd`` — total cost estimate in US dollars.
        * ``modalities`` — counted units by modality.
        * ``token_breakdown`` — token-equivalent totals by modality.
        * ``per_modality_usd`` — cost estimate per modality.
    """
    per_file = count_pdf_pages(docs_dir)
    modalities = count_multimodal_inputs(docs_dir)
    multimodal = estimate_multimodal_cost_from_counts(modalities)

    return {
        "per_file": per_file,
        "pages": sum(per_file.values()),
        "tokens_per_page": estimate_tokens_per_page(dpi),
        "total_tokens": multimodal["total_tokens"],
        "estimated_usd": multimodal["estimated_usd"],
        "modalities": modalities,
        "token_breakdown": multimodal["token_breakdown"],
        "per_modality_usd": multimodal["per_modality_usd"],
    }
