"""Shared embedding output contract and path helpers for vlmembed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

# ---------------------------------------------------------------------------
# Default directories
# ---------------------------------------------------------------------------

DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_EMBED_DIR = Path("embeddings")

# Subdirectories under embed_dir
IMAGES_SUBDIR = Path("images")
DB_SUBDIR = Path("db")

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

ENV_API_KEY = "OPENROUTER_API_KEY"
ENV_MODEL = "VLMEMBED_MODEL"
ENV_DPI = "VLMEMBED_DPI"
ENV_IMAGE_FORMAT = "VLMEMBED_IMAGE_FORMAT"
ENV_DIMENSIONS = "VLMEMBED_DIMENSIONS"
ENV_MAX_WORKERS = "VLMEMBED_MAX_WORKERS"
ENV_MAX_RETRIES = "VLMEMBED_MAX_RETRIES"

# Hard-coded defaults
DEFAULT_MODEL = "google/gemini-embedding-2-preview"
DEFAULT_DPI = 200
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_DIMENSIONS = 3072
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3

# OpenRouter endpoint
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_images_dir(embed_dir: Path) -> Path:
    """Return the page-image cache root under *embed_dir*."""
    return embed_dir / IMAGES_SUBDIR


def get_doc_images_dir(embed_dir: Path, doc_hash: str) -> Path:
    """Return the per-document image cache directory under *embed_dir*."""
    return get_images_dir(embed_dir) / doc_hash


def get_db_dir(embed_dir: Path) -> Path:
    """Return the ChromaDB persistence directory under *embed_dir*."""
    return embed_dir / DB_SUBDIR


def get_project_directories(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    embed_dir: Path = DEFAULT_EMBED_DIR,
) -> dict[str, Path]:
    """Return the directories that make up the standard vlmembed workspace."""
    return {
        "docs": docs_dir,
        "embeddings root": embed_dir,
        "page images": get_images_dir(embed_dir),
        "vector database": get_db_dir(embed_dir),
    }


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------


class PageMetadata(TypedDict):
    """Metadata stored alongside each page embedding in ChromaDB."""

    doc_path: str
    page_number: int
    doc_hash: str
    settings_hash: str
    image_cache_path: str


class EmbedResult(TypedDict):
    """Result returned for a single embedded page."""

    page_id: str
    embedding: list[float]
    metadata: PageMetadata


class SearchResult(TypedDict):
    """One result entry returned from a semantic search query."""

    page_id: str
    metadata: PageMetadata
    distance: float


# ---------------------------------------------------------------------------
# Frozen dataclass for project-path validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectPathStatus:
    """Validation status for one expected project directory."""

    label: str
    path: Path
    exists: bool
