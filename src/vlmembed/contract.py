"""Shared embedding output contract and path helpers for search-anything."""

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

ENV_API_KEY = "GOOGLE_API_KEY"
ENV_USE_ENTERPRISE = "GOOGLE_GENAI_USE_ENTERPRISE"
ENV_MODEL = "SEARCH_MODEL"
ENV_DPI = "SEARCH_DPI"
ENV_IMAGE_FORMAT = "SEARCH_IMAGE_FORMAT"
ENV_DIMENSIONS = "SEARCH_DIMENSIONS"
ENV_MAX_WORKERS = "SEARCH_MAX_WORKERS"
ENV_MAX_RETRIES = "SEARCH_MAX_RETRIES"

# Hard-coded defaults
DEFAULT_MODEL = "gemini-embedding-2"
DEFAULT_DPI = 300
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_DIMENSIONS = 3072
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_USE_ENTERPRISE = True

# Provider and schema constants
EMBEDDING_PROVIDER = "google-genai"
STORE_SCHEMA_VERSION = "2"

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
    """Return the directories that make up the standard search-anything workspace."""
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


class StoreMetadata(TypedDict):
    """Metadata persisted for validating vector-store compatibility."""

    provider: str
    schema_version: str
    model: str
    dimensions: int


# ---------------------------------------------------------------------------
# Frozen dataclass for project-path validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectPathStatus:
    """Validation status for one expected project directory."""

    label: str
    path: Path
    exists: bool
