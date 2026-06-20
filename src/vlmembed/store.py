"""ChromaDB vector store for search-anything."""

from __future__ import annotations

import hashlib
import json
import shutil
import string
from pathlib import Path

import chromadb

from vlmembed.contract import (
    EMBEDDING_PROVIDER,
    STORE_SCHEMA_VERSION,
    StoreMetadata,
    get_db_dir,
    get_images_dir,
)

_QUERY_CACHE_FILE = "query_cache.json"
_STORE_META_FILE = "store_meta.json"


# ---------------------------------------------------------------------------
# Query embedding cache
# ---------------------------------------------------------------------------


def normalize_query(query: str) -> str:
    """Lowercase *query* and strip all punctuation characters.

    Two queries are considered identical if their normalized forms match,
    so repeated searches like ``"cats!"`` and ``"Cats"`` share one cached
    embedding.

    Args:
        query: Raw query string entered by the user.

    Returns:
        Lowercased string with all ``string.punctuation`` characters removed.
    """
    table = str.maketrans("", "", string.punctuation)
    return query.lower().translate(table)


def _query_cache_key(normalized: str, model: str, dimensions: int) -> str:
    """Return a stable SHA-256 hex key for a normalized query + settings pair.

    Using a hash keeps the JSON file tidy regardless of query length.

    Args:
        normalized: Output of :func:`normalize_query`.
        model: Embedding model identifier.
        dimensions: Embedding dimensionality.

    Returns:
        64-character lowercase hex string.
    """
    # Null-byte separators prevent cross-field collisions
    payload = f"{normalized}\x00{model}\x00{dimensions}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _store_meta_path(embed_dir: Path) -> Path:
    """Return the path to the persisted store metadata file."""
    return embed_dir / _STORE_META_FILE


def _expected_store_metadata(model: str, dimensions: int) -> StoreMetadata:
    """Build the expected store metadata for the current runtime settings."""
    return {
        "provider": EMBEDDING_PROVIDER,
        "schema_version": STORE_SCHEMA_VERSION,
        "model": model,
        "dimensions": dimensions,
    }


def load_store_metadata(embed_dir: Path) -> StoreMetadata | None:
    """Load store metadata from disk, returning ``None`` when absent."""
    path = _store_meta_path(embed_dir)
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Corrupt store metadata file: {path}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid store metadata format in: {path}")

    required = {"provider", "schema_version", "model", "dimensions"}
    missing = required.difference(raw)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Store metadata file is missing required keys ({missing_str}): {path}"
        )

    dimensions = raw["dimensions"]
    if not isinstance(dimensions, int):
        raise RuntimeError(f"Store metadata dimensions must be int in: {path}")

    return {
        "provider": str(raw["provider"]),
        "schema_version": str(raw["schema_version"]),
        "model": str(raw["model"]),
        "dimensions": dimensions,
    }


def save_store_metadata(embed_dir: Path, metadata: StoreMetadata) -> None:
    """Persist store metadata under the embeddings directory."""
    embed_dir.mkdir(parents=True, exist_ok=True)
    _store_meta_path(embed_dir).write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def ensure_store_compatibility(
    embed_dir: Path,
    *,
    model: str,
    dimensions: int,
) -> StoreMetadata:
    """Validate store metadata and fail fast on incompatible schema/provider.

    If metadata is missing, this initializes it with the current expected values.
    """
    expected = _expected_store_metadata(model=model, dimensions=dimensions)
    existing = load_store_metadata(embed_dir)

    if existing is None:
        save_store_metadata(embed_dir, expected)
        return expected

    mismatches: list[str] = []
    for key in ("provider", "schema_version", "dimensions"):
        if existing[key] != expected[key]:
            mismatches.append(
                f"{key}: expected {expected[key]!r}, found {existing[key]!r}"
            )

    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            "Store metadata mismatch detected. "
            f"{details}. Delete '{embed_dir}' and re-run 'search embed' to rebuild."
        )

    if existing["model"] != model:
        updated: StoreMetadata = {
            "provider": existing["provider"],
            "schema_version": existing["schema_version"],
            "model": model,
            "dimensions": existing["dimensions"],
        }
        save_store_metadata(embed_dir, updated)
        return updated

    return existing


def load_query_cache(embed_dir: Path) -> dict[str, list[float]]:
    """Load the persisted query-embedding cache from *embed_dir*.

    Args:
        embed_dir: Root embeddings directory.

    Returns:
        Dict mapping cache keys to embedding vectors.  Returns ``{}`` if the
        cache file does not yet exist or cannot be parsed.
    """
    cache_path = embed_dir / _QUERY_CACHE_FILE
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_query_cache(embed_dir: Path, cache: dict[str, list[float]]) -> None:
    """Persist *cache* to *embed_dir*.

    Args:
        embed_dir: Root embeddings directory.
        cache: Dict mapping cache keys to embedding vectors.
    """
    cache_path = embed_dir / _QUERY_CACHE_FILE
    cache_path.write_text(
        json.dumps(cache, separators=(",", ":")),
        encoding="utf-8",
    )


def get_cached_embedding(
    embed_dir: Path,
    query: str,
    model: str,
    dimensions: int,
    *,
    cache: dict[str, list[float]] | None = None,
) -> list[float] | None:
    """Return the cached embedding for *query*, or ``None`` on a cache miss.

    Args:
        embed_dir: Root embeddings directory (used to load the cache when
            *cache* is ``None``).
        query: Raw query string.
        model: Embedding model identifier.
        dimensions: Embedding dimensionality.
        cache: Optional pre-loaded cache dict.  Avoids a redundant file read
            when the caller already holds the cache.

    Returns:
        The cached embedding vector, or ``None`` if not present.
    """
    if cache is None:
        cache = load_query_cache(embed_dir)
    key = _query_cache_key(normalize_query(query), model, dimensions)
    return cache.get(key)


def set_cached_embedding(
    embed_dir: Path,
    query: str,
    model: str,
    dimensions: int,
    embedding: list[float],
    *,
    cache: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """Store *embedding* in the cache and persist it, then return the updated cache.

    Args:
        embed_dir: Root embeddings directory.
        query: Raw query string.
        model: Embedding model identifier.
        dimensions: Embedding dimensionality.
        embedding: Embedding vector to cache.
        cache: Optional pre-loaded cache dict.  Avoids a redundant file read
            when the caller already holds the cache.

    Returns:
        The updated cache dict (mutated in-place when *cache* is provided).
    """
    if cache is None:
        cache = load_query_cache(embed_dir)
    key = _query_cache_key(normalize_query(query), model, dimensions)
    cache[key] = embedding
    save_query_cache(embed_dir, cache)
    return cache


# ---------------------------------------------------------------------------
# ChromaDB collection helpers
# ---------------------------------------------------------------------------

_COLLECTION_NAME = "pdf_pages"


def get_collection(embed_dir: Path):
    """Open or create the persistent ChromaDB collection.

    Args:
        embed_dir: Root embeddings directory.  The database is stored at
            ``embed_dir/db/``.

    Returns:
        A ChromaDB ``Collection`` object.
    """
    db_dir = get_db_dir(embed_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def page_exists(collection, page_id: str) -> bool:
    """Return whether a page embedding already exists in the collection.

    Args:
        collection: ChromaDB ``Collection`` object.
        page_id: Composite page identifier ``"{doc_hash}_{page_idx}"``.

    Returns:
        ``True`` if the page is present, ``False`` otherwise.
    """
    result = collection.get(ids=[page_id], include=[])
    return len(result["ids"]) > 0


def upsert_page(
    collection,
    page_id: str,
    embedding: list[float],
    metadata: dict,
) -> None:
    """Insert or replace a page embedding in the collection.

    Args:
        collection: ChromaDB ``Collection`` object.
        page_id: Composite page identifier ``"{doc_hash}_{page_idx}"``.
        embedding: Embedding vector.
        metadata: Metadata dict conforming to
            :class:`~vlmembed.contract.PageMetadata`.
    """
    collection.upsert(
        ids=[page_id],
        embeddings=[embedding],
        metadatas=[metadata],
    )


def search(
    collection,
    query_embedding: list[float],
    n_results: int = 5,
) -> list[dict]:
    """Query the collection and return the closest page embeddings.

    Args:
        collection: ChromaDB ``Collection`` object.
        query_embedding: Query vector (same dimensionality as stored embeddings).
        n_results: Maximum number of results to return.

    Returns:
        List of dicts conforming to :class:`~vlmembed.contract.SearchResult`.
    """
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return [
        {"page_id": page_id, "metadata": metadata, "distance": distance}
        for page_id, metadata, distance in zip(
            result["ids"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def reset_store(embed_dir: Path, *, remove_images: bool = True) -> list[Path]:
    """Remove persistent store artifacts and return the removed paths."""
    targets: list[Path] = [
        get_db_dir(embed_dir),
        embed_dir / _QUERY_CACHE_FILE,
        _store_meta_path(embed_dir),
    ]
    if remove_images:
        targets.append(get_images_dir(embed_dir))

    removed: list[Path] = []
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(target)
        elif target.is_file():
            target.unlink()
            removed.append(target)

    return removed
