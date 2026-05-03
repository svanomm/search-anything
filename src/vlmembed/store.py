"""ChromaDB vector store for vlmembed — stub implementation.

The public functions defined here are called by :mod:`vlmembed.embed`.
This module will be fully implemented in Phase 3; for now each function
raises :class:`NotImplementedError` so that the interface is stable and
tests can patch individual functions without importing chromadb.
"""

from __future__ import annotations

from pathlib import Path


def get_collection(embed_dir: Path):
    """Open or create the persistent ChromaDB collection.

    Args:
        embed_dir: Root embeddings directory.  The database is stored at
            ``embed_dir/db/``.

    Returns:
        A ChromaDB ``Collection`` object.

    Raises:
        NotImplementedError: Until Phase 3 is implemented.
    """
    raise NotImplementedError(
        "store.get_collection() will be implemented in Phase 3"
    )


def page_exists(collection, page_id: str) -> bool:
    """Return whether a page embedding already exists in the collection.

    Args:
        collection: ChromaDB ``Collection`` object.
        page_id: Composite page identifier ``"{doc_hash}_{page_idx}"``.

    Returns:
        ``True`` if the page is present, ``False`` otherwise.

    Raises:
        NotImplementedError: Until Phase 3 is implemented.
    """
    raise NotImplementedError(
        "store.page_exists() will be implemented in Phase 3"
    )


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

    Raises:
        NotImplementedError: Until Phase 3 is implemented.
    """
    raise NotImplementedError(
        "store.upsert_page() will be implemented in Phase 3"
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

    Raises:
        NotImplementedError: Until Phase 3 is implemented.
    """
    raise NotImplementedError(
        "store.search() will be implemented in Phase 3"
    )
