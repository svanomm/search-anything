"""ChromaDB vector store for vlmembed."""

from __future__ import annotations

from pathlib import Path

import chromadb

from vlmembed.contract import get_db_dir

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
