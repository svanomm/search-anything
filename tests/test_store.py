"""Tests for vlmembed.store (Phase 3)."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from vlmembed.store import get_collection, page_exists, search, upsert_page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 4  # small dimension keeps tests fast


def _make_collection():
    """Return an in-memory ChromaDB collection for isolated testing.

    A UUID suffix is used so each call gets a fresh, isolated collection even
    if the EphemeralClient shares in-process state between calls.
    """
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        name=f"pdf_pages_{uuid.uuid4().hex}",
        metadata={"hnsw:space": "cosine"},
    )


def _sample_metadata(page_number: int = 0) -> dict:
    return {
        "doc_path": "docs/sample.pdf",
        "page_number": page_number,
        "doc_hash": "abc123",
        "settings_hash": "def456",
        "image_cache_path": f"embeddings/images/abc123/page_{page_number}.png",
    }


def _unit_vec(i: int, dim: int = _DIM) -> list[float]:
    """Return a simple deterministic unit-ish vector."""
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


# ---------------------------------------------------------------------------
# get_collection
# ---------------------------------------------------------------------------


class TestGetCollection:
    def test_returns_collection_with_correct_name(self, tmp_path):
        col = get_collection(tmp_path)
        assert col.name == "pdf_pages"

    def test_creates_db_dir(self, tmp_path):
        get_collection(tmp_path)
        assert (tmp_path / "db").is_dir()

    def test_idempotent_when_called_twice(self, tmp_path):
        col1 = get_collection(tmp_path)
        col2 = get_collection(tmp_path)
        assert col1.name == col2.name

    def test_collection_is_empty_on_creation(self, tmp_path):
        col = get_collection(tmp_path)
        assert col.count() == 0


# ---------------------------------------------------------------------------
# page_exists
# ---------------------------------------------------------------------------


class TestPageExists:
    def test_returns_false_for_missing_page(self):
        col = _make_collection()
        assert page_exists(col, "nonexistent_id") is False

    def test_returns_true_after_upsert(self):
        col = _make_collection()
        col.upsert(
            ids=["doc1_0"],
            embeddings=[_unit_vec(0)],
            metadatas=[_sample_metadata(0)],
        )
        assert page_exists(col, "doc1_0") is True

    def test_does_not_affect_other_pages(self):
        col = _make_collection()
        col.upsert(
            ids=["doc1_0"],
            embeddings=[_unit_vec(0)],
            metadatas=[_sample_metadata(0)],
        )
        assert page_exists(col, "doc1_1") is False

    def test_uses_cheap_include_empty(self):
        """page_exists should call collection.get with include=[]."""
        col = MagicMock()
        col.get.return_value = {"ids": []}
        page_exists(col, "any_id")
        col.get.assert_called_once_with(ids=["any_id"], include=[])


# ---------------------------------------------------------------------------
# upsert_page
# ---------------------------------------------------------------------------


class TestUpsertPage:
    def test_page_visible_after_upsert(self):
        col = _make_collection()
        upsert_page(col, "doc1_0", _unit_vec(0), _sample_metadata(0))
        assert col.count() == 1

    def test_metadata_round_trips(self):
        col = _make_collection()
        meta = _sample_metadata(3)
        upsert_page(col, "doc1_3", _unit_vec(1), meta)
        result = col.get(ids=["doc1_3"], include=["metadatas"])
        assert result["metadatas"][0] == meta

    def test_upsert_overwrites_existing(self):
        col = _make_collection()
        upsert_page(col, "doc1_0", _unit_vec(0), _sample_metadata(0))
        new_meta = _sample_metadata(0)
        new_meta["settings_hash"] = "new_hash"
        upsert_page(col, "doc1_0", _unit_vec(1), new_meta)
        assert col.count() == 1
        result = col.get(ids=["doc1_0"], include=["metadatas"])
        assert result["metadatas"][0]["settings_hash"] == "new_hash"

    def test_multiple_pages_stored_independently(self):
        col = _make_collection()
        for i in range(3):
            upsert_page(col, f"doc1_{i}", _unit_vec(i), _sample_metadata(i))
        assert col.count() == 3


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def _populated_collection(self, n: int = 3):
        col = _make_collection()
        for i in range(n):
            col.upsert(
                ids=[f"doc1_{i}"],
                embeddings=[_unit_vec(i)],
                metadatas=[_sample_metadata(i)],
            )
        return col

    def test_returns_list_of_dicts(self):
        col = self._populated_collection()
        results = search(col, _unit_vec(0), n_results=1)
        assert isinstance(results, list)
        assert isinstance(results[0], dict)

    def test_result_has_required_keys(self):
        col = self._populated_collection()
        result = search(col, _unit_vec(0), n_results=1)[0]
        assert "page_id" in result
        assert "metadata" in result
        assert "distance" in result

    def test_closest_page_is_first(self):
        col = self._populated_collection(n=_DIM)
        results = search(col, _unit_vec(2), n_results=_DIM)
        assert results[0]["page_id"] == "doc1_2"

    def test_n_results_limits_output(self):
        col = self._populated_collection(n=_DIM)
        results = search(col, _unit_vec(0), n_results=2)
        assert len(results) == 2

    def test_distance_is_float(self):
        col = self._populated_collection()
        result = search(col, _unit_vec(0), n_results=1)[0]
        assert isinstance(result["distance"], float)

    def test_metadata_matches_upserted_data(self):
        col = _make_collection()
        meta = _sample_metadata(7)
        col.upsert(ids=["doc1_7"], embeddings=[_unit_vec(0)], metadatas=[meta])
        results = search(col, _unit_vec(0), n_results=1)
        assert results[0]["metadata"] == meta
        assert results[0]["page_id"] == "doc1_7"
