"""Tests for search_anything.store (Phase 3)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import chromadb
import pytest

from search_anything.store import (
    ensure_store_compatibility,
    get_collection,
    get_cached_embedding,
    load_store_metadata,
    load_query_cache,
    normalize_query,
    page_exists,
    reset_store,
    save_query_cache,
    search,
    set_cached_embedding,
    upsert_page,
)


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


def _sample_metadata(page_number: int = 1) -> dict:
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


class TestStoreMetadataCompatibility:
    def test_initializes_metadata_when_missing(self, tmp_path):
        metadata = ensure_store_compatibility(
            tmp_path,
            model="gemini-embedding-2",
            dimensions=3072,
        )
        assert metadata["provider"] == "google-genai"
        assert metadata["schema_version"] == "2"

        loaded = load_store_metadata(tmp_path)
        assert loaded == metadata

    def test_raises_on_provider_mismatch(self, tmp_path):
        (tmp_path / "store_meta.json").write_text(
            json.dumps(
                {
                    "provider": "other-provider",
                    "schema_version": "2",
                    "model": "gemini-embedding-2",
                    "dimensions": 3072,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="Store metadata mismatch"):
            ensure_store_compatibility(
                tmp_path,
                model="gemini-embedding-2",
                dimensions=3072,
            )


class TestResetStore:
    def test_removes_all_store_artifacts(self, tmp_path):
        (tmp_path / "db").mkdir(parents=True)
        (tmp_path / "db" / "chroma.sqlite3").write_text("db", encoding="utf-8")
        (tmp_path / "query_cache.json").write_text("{}", encoding="utf-8")
        (tmp_path / "store_meta.json").write_text(
            '{"provider":"google-genai","schema_version":"2","model":"m","dimensions":1}',
            encoding="utf-8",
        )
        (tmp_path / "images" / "x").mkdir(parents=True)
        (tmp_path / "images" / "x" / "page_1.png").write_bytes(b"img")

        removed = reset_store(tmp_path, remove_images=True)

        assert (tmp_path / "db").exists() is False
        assert (tmp_path / "query_cache.json").exists() is False
        assert (tmp_path / "store_meta.json").exists() is False
        assert (tmp_path / "images").exists() is False
        assert len(removed) == 4

    def test_keeps_images_when_requested(self, tmp_path):
        (tmp_path / "db").mkdir(parents=True)
        (tmp_path / "query_cache.json").write_text("{}", encoding="utf-8")
        (tmp_path / "store_meta.json").write_text(
            '{"provider":"google-genai","schema_version":"2","model":"m","dimensions":1}',
            encoding="utf-8",
        )
        (tmp_path / "images" / "x").mkdir(parents=True)

        reset_store(tmp_path, remove_images=False)

        assert (tmp_path / "db").exists() is False
        assert (tmp_path / "query_cache.json").exists() is False
        assert (tmp_path / "store_meta.json").exists() is False
        assert (tmp_path / "images").exists() is True

    def test_updates_model_without_failing(self, tmp_path):
        (tmp_path / "store_meta.json").write_text(
            json.dumps(
                {
                    "provider": "google-genai",
                    "schema_version": "2",
                    "model": "old-model",
                    "dimensions": 3072,
                }
            ),
            encoding="utf-8",
        )

        ensure_store_compatibility(
            tmp_path,
            model="gemini-embedding-2",
            dimensions=3072,
        )
        updated = load_store_metadata(tmp_path)
        assert updated is not None
        assert updated["model"] == "gemini-embedding-2"

    def test_raises_on_corrupt_metadata_file(self, tmp_path):
        (tmp_path / "store_meta.json").write_text("not-json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Corrupt store metadata"):
            ensure_store_compatibility(
                tmp_path,
                model="gemini-embedding-2",
                dimensions=3072,
            )


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


# ---------------------------------------------------------------------------
# normalize_query
# ---------------------------------------------------------------------------


class TestNormalizeQuery:
    def test_lowercases(self):
        assert normalize_query("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert normalize_query("cats!") == "cats"

    def test_removes_all_punctuation_chars(self):
        assert normalize_query("What's this? A test.") == "whats this a test"

    def test_empty_string(self):
        assert normalize_query("") == ""

    def test_already_normalized(self):
        assert normalize_query("hello world") == "hello world"

    def test_equivalent_after_normalization(self):
        assert normalize_query("Cats!") == normalize_query("cats")


# ---------------------------------------------------------------------------
# load_query_cache / save_query_cache
# ---------------------------------------------------------------------------


class TestQueryCachePersistence:
    def test_load_returns_empty_dict_when_file_missing(self, tmp_path):
        assert load_query_cache(tmp_path) == {}

    def test_save_then_load_round_trips(self, tmp_path):
        cache = {"key1": [0.1, 0.2, 0.3]}
        save_query_cache(tmp_path, cache)
        loaded = load_query_cache(tmp_path)
        assert loaded == cache

    def test_load_returns_empty_dict_on_corrupt_file(self, tmp_path):
        (tmp_path / "query_cache.json").write_text("NOT JSON", encoding="utf-8")
        assert load_query_cache(tmp_path) == {}

    def test_save_overwrites_previous_cache(self, tmp_path):
        save_query_cache(tmp_path, {"a": [1.0]})
        save_query_cache(tmp_path, {"b": [2.0]})
        loaded = load_query_cache(tmp_path)
        assert "b" in loaded
        assert "a" not in loaded


# ---------------------------------------------------------------------------
# get_cached_embedding / set_cached_embedding
# ---------------------------------------------------------------------------


class TestGetSetCachedEmbedding:
    _MODEL = "gemini-embedding-2"
    _DIM = 8
    _EMB = [0.1] * 8

    def test_miss_returns_none(self, tmp_path):
        result = get_cached_embedding(tmp_path, "hello", self._MODEL, self._DIM)
        assert result is None

    def test_hit_returns_embedding(self, tmp_path):
        set_cached_embedding(tmp_path, "hello", self._MODEL, self._DIM, self._EMB)
        result = get_cached_embedding(tmp_path, "hello", self._MODEL, self._DIM)
        assert result == self._EMB

    def test_case_insensitive(self, tmp_path):
        set_cached_embedding(tmp_path, "Hello", self._MODEL, self._DIM, self._EMB)
        result = get_cached_embedding(tmp_path, "hello", self._MODEL, self._DIM)
        assert result == self._EMB

    def test_punctuation_invariant(self, tmp_path):
        set_cached_embedding(tmp_path, "cats!", self._MODEL, self._DIM, self._EMB)
        result = get_cached_embedding(tmp_path, "cats", self._MODEL, self._DIM)
        assert result == self._EMB

    def test_different_model_is_cache_miss(self, tmp_path):
        set_cached_embedding(tmp_path, "hello", self._MODEL, self._DIM, self._EMB)
        result = get_cached_embedding(tmp_path, "hello", "other/model", self._DIM)
        assert result is None

    def test_different_dimensions_is_cache_miss(self, tmp_path):
        set_cached_embedding(tmp_path, "hello", self._MODEL, self._DIM, self._EMB)
        result = get_cached_embedding(tmp_path, "hello", self._MODEL, 16)
        assert result is None

    def test_set_returns_updated_cache(self, tmp_path):
        cache = set_cached_embedding(tmp_path, "hello", self._MODEL, self._DIM, self._EMB)
        assert isinstance(cache, dict)
        assert len(cache) == 1

    def test_accepts_pre_loaded_cache(self, tmp_path):
        preloaded: dict = {}
        set_cached_embedding(
            tmp_path, "hello", self._MODEL, self._DIM, self._EMB, cache=preloaded
        )
        assert len(preloaded) == 1
        result = get_cached_embedding(
            tmp_path, "hello", self._MODEL, self._DIM, cache=preloaded
        )
        assert result == self._EMB
