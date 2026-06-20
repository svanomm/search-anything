"""Tests for vlmembed.contract."""

from pathlib import Path

import pytest

from vlmembed.contract import (
    DEFAULT_DIMENSIONS,
    DEFAULT_DPI,
    DEFAULT_DOCS_DIR,
    DEFAULT_EMBED_DIR,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MODEL,
    DEFAULT_USE_ENTERPRISE,
    EMBEDDING_PROVIDER,
    ENV_API_KEY,
    ENV_DIMENSIONS,
    ENV_DPI,
    ENV_IMAGE_FORMAT,
    ENV_MAX_RETRIES,
    ENV_MAX_WORKERS,
    ENV_MODEL,
    ENV_USE_ENTERPRISE,
    STORE_SCHEMA_VERSION,
    EmbedResult,
    PageMetadata,
    ProjectPathStatus,
    SearchResult,
    StoreMetadata,
    get_db_dir,
    get_doc_images_dir,
    get_images_dir,
    get_project_directories,
)


class TestDefaults:
    def test_docs_dir(self):
        assert DEFAULT_DOCS_DIR == Path("docs")

    def test_embed_dir(self):
        assert DEFAULT_EMBED_DIR == Path("embeddings")

    def test_model(self):
        assert DEFAULT_MODEL == "gemini-embedding-2"

    def test_dpi(self):
        assert DEFAULT_DPI == 300

    def test_image_format(self):
        assert DEFAULT_IMAGE_FORMAT == "png"

    def test_dimensions(self):
        assert DEFAULT_DIMENSIONS == 3072

    def test_max_workers(self):
        assert DEFAULT_MAX_WORKERS == 4

    def test_max_retries(self):
        assert DEFAULT_MAX_RETRIES == 3

    def test_use_enterprise_default(self):
        assert DEFAULT_USE_ENTERPRISE is True

    def test_embedding_provider(self):
        assert EMBEDDING_PROVIDER == "google-genai"

    def test_schema_version(self):
        assert STORE_SCHEMA_VERSION == "2"

class TestEnvVarNames:
    def test_api_key(self):
        assert ENV_API_KEY == "GOOGLE_API_KEY"

    def test_use_enterprise(self):
        assert ENV_USE_ENTERPRISE == "GOOGLE_GENAI_USE_ENTERPRISE"

    def test_model(self):
        assert ENV_MODEL == "VLMEMBED_MODEL"

    def test_dpi(self):
        assert ENV_DPI == "VLMEMBED_DPI"

    def test_image_format(self):
        assert ENV_IMAGE_FORMAT == "VLMEMBED_IMAGE_FORMAT"

    def test_dimensions(self):
        assert ENV_DIMENSIONS == "VLMEMBED_DIMENSIONS"

    def test_max_workers(self):
        assert ENV_MAX_WORKERS == "VLMEMBED_MAX_WORKERS"

    def test_max_retries(self):
        assert ENV_MAX_RETRIES == "VLMEMBED_MAX_RETRIES"


class TestPathHelpers:
    def setup_method(self):
        self.embed_dir = Path("embeddings")

    def test_images_dir(self):
        assert get_images_dir(self.embed_dir) == Path("embeddings/images")

    def test_doc_images_dir(self):
        assert get_doc_images_dir(self.embed_dir, "abc123") == Path(
            "embeddings/images/abc123"
        )

    def test_db_dir(self):
        assert get_db_dir(self.embed_dir) == Path("embeddings/db")

    def test_project_directories_keys(self):
        dirs = get_project_directories()
        assert "docs" in dirs
        assert "embeddings root" in dirs
        assert "page images" in dirs
        assert "vector database" in dirs

    def test_project_directories_custom(self):
        dirs = get_project_directories(
            docs_dir=Path("my_docs"), embed_dir=Path("my_embed")
        )
        assert dirs["docs"] == Path("my_docs")
        assert dirs["embeddings root"] == Path("my_embed")
        assert dirs["page images"] == Path("my_embed/images")
        assert dirs["vector database"] == Path("my_embed/db")


class TestProjectPathStatus:
    def test_fields(self):
        status = ProjectPathStatus(label="docs", path=Path("docs"), exists=True)
        assert status.label == "docs"
        assert status.path == Path("docs")
        assert status.exists is True

    def test_frozen(self):
        status = ProjectPathStatus(label="docs", path=Path("docs"), exists=False)
        with pytest.raises((AttributeError, TypeError)):
            status.label = "other"  # type: ignore[misc]


class TestTypedDicts:
    def test_page_metadata_structure(self):
        meta: PageMetadata = {
            "doc_path": "docs/test.pdf",
            "page_number": 2,
            "doc_hash": "abc123",
            "settings_hash": "def456",
            "image_cache_path": "embeddings/images/abc123/page_2.png",
        }
        assert meta["page_number"] == 2
        assert meta["doc_hash"] == "abc123"

    def test_embed_result_structure(self):
        result: EmbedResult = {
            "page_id": "abc123_2",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "doc_path": "docs/test.pdf",
                "page_number": 2,
                "doc_hash": "abc123",
                "settings_hash": "def456",
                "image_cache_path": "embeddings/images/abc123/page_2.png",
            },
        }
        assert result["page_id"] == "abc123_2"
        assert len(result["embedding"]) == 3

    def test_search_result_structure(self):
        result: SearchResult = {
            "page_id": "abc123_2",
            "metadata": {
                "doc_path": "docs/test.pdf",
                "page_number": 2,
                "doc_hash": "abc123",
                "settings_hash": "def456",
                "image_cache_path": "embeddings/images/abc123/page_2.png",
            },
            "distance": 0.05,
        }
        assert result["distance"] == 0.05

    def test_store_metadata_structure(self):
        metadata: StoreMetadata = {
            "provider": "google-genai",
            "schema_version": "2",
            "model": "gemini-embedding-2",
            "dimensions": 3072,
        }
        assert metadata["provider"] == "google-genai"
        assert metadata["dimensions"] == 3072
