"""Unit tests for vlmembed.embed."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import fitz
import pytest
import requests

from vlmembed.embed import (
    compute_doc_hash,
    compute_settings_hash,
    embed_all_pdfs,
    embed_image_page,
    embed_text_query,
    render_page_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(path: Path, num_pages: int = 1) -> Path:
    """Create a minimal valid PDF with *num_pages* blank pages at *path*."""
    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page()
    doc.save(str(path))
    doc.close()
    return path


def _fake_embedding(dimensions: int = 3) -> list[float]:
    """Return a deterministic fake embedding vector."""
    return [0.1 * i for i in range(dimensions)]


def _mock_embed_response(embedding: list[float]) -> MagicMock:
    """Build a mock requests.Response that looks like the OpenRouter API."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"embedding": embedding}]}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# compute_doc_hash
# ---------------------------------------------------------------------------


class TestComputeDocHash:
    def test_returns_64_char_hex_string(self, tmp_path):
        pdf = _make_pdf(tmp_path / "a.pdf")
        result = compute_doc_hash(pdf)
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_is_deterministic(self, tmp_path):
        pdf = _make_pdf(tmp_path / "b.pdf")
        assert compute_doc_hash(pdf) == compute_doc_hash(pdf)

    def test_matches_sha256_of_file_bytes(self, tmp_path):
        pdf = _make_pdf(tmp_path / "c.pdf")
        expected = hashlib.sha256(pdf.read_bytes()).hexdigest()
        assert compute_doc_hash(pdf) == expected

    def test_accepts_str_path(self, tmp_path):
        pdf = _make_pdf(tmp_path / "d.pdf")
        assert compute_doc_hash(str(pdf)) == compute_doc_hash(pdf)

    def test_different_content_different_hash(self, tmp_path):
        pdf_a = _make_pdf(tmp_path / "e1.pdf", num_pages=1)
        pdf_b = _make_pdf(tmp_path / "e2.pdf", num_pages=2)
        assert compute_doc_hash(pdf_a) != compute_doc_hash(pdf_b)


# ---------------------------------------------------------------------------
# compute_settings_hash
# ---------------------------------------------------------------------------


class TestComputeSettingsHash:
    _BASE = dict(model="m", dpi=200, image_format="png", dimensions=3072)

    def test_returns_64_char_hex_string(self):
        result = compute_settings_hash(**self._BASE)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_is_deterministic(self):
        assert compute_settings_hash(**self._BASE) == compute_settings_hash(**self._BASE)

    def test_matches_manual_sha256(self):
        settings = {"dimensions": 3072, "dpi": 200, "image_format": "png", "model": "m"}
        serialized = json.dumps(settings, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        expected = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        assert compute_settings_hash(**self._BASE) == expected

    def test_different_model_different_hash(self):
        h1 = compute_settings_hash(**{**self._BASE, "model": "model-a"})
        h2 = compute_settings_hash(**{**self._BASE, "model": "model-b"})
        assert h1 != h2

    def test_different_dpi_different_hash(self):
        h1 = compute_settings_hash(**{**self._BASE, "dpi": 100})
        h2 = compute_settings_hash(**{**self._BASE, "dpi": 300})
        assert h1 != h2

    def test_different_dimensions_different_hash(self):
        h1 = compute_settings_hash(**{**self._BASE, "dimensions": 512})
        h2 = compute_settings_hash(**{**self._BASE, "dimensions": 3072})
        assert h1 != h2

    def test_different_image_format_different_hash(self):
        h1 = compute_settings_hash(**{**self._BASE, "image_format": "png"})
        h2 = compute_settings_hash(**{**self._BASE, "image_format": "jpeg"})
        assert h1 != h2


# ---------------------------------------------------------------------------
# render_page_image
# ---------------------------------------------------------------------------


class TestRenderPageImage:
    def test_returns_base64_string_and_path(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r.pdf")
        images_dir = tmp_path / "images"
        b64, path = render_page_image(pdf, 0, images_dir=images_dir)
        assert isinstance(b64, str)
        assert isinstance(path, Path)

    def test_base64_decodes_to_non_empty_bytes(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r2.pdf")
        images_dir = tmp_path / "images"
        b64, _ = render_page_image(pdf, 0, images_dir=images_dir)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_saves_file_to_images_dir(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r3.pdf")
        images_dir = tmp_path / "images"
        _, path = render_page_image(pdf, 0, images_dir=images_dir)
        assert path.exists()
        assert path.parent == images_dir
        assert path.name == "page_0.png"

    def test_creates_images_dir_if_missing(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r4.pdf")
        images_dir = tmp_path / "nested" / "images"
        assert not images_dir.exists()
        render_page_image(pdf, 0, images_dir=images_dir)
        assert images_dir.exists()

    def test_png_format_default(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r5.pdf")
        images_dir = tmp_path / "images"
        _, path = render_page_image(pdf, 0, images_dir=images_dir)
        assert path.suffix == ".png"

    def test_jpeg_format(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r6.pdf")
        images_dir = tmp_path / "images"
        _, path = render_page_image(pdf, 0, image_format="jpeg", images_dir=images_dir)
        assert path.suffix == ".jpg"

    def test_saved_bytes_match_base64(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r7.pdf")
        images_dir = tmp_path / "images"
        b64, path = render_page_image(pdf, 0, images_dir=images_dir)
        assert path.read_bytes() == base64.b64decode(b64)

    def test_no_images_dir_does_not_save(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r8.pdf")
        b64, path = render_page_image(pdf, 0)
        assert isinstance(b64, str)
        assert not path.is_absolute()
        # File should NOT have been created on disk
        assert not path.exists()

    def test_invalid_format_raises_value_error(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r9.pdf")
        with pytest.raises(ValueError, match="Unsupported image format"):
            render_page_image(pdf, 0, image_format="bmp")

    def test_second_page(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r10.pdf", num_pages=3)
        images_dir = tmp_path / "images"
        _, path = render_page_image(pdf, 2, images_dir=images_dir)
        assert path.name == "page_2.png"

    def test_page_idx_out_of_range_raises(self, tmp_path):
        pdf = _make_pdf(tmp_path / "r11.pdf", num_pages=1)
        with pytest.raises(Exception):
            render_page_image(pdf, 5)


# ---------------------------------------------------------------------------
# embed_image_page
# ---------------------------------------------------------------------------


class TestEmbedImagePage:
    _FAKE_B64 = base64.b64encode(b"fake-image-data").decode()
    _EMBEDDING = [0.1, 0.2, 0.3]

    @patch("vlmembed.embed.requests.post")
    def test_returns_embedding_list(self, mock_post):
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        result = embed_image_page(
            self._FAKE_B64, model="m", api_key="key", dimensions=3
        )
        assert result == self._EMBEDDING

    @patch("vlmembed.embed.requests.post")
    def test_posts_to_openrouter_url(self, mock_post):
        from vlmembed.contract import OPENROUTER_EMBEDDINGS_URL
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        embed_image_page(self._FAKE_B64, model="m", api_key="key", dimensions=3)
        assert mock_post.call_args[0][0] == OPENROUTER_EMBEDDINGS_URL

    @patch("vlmembed.embed.requests.post")
    def test_auth_header_uses_bearer_token(self, mock_post):
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        embed_image_page(self._FAKE_B64, model="m", api_key="my-secret", dimensions=3)
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-secret"

    @patch("vlmembed.embed.requests.post")
    def test_payload_contains_image_url_input(self, mock_post):
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        embed_image_page(self._FAKE_B64, model="mymodel", api_key="key", dimensions=16)
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "mymodel"
        assert payload["dimensions"] == 16
        input_item = payload["input"][0]
        content = input_item["content"][0]
        assert content["type"] == "image_url"
        assert self._FAKE_B64 in content["image_url"]["url"]

    @patch("vlmembed.embed.requests.post")
    def test_png_mime_type(self, mock_post):
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        embed_image_page(
            self._FAKE_B64, model="m", api_key="k", dimensions=3, image_format="png"
        )
        payload = mock_post.call_args[1]["json"]
        url = payload["input"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    @patch("vlmembed.embed.requests.post")
    def test_jpeg_mime_type(self, mock_post):
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        embed_image_page(
            self._FAKE_B64, model="m", api_key="k", dimensions=3, image_format="jpeg"
        )
        payload = mock_post.call_args[1]["json"]
        url = payload["input"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    @patch("vlmembed.embed.requests.post")
    def test_http_error_is_propagated(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_post.return_value = mock_resp
        with pytest.raises(requests.HTTPError):
            embed_image_page(self._FAKE_B64, model="m", api_key="k", dimensions=3)

    @patch("vlmembed.embed.requests.post")
    def test_timeout_is_set(self, mock_post):
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        embed_image_page(self._FAKE_B64, model="m", api_key="k", dimensions=3)
        assert mock_post.call_args[1]["timeout"] == 60


# ---------------------------------------------------------------------------
# embed_text_query
# ---------------------------------------------------------------------------


class TestEmbedTextQuery:
    _EMBEDDING = [0.5, 0.6, 0.7]

    @patch("vlmembed.embed.requests.post")
    def test_returns_embedding_list(self, mock_post):
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        result = embed_text_query("hello world", model="m", api_key="k", dimensions=3)
        assert result == self._EMBEDDING

    @patch("vlmembed.embed.requests.post")
    def test_input_is_plain_string(self, mock_post):
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        embed_text_query("search query", model="mymodel", api_key="k", dimensions=8)
        payload = mock_post.call_args[1]["json"]
        assert payload["input"] == "search query"
        assert payload["model"] == "mymodel"
        assert payload["dimensions"] == 8

    @patch("vlmembed.embed.requests.post")
    def test_auth_header_uses_bearer_token(self, mock_post):
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        embed_text_query("q", model="m", api_key="tok", dimensions=3)
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer tok"

    @patch("vlmembed.embed.requests.post")
    def test_posts_to_openrouter_url(self, mock_post):
        from vlmembed.contract import OPENROUTER_EMBEDDINGS_URL
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        embed_text_query("q", model="m", api_key="k", dimensions=3)
        assert mock_post.call_args[0][0] == OPENROUTER_EMBEDDINGS_URL

    @patch("vlmembed.embed.requests.post")
    def test_http_error_propagated(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("401")
        mock_post.return_value = mock_resp
        with pytest.raises(requests.HTTPError):
            embed_text_query("q", model="m", api_key="k", dimensions=3)

    @patch("vlmembed.embed.requests.post")
    def test_timeout_is_set(self, mock_post):
        mock_post.return_value = _mock_embed_response(self._EMBEDDING)
        embed_text_query("q", model="m", api_key="k", dimensions=3)
        assert mock_post.call_args[1]["timeout"] == 60


# ---------------------------------------------------------------------------
# embed_all_pdfs
# ---------------------------------------------------------------------------


class TestEmbedAllPdfs:
    """Tests for embed_all_pdfs using mocked store and API calls."""

    def _setup_store_mocks(self, mock_store, page_exists_return=False):
        mock_collection = MagicMock()
        mock_store.get_collection.return_value = mock_collection
        mock_store.page_exists.return_value = page_exists_return
        mock_store.upsert_page.return_value = None
        return mock_collection

    # -- validation --

    def test_missing_api_key_raises(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        with patch.dict(os.environ, {}, clear=True):
            with patch("vlmembed.embed.dotenv.load_dotenv"):
                with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                    embed_all_pdfs(docs_dir, tmp_path / "embed", api_key=None)

    def test_invalid_max_workers_raises(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        with pytest.raises(ValueError, match="max_workers"):
            embed_all_pdfs(docs_dir, tmp_path / "embed", api_key="k", max_workers=0)

    def test_missing_docs_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Docs directory"):
            embed_all_pdfs(
                tmp_path / "nonexistent",
                tmp_path / "embed",
                api_key="k",
            )

    def test_empty_docs_dir_returns_empty_list(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        result = embed_all_pdfs(docs_dir, tmp_path / "embed", api_key="k")
        assert result == []

    # -- skip logic --

    @patch("vlmembed.store.get_collection")
    @patch("vlmembed.store.page_exists")
    @patch("vlmembed.store.upsert_page")
    def test_skips_already_embedded_pages(
        self, mock_upsert, mock_exists, mock_get_coll, tmp_path
    ):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        _make_pdf(docs_dir / "a.pdf", num_pages=2)

        mock_get_coll.return_value = MagicMock()
        mock_exists.return_value = True  # All pages already in store

        with patch("vlmembed.embed.dotenv.load_dotenv"):
            result = embed_all_pdfs(
                docs_dir, tmp_path / "embed", api_key="testkey"
            )

        assert result == []
        mock_upsert.assert_not_called()

    # -- happy path --

    @patch("vlmembed.store.get_collection")
    @patch("vlmembed.store.page_exists")
    @patch("vlmembed.store.upsert_page")
    @patch("vlmembed.embed.requests.post")
    def test_embeds_new_pages(
        self, mock_post, mock_upsert, mock_exists, mock_get_coll, tmp_path
    ):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        _make_pdf(docs_dir / "b.pdf", num_pages=2)

        mock_get_coll.return_value = MagicMock()
        mock_exists.return_value = False  # Nothing in store yet
        fake_emb = _fake_embedding(4)
        mock_post.return_value = _mock_embed_response(fake_emb)

        with patch("vlmembed.embed.dotenv.load_dotenv"):
            results = embed_all_pdfs(
                docs_dir,
                tmp_path / "embed",
                api_key="testkey",
                model="test-model",
                dimensions=4,
                max_workers=1,
            )

        assert len(results) == 2  # 2 pages
        assert mock_upsert.call_count == 2

        for r in results:
            assert r["embedding"] == fake_emb
            assert r["metadata"]["doc_path"].endswith("b.pdf")
            assert r["metadata"]["doc_hash"] != ""
            assert r["metadata"]["settings_hash"] != ""

    @patch("vlmembed.store.get_collection")
    @patch("vlmembed.store.page_exists")
    @patch("vlmembed.store.upsert_page")
    @patch("vlmembed.embed.requests.post")
    def test_page_ids_contain_doc_hash_and_index(
        self, mock_post, mock_upsert, mock_exists, mock_get_coll, tmp_path
    ):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        pdf = _make_pdf(docs_dir / "c.pdf", num_pages=1)

        mock_get_coll.return_value = MagicMock()
        mock_exists.return_value = False
        mock_post.return_value = _mock_embed_response(_fake_embedding(4))

        with patch("vlmembed.embed.dotenv.load_dotenv"):
            results = embed_all_pdfs(
                docs_dir,
                tmp_path / "embed",
                api_key="k",
                dimensions=4,
                max_workers=1,
            )

        expected_hash = compute_doc_hash(pdf)
        assert results[0]["page_id"] == f"{expected_hash}_0"

    @patch("vlmembed.store.get_collection")
    @patch("vlmembed.store.page_exists")
    @patch("vlmembed.store.upsert_page")
    @patch("vlmembed.embed.requests.post")
    def test_image_files_are_cached_to_disk(
        self, mock_post, mock_upsert, mock_exists, mock_get_coll, tmp_path
    ):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        pdf = _make_pdf(docs_dir / "d.pdf", num_pages=1)
        embed_dir = tmp_path / "embed"

        mock_get_coll.return_value = MagicMock()
        mock_exists.return_value = False
        mock_post.return_value = _mock_embed_response(_fake_embedding(4))

        with patch("vlmembed.embed.dotenv.load_dotenv"):
            results = embed_all_pdfs(
                docs_dir, embed_dir, api_key="k", dimensions=4, max_workers=1
            )

        image_cache = Path(results[0]["metadata"]["image_cache_path"])
        assert image_cache.exists()
        assert image_cache.name == "page_0.png"

    @patch("vlmembed.store.get_collection")
    @patch("vlmembed.store.page_exists")
    @patch("vlmembed.store.upsert_page")
    @patch("vlmembed.embed.requests.post")
    def test_settings_hash_in_metadata(
        self, mock_post, mock_upsert, mock_exists, mock_get_coll, tmp_path
    ):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        _make_pdf(docs_dir / "e.pdf", num_pages=1)

        mock_get_coll.return_value = MagicMock()
        mock_exists.return_value = False
        mock_post.return_value = _mock_embed_response(_fake_embedding(4))

        with patch("vlmembed.embed.dotenv.load_dotenv"):
            results = embed_all_pdfs(
                docs_dir,
                tmp_path / "embed",
                api_key="k",
                model="test-model",
                dpi=150,
                image_format="png",
                dimensions=4,
                max_workers=1,
            )

        expected_hash = compute_settings_hash(
            model="test-model", dpi=150, image_format="png", dimensions=4
        )
        assert results[0]["metadata"]["settings_hash"] == expected_hash

    # -- retry logic --

    @patch("vlmembed.store.get_collection")
    @patch("vlmembed.store.page_exists")
    @patch("vlmembed.store.upsert_page")
    @patch("vlmembed.embed.requests.post")
    def test_retries_on_transient_error(
        self, mock_post, mock_upsert, mock_exists, mock_get_coll, tmp_path
    ):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        _make_pdf(docs_dir / "f.pdf", num_pages=1)

        mock_get_coll.return_value = MagicMock()
        mock_exists.return_value = False

        # First call raises, second succeeds.
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = requests.HTTPError("503")
        ok_resp = _mock_embed_response(_fake_embedding(4))
        mock_post.side_effect = [fail_resp, ok_resp]

        with patch("vlmembed.embed.dotenv.load_dotenv"):
            results = embed_all_pdfs(
                docs_dir,
                tmp_path / "embed",
                api_key="k",
                dimensions=4,
                max_workers=1,
                max_retries=2,
            )

        assert len(results) == 1
        assert mock_post.call_count == 2

    @patch("vlmembed.store.get_collection")
    @patch("vlmembed.store.page_exists")
    @patch("vlmembed.store.upsert_page")
    @patch("vlmembed.embed.requests.post")
    def test_raises_after_all_retries_exhausted(
        self, mock_post, mock_upsert, mock_exists, mock_get_coll, tmp_path
    ):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        _make_pdf(docs_dir / "g.pdf", num_pages=1)

        mock_get_coll.return_value = MagicMock()
        mock_exists.return_value = False

        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_post.return_value = fail_resp

        with patch("vlmembed.embed.dotenv.load_dotenv"):
            with pytest.raises(RuntimeError, match="failed after"):
                embed_all_pdfs(
                    docs_dir,
                    tmp_path / "embed",
                    api_key="k",
                    dimensions=4,
                    max_workers=1,
                    max_retries=2,
                )

        assert mock_post.call_count == 2

    # -- multi-PDF --

    @patch("vlmembed.store.get_collection")
    @patch("vlmembed.store.page_exists")
    @patch("vlmembed.store.upsert_page")
    @patch("vlmembed.embed.requests.post")
    def test_processes_multiple_pdfs(
        self, mock_post, mock_upsert, mock_exists, mock_get_coll, tmp_path
    ):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        _make_pdf(docs_dir / "p1.pdf", num_pages=1)
        _make_pdf(docs_dir / "p2.pdf", num_pages=2)

        mock_get_coll.return_value = MagicMock()
        mock_exists.return_value = False
        mock_post.return_value = _mock_embed_response(_fake_embedding(4))

        with patch("vlmembed.embed.dotenv.load_dotenv"):
            results = embed_all_pdfs(
                docs_dir,
                tmp_path / "embed",
                api_key="k",
                dimensions=4,
                max_workers=2,
            )

        assert len(results) == 3  # 1 + 2 pages

    @patch("vlmembed.store.get_collection")
    @patch("vlmembed.store.page_exists")
    @patch("vlmembed.store.upsert_page")
    @patch("vlmembed.embed.requests.post")
    def test_partial_skip_only_embeds_new(
        self, mock_post, mock_upsert, mock_exists, mock_get_coll, tmp_path
    ):
        """When some pages exist and some don't, only new ones are embedded."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        _make_pdf(docs_dir / "h.pdf", num_pages=3)

        mock_get_coll.return_value = MagicMock()
        # page 0 exists, pages 1+2 don't
        mock_exists.side_effect = [True, False, False]
        mock_post.return_value = _mock_embed_response(_fake_embedding(4))

        with patch("vlmembed.embed.dotenv.load_dotenv"):
            results = embed_all_pdfs(
                docs_dir,
                tmp_path / "embed",
                api_key="k",
                dimensions=4,
                max_workers=1,
            )

        assert len(results) == 2
        assert mock_upsert.call_count == 2
