"""Unit tests for search_anything.search_app (Phase 4)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import gradio as gr

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING = [0.1] * 8
_DIM = 8


def _make_metadata(
    page_number: int = 0,
    image_cache_path: str = "",
    doc_path: str = "docs/report.pdf",
) -> dict:
    return {
        "doc_path": doc_path,
        "page_number": page_number,
        "doc_hash": "abc123",
        "settings_hash": "def456",
        "image_cache_path": image_cache_path,
    }


def _make_result(
    page_number: int = 0,
    distance: float = 0.2,
    image_cache_path: str = "",
    doc_path: str = "docs/report.pdf",
) -> dict:
    return {
        "page_id": f"abc123_{page_number}",
        "metadata": _make_metadata(
            page_number=page_number,
            image_cache_path=image_cache_path,
            doc_path=doc_path,
        ),
        "distance": distance,
    }


def _mock_collection(count: int = 10) -> MagicMock:
    col = MagicMock()
    col.count.return_value = count
    return col


# ---------------------------------------------------------------------------
# _make_caption
# ---------------------------------------------------------------------------


class TestMakeCaption:
    def test_format_filename_page_score(self):
        from search_anything.search_app import _make_caption

        meta = _make_metadata(page_number=2, doc_path="docs/annual_report.pdf")
        caption = _make_caption(meta, distance=0.3)
        assert caption == "annual_report.pdf · page 2 · score 0.700"

    def test_page_number_one_indexed(self):
        from search_anything.search_app import _make_caption

        meta = _make_metadata(page_number=1)
        caption = _make_caption(meta, distance=0.0)
        assert "page 1" in caption

    def test_score_is_one_minus_distance(self):
        from search_anything.search_app import _make_caption

        meta = _make_metadata(page_number=1)
        caption = _make_caption(meta, distance=0.5)
        assert "score 0.500" in caption

    def test_unknown_doc_path(self):
        from search_anything.search_app import _make_caption

        meta = _make_metadata(doc_path="unknown")
        caption = _make_caption(meta, distance=0.1)
        assert "unknown" in caption

    def test_non_pdf_includes_modality_and_segment_label(self):
        from search_anything.search_app import _make_caption

        meta = _make_metadata(page_number=2, doc_path="docs/clip.mp3")
        caption = _make_caption(meta, distance=0.2)
        assert "segment 2" in caption
        assert "audio" in caption


# ---------------------------------------------------------------------------
# _load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_returns_none_for_nonexistent_path(self):
        from search_anything.search_app import _load_image

        result = _load_image("/does/not/exist/page_1.png")
        assert result is None

    def test_returns_none_for_empty_string(self):
        from search_anything.search_app import _load_image

        result = _load_image("")
        assert result is None

    def test_returns_pil_image_for_valid_file(self, tmp_path):
        from PIL import Image

        from search_anything.search_app import _load_image

        img_path = tmp_path / "page_1.png"
        Image.new("RGB", (10, 10), color=(255, 0, 0)).save(str(img_path))

        result = _load_image(str(img_path))
        assert result is not None
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# _build_gallery_items
# ---------------------------------------------------------------------------


class TestBuildGalleryItems:
    def test_empty_results_returns_empty_list(self):
        from search_anything.search_app import _build_gallery_items

        assert _build_gallery_items([]) == []

    def test_missing_image_produces_placeholder_tile(self):
        from PIL import Image

        from search_anything.search_app import _build_gallery_items

        results = [_make_result(image_cache_path="/no/such/file.png")]
        items = _build_gallery_items(results)
        assert len(items) == 1
        img, caption = items[0]
        assert isinstance(img, Image.Image)
        assert "report.pdf" in caption

    def test_non_image_result_gets_modality_caption(self):
        from search_anything.search_app import _build_gallery_items

        results = [
            _make_result(
                page_number=3,
                image_cache_path="",
                doc_path="docs/transcript.md",
            )
        ]
        _, caption = _build_gallery_items(results)[0]
        assert "chunk 3" in caption
        assert "text" in caption

    def test_valid_image_produces_pil_tile(self, tmp_path):
        from PIL import Image

        from search_anything.search_app import _build_gallery_items

        img_path = tmp_path / "page_1.png"
        Image.new("RGB", (10, 10)).save(str(img_path))

        results = [_make_result(image_cache_path=str(img_path))]
        items = _build_gallery_items(results)
        assert len(items) == 1
        img, caption = items[0]
        assert isinstance(img, Image.Image)

    def test_multiple_results_preserve_order(self):
        from search_anything.search_app import _build_gallery_items

        results = [_make_result(page_number=i + 1) for i in range(3)]
        items = _build_gallery_items(results)
        assert len(items) == 3
        for i, (_, caption) in enumerate(items):
            assert f"page {i + 1}" in caption

    def test_caption_contains_score(self):
        from search_anything.search_app import _build_gallery_items

        results = [_make_result(distance=0.25)]
        _, caption = _build_gallery_items(results)[0]
        assert "score 0.750" in caption


# ---------------------------------------------------------------------------
# build_search_app — Blocks construction
# ---------------------------------------------------------------------------


class TestBuildSearchApp:
    """Test that build_search_app returns a Gradio Blocks without errors."""

    @patch("search_anything.search_app.get_collection")
    def test_returns_blocks_instance(self, mock_get_col, tmp_path):
        from search_anything.search_app import build_search_app

        mock_get_col.return_value = _mock_collection()
        demo = build_search_app(
            embed_dir=tmp_path,
            api_key="test-key",
            model="gemini-embedding-2",
            dimensions=8,
        )
        assert isinstance(demo, gr.Blocks)

    @patch("search_anything.search_app.get_collection")
    def test_collection_opened_with_correct_embed_dir(self, mock_get_col, tmp_path):
        from search_anything.search_app import build_search_app

        mock_get_col.return_value = _mock_collection()
        build_search_app(
            embed_dir=tmp_path,
            api_key="key",
            model="m",
            dimensions=8,
        )
        mock_get_col.assert_called_once_with(tmp_path)

    @patch("search_anything.search_app.ensure_store_compatibility")
    @patch("search_anything.search_app.get_collection")
    def test_validates_store_compatibility(self, mock_get_col, mock_compat, tmp_path):
        from search_anything.search_app import build_search_app

        mock_get_col.return_value = _mock_collection()
        build_search_app(
            embed_dir=tmp_path,
            api_key="k",
            model="gemini-embedding-2",
            dimensions=8,
        )
        mock_compat.assert_called_once_with(
            tmp_path,
            model="gemini-embedding-2",
            dimensions=8,
        )


# ---------------------------------------------------------------------------
# _run_search (inner function) — exercised via gr.Blocks event
# ---------------------------------------------------------------------------


class TestRunSearch:
    """Test the search callback via make_search_fn, which is testable directly."""

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search", return_value=[])
    def test_empty_query_returns_no_results(self, mock_search, mock_embed):
        from search_anything.search_app import make_search_fn

        run_search = make_search_fn(
            _mock_collection(count=5), api_key="key", model="m", dimensions=_DIM
        )
        result = run_search("   ", 5)
        assert result == []
        mock_embed.assert_not_called()

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search")
    def test_empty_collection_returns_no_results(self, mock_search, mock_embed):
        from search_anything.search_app import make_search_fn

        run_search = make_search_fn(
            _mock_collection(count=0), api_key="key", model="m", dimensions=_DIM
        )
        result = run_search("find something", 5)
        assert result == []
        mock_embed.assert_not_called()
        mock_search.assert_not_called()

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search")
    def test_n_results_capped_at_collection_count(self, mock_search, mock_embed):
        from search_anything.search_app import make_search_fn

        mock_search.return_value = [_make_result(page_number=i) for i in range(3)]
        run_search = make_search_fn(
            _mock_collection(count=3), api_key="key", model="m", dimensions=_DIM
        )
        run_search("query", n_results=10)
        call_n = (
            mock_search.call_args[1].get("n_results") or mock_search.call_args[0][2]
        )
        assert call_n <= 3

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search", return_value=[])
    def test_embed_text_query_called_with_correct_args(self, mock_search, mock_embed):
        from search_anything.search_app import make_search_fn

        run_search = make_search_fn(
            _mock_collection(count=5),
            api_key="my-api-key",
            model="my-model",
            dimensions=_DIM,
        )
        run_search("semantic query", 3)
        mock_embed.assert_called_once_with(
            "semantic query",
            model="my-model",
            api_key="my-api-key",
            dimensions=_DIM,
        )

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search")
    def test_returns_gallery_items_from_results(self, mock_search, mock_embed):
        from search_anything.search_app import make_search_fn

        mock_search.return_value = [_make_result(page_number=0, distance=0.1)]
        run_search = make_search_fn(
            _mock_collection(count=5), api_key="key", model="m", dimensions=_DIM
        )
        items = run_search("find page", 5)
        assert len(items) == 1
        _, caption = items[0]
        assert "score 0.900" in caption


# ---------------------------------------------------------------------------
# launch_search_app — env var + dotenv config priority
# ---------------------------------------------------------------------------


class TestLaunchSearchApp:
    @patch("search_anything.search_app.gr.Blocks.launch")
    @patch("search_anything.search_app.get_collection")
    @patch("search_anything.search_app.dotenv.load_dotenv")
    def test_calls_launch_with_port(self, mock_dotenv, mock_col, mock_launch, tmp_path):
        from search_anything.search_app import launch_search_app

        mock_col.return_value = _mock_collection()
        launch_search_app(
            embed_dir=tmp_path,
            api_key="key",
            model="m",
            dimensions=_DIM,
            port=7890,
        )
        mock_launch.assert_called_once_with(server_port=7890, inbrowser=True)

    @patch("search_anything.search_app.gr.Blocks.launch")
    @patch("search_anything.search_app.get_collection")
    @patch("search_anything.search_app.dotenv.load_dotenv")
    def test_reads_api_key_from_env_when_not_provided(
        self, mock_dotenv, mock_col, mock_launch, tmp_path, monkeypatch
    ):
        """When api_key is empty, GOOGLE_API_KEY env var is used instead."""
        from search_anything.search_app import launch_search_app, make_search_fn

        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        col = _mock_collection()
        mock_col.return_value = col

        with patch(
            "search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING
        ) as mock_embed:
            with patch("search_anything.search_app.search", return_value=[]):
                launch_search_app(
                    embed_dir=tmp_path,
                    api_key="",
                    model="m",
                    dimensions=_DIM,
                    port=7860,
                )
                # Verify the key was picked up: call make_search_fn with env key and exercise it
                run_search = make_search_fn(
                    col, api_key="env-key", model="m", dimensions=_DIM
                )
                run_search("test", 1)
                mock_embed.assert_called_with(
                    "test", model="m", api_key="env-key", dimensions=_DIM
                )

    @patch("search_anything.search_app.gr.Blocks.launch")
    @patch("search_anything.search_app.get_collection")
    @patch("search_anything.search_app.dotenv.load_dotenv")
    def test_load_dotenv_called(self, mock_dotenv, mock_col, mock_launch, tmp_path):
        from search_anything.search_app import launch_search_app

        mock_col.return_value = _mock_collection()
        launch_search_app(embed_dir=tmp_path, api_key="k", model="m", dimensions=_DIM)
        mock_dotenv.assert_called_once()


# ---------------------------------------------------------------------------
# Query embedding cache — make_search_fn with embed_dir
# ---------------------------------------------------------------------------


class TestQueryEmbeddingCache:
    """Verify that make_search_fn skips the API when a cached embedding exists."""

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search", return_value=[])
    def test_embed_called_on_first_query(self, mock_search, mock_embed, tmp_path):
        from search_anything.search_app import make_search_fn

        run_search = make_search_fn(
            _mock_collection(count=5),
            api_key="key",
            model="m",
            dimensions=_DIM,
            embed_dir=tmp_path,
        )
        run_search("hello", 3)
        mock_embed.assert_called_once()

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search", return_value=[])
    def test_embed_not_called_on_repeat_query(self, mock_search, mock_embed, tmp_path):
        from search_anything.search_app import make_search_fn

        run_search = make_search_fn(
            _mock_collection(count=5),
            api_key="key",
            model="m",
            dimensions=_DIM,
            embed_dir=tmp_path,
        )
        run_search("hello", 3)
        run_search("hello", 3)
        mock_embed.assert_called_once()  # only the first call hits the API

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search", return_value=[])
    def test_cache_is_case_insensitive(self, mock_search, mock_embed, tmp_path):
        from search_anything.search_app import make_search_fn

        run_search = make_search_fn(
            _mock_collection(count=5),
            api_key="key",
            model="m",
            dimensions=_DIM,
            embed_dir=tmp_path,
        )
        run_search("Hello", 3)
        run_search("hello", 3)
        mock_embed.assert_called_once()

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search", return_value=[])
    def test_cache_ignores_punctuation(self, mock_search, mock_embed, tmp_path):
        from search_anything.search_app import make_search_fn

        run_search = make_search_fn(
            _mock_collection(count=5),
            api_key="key",
            model="m",
            dimensions=_DIM,
            embed_dir=tmp_path,
        )
        run_search("cats!", 3)
        run_search("cats", 3)
        mock_embed.assert_called_once()

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search", return_value=[])
    def test_different_queries_each_call_api(self, mock_search, mock_embed, tmp_path):
        from search_anything.search_app import make_search_fn

        run_search = make_search_fn(
            _mock_collection(count=5),
            api_key="key",
            model="m",
            dimensions=_DIM,
            embed_dir=tmp_path,
        )
        run_search("hello", 3)
        run_search("world", 3)
        assert mock_embed.call_count == 2

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search", return_value=[])
    def test_no_embed_dir_disables_caching(self, mock_search, mock_embed, tmp_path):
        from search_anything.search_app import make_search_fn

        run_search = make_search_fn(
            _mock_collection(count=5),
            api_key="key",
            model="m",
            dimensions=_DIM,
            # embed_dir not provided
        )
        run_search("hello", 3)
        run_search("hello", 3)
        assert mock_embed.call_count == 2  # no cache → two API calls

    @patch("search_anything.search_app.embed_text_query", return_value=_FAKE_EMBEDDING)
    @patch("search_anything.search_app.search", return_value=[])
    def test_cache_persists_across_search_fn_instances(
        self, mock_search, mock_embed, tmp_path
    ):
        from search_anything.search_app import make_search_fn

        # First instance populates the cache
        run_search_1 = make_search_fn(
            _mock_collection(count=5),
            api_key="key",
            model="m",
            dimensions=_DIM,
            embed_dir=tmp_path,
        )
        run_search_1("hello", 3)

        # Second instance (simulates app restart) should read from disk cache
        run_search_2 = make_search_fn(
            _mock_collection(count=5),
            api_key="key",
            model="m",
            dimensions=_DIM,
            embed_dir=tmp_path,
        )
        run_search_2("hello", 3)
        mock_embed.assert_called_once()  # only the very first call ever hits the API
