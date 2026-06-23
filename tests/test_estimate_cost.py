"""Unit tests for search_anything.estimate_cost."""

from __future__ import annotations

from pathlib import Path

import fitz

from search_anything.estimate_cost import (
    _PRICE_PER_M_TOKENS,
    count_multimodal_inputs,
    count_pdf_pages,
    estimate_cost,
    estimate_cost_from_page_counts,
    estimate_multimodal_cost_from_counts,
    _estimate_text_tokens_from_content,
    estimate_tokens_per_page,
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


# ---------------------------------------------------------------------------
# count_pdf_pages
# ---------------------------------------------------------------------------


class TestCountPdfPages:
    def test_returns_empty_dict_when_no_pdfs(self, tmp_path):
        result = count_pdf_pages(tmp_path)
        assert result == {}

    def test_returns_empty_dict_for_nonexistent_dir(self, tmp_path):
        result = count_pdf_pages(tmp_path / "nonexistent")
        assert result == {}

    def test_counts_single_pdf_correctly(self, tmp_path):
        _make_pdf(tmp_path / "doc.pdf", num_pages=3)
        result = count_pdf_pages(tmp_path)
        assert result == {"doc.pdf": 3}

    def test_counts_multiple_pdfs(self, tmp_path):
        _make_pdf(tmp_path / "a.pdf", num_pages=2)
        _make_pdf(tmp_path / "b.pdf", num_pages=5)
        result = count_pdf_pages(tmp_path)
        assert result == {"a.pdf": 2, "b.pdf": 5}

    def test_keys_are_filenames_not_paths(self, tmp_path):
        _make_pdf(tmp_path / "report.pdf", num_pages=1)
        result = count_pdf_pages(tmp_path)
        assert list(result.keys()) == ["report.pdf"]

    def test_ignores_non_pdf_files(self, tmp_path):
        _make_pdf(tmp_path / "real.pdf", num_pages=1)
        (tmp_path / "notes.txt").write_text("hello")
        result = count_pdf_pages(tmp_path)
        assert "notes.txt" not in result
        assert "real.pdf" in result

    def test_single_page_pdf(self, tmp_path):
        _make_pdf(tmp_path / "single.pdf", num_pages=1)
        result = count_pdf_pages(tmp_path)
        assert result["single.pdf"] == 1

    def test_results_sorted_alphabetically(self, tmp_path):
        _make_pdf(tmp_path / "z.pdf", num_pages=1)
        _make_pdf(tmp_path / "a.pdf", num_pages=1)
        _make_pdf(tmp_path / "m.pdf", num_pages=1)
        result = count_pdf_pages(tmp_path)
        assert list(result.keys()) == ["a.pdf", "m.pdf", "z.pdf"]

    def test_accepts_path_and_str(self, tmp_path):
        _make_pdf(tmp_path / "x.pdf", num_pages=2)
        assert count_pdf_pages(tmp_path) == count_pdf_pages(str(tmp_path))


# ---------------------------------------------------------------------------
# estimate_tokens_per_page
# ---------------------------------------------------------------------------


class TestEstimateTokensPerPage:
    def test_returns_positive_int(self):
        result = estimate_tokens_per_page(dpi=200)
        assert isinstance(result, int)
        assert result > 0

    def test_very_low_dpi_returns_at_least_one(self):
        # Even at dpi=1 (degenerate case), must return >= 1.
        result = estimate_tokens_per_page(dpi=1)
        assert result >= 1

    def test_default_dpi(self):
        from search_anything.contract import DEFAULT_DPI

        assert estimate_tokens_per_page() == estimate_tokens_per_page(dpi=DEFAULT_DPI)


class TestTextTokenEstimate:
    def test_empty_text_is_zero(self):
        assert _estimate_text_tokens_from_content("") == 0

    def test_non_empty_text_positive(self):
        assert _estimate_text_tokens_from_content("hello world") > 0

    def test_overlap_increases_estimate_for_long_text(self):
        short = _estimate_text_tokens_from_content("a" * 400)
        long = _estimate_text_tokens_from_content("a" * 3000)
        assert long > short


class TestCountMultimodalInputs:
    def test_returns_zero_counts_for_missing_dir(self, tmp_path):
        counts = count_multimodal_inputs(tmp_path / "missing")
        assert counts == {
            "pdf_pages": 0,
            "images": 0,
            "text_tokens": 0,
            "audio_seconds": 0.0,
            "video_frames": 0,
        }

    def test_counts_recursive_multimodal_files(self, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir(parents=True)

        _make_pdf(tmp_path / "a.pdf", num_pages=2)
        (nested / "photo.png").write_bytes(b"img")
        (nested / "note.md").write_text("# title\n\nbody", encoding="utf-8")
        # ~2 seconds by assumption
        (nested / "clip.wav").write_bytes(b"x" * 40_000)
        (nested / "video.mp4").write_bytes(b"video")

        counts = count_multimodal_inputs(tmp_path)
        assert counts["pdf_pages"] == 2
        assert counts["images"] == 1
        assert counts["text_tokens"] > 0
        assert counts["audio_seconds"] > 0.0
        assert counts["video_frames"] == 120


class TestEstimateMultimodalCostFromCounts:
    def test_outputs_expected_keys(self):
        result = estimate_multimodal_cost_from_counts(
            {
                "pdf_pages": 2,
                "images": 1,
                "text_tokens": 100,
                "audio_seconds": 4.0,
                "video_frames": 120,
            }
        )
        assert set(result.keys()) == {
            "token_breakdown",
            "per_modality_usd",
            "total_tokens",
            "estimated_usd",
        }

    def test_total_tokens_matches_breakdown_sum(self):
        result = estimate_multimodal_cost_from_counts(
            {
                "pdf_pages": 1,
                "images": 1,
                "text_tokens": 50,
                "audio_seconds": 2.0,
                "video_frames": 10,
            }
        )
        assert result["total_tokens"] == sum(result["token_breakdown"].values())

    def test_estimated_usd_matches_rate(self):
        result = estimate_multimodal_cost_from_counts(
            {
                "pdf_pages": 1,
                "images": 0,
                "text_tokens": 0,
                "audio_seconds": 0.0,
                "video_frames": 0,
            }
        )
        expected = result["total_tokens"] * _PRICE_PER_M_TOKENS / 1_000_000
        assert abs(result["estimated_usd"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_returns_expected_keys(self, tmp_path):
        result = estimate_cost(docs_dir=tmp_path, dpi=200)
        assert set(result.keys()) == {
            "per_file",
            "pages",
            "tokens_per_page",
            "total_tokens",
            "estimated_usd",
            "modalities",
            "token_breakdown",
            "per_modality_usd",
        }

    def test_empty_dir_gives_zero_cost(self, tmp_path):
        result = estimate_cost(docs_dir=tmp_path, dpi=200)
        assert result["pages"] == 0
        assert result["total_tokens"] == 0
        assert result["estimated_usd"] == 0.0
        assert result["per_file"] == {}

    def test_nonexistent_dir_gives_zero_cost(self, tmp_path):
        result = estimate_cost(docs_dir=tmp_path / "missing", dpi=200)
        assert result["pages"] == 0
        assert result["total_tokens"] == 0

    def test_total_pages_sums_all_pdfs(self, tmp_path):
        _make_pdf(tmp_path / "a.pdf", num_pages=3)
        _make_pdf(tmp_path / "b.pdf", num_pages=7)
        result = estimate_cost(docs_dir=tmp_path, dpi=200)
        assert result["pages"] == 10

    def test_total_tokens_equals_pages_times_tokens_per_page(self, tmp_path):
        _make_pdf(tmp_path / "doc.pdf", num_pages=4)
        result = estimate_cost(docs_dir=tmp_path, dpi=200)
        assert result["total_tokens"] == result["pages"] * result["tokens_per_page"]

    def test_estimated_usd_formula(self, tmp_path):
        _make_pdf(tmp_path / "doc.pdf", num_pages=2)
        result = estimate_cost(docs_dir=tmp_path, dpi=200)
        expected = result["total_tokens"] * _PRICE_PER_M_TOKENS / 1_000_000
        assert abs(result["estimated_usd"] - expected) < 1e-9

    def test_per_file_contents(self, tmp_path):
        _make_pdf(tmp_path / "x.pdf", num_pages=5)
        result = estimate_cost(docs_dir=tmp_path, dpi=200)
        assert result["per_file"] == {"x.pdf": 5}

    def test_multimodal_counts_appear_in_result(self, tmp_path):
        _make_pdf(tmp_path / "x.pdf", num_pages=1)
        (tmp_path / "photo.png").write_bytes(b"img")
        (tmp_path / "note.txt").write_text("hello world", encoding="utf-8")

        result = estimate_cost(docs_dir=tmp_path, dpi=200)
        assert result["modalities"]["pdf_pages"] == 1
        assert result["modalities"]["images"] == 1
        assert result["modalities"]["text_tokens"] > 0

    def test_default_docs_dir_used_when_not_specified(self):
        # Python evaluates default argument values at definition time, so we
        # cannot monkey-patch them; instead verify the signature itself.
        import inspect

        sig = inspect.signature(estimate_cost)
        from search_anything.contract import DEFAULT_DOCS_DIR as _DEFAULT

        assert sig.parameters["docs_dir"].default == _DEFAULT


# ---------------------------------------------------------------------------
# estimate_cost_from_page_counts
# ---------------------------------------------------------------------------


class TestEstimateCostFromPageCounts:
    def test_expected_shape_and_totals(self):
        per_file = {"a.pdf": 2, "b.pdf": 3}
        result = estimate_cost_from_page_counts(per_file, dpi=200)

        assert result["per_file"] == per_file
        assert result["pages"] == 5
        assert result["total_tokens"] == result["pages"] * result["tokens_per_page"]

    def test_empty_mapping_has_zero_pages_and_cost(self):
        result = estimate_cost_from_page_counts({}, dpi=200)
        assert result["pages"] == 0
        assert result["total_tokens"] == 0
        assert result["estimated_usd"] == 0.0
