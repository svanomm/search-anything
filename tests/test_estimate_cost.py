"""Unit tests for vlmembed.estimate_cost."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from vlmembed.estimate_cost import (
    _PAGE_HEIGHT_INCHES,
    _PAGE_WIDTH_INCHES,
    _PIXELS_PER_TOKEN,
    _PRICE_PER_M_TOKENS,
    count_pdf_pages,
    estimate_cost,
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

    def test_formula_at_200_dpi(self):
        dpi = 200
        width_px = int(_PAGE_WIDTH_INCHES * dpi)
        height_px = int(_PAGE_HEIGHT_INCHES * dpi)
        expected = max(1, (width_px * height_px) // _PIXELS_PER_TOKEN)
        assert estimate_tokens_per_page(dpi=dpi) == expected

    def test_higher_dpi_gives_more_tokens(self):
        assert estimate_tokens_per_page(dpi=300) > estimate_tokens_per_page(dpi=100)

    def test_very_low_dpi_returns_at_least_one(self):
        # Even at dpi=1 (degenerate case), must return >= 1.
        result = estimate_tokens_per_page(dpi=1)
        assert result >= 1

    def test_default_dpi(self):
        from vlmembed.contract import DEFAULT_DPI

        assert estimate_tokens_per_page() == estimate_tokens_per_page(dpi=DEFAULT_DPI)

    def test_tokens_scale_quadratically_with_dpi(self):
        # Doubling DPI should ~quadruple the token count (area scales as DPI²).
        low = estimate_tokens_per_page(dpi=100)
        high = estimate_tokens_per_page(dpi=200)
        ratio = high / low
        assert 3.5 < ratio < 4.5


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

    def test_dpi_affects_tokens_per_page(self, tmp_path):
        _make_pdf(tmp_path / "doc.pdf", num_pages=1)
        low = estimate_cost(docs_dir=tmp_path, dpi=100)
        high = estimate_cost(docs_dir=tmp_path, dpi=300)
        assert high["tokens_per_page"] > low["tokens_per_page"]

    def test_default_docs_dir_used_when_not_specified(self):
        # Python evaluates default argument values at definition time, so we
        # cannot monkey-patch them; instead verify the signature itself.
        import inspect

        sig = inspect.signature(estimate_cost)
        from vlmembed.contract import DEFAULT_DOCS_DIR as _DEFAULT

        assert sig.parameters["docs_dir"].default == _DEFAULT
