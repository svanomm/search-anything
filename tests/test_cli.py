"""Unit tests for vlmembed.cli."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from vlmembed.cli import (
    _build_parser,
    _interactive_menu,
    _resolve_int,
    _resolve_str,
    cmd_embed,
    cmd_estimate_cost,
    cmd_init,
    cmd_search,
    main,
)
from vlmembed.contract import (
    DEFAULT_DIMENSIONS,
    DEFAULT_DPI,
    DEFAULT_DOCS_DIR,
    DEFAULT_EMBED_DIR,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MODEL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(**kwargs) -> argparse.Namespace:
    """Build a Namespace with the given kwargs, adding None for missing embed fields."""
    defaults = {
        "docs_dir": str(DEFAULT_DOCS_DIR),
        "embed_dir": str(DEFAULT_EMBED_DIR),
        "api_key": None,
        "model": None,
        "dpi": None,
        "image_format": None,
        "dimensions": None,
        "max_workers": None,
        "max_retries": None,
        "port": 7860,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# _resolve_int / _resolve_str helpers
# ---------------------------------------------------------------------------


class TestResolveInt:
    def test_cli_val_wins(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "99")
        assert _resolve_int(42, "MY_KEY", 0) == 42

    def test_env_var_used_when_cli_none(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "7")
        assert _resolve_int(None, "MY_KEY", 0) == 7

    def test_default_used_when_both_absent(self, monkeypatch):
        monkeypatch.delenv("MY_KEY", raising=False)
        assert _resolve_int(None, "MY_KEY", 55) == 55

    def test_zero_cli_val_is_respected(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "99")
        # 0 is falsy but is a valid explicit value — only None means "not set".
        assert _resolve_int(0, "MY_KEY", 55) == 0


class TestResolveStr:
    def test_cli_val_wins(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "env_val")
        assert _resolve_str("cli_val", "MY_KEY", "default") == "cli_val"

    def test_env_var_used_when_cli_none(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "env_val")
        assert _resolve_str(None, "MY_KEY", "default") == "env_val"

    def test_env_var_used_when_cli_empty_string(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "env_val")
        assert _resolve_str("", "MY_KEY", "default") == "env_val"

    def test_default_used_when_both_absent(self, monkeypatch):
        monkeypatch.delenv("MY_KEY", raising=False)
        assert _resolve_str(None, "MY_KEY", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# _build_parser — argument defaults and types
# ---------------------------------------------------------------------------


class TestBuildParser:
    def setup_method(self):
        self.parser = _build_parser()

    def test_no_subcommand_gives_none(self):
        args = self.parser.parse_args([])
        assert args.subcommand is None

    # init
    def test_init_docs_dir_default(self):
        args = self.parser.parse_args(["init"])
        assert args.docs_dir == str(DEFAULT_DOCS_DIR)

    def test_init_embed_dir_default(self):
        args = self.parser.parse_args(["init"])
        assert args.embed_dir == str(DEFAULT_EMBED_DIR)

    def test_init_custom_dirs(self):
        args = self.parser.parse_args(["init", "--docs-dir", "my_docs", "--embed-dir", "my_emb"])
        assert args.docs_dir == "my_docs"
        assert args.embed_dir == "my_emb"

    # embed — defaults
    def test_embed_defaults(self):
        args = self.parser.parse_args(["embed"])
        assert args.docs_dir == str(DEFAULT_DOCS_DIR)
        assert args.embed_dir == str(DEFAULT_EMBED_DIR)
        assert args.api_key is None
        assert args.model is None
        assert args.dpi is None
        assert args.image_format is None
        assert args.dimensions is None
        assert args.max_workers is None
        assert args.max_retries is None

    def test_embed_accepts_all_flags(self):
        args = self.parser.parse_args([
            "embed",
            "--api-key", "key123",
            "--model", "mymodel",
            "--dpi", "150",
            "--format", "jpeg",
            "--dimensions", "512",
            "--max-workers", "2",
            "--max-retries", "1",
        ])
        assert args.api_key == "key123"
        assert args.model == "mymodel"
        assert args.dpi == 150
        assert args.image_format == "jpeg"
        assert args.dimensions == 512
        assert args.max_workers == 2
        assert args.max_retries == 1

    def test_embed_dpi_is_int(self):
        args = self.parser.parse_args(["embed", "--dpi", "300"])
        assert isinstance(args.dpi, int)

    # search
    def test_search_port_default(self):
        args = self.parser.parse_args(["search"])
        assert args.port == 7860

    def test_search_custom_port(self):
        args = self.parser.parse_args(["search", "--port", "8080"])
        assert args.port == 8080

    def test_search_port_is_int(self):
        args = self.parser.parse_args(["search", "--port", "9000"])
        assert isinstance(args.port, int)

    # estimate-cost
    def test_estimate_cost_docs_dir_default(self):
        args = self.parser.parse_args(["estimate-cost"])
        assert args.docs_dir == str(DEFAULT_DOCS_DIR)

    def test_estimate_cost_custom_docs_dir(self):
        args = self.parser.parse_args(["estimate-cost", "--docs-dir", "other_docs"])
        assert args.docs_dir == "other_docs"

    def test_estimate_cost_dpi_is_int(self):
        args = self.parser.parse_args(["estimate-cost", "--dpi", "100"])
        assert isinstance(args.dpi, int)
        assert args.dpi == 100


# ---------------------------------------------------------------------------
# cmd_init
# ---------------------------------------------------------------------------


class TestCmdInit:
    def test_creates_all_directories(self, tmp_path):
        args = argparse.Namespace(
            docs_dir=str(tmp_path / "docs"),
            embed_dir=str(tmp_path / "embeddings"),
        )
        result = cmd_init(args)
        assert result == 0
        assert (tmp_path / "docs").is_dir()
        assert (tmp_path / "embeddings").is_dir()
        assert (tmp_path / "embeddings" / "images").is_dir()
        assert (tmp_path / "embeddings" / "db").is_dir()

    def test_returns_zero(self, tmp_path):
        args = argparse.Namespace(
            docs_dir=str(tmp_path / "docs"),
            embed_dir=str(tmp_path / "embeddings"),
        )
        assert cmd_init(args) == 0

    def test_idempotent_when_dirs_exist(self, tmp_path):
        args = argparse.Namespace(
            docs_dir=str(tmp_path / "docs"),
            embed_dir=str(tmp_path / "embeddings"),
        )
        cmd_init(args)
        # Second call must not raise.
        assert cmd_init(args) == 0

    def test_uses_custom_docs_and_embed_dirs(self, tmp_path):
        custom_docs = tmp_path / "my_docs"
        custom_embed = tmp_path / "my_embed"
        args = argparse.Namespace(
            docs_dir=str(custom_docs),
            embed_dir=str(custom_embed),
        )
        cmd_init(args)
        assert custom_docs.is_dir()
        assert custom_embed.is_dir()


# ---------------------------------------------------------------------------
# cmd_embed — config priority
# ---------------------------------------------------------------------------


class TestCmdEmbed:
    """Tests for the embed subcommand; embed_all_pdfs is always mocked."""

    _PATCH = "vlmembed.embed.embed_all_pdfs"

    def _run(self, args: argparse.Namespace, env: dict | None = None):
        env = env or {}
        with patch(self._PATCH, return_value=[]) as mock_fn:
            with patch.dict(os.environ, env, clear=False):
                with patch("dotenv.load_dotenv"):
                    result = cmd_embed(args)
        return result, mock_fn

    def test_returns_zero(self, tmp_path):
        args = _ns(docs_dir=str(tmp_path / "docs"), embed_dir=str(tmp_path / "emb"))
        rc, _ = self._run(args)
        assert rc == 0

    def test_calls_embed_all_pdfs_once(self, tmp_path):
        args = _ns()
        _, mock_fn = self._run(args)
        mock_fn.assert_called_once()

    def test_cli_api_key_passed(self, tmp_path):
        args = _ns(api_key="mykey")
        _, mock_fn = self._run(args)
        _, kwargs = mock_fn.call_args
        assert kwargs["api_key"] == "mykey"

    def test_api_key_none_when_not_provided(self):
        """CLI passes None to embed_all_pdfs when no key; it handles env fallback."""
        args = _ns(api_key=None)
        _, mock_fn = self._run(args)
        _, kwargs = mock_fn.call_args
        assert kwargs["api_key"] is None

    def test_cli_model_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("VLMEMBED_MODEL", "env-model")
        args = _ns(model="cli-model")
        _, mock_fn = self._run(args, env={"VLMEMBED_MODEL": "env-model"})
        _, kwargs = mock_fn.call_args
        assert kwargs["model"] == "cli-model"

    def test_env_model_used_when_no_cli_arg(self):
        args = _ns(model=None)
        _, mock_fn = self._run(args, env={"VLMEMBED_MODEL": "env-model"})
        _, kwargs = mock_fn.call_args
        assert kwargs["model"] == "env-model"

    def test_default_model_when_neither_set(self):
        args = _ns(model=None)
        env = {k: "" for k in ["VLMEMBED_MODEL"]}
        _, mock_fn = self._run(args, env=env)
        _, kwargs = mock_fn.call_args
        assert kwargs["model"] == DEFAULT_MODEL

    def test_cli_dpi_wins_over_env(self):
        args = _ns(dpi=150)
        _, mock_fn = self._run(args, env={"VLMEMBED_DPI": "300"})
        _, kwargs = mock_fn.call_args
        assert kwargs["dpi"] == 150

    def test_env_dpi_used_when_no_cli_arg(self):
        args = _ns(dpi=None)
        _, mock_fn = self._run(args, env={"VLMEMBED_DPI": "100"})
        _, kwargs = mock_fn.call_args
        assert kwargs["dpi"] == 100

    def test_default_dpi_when_neither_set(self):
        args = _ns(dpi=None)
        env = {"VLMEMBED_DPI": ""}
        _, mock_fn = self._run(args, env=env)
        _, kwargs = mock_fn.call_args
        assert kwargs["dpi"] == DEFAULT_DPI

    def test_cli_image_format_wins(self):
        args = _ns(image_format="jpeg")
        _, mock_fn = self._run(args, env={"VLMEMBED_IMAGE_FORMAT": "png"})
        _, kwargs = mock_fn.call_args
        assert kwargs["image_format"] == "jpeg"

    def test_env_image_format_used(self):
        args = _ns(image_format=None)
        _, mock_fn = self._run(args, env={"VLMEMBED_IMAGE_FORMAT": "jpeg"})
        _, kwargs = mock_fn.call_args
        assert kwargs["image_format"] == "jpeg"

    def test_default_image_format_fallback(self):
        args = _ns(image_format=None)
        env = {"VLMEMBED_IMAGE_FORMAT": ""}
        _, mock_fn = self._run(args, env=env)
        _, kwargs = mock_fn.call_args
        assert kwargs["image_format"] == DEFAULT_IMAGE_FORMAT

    def test_cli_dimensions_wins(self):
        args = _ns(dimensions=512)
        _, mock_fn = self._run(args, env={"VLMEMBED_DIMENSIONS": "1024"})
        _, kwargs = mock_fn.call_args
        assert kwargs["dimensions"] == 512

    def test_env_dimensions_used(self):
        args = _ns(dimensions=None)
        _, mock_fn = self._run(args, env={"VLMEMBED_DIMENSIONS": "768"})
        _, kwargs = mock_fn.call_args
        assert kwargs["dimensions"] == 768

    def test_default_dimensions_fallback(self):
        args = _ns(dimensions=None)
        env = {"VLMEMBED_DIMENSIONS": ""}
        _, mock_fn = self._run(args, env=env)
        _, kwargs = mock_fn.call_args
        assert kwargs["dimensions"] == DEFAULT_DIMENSIONS

    def test_cli_max_workers_wins(self):
        args = _ns(max_workers=2)
        _, mock_fn = self._run(args, env={"VLMEMBED_MAX_WORKERS": "8"})
        _, kwargs = mock_fn.call_args
        assert kwargs["max_workers"] == 2

    def test_env_max_workers_used(self):
        args = _ns(max_workers=None)
        _, mock_fn = self._run(args, env={"VLMEMBED_MAX_WORKERS": "6"})
        _, kwargs = mock_fn.call_args
        assert kwargs["max_workers"] == 6

    def test_default_max_workers_fallback(self):
        args = _ns(max_workers=None)
        env = {"VLMEMBED_MAX_WORKERS": ""}
        _, mock_fn = self._run(args, env=env)
        _, kwargs = mock_fn.call_args
        assert kwargs["max_workers"] == DEFAULT_MAX_WORKERS

    def test_cli_max_retries_wins(self):
        args = _ns(max_retries=1)
        _, mock_fn = self._run(args, env={"VLMEMBED_MAX_RETRIES": "5"})
        _, kwargs = mock_fn.call_args
        assert kwargs["max_retries"] == 1

    def test_env_max_retries_used(self):
        args = _ns(max_retries=None)
        _, mock_fn = self._run(args, env={"VLMEMBED_MAX_RETRIES": "7"})
        _, kwargs = mock_fn.call_args
        assert kwargs["max_retries"] == 7

    def test_default_max_retries_fallback(self):
        args = _ns(max_retries=None)
        env = {"VLMEMBED_MAX_RETRIES": ""}
        _, mock_fn = self._run(args, env=env)
        _, kwargs = mock_fn.call_args
        assert kwargs["max_retries"] == DEFAULT_MAX_RETRIES

    def test_docs_dir_and_embed_dir_forwarded(self, tmp_path):
        args = _ns(
            docs_dir=str(tmp_path / "d"),
            embed_dir=str(tmp_path / "e"),
        )
        _, mock_fn = self._run(args)
        pos, _ = mock_fn.call_args
        assert pos[0] == tmp_path / "d"
        assert pos[1] == tmp_path / "e"


# ---------------------------------------------------------------------------
# cmd_search — config priority
# ---------------------------------------------------------------------------


class TestCmdSearch:
    _PATCH = "vlmembed.search_app.launch_search_app"

    def _run(self, args: argparse.Namespace, env: dict | None = None):
        env = env or {}
        with patch(self._PATCH) as mock_fn:
            with patch.dict(os.environ, env, clear=False):
                with patch("dotenv.load_dotenv"):
                    result = cmd_search(args)
        return result, mock_fn

    def _search_ns(self, **kwargs):
        defaults = {
            "embed_dir": str(DEFAULT_EMBED_DIR),
            "api_key": None,
            "model": None,
            "dimensions": None,
            "port": 7860,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_returns_zero(self):
        args = self._search_ns()
        rc, _ = self._run(args)
        assert rc == 0

    def test_calls_launch_search_app_once(self):
        args = self._search_ns()
        _, mock_fn = self._run(args)
        mock_fn.assert_called_once()

    def test_cli_api_key_forwarded(self):
        args = self._search_ns(api_key="mykey")
        _, mock_fn = self._run(args)
        _, kwargs = mock_fn.call_args
        assert kwargs["api_key"] == "mykey"

    def test_empty_api_key_forwarded_for_env_fallback(self):
        """Empty api_key is forwarded so launch_search_app can pick up env var."""
        args = self._search_ns(api_key=None)
        _, mock_fn = self._run(args)
        _, kwargs = mock_fn.call_args
        assert kwargs["api_key"] == ""

    def test_cli_model_wins_over_env(self):
        args = self._search_ns(model="cli-model")
        _, mock_fn = self._run(args, env={"VLMEMBED_MODEL": "env-model"})
        _, kwargs = mock_fn.call_args
        assert kwargs["model"] == "cli-model"

    def test_env_model_used_when_no_cli_arg(self):
        args = self._search_ns(model=None)
        _, mock_fn = self._run(args, env={"VLMEMBED_MODEL": "env-model"})
        _, kwargs = mock_fn.call_args
        assert kwargs["model"] == "env-model"

    def test_default_model_when_neither_set(self):
        args = self._search_ns(model=None)
        env = {"VLMEMBED_MODEL": ""}
        _, mock_fn = self._run(args, env=env)
        _, kwargs = mock_fn.call_args
        assert kwargs["model"] == DEFAULT_MODEL

    def test_cli_dimensions_wins(self):
        args = self._search_ns(dimensions=512)
        _, mock_fn = self._run(args, env={"VLMEMBED_DIMENSIONS": "1024"})
        _, kwargs = mock_fn.call_args
        assert kwargs["dimensions"] == 512

    def test_env_dimensions_used(self):
        args = self._search_ns(dimensions=None)
        _, mock_fn = self._run(args, env={"VLMEMBED_DIMENSIONS": "768"})
        _, kwargs = mock_fn.call_args
        assert kwargs["dimensions"] == 768

    def test_default_dimensions_fallback(self):
        args = self._search_ns(dimensions=None)
        env = {"VLMEMBED_DIMENSIONS": ""}
        _, mock_fn = self._run(args, env=env)
        _, kwargs = mock_fn.call_args
        assert kwargs["dimensions"] == DEFAULT_DIMENSIONS

    def test_port_forwarded(self):
        args = self._search_ns(port=8765)
        _, mock_fn = self._run(args)
        _, kwargs = mock_fn.call_args
        assert kwargs["port"] == 8765

    def test_embed_dir_forwarded(self, tmp_path):
        args = self._search_ns(embed_dir=str(tmp_path / "emb"))
        _, mock_fn = self._run(args)
        _, kwargs = mock_fn.call_args
        assert kwargs["embed_dir"] == tmp_path / "emb"


# ---------------------------------------------------------------------------
# cmd_estimate_cost
# ---------------------------------------------------------------------------


class TestCmdEstimateCost:
    _PATCH = "vlmembed.estimate_cost.estimate_cost"

    def _run(self, args: argparse.Namespace, return_value: dict | None = None):
        default_return = {
            "per_file": {"a.pdf": 3},
            "pages": 3,
            "tokens_per_page": 58593,
            "total_tokens": 175779,
            "estimated_usd": 0.079,
        }
        rv = return_value if return_value is not None else default_return
        with patch(self._PATCH, return_value=rv) as mock_fn:
            result = cmd_estimate_cost(args)
        return result, mock_fn

    def _est_ns(self, **kwargs):
        defaults = {"docs_dir": str(DEFAULT_DOCS_DIR), "dpi": None}
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_returns_zero_when_pdfs_found(self):
        args = self._est_ns()
        rc, _ = self._run(args)
        assert rc == 0

    def test_returns_zero_when_no_pdfs(self):
        args = self._est_ns()
        rv = {"per_file": {}, "pages": 0, "tokens_per_page": 58593, "total_tokens": 0, "estimated_usd": 0.0}
        rc, _ = self._run(args, return_value=rv)
        assert rc == 0

    def test_calls_estimate_cost_once(self):
        args = self._est_ns()
        _, mock_fn = self._run(args)
        mock_fn.assert_called_once()

    def test_docs_dir_forwarded(self, tmp_path):
        args = self._est_ns(docs_dir=str(tmp_path / "custom_docs"))
        _, mock_fn = self._run(args)
        _, kwargs = mock_fn.call_args
        assert kwargs["docs_dir"] == tmp_path / "custom_docs"

    def test_default_dpi_used_when_none(self):
        args = self._est_ns(dpi=None)
        _, mock_fn = self._run(args)
        _, kwargs = mock_fn.call_args
        assert kwargs["dpi"] == DEFAULT_DPI

    def test_cli_dpi_forwarded(self):
        args = self._est_ns(dpi=300)
        _, mock_fn = self._run(args)
        _, kwargs = mock_fn.call_args
        assert kwargs["dpi"] == 300


# ---------------------------------------------------------------------------
# main() — subcommand routing
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_args_triggers_interactive_menu(self):
        with patch("vlmembed.cli._interactive_menu", return_value=0) as mock_menu:
            rc = main([])
        mock_menu.assert_called_once()
        assert rc == 0

    def test_init_routed_correctly(self, tmp_path):
        with patch("vlmembed.cli.cmd_init", return_value=0) as mock_cmd:
            rc = main(["init"])
        mock_cmd.assert_called_once()
        assert rc == 0

    def test_embed_routed_correctly(self):
        with patch("vlmembed.cli.cmd_embed", return_value=0) as mock_cmd:
            rc = main(["embed"])
        mock_cmd.assert_called_once()
        assert rc == 0

    def test_search_routed_correctly(self):
        with patch("vlmembed.cli.cmd_search", return_value=0) as mock_cmd:
            rc = main(["search"])
        mock_cmd.assert_called_once()
        assert rc == 0

    def test_estimate_cost_routed_correctly(self):
        with patch("vlmembed.cli.cmd_estimate_cost", return_value=0) as mock_cmd:
            rc = main(["estimate-cost"])
        mock_cmd.assert_called_once()
        assert rc == 0

    def test_init_passes_parsed_namespace(self, tmp_path):
        with patch("vlmembed.cli.cmd_init", return_value=0) as mock_cmd:
            main(["init", "--docs-dir", str(tmp_path / "d"), "--embed-dir", str(tmp_path / "e")])
        args = mock_cmd.call_args[0][0]
        assert args.docs_dir == str(tmp_path / "d")
        assert args.embed_dir == str(tmp_path / "e")

    def test_embed_passes_parsed_flags(self):
        with patch("vlmembed.cli.cmd_embed", return_value=0) as mock_cmd:
            main(["embed", "--dpi", "150", "--api-key", "k"])
        args = mock_cmd.call_args[0][0]
        assert args.dpi == 150
        assert args.api_key == "k"

    def test_search_passes_port(self):
        with patch("vlmembed.cli.cmd_search", return_value=0) as mock_cmd:
            main(["search", "--port", "9999"])
        args = mock_cmd.call_args[0][0]
        assert args.port == 9999


# ---------------------------------------------------------------------------
# _interactive_menu — EOFError / KeyboardInterrupt exits cleanly
# ---------------------------------------------------------------------------


class TestInteractiveMenu:
    def test_eof_returns_zero(self):
        with patch("builtins.input", side_effect=EOFError):
            assert _interactive_menu() == 0

    def test_keyboard_interrupt_returns_zero(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert _interactive_menu() == 0

    def test_quit_choice_returns_zero(self):
        with patch("builtins.input", return_value="5"):
            assert _interactive_menu() == 0

    def test_q_choice_returns_zero(self):
        with patch("builtins.input", return_value="q"):
            assert _interactive_menu() == 0

    def test_invalid_then_quit(self):
        inputs = iter(["bad_input", "5"])
        with patch("builtins.input", side_effect=inputs):
            assert _interactive_menu() == 0

    def test_choice_1_calls_cmd_init(self):
        inputs = iter(["1", "5"])
        with patch("builtins.input", side_effect=inputs):
            with patch("vlmembed.cli.cmd_init", return_value=0) as mock_init:
                _interactive_menu()
        mock_init.assert_called_once()

    def test_choice_2_calls_cmd_embed(self):
        inputs = iter(["2", "5"])
        with patch("builtins.input", side_effect=inputs):
            with patch("vlmembed.cli.cmd_embed", return_value=0) as mock_embed:
                with patch("dotenv.load_dotenv"):
                    _interactive_menu()
        mock_embed.assert_called_once()

    def test_choice_3_calls_cmd_search(self):
        inputs = iter(["3", "5"])
        with patch("builtins.input", side_effect=inputs):
            with patch("vlmembed.cli.cmd_search", return_value=0) as mock_search:
                with patch("dotenv.load_dotenv"):
                    _interactive_menu()
        mock_search.assert_called_once()

    def test_choice_4_calls_cmd_estimate_cost(self):
        inputs = iter(["4", "5"])
        with patch("builtins.input", side_effect=inputs):
            with patch("vlmembed.cli.cmd_estimate_cost", return_value=0) as mock_est:
                _interactive_menu()
        mock_est.assert_called_once()
