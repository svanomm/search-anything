"""Command-line interface for search-anything."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import dotenv

from search_anything.contract import (
    DEFAULT_DIMENSIONS,
    DEFAULT_DPI,
    DEFAULT_DOCS_DIR,
    DEFAULT_EMBED_DIR,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MODEL,
    DEFAULT_USE_ENTERPRISE,
    ENV_API_KEY,
    ENV_DIMENSIONS,
    ENV_DPI,
    ENV_IMAGE_FORMAT,
    ENV_MAX_RETRIES,
    ENV_MAX_WORKERS,
    ENV_MODEL,
    ENV_USE_ENTERPRISE,
    ProjectPathStatus,
    get_project_directories,
)

# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------

_R = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"

_BANNER = (
    f"\n{_BOLD}{_CYAN}"
    "╔══════════════════════════════════════╗\n"
    "║      search-anything  v0.1.0         ║\n"
    "║  Multimodal PDF embedding & search   ║\n"
    "╚══════════════════════════════════════╝"
    f"{_R}\n"
)

_MENU_OPTIONS = [
    ("1", "Init          — create docs/ and embeddings/ structure"),
    ("2", "Embed         — recursively process supported files"),
    ("3", "Search        — launch the Gradio semantic search UI"),
    ("4", "Estimate cost — estimate embedding cost for docs/"),
    ("5", "Quit"),
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_int(cli_val: int | None, env_key: str, default: int) -> int:
    """Return *cli_val* if set, else the env var cast to int, else *default*."""
    if cli_val is not None:
        return cli_val
    env_val = os.environ.get(env_key)
    return int(env_val) if env_val else default


def _resolve_str(cli_val: str | None, env_key: str, default: str) -> str:
    """Return *cli_val* if non-empty, else the env var, else *default*."""
    if cli_val:
        return cli_val
    return os.environ.get(env_key) or default


def _ensure_google_enterprise_env() -> None:
    """Ensure the Google enterprise mode env var has a stable default."""
    if DEFAULT_USE_ENTERPRISE:
        os.environ.setdefault(ENV_USE_ENTERPRISE, "True")


def _print_path_status(label: str, path: Path, exists: bool) -> None:
    tick = f"{_GREEN}✓{_R}" if exists else f"{_YELLOW}·{_R}"
    print(f"  {tick} {_DIM}{label}:{_R} {path}")


def _prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    """Prompt for a yes/no answer and return the parsed boolean value."""
    suffix = "Y/n" if default else "y/N"
    while True:
        try:
            answer = input(f"{prompt} [{suffix}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False

        print(f"{_YELLOW}Please answer y or n.{_R}")


def _estimate_pending_embedding_cost(
    *,
    docs_dir: Path,
    embed_dir: Path,
    dpi: int,
) -> dict:
    """Estimate cost for pending PDF pages discovered recursively."""
    import fitz  # noqa: PLC0415

    from search_anything.embed import compute_doc_hash  # noqa: PLC0415
    from search_anything.estimate_cost import estimate_cost_from_page_counts  # noqa: PLC0415
    from search_anything.store import get_collection, page_exists  # noqa: PLC0415

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    pdf_files = sorted(docs_dir.rglob("*.pdf"))
    if not pdf_files:
        return estimate_cost_from_page_counts({}, dpi=dpi)

    collection = get_collection(embed_dir)
    per_file: dict[str, int] = {}

    for pdf_path in pdf_files:
        doc_hash = compute_doc_hash(pdf_path)
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)

        pending_pages = 0
        for page_idx in range(page_count):
            page_id = f"{doc_hash}_{page_idx}"
            if not page_exists(collection, page_id):
                pending_pages += 1

        if pending_pages > 0:
            per_file[pdf_path.name] = pending_pages

    return estimate_cost_from_page_counts(per_file, dpi=dpi)


# ---------------------------------------------------------------------------
# Subcommand: init
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    """Create the standard search-anything workspace directory structure."""
    docs_dir = Path(args.docs_dir)
    embed_dir = Path(args.embed_dir)

    print(f"\n{_BOLD}Initialising project structure…{_R}\n")

    directories = get_project_directories(docs_dir=docs_dir, embed_dir=embed_dir)
    statuses: list[ProjectPathStatus] = []
    for label, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        statuses.append(ProjectPathStatus(label=label, path=path, exists=path.exists()))

    for s in statuses:
        _print_path_status(s.label, s.path, s.exists)

    print(
        f"\n{_GREEN}Done.{_R} Drop supported files into {_BOLD}{docs_dir}{_R} "
        f"then run {_BOLD}search embed{_R}.\n"
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: embed
# ---------------------------------------------------------------------------


def cmd_embed(args: argparse.Namespace) -> int:
    """Embed all unprocessed supported files into the ChromaDB vector store."""
    from search_anything.embed import embed_all_pdfs  # noqa: PLC0415

    dotenv.load_dotenv()
    _ensure_google_enterprise_env()

    api_key = _resolve_str(args.api_key, ENV_API_KEY, "")
    model = _resolve_str(args.model, ENV_MODEL, DEFAULT_MODEL)
    dpi = _resolve_int(args.dpi, ENV_DPI, DEFAULT_DPI)
    image_format = _resolve_str(args.image_format, ENV_IMAGE_FORMAT, DEFAULT_IMAGE_FORMAT)
    dimensions = _resolve_int(args.dimensions, ENV_DIMENSIONS, DEFAULT_DIMENSIONS)
    max_workers = _resolve_int(args.max_workers, ENV_MAX_WORKERS, DEFAULT_MAX_WORKERS)
    max_retries = _resolve_int(args.max_retries, ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES)

    docs_dir = Path(args.docs_dir)
    embed_dir = Path(args.embed_dir)

    print(f"\n{_BOLD}Estimating embedding cost for pending PDF pages…{_R}\n")
    estimate = _estimate_pending_embedding_cost(
        docs_dir=docs_dir,
        embed_dir=embed_dir,
        dpi=dpi,
    )

    if estimate["per_file"]:
        for filename, pages in estimate["per_file"].items():
            print(f"  {_DIM}{filename}:{_R} {pages} pending page(s)")

        print(f"\n  Pending pages:   {_BOLD}{estimate['pages']}{_R}")
        print(f"  Tokens/page:     {_BOLD}{estimate['tokens_per_page']:,}{_R}")
        print(f"  Total tokens:    {_BOLD}{estimate['total_tokens']:,}{_R}")
        print(f"  Estimated cost:  {_BOLD}{_GREEN}${estimate['estimated_usd']:.4f}{_R}")
        print(
            f"\n  {_DIM}Disclaimer: estimate based on ~${0.2}/M tokens at {dpi} DPI "
            f"(US Letter page size). Actual cost may vary.{_R}\n"
        )
    else:
        print(
            f"  {_YELLOW}No pending PDF pages found for cost estimation.{_R}\n"
            f"  {_DIM}Non-PDF files may still produce new embeddings.{_R}\n"
        )

    if not getattr(args, "yes", False) and not _prompt_yes_no(
        "Proceed with embedding supported files?",
        default=False,
    ):
        print(f"\n{_YELLOW}Embedding cancelled.{_R}\n")
        return 0

    print(f"\n{_BOLD}Embedding files…{_R}\n")
    results = embed_all_pdfs(
        docs_dir,
        embed_dir,
        api_key=api_key or None,
        model=model,
        dpi=dpi,
        image_format=image_format,
        dimensions=dimensions,
        max_workers=max_workers,
        max_retries=max_retries,
    )
    n = len(results)
    print(f"\n{_GREEN}Done.{_R} {_BOLD}{n}{_R} item(s) newly embedded.\n")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: search
# ---------------------------------------------------------------------------


def cmd_search(args: argparse.Namespace) -> int:
    """Launch the Gradio semantic search UI."""
    from search_anything.search_app import launch_search_app  # noqa: PLC0415

    # launch_search_app calls dotenv.load_dotenv() internally; load it here
    # too so that env-var fallback for model/dimensions works correctly.
    dotenv.load_dotenv()
    _ensure_google_enterprise_env()

    # api_key: pass as-is (empty string means launch_search_app will use env).
    api_key = args.api_key or ""
    model = _resolve_str(args.model, ENV_MODEL, DEFAULT_MODEL)
    dimensions = _resolve_int(args.dimensions, ENV_DIMENSIONS, DEFAULT_DIMENSIONS)
    embed_dir = Path(args.embed_dir)
    port = args.port

    print(f"\n{_BOLD}Launching search UI at http://localhost:{port}{_R}\n")
    launch_search_app(
        embed_dir=embed_dir,
        api_key=api_key,
        model=model,
        dimensions=dimensions,
        port=port,
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: estimate-cost
# ---------------------------------------------------------------------------


def cmd_estimate_cost(args: argparse.Namespace) -> int:
    """Estimate the cost to embed supported files in the docs directory."""
    from search_anything.estimate_cost import estimate_cost  # noqa: PLC0415

    docs_dir = Path(args.docs_dir)
    dpi = _resolve_int(getattr(args, "dpi", None), ENV_DPI, DEFAULT_DPI)

    print(f"\n{_BOLD}Estimating embedding cost…{_R}\n")
    result = estimate_cost(docs_dir=docs_dir, dpi=dpi)

    modalities = result.get("modalities", {})
    if not any(modalities.values()) and not result["per_file"]:
        print(f"  {_YELLOW}No supported files found in {docs_dir}{_R}\n")
        return 0

    for filename, pages in result["per_file"].items():
        print(f"  {_DIM}[pdf]{_R} {filename}: {pages} page(s)")

    print("\n  Modality unit counts:")
    print(f"    PDFs (pages):  {_BOLD}{modalities.get('pdf_pages', 0)}{_R}")
    print(f"    Images:        {_BOLD}{modalities.get('images', 0)}{_R}")
    print(f"    Text tokens:   {_BOLD}{modalities.get('text_tokens', 0):,}{_R}")
    print(f"    Audio seconds: {_BOLD}{modalities.get('audio_seconds', 0):,.2f}{_R}")
    print(f"    Video frames:  {_BOLD}{modalities.get('video_frames', 0):,}{_R}")

    token_breakdown = result.get("token_breakdown", {})
    per_modality_usd = result.get("per_modality_usd", {})
    print("\n  Token-equivalent breakdown:")
    for modality in ("pdf", "image", "text", "audio", "video"):
        tokens = token_breakdown.get(modality, 0)
        cost = per_modality_usd.get(modality, 0.0)
        print(
            f"    {modality:>5}: {_BOLD}{tokens:,}{_R} tokens (~${cost:.4f})"
        )

    print(f"\n  Total tokens:    {_BOLD}{result['total_tokens']:,}{_R}")
    print(f"  Estimated cost:  {_BOLD}{_GREEN}${result['estimated_usd']:.4f}{_R}")
    print(
        f"\n  {_DIM}Disclaimer: estimate uses midpoint token assumptions and "
        f"~${0.2}/M token-equivalents. Actual provider billing may vary.{_R}\n"
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: reset-store
# ---------------------------------------------------------------------------


def cmd_reset_store(args: argparse.Namespace) -> int:
    """Remove persisted store artifacts so embeddings can be rebuilt cleanly."""
    from search_anything.store import reset_store  # noqa: PLC0415

    embed_dir = Path(args.embed_dir)

    if not getattr(args, "yes", False) and not _prompt_yes_no(
        f"Reset store under {embed_dir}? This will remove DB/cache/image artifacts.",
        default=False,
    ):
        print(f"\n{_YELLOW}Reset cancelled.{_R}\n")
        return 0

    removed = reset_store(embed_dir, remove_images=True)
    if not removed:
        print(f"\n{_YELLOW}No store artifacts found under {embed_dir}.{_R}\n")
        return 0

    print(f"\n{_GREEN}Removed store artifacts:{_R}")
    for path in removed:
        print(f"  {_DIM}- {_R}{path}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------


def _interactive_menu() -> int:
    """Show an interactive TUI menu and dispatch the chosen action."""
    print(_BANNER)
    while True:
        print(f"{_BOLD}What would you like to do?{_R}\n")
        for key, label in _MENU_OPTIONS:
            print(f"  {_CYAN}{key}{_R}  {label}")
        print()
        try:
            choice = input(f"{_BOLD}>{_R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_DIM}Bye.{_R}\n")
            return 0

        if choice == "1":
            ns = argparse.Namespace(
                docs_dir=str(DEFAULT_DOCS_DIR),
                embed_dir=str(DEFAULT_EMBED_DIR),
            )
            cmd_init(ns)
        elif choice == "2":
            ns = argparse.Namespace(
                docs_dir=str(DEFAULT_DOCS_DIR),
                embed_dir=str(DEFAULT_EMBED_DIR),
                api_key=None,
                model=None,
                dpi=None,
                image_format=None,
                dimensions=None,
                max_workers=None,
                max_retries=None,
                yes=False,
            )
            cmd_embed(ns)
        elif choice == "3":
            ns = argparse.Namespace(
                embed_dir=str(DEFAULT_EMBED_DIR),
                api_key=None,
                model=None,
                dimensions=None,
                port=7860,
            )
            cmd_search(ns)
        elif choice == "4":
            ns = argparse.Namespace(
                docs_dir=str(DEFAULT_DOCS_DIR),
                dpi=None,
            )
            cmd_estimate_cost(ns)
        elif choice in {"5", "q", "quit", "exit"}:
            print(f"\n{_DIM}Bye.{_R}\n")
            return 0
        else:
            print(f"\n{_YELLOW}Unknown option {choice!r}. Enter 1–5.{_R}\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search",
        description="Multimodal embedding and semantic search via Google Gemini API and ChromaDB.",
    )
    sub = parser.add_subparsers(dest="subcommand")

    # --- init ---
    p_init = sub.add_parser("init", help="Create the project directory structure.")
    p_init.add_argument(
        "--docs-dir",
        default=str(DEFAULT_DOCS_DIR),
        help=f"Documents directory (default: {DEFAULT_DOCS_DIR}).",
    )
    p_init.add_argument(
        "--embed-dir",
        default=str(DEFAULT_EMBED_DIR),
        help=f"Embeddings directory (default: {DEFAULT_EMBED_DIR}).",
    )

    # --- embed ---
    p_embed = sub.add_parser(
        "embed",
        help="Recursively embed supported files into the vector store.",
    )
    p_embed.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR))
    p_embed.add_argument("--embed-dir", default=str(DEFAULT_EMBED_DIR))
    p_embed.add_argument("--api-key", default=None, help="Google API key.")
    p_embed.add_argument(
        "--model",
        default=None,
        help=f"Embedding model (default: {DEFAULT_MODEL}).",
    )
    p_embed.add_argument(
        "--dpi",
        type=int,
        default=None,
        help=f"Render DPI (default: {DEFAULT_DPI}).",
    )
    p_embed.add_argument(
        "--format",
        dest="image_format",
        default=None,
        help=f"Image format: png or jpeg (default: {DEFAULT_IMAGE_FORMAT}).",
    )
    p_embed.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help=f"Embedding dimensions (default: {DEFAULT_DIMENSIONS}).",
    )
    p_embed.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=f"Thread pool size (default: {DEFAULT_MAX_WORKERS}).",
    )
    p_embed.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help=f"Retries per embedding task (default: {DEFAULT_MAX_RETRIES}).",
    )
    p_embed.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and proceed with embedding.",
    )

    # --- search ---
    p_search = sub.add_parser("search", help="Launch the Gradio semantic search UI.")
    p_search.add_argument("--embed-dir", default=str(DEFAULT_EMBED_DIR))
    p_search.add_argument("--api-key", default=None, help="Google API key.")
    p_search.add_argument(
        "--model",
        default=None,
        help=f"Embedding model (default: {DEFAULT_MODEL}).",
    )
    p_search.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help=f"Embedding dimensions (default: {DEFAULT_DIMENSIONS}).",
    )
    p_search.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port for Gradio (default: 7860).",
    )

    # --- estimate-cost ---
    p_est = sub.add_parser("estimate-cost", help="Estimate embedding cost for PDFs in docs/.")
    p_est.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR))
    p_est.add_argument(
        "--dpi",
        type=int,
        default=None,
        help=f"Render DPI used for token estimation (default: {DEFAULT_DPI}).",
    )

    # --- reset-store ---
    p_reset = sub.add_parser(
        "reset-store",
        help="Delete persisted embedding store artifacts for a clean rebuild.",
    )
    p_reset.add_argument("--embed-dir", default=str(DEFAULT_EMBED_DIR))
    p_reset.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and reset immediately.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* (or ``sys.argv[1:]``) and dispatch to the right subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        return _interactive_menu()
    if args.subcommand == "init":
        return cmd_init(args)
    if args.subcommand == "embed":
        return cmd_embed(args)
    if args.subcommand == "search":
        return cmd_search(args)
    if args.subcommand == "estimate-cost":
        return cmd_estimate_cost(args)
    if args.subcommand == "reset-store":
        return cmd_reset_store(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
