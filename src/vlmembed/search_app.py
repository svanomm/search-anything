"""Gradio-based semantic search UI for vlmembed.

Launch with :func:`launch_search_app`.  Called from ``cli.py`` in Phase 6.
"""

from __future__ import annotations

import os
from pathlib import Path

import dotenv
import gradio as gr

from vlmembed.contract import (
    DEFAULT_DIMENSIONS,
    DEFAULT_EMBED_DIR,
    DEFAULT_MODEL,
    ENV_API_KEY,
)
from vlmembed.embed import embed_text_query
from vlmembed.store import (
    get_cached_embedding,
    get_collection,
    search,
    set_cached_embedding,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_TEXT = "Image not available"


def _load_image(image_cache_path: str):
    """Return a PIL image from *image_cache_path*, or ``None`` on failure."""
    try:
        from PIL import Image  # PIL is a transitive dep of gradio

        path = Path(image_cache_path)
        if path.is_file():
            return Image.open(path)
    except Exception:
        pass
    return None


def _make_caption(metadata: dict, distance: float) -> str:
    """Format a result caption from metadata and cosine distance."""
    filename = Path(metadata.get("doc_path", "unknown")).name
    page_num = metadata.get("page_number", 0)
    similarity = 1.0 - distance
    return f"{filename} · page {page_num + 1} · score {similarity:.3f}"


def _build_gallery_items(results: list[dict]) -> list[tuple]:
    """Convert search results into (image | None, caption) tuples for the gallery."""
    items: list[tuple] = []
    for r in results:
        meta = r["metadata"]
        caption = _make_caption(meta, r["distance"])
        img = _load_image(meta.get("image_cache_path", ""))
        if img is not None:
            items.append((img, caption))
        else:
            # Gradio gallery accepts a path string; pass None to show a blank tile
            items.append((None, caption))
    return items


# ---------------------------------------------------------------------------
# Search callback factory (testable independently of Gradio)
# ---------------------------------------------------------------------------


def make_search_fn(
    collection,
    *,
    api_key: str,
    model: str,
    dimensions: int,
    embed_dir: Path | None = None,
):
    """Return a callable ``(query, n_results) -> list[tuple]`` for the search UI.

    Factored out of :func:`build_search_app` so it can be unit-tested without
    introspecting Gradio internals.

    When *embed_dir* is provided, query embeddings are cached on disk so that
    repeated identical queries (same text after lowercasing and stripping
    punctuation) do not incur additional API calls.

    Args:
        collection: ChromaDB ``Collection`` object.
        api_key: OpenRouter API key.
        model: Embedding model identifier.
        dimensions: Embedding dimensionality.
        embed_dir: Root embeddings directory used to persist the query cache.
            When ``None`` caching is disabled.

    Returns:
        A function ``(query: str, n_results: int) -> list[tuple]``.
    """

    def _run_search(query: str, n_results: int) -> list[tuple]:
        if not query.strip():
            return []
        count = collection.count()
        if count == 0:
            return []
        safe_n = min(int(n_results), count)

        if embed_dir is not None:
            query_embedding = get_cached_embedding(
                embed_dir, query, model, dimensions
            )
        else:
            query_embedding = None

        if query_embedding is None:
            query_embedding = embed_text_query(
                query,
                model=model,
                api_key=api_key,
                dimensions=dimensions,
            )
            if embed_dir is not None:
                set_cached_embedding(
                    embed_dir, query, model, dimensions, query_embedding
                )

        results = search(collection, query_embedding, n_results=safe_n)
        return _build_gallery_items(results)

    return _run_search


# ---------------------------------------------------------------------------
# Gradio app builder
# ---------------------------------------------------------------------------


def build_search_app(
    embed_dir: Path,
    api_key: str,
    model: str,
    dimensions: int,
) -> gr.Blocks:
    """Build and return the Gradio Blocks demo (does not launch it).

    Args:
        embed_dir: Root embeddings directory containing the ChromaDB database.
        api_key: OpenRouter API key used to embed the text query.
        model: Embedding model identifier.
        dimensions: Embedding dimensionality.

    Returns:
        A :class:`gradio.Blocks` instance.
    """
    collection = get_collection(embed_dir)
    _run_search = make_search_fn(
        collection,
        api_key=api_key,
        model=model,
        dimensions=dimensions,
        embed_dir=embed_dir,
    )

    with gr.Blocks(title="vlmembed — PDF Semantic Search") as demo:
        gr.Markdown("# PDF Semantic Search\nEnter a text query to find the most relevant PDF pages.")

        with gr.Row():
            query_box = gr.Textbox(
                label="Query",
                placeholder="Describe what you're looking for…",
                lines=2,
                scale=4,
            )
            n_slider = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Top N results",
                scale=1,
            )

        search_btn = gr.Button("Search", variant="primary")

        gallery = gr.Gallery(
            label="Results",
            columns=3,
            object_fit="contain",
            height="auto",
        )

        search_btn.click(
            fn=_run_search,
            inputs=[query_box, n_slider],
            outputs=gallery,
        )
        query_box.submit(
            fn=_run_search,
            inputs=[query_box, n_slider],
            outputs=gallery,
        )

    return demo


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def launch_search_app(
    embed_dir: Path = DEFAULT_EMBED_DIR,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    dimensions: int = DEFAULT_DIMENSIONS,
    port: int = 7860,
) -> None:
    """Build and launch the Gradio search UI.

    Config priority: explicit args > env vars > ``.env`` > hard-coded defaults.

    Args:
        embed_dir: Root embeddings directory.
        api_key: OpenRouter API key.  Falls back to the ``OPENROUTER_API_KEY``
            environment variable when empty.
        model: Embedding model identifier.
        dimensions: Embedding dimensionality.
        port: Local port on which to serve the Gradio app.
    """
    dotenv.load_dotenv()
    if not api_key:
        api_key = os.environ.get(ENV_API_KEY, "")

    demo = build_search_app(
        embed_dir=embed_dir,
        api_key=api_key,
        model=model,
        dimensions=dimensions,
    )
    demo.launch(server_port=port)
