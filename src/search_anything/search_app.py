"""Gradio-based semantic search UI for search_anything.

Launch with :func:`launch_search_app`.  Called from ``cli.py`` in Phase 6.
"""

from __future__ import annotations

import os
from pathlib import Path

import dotenv
import gradio as gr

from search_anything.contract import (
    DEFAULT_DIMENSIONS,
    DEFAULT_EMBED_DIR,
    DEFAULT_MODEL,
    DEFAULT_USE_ENTERPRISE,
    ENV_API_KEY,
    ENV_USE_ENTERPRISE,
)
from search_anything.embed import embed_text_query
from search_anything.store import (
    ensure_store_compatibility,
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


def _detect_modality(metadata: dict, page_id: str = "") -> str:
    """Infer modality from doc path extension, falling back to page-id suffixes."""
    suffix = Path(metadata.get("doc_path", "")).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".heif", ".avif"}:
        return "image"
    if suffix in {".txt", ".md", ".markdown"}:
        return "text"
    if suffix in {".mp3", ".wav"}:
        return "audio"
    if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        return "video"

    if "_txt_" in page_id:
        return "text"
    if "_aud_" in page_id:
        return "audio"
    if "_vid_" in page_id:
        return "video"
    if "_img_" in page_id:
        return "image"

    return "unknown"


def _position_label(metadata: dict, modality: str) -> str:
    """Return a modality-aware ordinal label (page/chunk/segment)."""
    index = int(metadata.get("page_number", 1))
    if modality == "pdf":
        return f"page {index}"
    if modality == "text":
        return f"chunk {index}"
    if modality in {"audio", "video"}:
        return f"segment {index}"
    if modality == "image":
        return "image"
    return f"item {index}"


def _make_placeholder_tile(metadata: dict, modality: str):
    """Create a placeholder preview image for non-image search results."""
    from PIL import Image, ImageDraw

    colors = {
        "text": (47, 92, 171),
        "audio": (176, 100, 36),
        "video": (31, 122, 96),
        "unknown": (82, 88, 95),
    }
    background = colors.get(modality, colors["unknown"])

    filename = Path(metadata.get("doc_path", "unknown")).name
    label = _position_label(metadata, modality)

    image = Image.new("RGB", (900, 640), color=background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 876, 616), outline=(235, 235, 235), width=4)
    draw.text((64, 120), modality.upper(), fill=(255, 255, 255))
    draw.text((64, 240), filename, fill=(245, 245, 245))
    draw.text((64, 320), label, fill=(245, 245, 245))
    draw.text((64, 520), _PLACEHOLDER_TEXT, fill=(230, 230, 230))
    return image


def _make_caption(metadata: dict, distance: float, *, modality: str | None = None) -> str:
    """Format a result caption from metadata and cosine distance."""
    modality = modality or _detect_modality(metadata)
    filename = Path(metadata.get("doc_path", "unknown")).name
    label = _position_label(metadata, modality)
    similarity = 1.0 - distance
    if modality == "pdf":
        return f"{filename} · {label} · score {similarity:.3f}"
    return f"{filename} · {label} · {modality} · score {similarity:.3f}"


def _build_gallery_items(results: list[dict]) -> list[tuple]:
    """Convert search results into (image | None, caption) tuples for the gallery."""
    items: list[tuple] = []
    for r in results:
        meta = r["metadata"]
        modality = _detect_modality(meta, r.get("page_id", ""))
        caption = _make_caption(meta, r["distance"], modality=modality)
        img = _load_image(meta.get("image_cache_path", ""))
        if img is None:
            img = _make_placeholder_tile(meta, modality)
        items.append((img, caption))
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
        api_key: Google API key.
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
        api_key: Google API key used to embed the text query.
        model: Embedding model identifier.
        dimensions: Embedding dimensionality.

    Returns:
        A :class:`gradio.Blocks` instance.
    """
    ensure_store_compatibility(
        embed_dir,
        model=model,
        dimensions=dimensions,
    )
    collection = get_collection(embed_dir)
    _run_search = make_search_fn(
        collection,
        api_key=api_key,
        model=model,
        dimensions=dimensions,
        embed_dir=embed_dir,
    )

    with gr.Blocks(title="search_anything — PDF Semantic Search") as demo:
        gr.Markdown("# PDF Semantic Search\nEnter a text query to find the most relevant PDF pages.")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=320):
                query_box = gr.Textbox(
                    label="Query",
                    placeholder="Describe what you're looking for…",
                    lines=3,
                )
                n_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Top N results",
                )
                search_btn = gr.Button("Search", variant="primary")

            with gr.Column(scale=3, min_width=640):
                gallery = gr.Gallery(
                    label="Results",
                    columns=2,
                    object_fit="contain",
                    height="78vh",
                    preview=True,
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
        api_key: Google API key.  Falls back to the ``GOOGLE_API_KEY``
            environment variable when empty.
        model: Embedding model identifier.
        dimensions: Embedding dimensionality.
        port: Local port on which to serve the Gradio app.
    """
    dotenv.load_dotenv()
    if DEFAULT_USE_ENTERPRISE:
        os.environ.setdefault(ENV_USE_ENTERPRISE, "True")
    if not api_key:
        api_key = os.environ.get(ENV_API_KEY, "")

    demo = build_search_app(
        embed_dir=embed_dir,
        api_key=api_key,
        model=model,
        dimensions=dimensions,
    )
    demo.launch(server_port=port, inbrowser=True)
