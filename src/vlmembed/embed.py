"""PDF rendering and OpenRouter-based multimodal page embedding."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
import fitz
import requests
from tqdm import tqdm

from vlmembed.contract import (
    DEFAULT_DIMENSIONS,
    DEFAULT_DPI,
    DEFAULT_DOCS_DIR,
    DEFAULT_EMBED_DIR,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MODEL,
    OPENROUTER_EMBEDDINGS_URL,
    EmbedResult,
    PageMetadata,
    get_doc_images_dir,
)


def compute_doc_hash(pdf_path: Path | str) -> str:
    """Return the SHA-256 hex digest of a PDF file's raw bytes.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()


def compute_settings_hash(
    *,
    model: str,
    dpi: int,
    image_format: str,
    dimensions: int,
) -> str:
    """Return a stable SHA-256 hex digest for the given embedding settings.

    The settings are JSON-serialised with sorted keys so that the hash is
    independent of the order in which keyword arguments are supplied.

    Args:
        model: Embedding model identifier.
        dpi: Page render resolution.
        image_format: Image format used for rendering (``"png"`` or ``"jpeg"``).
        dimensions: Number of embedding dimensions requested.

    Returns:
        64-character lowercase hex string.
    """
    settings: dict[str, str | int] = {
        "dimensions": dimensions,
        "dpi": dpi,
        "image_format": image_format,
        "model": model,
    }
    serialized = json.dumps(
        settings, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def render_page_image(
    pdf_path: Path | str,
    page_idx: int,
    dpi: int = DEFAULT_DPI,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    images_dir: Path | None = None,
) -> tuple[str, Path]:
    """Render a single PDF page to an image, optionally cache it, and return it.

    Args:
        pdf_path: Path to the source PDF file.
        page_idx: Zero-based page index to render.
        dpi: Render resolution in dots per inch.
        image_format: Output image format – ``"png"`` (default) or ``"jpeg"``.
        images_dir: Per-document image cache directory.  When provided the
            rendered image is written to ``images_dir/page_{page_idx}.<ext>``.
            When ``None`` the image is not persisted and the returned path is
            a relative filename with no parent directory.

    Returns:
        ``(base64_str, saved_path)`` where *base64_str* is the base64-encoded
        image bytes and *saved_path* is the ``Path`` where the image was
        written (or would be written if *images_dir* were provided).

    Raises:
        ValueError: If *image_format* is not ``"png"`` or ``"jpeg"``.
        fitz.FileNotFoundError: If *pdf_path* does not exist.
        IndexError: If *page_idx* is out of range for the document.
    """
    if image_format not in {"png", "jpeg"}:
        raise ValueError(
            f"Unsupported image format {image_format!r}. Must be 'png' or 'jpeg'."
        )

    pdf_path = Path(pdf_path)
    ext = "jpg" if image_format == "jpeg" else "png"
    fitz_output = "jpeg" if image_format == "jpeg" else "png"

    with fitz.open(pdf_path) as doc:
        page = doc[page_idx]
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        image_bytes = pix.tobytes(output=fitz_output)

    base64_str = base64.b64encode(image_bytes).decode("utf-8")

    if images_dir is not None:
        images_dir = Path(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        saved_path = images_dir / f"page_{page_idx}.{ext}"
        saved_path.write_bytes(image_bytes)
    else:
        saved_path = Path(f"page_{page_idx}.{ext}")

    return base64_str, saved_path


def embed_image_page(
    base64_str: str,
    *,
    model: str,
    api_key: str,
    dimensions: int,
    image_format: str = DEFAULT_IMAGE_FORMAT,
) -> list[float]:
    """Embed a single page image via the OpenRouter embeddings endpoint.

    Uses the multimodal ``content`` array input format required by
    Gemini Embedding 2 (not supported by the OpenAI SDK).

    Args:
        base64_str: Base64-encoded page image.
        model: Embedding model identifier (e.g. ``"google/gemini-embedding-2-preview"``).
        api_key: OpenRouter API key.
        dimensions: Number of embedding dimensions to request.
        image_format: Image format used to construct the data-URI MIME type.

    Returns:
        Embedding vector as a list of floats.

    Raises:
        requests.HTTPError: On a non-2xx HTTP response.
    """
    mime = "image/jpeg" if image_format == "jpeg" else "image/png"
    payload: dict = {
        "model": model,
        "input": [
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{base64_str}"},
                    }
                ]
            }
        ],
        "dimensions": dimensions,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        OPENROUTER_EMBEDDINGS_URL, json=payload, headers=headers, timeout=60
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def embed_text_query(
    text: str,
    *,
    model: str,
    api_key: str,
    dimensions: int,
) -> list[float]:
    """Embed a text query via the OpenRouter embeddings endpoint.

    Uses the same model as image embeddings for cross-modal retrieval —
    Gemini Embedding 2 maps both text and images to a shared vector space.

    Args:
        text: Query string to embed.
        model: Embedding model identifier.
        api_key: OpenRouter API key.
        dimensions: Number of embedding dimensions to request.

    Returns:
        Embedding vector as a list of floats.

    Raises:
        requests.HTTPError: On a non-2xx HTTP response.
    """
    payload: dict = {
        "model": model,
        "input": text,
        "dimensions": dimensions,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        OPENROUTER_EMBEDDINGS_URL, json=payload, headers=headers, timeout=60
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def embed_all_pdfs(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    embed_dir: Path = DEFAULT_EMBED_DIR,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    dpi: int = DEFAULT_DPI,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    dimensions: int = DEFAULT_DIMENSIONS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[EmbedResult]:
    """Embed all unprocessed PDF pages in *docs_dir* into the ChromaDB store.

    Pages that are already present in the vector store (matched by their
    composite page ID ``{doc_hash}_{page_idx}``) are skipped, enabling
    fast incremental updates when new PDFs are added.

    Args:
        docs_dir: Directory containing source PDF files.
        embed_dir: Root directory for embedding artefacts (image cache + DB).
        api_key: OpenRouter API key; overrides the ``OPENROUTER_API_KEY``
            environment variable and any ``.env`` file.
        model: Embedding model identifier.
        dpi: Page render resolution in dots per inch.
        image_format: Image format for rendering (``"png"`` or ``"jpeg"``).
        dimensions: Number of embedding dimensions to request.
        max_workers: Thread pool size for parallel page embedding.
        max_retries: Number of attempts per page before raising an error.

    Returns:
        List of :class:`~vlmembed.contract.EmbedResult` dicts for newly
        embedded pages (already-present pages are not included).

    Raises:
        ValueError: If *api_key* cannot be resolved or *max_workers* < 1.
        FileNotFoundError: If *docs_dir* does not exist.
        RuntimeError: If any page fails after all retries are exhausted.
    """
    # Lazy import so embed.py stays testable without a full store implementation.
    from vlmembed import store as _store  # noqa: PLC0415

    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    dotenv.load_dotenv()
    resolved_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
    if not resolved_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required. Set it in your environment, "
            "pass --api-key, or add OPENROUTER_API_KEY=<key> to a .env file."
        )

    docs_dir = Path(docs_dir)
    embed_dir = Path(embed_dir)

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        return []

    settings_hash = compute_settings_hash(
        model=model,
        dpi=dpi,
        image_format=image_format,
        dimensions=dimensions,
    )

    collection = _store.get_collection(embed_dir)

    # Build the list of pages that are not yet in the store.
    pending: list[tuple[Path, int, str, str, Path]] = []
    for pdf_path in pdf_files:
        doc_hash = compute_doc_hash(pdf_path)
        images_dir = get_doc_images_dir(embed_dir, doc_hash)
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
        for page_idx in range(page_count):
            page_id = f"{doc_hash}_{page_idx}"
            if not _store.page_exists(collection, page_id):
                pending.append((pdf_path, page_idx, doc_hash, page_id, images_dir))

    if not pending:
        return []

    def _embed_task(
        task: tuple[Path, int, str, str, Path],
    ) -> EmbedResult:
        pdf_path, page_idx, doc_hash, page_id, task_images_dir = task
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                base64_str, image_path = render_page_image(
                    pdf_path, page_idx, dpi, image_format, task_images_dir
                )
                embedding = embed_image_page(
                    base64_str,
                    model=model,
                    api_key=resolved_key,
                    dimensions=dimensions,
                    image_format=image_format,
                )
                metadata: PageMetadata = {
                    "doc_path": str(pdf_path),
                    "page_number": page_idx + 1,
                    "doc_hash": doc_hash,
                    "settings_hash": settings_hash,
                    "image_cache_path": str(image_path),
                }
                return EmbedResult(
                    page_id=page_id,
                    embedding=embedding,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries - 1:
                    continue
        raise RuntimeError(
            f"Page {page_idx} of '{pdf_path.name}' failed after "
            f"{max_retries} attempt(s)"
        ) from last_exc

    results: list[EmbedResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_embed_task, task): task for task in pending}
        with tqdm(
            total=len(pending), desc="Embedding pages", leave=False
        ) as pbar:
            for future in as_completed(futures):
                result = future.result()  # re-raises any RuntimeError from _embed_task
                _store.upsert_page(
                    collection,
                    result["page_id"],
                    result["embedding"],
                    result["metadata"],
                )
                results.append(result)
                pbar.update(1)

    return results
