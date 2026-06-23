"""Multimodal document ingestion and Gemini Embedding 2 embedding helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dotenv
import fitz
from google import genai
from google.genai import types
from tqdm import tqdm

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
    ENV_USE_ENTERPRISE,
    EmbedResult,
    PageMetadata,
    get_doc_images_dir,
)

_IMAGE_MIME_BY_SUFFIX = {
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_AUDIO_MIME_BY_SUFFIX = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
}

_VIDEO_MIME_BY_SUFFIX = {
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}

_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
_PDF_SUFFIXES = {".pdf"}
_SUPPORTED_SUFFIXES = (
    _PDF_SUFFIXES
    | _TEXT_SUFFIXES
    | set(_IMAGE_MIME_BY_SUFFIX)
    | set(_AUDIO_MIME_BY_SUFFIX)
    | set(_VIDEO_MIME_BY_SUFFIX)
)

_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 200
_MEDIA_SEGMENT_COUNT = 3
_VIDEO_SEGMENT_WINDOWS = (("0s", "40s"), ("40s", "80s"), ("80s", "120s"))


@dataclass(frozen=True)
class _PendingEmbedTask:
    page_id: str
    source_path: Path
    doc_hash: str
    kind: Literal["pdf", "image", "text", "audio", "video"]
    ordinal: int
    images_dir: Path | None = None
    text_chunk: str | None = None
    mime_type: str | None = None
    byte_range: tuple[int, int] | None = None
    video_offsets: tuple[str, str] | None = None


def _iter_supported_files(docs_dir: Path) -> list[Path]:
    """Return all supported files under *docs_dir*, recursively."""
    return sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def _resolve_media_mime(path: Path) -> str:
    """Return a best-effort MIME type for supported non-text media files."""
    suffix = path.suffix.lower()
    if suffix in _IMAGE_MIME_BY_SUFFIX:
        return _IMAGE_MIME_BY_SUFFIX[suffix]
    if suffix in _AUDIO_MIME_BY_SUFFIX:
        return _AUDIO_MIME_BY_SUFFIX[suffix]
    if suffix in _VIDEO_MIME_BY_SUFFIX:
        return _VIDEO_MIME_BY_SUFFIX[suffix]
    raise ValueError(f"Unsupported media extension: {suffix}")


def _fallback_char_chunks(text: str, chunk_size: int) -> list[str]:
    """Split *text* into fixed-size chunks as a robust fallback."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _apply_text_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prefix each chunk with trailing context from the previous chunk."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    merged: list[str] = []
    previous_source = ""
    for chunk in chunks:
        current = chunk.strip()
        if not current:
            continue
        if previous_source:
            prefix = previous_source[-overlap:]
            if prefix and not current.startswith(prefix):
                current = f"{prefix}\n{current}"
        merged.append(current)
        previous_source = chunk
    return merged


def _chunk_text_content(text: str) -> list[str]:
    """Chunk text with Chonkie's markdown recursive chunker."""
    clean_text = text.strip()
    if not clean_text:
        return []

    chunks: list[str] = []
    try:
        from chonkie import RecursiveChunker  # noqa: PLC0415

        try:
            chunker = RecursiveChunker.from_recipe(
                name="markdown",
                lang="en",
                tokenizer="character",
                chunk_size=_TEXT_CHUNK_SIZE,
                min_characters_per_chunk=24,
            )
        except Exception:  # noqa: BLE001
            chunker = RecursiveChunker(
                tokenizer="character",
                chunk_size=_TEXT_CHUNK_SIZE,
                min_characters_per_chunk=24,
            )

        raw_chunks = chunker.chunk(clean_text) if hasattr(chunker, "chunk") else chunker(clean_text)
        for raw_chunk in raw_chunks:
            if isinstance(raw_chunk, str):
                value = raw_chunk
            else:
                value = getattr(raw_chunk, "text", str(raw_chunk))
            value = value.strip()
            if value:
                chunks.append(value)
    except Exception:  # noqa: BLE001
        chunks = _fallback_char_chunks(clean_text, _TEXT_CHUNK_SIZE)

    return _apply_text_overlap(chunks, _TEXT_CHUNK_OVERLAP)


def _chunk_text_file(path: Path) -> list[str]:
    """Read and chunk a text or markdown file."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return _chunk_text_content(text)


def _build_audio_byte_windows(total_size: int) -> list[tuple[int, int]]:
    """Return beginning/middle/end byte windows for audio segmentation."""
    if total_size <= 0:
        return []
    if total_size <= _MEDIA_SEGMENT_COUNT:
        return [(0, total_size)]

    window = max(total_size // _MEDIA_SEGMENT_COUNT, 1)
    starts = [
        0,
        max((total_size - window) // 2, 0),
        max(total_size - window, 0),
    ]

    windows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for start in starts:
        end = min(start + window, total_size)
        if end <= start:
            continue
        item = (start, end)
        if item not in seen:
            seen.add(item)
            windows.append(item)
    return windows


def _format_document_text(text: str, title: str | None = None) -> str:
    """Format text input using Gemini retrieval guidance for documents."""
    normalized_title = (title or "").strip() or "none"
    return f"title: {normalized_title} | text: {text}"


def _build_genai_client(api_key: str):
    """Return a configured google-genai client for embedding requests."""
    if DEFAULT_USE_ENTERPRISE:
        os.environ.setdefault(ENV_USE_ENTERPRISE, "True")
    return genai.Client(api_key=api_key)


def _extract_embedding_values(response) -> list[float]:
    """Return the first embedding values list from a google-genai response."""
    return list(response.embeddings[0].values)


def compute_doc_hash(pdf_path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes.

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
        saved_path = images_dir / f"page_{page_idx + 1}.{ext}"
        saved_path.write_bytes(image_bytes)
    else:
        saved_path = Path(f"page_{page_idx + 1}.{ext}")

    return base64_str, saved_path


def embed_image_page(
    base64_str: str,
    *,
    model: str,
    api_key: str,
    dimensions: int,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    mime_type: str | None = None,
) -> list[float]:
    """Embed a single page image using the Google Gemini API.

    Args:
        base64_str: Base64-encoded page image.
        model: Embedding model identifier (e.g. ``"gemini-embedding-2"``).
        api_key: Google API key.
        dimensions: Number of embedding dimensions to request.
        image_format: Image format used to construct the MIME type.

    Returns:
        Embedding vector as a list of floats.
    """
    mime = mime_type or ("image/jpeg" if image_format == "jpeg" else "image/png")
    image_bytes = base64.b64decode(base64_str)
    content = types.Content(
        parts=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime,
            )
        ]
    )
    client = _build_genai_client(api_key)
    response = client.models.embed_content(
        model=model,
        contents=[content],
        config=types.EmbedContentConfig(output_dimensionality=dimensions),
    )
    return _extract_embedding_values(response)


def embed_text_document(
    text: str,
    *,
    title: str,
    model: str,
    api_key: str,
    dimensions: int,
) -> list[float]:
    """Embed document text with Gemini-recommended retrieval formatting."""
    prepared = _format_document_text(text, title=title)
    client = _build_genai_client(api_key)
    response = client.models.embed_content(
        model=model,
        contents=[prepared],
        config=types.EmbedContentConfig(output_dimensionality=dimensions),
    )
    return _extract_embedding_values(response)


def _embed_media_part(
    file_bytes: bytes,
    *,
    mime_type: str,
    model: str,
    api_key: str,
    dimensions: int,
    segment_label: str | None = None,
    video_offsets: tuple[str, str] | None = None,
) -> list[float]:
    """Embed a non-text media part (image/audio/video)."""
    media_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    if video_offsets is not None:
        media_part.video_metadata = types.VideoMetadata(
            start_offset=video_offsets[0],
            end_offset=video_offsets[1],
            fps=1.0,
        )

    parts = [media_part]
    if segment_label:
        parts.insert(0, types.Part.from_text(text=f"segment: {segment_label}"))

    content = types.Content(parts=parts)
    client = _build_genai_client(api_key)
    response = client.models.embed_content(
        model=model,
        contents=[content],
        config=types.EmbedContentConfig(output_dimensionality=dimensions),
    )
    return _extract_embedding_values(response)


def embed_text_query(
    text: str,
    *,
    model: str,
    api_key: str,
    dimensions: int,
) -> list[float]:
    """Embed a text query using the Google Gemini API.

    Query text is prefixed with the retrieval task instruction recommended
    for Gemini Embedding 2 search use-cases.

    Args:
        text: Query string to embed.
        model: Embedding model identifier.
        api_key: Google API key.
        dimensions: Number of embedding dimensions to request.

    Returns:
        Embedding vector as a list of floats.
    """
    prepared = f"task: search result | query: {text}"
    client = _build_genai_client(api_key)
    response = client.models.embed_content(
        model=model,
        contents=[prepared],
        config=types.EmbedContentConfig(output_dimensionality=dimensions),
    )
    return _extract_embedding_values(response)


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
    """Embed all unprocessed supported files in *docs_dir* into ChromaDB.

    Supported types are discovered recursively under *docs_dir*:
    PDF, image, text/markdown, audio, and video. Existing vectors are skipped
    by stable IDs so the ingestion remains incremental across repeated runs.

    Args:
        docs_dir: Directory containing source PDF files.
        embed_dir: Root directory for embedding artefacts (image cache + DB).
        api_key: Google API key; overrides the ``GOOGLE_API_KEY``
            environment variable and any ``.env`` file.
        model: Embedding model identifier.
        dpi: PDF page render resolution in dots per inch.
        image_format: Image format for PDF rendering (``"png"`` or ``"jpeg"``).
        dimensions: Number of embedding dimensions to request.
        max_workers: Thread pool size for parallel embedding.
        max_retries: Number of attempts per task before raising an error.

    Returns:
        List of :class:`~search_anything.contract.EmbedResult` dicts for newly
        embedded tasks (already-present tasks are not included).

    Raises:
        ValueError: If *api_key* cannot be resolved or *max_workers* < 1.
        FileNotFoundError: If *docs_dir* does not exist.
        RuntimeError: If any task fails after all retries are exhausted.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    dotenv.load_dotenv()
    resolved_key = (api_key or os.environ.get(ENV_API_KEY, "")).strip()
    if not resolved_key:
        raise ValueError(
            f"{ENV_API_KEY} is required. Set it in your environment, "
            f"pass --api-key, or add {ENV_API_KEY}=<key> to a .env file."
        )

    # Lazy import so embed.py stays testable without a full store implementation.
    from search_anything import store as _store  # noqa: PLC0415

    docs_dir = Path(docs_dir)
    embed_dir = Path(embed_dir)

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    source_files = _iter_supported_files(docs_dir)
    if not source_files:
        return []

    settings_hash = compute_settings_hash(
        model=model,
        dpi=dpi,
        image_format=image_format,
        dimensions=dimensions,
    )

    _store.ensure_store_compatibility(
        embed_dir,
        model=model,
        dimensions=dimensions,
    )
    collection = _store.get_collection(embed_dir)

    # Build all pending tasks for recursive multimodal ingestion.
    pending: list[_PendingEmbedTask] = []
    for source_path in source_files:
        suffix = source_path.suffix.lower()
        doc_hash = compute_doc_hash(source_path)

        if suffix in _PDF_SUFFIXES:
            images_dir = get_doc_images_dir(embed_dir, doc_hash)
            with fitz.open(source_path) as doc:
                page_count = len(doc)
            for page_idx in range(page_count):
                page_id = f"{doc_hash}_{page_idx}"
                if not _store.page_exists(collection, page_id):
                    pending.append(
                        _PendingEmbedTask(
                            page_id=page_id,
                            source_path=source_path,
                            doc_hash=doc_hash,
                            kind="pdf",
                            ordinal=page_idx,
                            images_dir=images_dir,
                        )
                    )
            continue

        if suffix in _TEXT_SUFFIXES:
            for idx, chunk in enumerate(_chunk_text_file(source_path)):
                page_id = f"{doc_hash}_txt_{idx}"
                if not _store.page_exists(collection, page_id):
                    pending.append(
                        _PendingEmbedTask(
                            page_id=page_id,
                            source_path=source_path,
                            doc_hash=doc_hash,
                            kind="text",
                            ordinal=idx,
                            text_chunk=chunk,
                        )
                    )
            continue

        if suffix in _IMAGE_MIME_BY_SUFFIX:
            page_id = f"{doc_hash}_img_0"
            if not _store.page_exists(collection, page_id):
                pending.append(
                    _PendingEmbedTask(
                        page_id=page_id,
                        source_path=source_path,
                        doc_hash=doc_hash,
                        kind="image",
                        ordinal=0,
                        mime_type=_resolve_media_mime(source_path),
                    )
                )
            continue

        if suffix in _AUDIO_MIME_BY_SUFFIX:
            windows = _build_audio_byte_windows(source_path.stat().st_size)
            for idx, byte_range in enumerate(windows):
                page_id = f"{doc_hash}_aud_{idx}"
                if not _store.page_exists(collection, page_id):
                    pending.append(
                        _PendingEmbedTask(
                            page_id=page_id,
                            source_path=source_path,
                            doc_hash=doc_hash,
                            kind="audio",
                            ordinal=idx,
                            mime_type=_resolve_media_mime(source_path),
                            byte_range=byte_range,
                        )
                    )
            continue

        if suffix in _VIDEO_MIME_BY_SUFFIX:
            for idx, offsets in enumerate(_VIDEO_SEGMENT_WINDOWS):
                page_id = f"{doc_hash}_vid_{idx}"
                if not _store.page_exists(collection, page_id):
                    pending.append(
                        _PendingEmbedTask(
                            page_id=page_id,
                            source_path=source_path,
                            doc_hash=doc_hash,
                            kind="video",
                            ordinal=idx,
                            mime_type=_resolve_media_mime(source_path),
                            video_offsets=offsets,
                        )
                    )

    if not pending:
        return []

    def _embed_task(task: _PendingEmbedTask) -> EmbedResult:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                image_cache_path = ""
                if task.kind == "pdf":
                    base64_str, image_path = render_page_image(
                        task.source_path,
                        task.ordinal,
                        dpi,
                        image_format,
                        task.images_dir,
                    )
                    embedding = embed_image_page(
                        base64_str,
                        model=model,
                        api_key=resolved_key,
                        dimensions=dimensions,
                        image_format=image_format,
                    )
                    image_cache_path = str(image_path)
                elif task.kind == "image":
                    base64_str = base64.b64encode(task.source_path.read_bytes()).decode(
                        "utf-8"
                    )
                    embedding = embed_image_page(
                        base64_str,
                        model=model,
                        api_key=resolved_key,
                        dimensions=dimensions,
                        image_format=image_format,
                        mime_type=task.mime_type,
                    )
                    image_cache_path = str(task.source_path)
                elif task.kind == "text":
                    embedding = embed_text_document(
                        task.text_chunk or "",
                        title=task.source_path.name,
                        model=model,
                        api_key=resolved_key,
                        dimensions=dimensions,
                    )
                elif task.kind == "audio":
                    file_bytes = task.source_path.read_bytes()
                    if task.byte_range is not None:
                        start, end = task.byte_range
                        segment_bytes = file_bytes[start:end]
                    else:
                        segment_bytes = file_bytes
                    if not segment_bytes:
                        segment_bytes = file_bytes
                    label = ("beginning", "middle", "end")[
                        min(task.ordinal, _MEDIA_SEGMENT_COUNT - 1)
                    ]
                    embedding = _embed_media_part(
                        segment_bytes,
                        mime_type=task.mime_type or "audio/wav",
                        model=model,
                        api_key=resolved_key,
                        dimensions=dimensions,
                        segment_label=label,
                    )
                elif task.kind == "video":
                    label = ("beginning", "middle", "end")[
                        min(task.ordinal, _MEDIA_SEGMENT_COUNT - 1)
                    ]
                    embedding = _embed_media_part(
                        task.source_path.read_bytes(),
                        mime_type=task.mime_type or "video/mp4",
                        model=model,
                        api_key=resolved_key,
                        dimensions=dimensions,
                        segment_label=label,
                        video_offsets=task.video_offsets,
                    )
                else:
                    raise ValueError(f"Unsupported task kind: {task.kind}")

                metadata: PageMetadata = {
                    "doc_path": str(task.source_path),
                    "page_number": task.ordinal + 1,
                    "doc_hash": task.doc_hash,
                    "settings_hash": settings_hash,
                    "image_cache_path": image_cache_path,
                }
                return EmbedResult(
                    page_id=task.page_id,
                    embedding=embedding,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries - 1:
                    continue
        raise RuntimeError(
            f"Task '{task.page_id}' from '{task.source_path.name}' failed after "
            f"{max_retries} attempt(s)"
        ) from last_exc

    results: list[EmbedResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_embed_task, task): task for task in pending}
        with tqdm(
            total=len(pending), desc="Embedding tasks", leave=False
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
