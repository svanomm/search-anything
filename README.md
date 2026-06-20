# search-anything

`search-anything` builds local multimodal embeddings from files under `docs/` and serves a semantic search UI on top of a ChromaDB index.

It uses the Google `gemini-embedding-2` model through `google-genai`, supports recursive ingestion, and stores vectors plus local metadata under `embeddings/`.

## What It Supports

Recursive ingestion scans `docs/` and embeds supported file types:

- PDF (`.pdf`): page-level image embeddings
- Image (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.heic`, `.heif`, `.avif`)
- Text/Markdown (`.txt`, `.md`, `.markdown`): chunked with `chonkie`
- Audio (`.mp3`, `.wav`): beginning/middle/end segment embeddings
- Video (`.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`): beginning/middle/end segment embeddings

## Defaults

- Provider: `google-genai`
- Model: `gemini-embedding-2`
- Dimensions: `3072`
- PDF render DPI: `300`
- PDF page image format: `png`
- Workers: `4`
- Retries per embedding task: `3`
- Enterprise mode env default: `GOOGLE_GENAI_USE_ENTERPRISE=True`

## Install

Requirements:

- Python `3.12+`
- `uv`

Install dependencies:

```bash
uv sync
```

## API Key Setup

Set a Google API key in your environment or `.env` file:

```bash
GOOGLE_API_KEY=your_key_here
```

Optional overrides:

- `GOOGLE_GENAI_USE_ENTERPRISE`
- `SEARCH_MODEL`
- `SEARCH_DPI`
- `SEARCH_IMAGE_FORMAT`
- `SEARCH_DIMENSIONS`
- `SEARCH_MAX_WORKERS`
- `SEARCH_MAX_RETRIES`

Config resolution order is:

1. CLI args
2. Environment variables
3. `.env`
4. Hard-coded defaults

## Quick Start

Initialize workspace directories:

```bash
uv run search init
```

Embed recursively from `docs/`:

```bash
uv run search embed
```

Launch search UI:

```bash
uv run search search
```

Estimate multimodal embedding cost:

```bash
uv run search estimate-cost
```

Reset local store artifacts (DB, query cache, metadata, cached images):

```bash
uv run search reset-store
```

Use `-y` with `embed` or `reset-store` to skip confirmation prompts.

## Commands

Interactive mode:

```bash
uv run search
```

Subcommands:

```bash
uv run search init [--docs-dir] [--embed-dir]
uv run search embed [--docs-dir] [--embed-dir] [--api-key] [--model] [--dpi] [--format] [--dimensions] [--max-workers] [--max-retries] [-y]
uv run search search [--embed-dir] [--api-key] [--model] [--dimensions] [--port]
uv run search estimate-cost [--docs-dir] [--dpi]
uv run search reset-store [--embed-dir] [-y]
```

## Search UI Notes

- Query embeddings use Gemini retrieval prompt formatting.
- Results display image previews when available.
- Non-image modalities render generated placeholder tiles with modality-aware captions.
- Query embeddings are cached on disk (`embeddings/query_cache.json`).

## Store Compatibility Guards

The project persists store metadata (`embeddings/store_meta.json`) and validates:

- provider
- schema version
- dimensions

On mismatch, embedding/search fails fast with a clear reset instruction.

## Development

Run tests:

```bash
uv run pytest -q
```

Run lint:

```bash
uv run ruff check
```

## License

MIT
