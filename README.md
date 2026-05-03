# vlm-embed

`vlm-embed` turns PDFs into page-level multimodal embeddings that you can search semantically.

The pipeline is simple: it scans a `docs/` folder for PDFs, renders each page to an image, sends those images to OpenRouter's embeddings API, and stores the resulting vectors in a local ChromaDB database.

The repo also keeps a cached image for each embedded page, so search results can show the original page preview instead of only a filename or score.

This project is useful when you want to search across PDFs by meaning rather than keyword match, especially when the relevant content may be inside diagrams, figures, slide layouts, tables, or scanned pages that are awkward to search with plain text tools.

Under the hood, `vlm-embed` uses OpenRouter-served multimodal embeddings and persists them in ChromaDB for local retrieval. The default model is `google/gemini-embedding-2-preview`, with page images rendered at 200 DPI and vectors stored at 3072 dimensions.

## Why this project uses multimodal embeddings

Traditional PDF search usually depends on extracted text. That breaks down when a page's meaning is carried by layout or imagery, for example:

- scanned pages with poor text extraction
- charts, figures, and diagrams
- presentation slides with sparse text
- tables where the structure matters as much as the words
- image-heavy documents where captions do not capture the full content

Multimodal embeddings help because the model sees the rendered page image directly and maps it into a shared vector space with text queries. That means you can search with natural language such as:

- `pages comparing quarterly revenue by region`
- `diagram showing service-to-service authentication flow`
- `slide with a timeline of rollout phases`

The query is embedded as text, the PDF pages are embedded as images, and ChromaDB retrieves the closest page vectors.

## What gets stored locally

After you run `vlmembed embed`, the repo writes two kinds of artifacts under `embeddings/`:

- cached page images under `embeddings/images/<doc_hash>/`
- a persistent ChromaDB database under `embeddings/db/`

Each stored page also carries metadata, including:

- the original PDF path
- the zero-based page number
- the PDF content hash
- a settings hash derived from model, DPI, format, and dimensions
- the cached image path for that page

This makes the embeddings reusable across sessions and lets the search UI display page previews directly from disk.

## Incremental behavior

The embedding command is designed for repeat runs.

Each page gets a stable page ID of the form `<doc_hash>_<page_idx>`. If that page ID already exists in the vector store, `vlm-embed` skips it instead of recomputing the embedding. In practice, that means you can add new PDFs to `docs/` and rerun the command without re-embedding everything.

The embed settings are also hashed and stored in metadata. That is useful for inspection, but the current incremental check is based on page ID presence, not on settings changes. If you change model, DPI, dimensions, or image format and want a fully refreshed index, remove the existing `embeddings/` directory first.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- `OPENROUTER_API_KEY` for embedding and search commands

Install the project environment:

```bash
uv sync
```

OpenRouter setup for first-time users:

1. Create an account at `https://openrouter.ai/` if you do not already have one.
2. Create an API key at `https://openrouter.ai/keys`.
3. Put the key in a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your_key_here
```

## Workspace layout

The default workspace layout looks like this:

```text
docs/
  your-files.pdf
embeddings/
  images/
    <doc_hash>/
      page_0.png
      page_1.png
  db/
```

You can create that structure with:

```bash
uv run vlmembed init
```

## Commands

```bash
uv run vlmembed
```

Running the command with no subcommand opens an interactive terminal menu with options to initialize the workspace, embed PDFs, launch search, estimate cost, or quit.

If you prefer direct subcommands, use the commands below.

### Initialize the project structure

```bash
uv run vlmembed init
```

Custom directories:

```bash
uv run vlmembed init --docs-dir my_docs --embed-dir my_embeddings
```

### Embed all PDFs in `docs/`

```bash
uv run vlmembed embed
```

Useful options:

```bash
uv run vlmembed embed \
  --docs-dir docs \
  --embed-dir embeddings \
  --model google/gemini-embedding-2-preview \
  --dpi 200 \
  --format png \
  --dimensions 3072 \
  --max-workers 4 \
  --max-retries 3
```

What this command does:

- finds all `*.pdf` files in the docs directory
- renders each page to an image
- sends each image to OpenRouter's embeddings endpoint
- stores vectors and metadata in ChromaDB
- caches the rendered page images locally

If `OPENROUTER_API_KEY` is not set and no `--api-key` is passed, the command exits with an error.

### Launch the semantic search UI

```bash
uv run vlmembed search
```

By default, the Gradio app launches on `http://localhost:7860`.

Example with custom options:

```bash
uv run vlmembed search --embed-dir embeddings --port 7860
```

The search flow is:

- you enter a natural-language query
- the query text is embedded through the same OpenRouter model
- ChromaDB retrieves the nearest page vectors
- the UI shows page images with filename, page number, and similarity score

If a cached image is missing, the result still appears, but the gallery tile is blank and only the caption is shown.

### Estimate embedding cost

```bash
uv run vlmembed estimate-cost
```

This command counts PDF pages in the docs directory and estimates cost using:

- US Letter page dimensions
- the configured render DPI
- a rough `256 pixels = 1 token` assumption
- an approximate price of `$0.45 / 1M tokens`

You can override the DPI used for the estimate:

```bash
uv run vlmembed estimate-cost --dpi 300
```

The estimate is intentionally approximate. Actual provider pricing and tokenization behavior may differ.

## Defaults and environment overrides

Config resolution follows this order:

- explicit CLI flags
- environment variables
- values loaded from `.env`
- hard-coded defaults

Supported environment overrides:

- `OPENROUTER_API_KEY`
- `VLMEMBED_MODEL`
- `VLMEMBED_DPI`
- `VLMEMBED_IMAGE_FORMAT`
- `VLMEMBED_DIMENSIONS`
- `VLMEMBED_MAX_WORKERS`
- `VLMEMBED_MAX_RETRIES`

Built-in defaults:

- model: `google/gemini-embedding-2-preview`
- DPI: `200`
- image format: `png`
- dimensions: `3072`
- max workers: `4`
- max retries: `3`
- search port: `7860`

## Notes and limitations

- Only top-level `*.pdf` files in the configured docs directory are embedded.
- The project does not extract OCR text; it indexes rendered page images.
- Search quality depends on the chosen embedding model and the visual clarity of the source PDFs.
- Re-running `embed` is incremental for already indexed page IDs, but changing settings does not automatically invalidate old embeddings.
- Supported render formats are `png` and `jpeg`.

## Development

Run the test suite:

```bash
uv run pytest
```

Run lint checks:

```bash
uv run ruff check
```

## License

MIT License.