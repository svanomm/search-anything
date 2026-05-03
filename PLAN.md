## Plan: vlm-embed — Multimodal PDF Embedding Project

**TL;DR**: Sister project to `vlmocr`. Mirrors its architecture exactly (Python 3.12+, uv/pyproject.toml, ANSI TUI, dotenv, tqdm, PyMuPDF, OpenRouter via `requests`) but embeds each PDF page image into a local **ChromaDB** vector store instead of doing OCR. Default model: `google/gemini-embedding-2-preview` at 3072 dimensions. Provides a **Gradio** web app for visual semantic search.

---

**Steps**

### Phase 1 — Scaffolding *(all parallel)*
1. Create `pyproject.toml` — hatchling build, uv runner, entry point `vlmembed = "vlmembed.cli:main"`, all deps
2. Create `src/vlmembed/__init__.py`
3. Create `contract.py` — TypedDicts for page metadata, path/env constants (`DEFAULT_DOCS_DIR = "docs/"`, `DEFAULT_EMBED_DIR = "embeddings/"`, env var names like `VLMEMBED_MODEL`)

### Phase 2 — Core Embedding Pipeline *(depends on Phase 1)*
4. Create `embed.py`:
   - `compute_doc_hash(pdf_path)` → SHA256 of file bytes
   - `compute_settings_hash(model, dpi, image_format, dimensions)` → SHA256
   - `render_page_image(pdf_path, page_idx, dpi, image_format, images_dir)` → renders via PyMuPDF `fitz.Matrix`, saves PNG to `embeddings/images/{doc_hash}/page_{idx}.png`, returns `(base64_str, saved_path)`
   - `embed_image_page(base64_str, model, api_key, dimensions)` → POST to `https://openrouter.ai/api/v1/embeddings` via `requests` using multimodal `content` array format (OpenAI SDK doesn't support this image format for embeddings): `"input": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,...}}]}]`
   - `embed_text_query(text, model, api_key, dimensions)` → same endpoint, `"input": "query string"` (cross-modal: Gemini Embedding 2 maps text+images to unified space)
   - `embed_all_pdfs(docs_dir, embed_dir, ...)` → orchestrates with `ThreadPoolExecutor` + `tqdm`, skips already-embedded pages via ChromaDB ID check

### Phase 3 — Vector Store *(depends on Phase 1)*
5. Create `store.py`:
   - `get_collection(embed_dir)` → opens/creates persistent ChromaDB at `embeddings/db/`, collection `"pdf_pages"` with `hnsw:space = "cosine"`
   - `page_exists(collection, page_id)` → `collection.get(ids=[page_id])` check
   - `upsert_page(collection, page_id, embedding, metadata)` — metadata: `doc_path`, `page_number`, `doc_hash`, `settings_hash`, `image_cache_path`
   - `search(collection, query_embedding, n_results)` → returns list of metadata + distances

### Phase 4 — Search App *(depends on Phase 2 + 3)*
6. Create `search_app.py` — Gradio `Blocks` UI:
   - Text box for query, slider for top-N results, Search button
   - On search: embed query text → query ChromaDB → load images from `image_cache_path` → display as gallery with captions (filename, page number, similarity score)
   - Launched via `gr.Blocks().launch(server_port=port)`

### Phase 5 — Cost Estimation *(parallel with Phase 4)*
7. Create `estimate_cost.py` — count total pages in `docs/`, estimate ≈ $0.45/M image tokens with DPI-based pixel estimate (with disclaimer)

### Phase 6 — CLI TUI *(depends on all above)*
8. Create `cli.py` — exact same ANSI TUI style as vlmocr:
   - `uv run vlmembed` → interactive menu (Init / Embed / Search / Estimate Cost / Quit)
   - `uv run vlmembed init` — create `docs/` + `embeddings/` structure
   - `uv run vlmembed embed [--docs-dir] [--embed-dir] [--api-key] [--model] [--dpi] [--format] [--dimensions] [--max-workers] [--max-retries]`
   - `uv run vlmembed search [--embed-dir] [--api-key] [--model] [--dimensions] [--port]`
   - `uv run vlmembed estimate-cost [--docs-dir]`
   - Config priority: CLI args → env vars (`VLMEMBED_*`) → `.env` → hard-coded defaults

---

**Relevant files**
- vlmocr `src/vlmocr/ocr.py` — template for `embed.py` (ThreadPoolExecutor + tqdm + retry pattern, PyMuPDF rendering, base64 encoding)
- vlmocr `src/vlmocr/cli.py` — template for `cli.py` (ANSI TUI, argparse subcommands, config priority chain)
- vlmocr `src/vlmocr/contract.py` — template for `contract.py` (TypedDict style, frozen dataclasses)

**Dependencies**
- `requests>=2.32.0` — multimodal embeddings API (image content array format not supported by OpenAI SDK)
- `pymupdf>=1.27.1` — PDF rendering
- `chromadb>=0.5.0` — local vector store
- `gradio>=5.0.0` — search UI
- `tqdm>=4.67.2` — progress bars
- `python-dotenv>=1.0.0` — `.env` loading

**Env vars**: `OPENROUTER_API_KEY` (required) + `VLMEMBED_MODEL`, `VLMEMBED_DPI` (200), `VLMEMBED_IMAGE_FORMAT` (png), `VLMEMBED_DIMENSIONS` (3072), `VLMEMBED_MAX_WORKERS` (4), `VLMEMBED_MAX_RETRIES` (3)

---

**Verification**
1. `uv run vlmembed init` → creates `docs/` and `embeddings/` dirs cleanly
2. Drop a PDF in `docs/`, run `uv run vlmembed embed` → progress bar appears, pages embedded, ChromaDB populated
3. Re-run `uv run vlmembed embed` on same PDF → 0 pages re-processed (hash deduplication confirmed)
4. `uv run vlmembed search` → Gradio launches at localhost, text query returns page image gallery with similarity scores
5. Add a second PDF, re-run embed → only new PDF's pages are processed

---

**Decisions**
- ChromaDB for the vector store (cosine similarity built in, persistent, simple API)
- Page images cached to `embeddings/images/{doc_hash}/` for fast display without re-rendering
- 3072 embedding dimensions (highest quality supported by Gemini Embedding 2)
- `requests` used directly for embedding API calls (OpenAI SDK doesn't support the multimodal `content` array input format for the `/embeddings` endpoint)
- Scope excludes: Gradio auth, multi-collection support, clustering/analysis tooling
