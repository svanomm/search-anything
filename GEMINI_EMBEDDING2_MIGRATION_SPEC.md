# Gemini Embedding 2 Migration Spec

This document locks implementation details for migrating this project from OpenRouter to Google Gemini Embedding 2.

## Scope and migration policy

- Provider: Google Gemini API directly via google-genai.
- Authentication mode: API key first.
- API key env var: GOOGLE_API_KEY.
- Legacy OpenRouter key compatibility: not supported.
- Ingestion scope: recursive scan under docs directory for supported file types.
- Migration strategy: hard reset existing embeddings during migration and rebuild index.

## SDK and baseline client usage

- Python SDK: google-genai.
- Preferred client pattern for this repo:
  - `genai.Client(api_key=...)` when explicit key is passed.
  - `genai.Client()` is acceptable when key is supplied through environment.
- Embedding call shape:
  - `client.models.embed_content(model="gemini-embedding-2", contents=[...], config=...)`

## Model and embedding settings

- Model id: gemini-embedding-2.
- Default output dimensionality: 3072.
- Optional output dimensionality: 128..3072 via EmbedContentConfig.output_dimensionality.

## Supported modalities

Gemini Embedding 2 accepts interleaved multimodal inputs:

- text
- image
- document (PDF)
- audio
- video

## Input limits

- Total shared context window: 8192 tokens.
- Image:
  - up to 6 images/request.
  - mime: image/jpeg, image/png, image/webp, image/bmp, image/heic, image/heif, image/avif.
- Document:
  - 1 PDF/request, up to 6 pages.
  - recommendation: 1 page for best quality.
- Audio:
  - up to 180 seconds/request.
  - mime: audio/mp3, audio/wav.
- Video:
  - 1 video/request, up to 120 frames.
  - default fps 1.
  - audio extraction optional through EmbedContentConfig.audio_track_extraction.

## Token budgeting guidance

- Approximate token equivalents from docs:
  - image: 258 tokens/image.
  - PDF: 258 tokens/page plus OCR text tokens when document OCR is enabled.
  - audio: 25 tokens/second.
  - video: 66 tokens/frame.
- Inputs beyond 8192 tokens may be truncated by service behavior.

## Retrieval prompt formatting guidance

For retrieval quality with gemini-embedding-2, include task instructions in content text:

- Query pattern: `task: search result | query: <text>`
- Document pattern: `title: <title_or_none> | text: <text>`

`task_type` is not used for gemini-embedding-2.

## Cost assumptions (to be encoded in estimate module)

Use Google multimodal embedding pricing categories:

- Text input: per 1M tokens.
- Image input: per image.
- Audio input: per second.
- Video input: per frame or service-defined mode pricing.
- PDF billed as image-equivalent pages in model documentation.

Implementation note: estimator remains approximate and must clearly label assumptions.

## Validation gates per phase

Each implementation phase must include:

1. targeted tests for changed behavior.
2. full suite run (`uv run pytest -q`) before phase commit.
3. lint pass (`uv run ruff check`) before phase commit.

## Rollout phases

1. lock migration spec and defaults.
2. refactor provider/config contract.
3. swap embedding provider implementation to google-genai.
4. add recursive multimodal ingestion.
5. extend store metadata and schema/version controls.
6. update CLI behavior and options.
7. adapt search UI for non-image modalities.
8. rewrite cost estimator for modality-aware pricing.
9. update tests and docs.
10. final e2e verification.
