# API Documentation

This API is intentionally simple. Use the `arlc/` client to hide boilerplate and keep your code focused on RAG logic.

## Platform model

- The competition is **API-only**. You run your own pipeline and submit results.
- The platform **does not** call your system; it only receives `submission.json`.
- Documents are **published by organizers** via `GET /documents`.

Offline ingestion/indexing time is **not** scored. Only per-answer **TTFT** from telemetry affects the score.

**TTFT measurement:** you measure latency within your own pipeline and report it in the submission telemetry. The platform does not measure it independently. TTFT is the time to the **first token of the final answer returned to the user** — not the first token produced by any intermediate model step (e.g. a query-rewriting model). If you are not using a streaming API, set `ttft_ms` equal to `total_time_ms`.

## Authentication

All requests require an API key:

```http
X-API-Key: your-api-key-here
```

## Base URL

```
https://platform.agentic-challenge.ai/api/v1
```

Environment variables:
```bash
export EVAL_BASE_URL="https://platform.agentic-challenge.ai/api/v1"
export EVAL_API_KEY="your-api-key"
```

---

## Quick usage (arlc)

```python
from arlc import EvaluationClient

client = EvaluationClient.from_env()
questions = client.download_questions()
client.download_documents("docs_corpus")

# ... build submission.json and code archive ...
result = client.submit_submission("submission.json", "code_archive.zip")
print(result)
```

---

## GET /questions

Download competition questions as JSON.

**Response:** list of items with `id`, `question`, `answer_type`.

Notes:
- `id` is a SHA-256 hash of the question text.
- `question` is the question text.
- `answer_type` indicates the expected answer format.

**Status codes:** `200`, `401`, `403`, `404`

---

## GET /documents

Download the document corpus for the current phase as a ZIP archive of PDFs.

**Important:** each phase (warm-up, final) has its own corpus. The warm-up corpus (~30 docs) and the final corpus (~300 docs) are distributed separately and may partially overlap. Index each corpus independently and use only the phase-specific corpus when generating answers for that phase.

**Document formats:** the corpus contains both digitally-born PDFs and scanned documents. Scanned pages may require OCR — ensure your ingestion pipeline handles both.

**Status codes:** `200`, `401`, `403`, `404`

---

## POST /submissions

Submit a `submission.json` **and** a code ZIP archive as multipart form data.

**Headers:**
```http
X-API-Key: your-api-key
```

**Request:** upload file fields:
- `file` (`application/json`) — submission JSON
- `code_archive` (`application/zip`) — ZIP archive with your code (max 25 MB)

Example curl:
```bash
curl -X POST "https://platform.agentic-challenge.ai/api/v1/submissions" \
  -H "X-API-Key: your-api-key" \
  -F "file=@submission.json" \
  -F "code_archive=@code_archive.zip"
```

**Response (202):**
```json
{
  "uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "queued",
  "phase": "warmup",
  "version": "1",
  "created_at": "2026-02-24T12:00:00Z",
  "payload_sha256": "abc123..."
}
```

**Status codes:** `202`, `400`, `401`, `429`

---

## GET /submissions/{uuid}/status

Check evaluation status and metrics.

**Status values:** `queued`, `processing`, `completed`, `error`

**Response (completed):**
```json
{
  "uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "phase": "final",
  "version": "final_v3",
  "status": "completed",
  "review_status": "active",
  "metrics": {
    "deterministic": 0.825,
    "assistant": 0.74,
    "grounding": 0.87,
    "telemetry": 0.95,
    "ttft_ms": 310,
    "ttft_multiplier": 1.0,
    "total_score": 0.783
  }
}
```

---

## Submission JSON format

**File-level fields:**
- `architecture_summary` — optional, max 500 chars
- `answers` — array of per-question answers

**Per answer:**
- `question_id` — must match dataset `id`
- `answer` — string, number, boolean, array of strings, or null (based on `answer_type`)
- `telemetry` — required telemetry object

### Telemetry (required)

```json
{
  "timing": {
    "ttft_ms": 320,
    "tpot_ms": 45,
    "total_time_ms": 1200
  },
  "retrieval": {
    "retrieved_chunk_pages": [
      {
        "doc_id": "443e04bc1a78940b3fcd5438d24b6c5f182a276d354a3108e738b193675de032",
        "page_numbers": [1, 2, 3]
      }
    ]
  },
  "usage": {
    "input_tokens": 512,
    "output_tokens": 128
  },
  "model_name": "gpt-4o-mini"
}
```

**Important telemetry rules:**
- `doc_id` must be the PDF filename (SHA-like string), not a human label.
- `page_numbers` must be **physical PDF page numbers** (1-based: first page of file = 1), not page labels printed on the page itself.
- Include **only pages actually used to generate the answer** — not all pages retrieved during search. Extra pages reduce precision and hurt your grounding score.
- Use **provider-reported token counts** when available; approximate tokenization is acceptable otherwise.
- Missing or malformed telemetry triggers a **telemetry penalty**.
- Unknown `doc_id` values are treated as malformed telemetry.
- The FAQ term “retrieved chunk IDs” refers to this `retrieved_chunk_pages` array.

### Telemetry example (structured multi-page)

```json
{
  "question_id": "cdddeb6a063f29cbea5f10b3dccbd83aa16849e1f3124e223d141d1578efeb0a",
  "answer": "Fursa Consulting",
  "telemetry": {
    "timing": {"ttft_ms": 1180, "tpot_ms": 52, "total_time_ms": 2440},
    "retrieval": {
      "retrieved_chunk_pages": [
        {"doc_id": "443e04bc1a78940b3fcd5438d24b6c5f182a276d354a3108e738b193675de032", "page_numbers": [1, 2]},
        {"doc_id": "443e04bc1a78940b3fcd5438d24b6c5f182a276d354a3108e738b193675de032", "page_numbers": [3, 4]}
      ]
    },
    "usage": {"input_tokens": 1420, "output_tokens": 188},
    "model_name": "participant-case10"
  }
}
```

---

## OpenAPI specification

Full specification: `openapi.yaml`

---

## Submission limits

- **Warm-up / public stage:** 10 submissions total
- **Private stage:** 2 submissions total
