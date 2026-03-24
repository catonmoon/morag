# RAG Challenge Starter Kit

A minimal, universal starter kit that hides API/submission infrastructure and lets you focus on RAG logic and DS experiments.

## What you get

- `arlc/` — a small client + submission/telemetry helpers (AGENTIC RAG Legal Challenge)
- `examples/` — naive end-to-end RAG pipelines (LlamaIndex, LangChain)
- `API.md` — precise API and submission format
- `EVALUATION.md` — scoring methodology with **Grounding-first** emphasis

## Platform model (participant-facing)

This is an **API-only** competition:
- You run your own pipeline locally or on your infrastructure.
- You **do not** host a public service for the organizers.
- The only required interaction is downloading resources and submitting `submission.json`
  **together with a ZIP archive of your code**.

Documents are **published by the organizers** via `GET /documents` as a ZIP of PDFs.
Participants never upload documents to the platform.

**Phase-specific corpora:** each competition phase has its own separate document corpus.
- **Warm-up phase:** ~30 documents, 100 questions.
- **Final phase:** ~300 documents, 900 questions.

The two corpora may partially overlap (some documents appear in both), but questions for each phase must be answered using **only the corpus provided for that phase**. Index and query each corpus independently.

**Document format note:** the corpus consists of PDFs in heterogeneous formats, including digitally-born PDFs and scanned documents. Scanned pages may require OCR. Make sure your ingestion pipeline handles both cases.

Processing time **before** answer generation (ingestion, indexing, offline preprocessing)
is **not** scored. Only per-answer **TTFT** reported in telemetry affects scoring.

If you use external LLM/embedding/search APIs, they must be **publicly accessible**
and reproducible. Local storage/indexing (e.g., vector DB, relational DB) is allowed
as part of your pipeline.

**Language:** Python is not mandatory — any reproducible language or stack is accepted.

**All questions and documents are in English.**

**API keys in code archive:** do not include actual secrets. Provide an `.env.example` listing required variable names; organizers will supply their own credentials when reproducing your solution.

## Quick start

1. Configure env (via `.env` or `export`):
   ```bash
   cp .env.example .env
   # Edit .env — set EVAL_API_KEY, OPENROUTER_API_KEY, etc.
   ```
   Or manually:
   ```bash
   export EVAL_API_KEY="your-api-key"
   export EVAL_BASE_URL="https://platform.agentic-challenge.ai/api/v1"
   export OPENROUTER_API_KEY="sk-or-v1-xxxxx"
   export OPENAI_API_KEY="sk-xxxxx"  # optional (embeddings)

2. Run an example:
   ```bash
   cd examples/llamaindex
   python naive_rag_llamaindex.py
   ```
3. You get:
   - `submission.json` created automatically
   - `code_archive.zip` created automatically (starter-kit + examples)
   - submission UUID printed to console

## Env vars (`.env` / `arlc.config`)

All examples use `arlc.get_config()` for keys and paths. See `.env.example`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_API_KEY` | — | Required. Evaluation API key from platform |
| `EVAL_BASE_URL` | prod URL | API base URL from platform |
| `OPENROUTER_API_KEY` | — | LLM/embeddings via OpenRouter |
| `OPENAI_API_KEY` | — | Optional. Embeddings via OpenAI directly |
| `SUBMISSION_PATH` | `submission.json` | Output path for submission |
| `CODE_ARCHIVE_PATH` | `code_archive.zip` | Output path for code archive |
| `DOCS_DIR` | `docs_corpus` | Directory for downloaded documents |

## Documentation

- **[Examples](examples/README.md)** — pipelines and configuration (use `from arlc import ...`)
- **[Evaluation System](EVALUATION.md)** — metrics, formulas, weighting, Grounding details
- **[API Documentation](API.md)** — endpoints, request/response, telemetry format
