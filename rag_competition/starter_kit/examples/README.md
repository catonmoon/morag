# Naive RAG Examples

These examples are intentionally minimal and rely on the `arlc/` layer to hide API and submission boilerplate.

Each script:
1. Downloads questions and documents
2. Builds a vector index
3. Answers all questions with **telemetry** (parsed by answer_type for streaming)
4. Creates and submits `submission.json`

## Available Examples

### 1. Submit (`submit.py`)

Unified script to create example submission (if missing) and submit to the API.

```bash
python submit.py
```

### 2. Telemetry (`telemetry_example.py`)

Demonstrates the exact telemetry calculation methodology (TTFT, retrieval refs, etc.).

### 3. LlamaIndex RAG (`llamaindex/naive_rag_llamaindex.py`)

Complete RAG pipeline using LlamaIndex:
- `EvaluationClient` for API
- `TelemetryTimer` for timing
- Normalized retrieval references (doc_id + page_numbers)

**Files:**
- `llamaindex/naive_rag_llamaindex.py` - Main script
- `llamaindex/requirements_llamaindex.txt` - Dependencies

### 4. LangChain RAG (`langchain/naive_rag_langchain.py`)

Complete RAG pipeline using LangChain:
- `EvaluationClient` for API
- `TelemetryTimer` for timing
- FAISS retrieval and normalized retrieval references

**Files:**
- `langchain/naive_rag_langchain.py` - Main script
- `langchain/requirements_langchain.txt` - Dependencies

## Installation

```bash
# Base dependencies (client, submit)
pip install -r ../requirements.txt

# LlamaIndex example
pip install -r llamaindex/requirements_llamaindex.txt

# LangChain example
pip install -r langchain/requirements_langchain.txt
```

## Setup

```bash
# Required: API keys
export EVAL_API_KEY="your-eval-api-key"
export EVAL_BASE_URL="https://platform.agentic-challenge.ai/api/v1"
export OPENROUTER_API_KEY="sk-or-v1-xxxxx"

# Optional: OpenAI API key for embeddings
export OPENAI_API_KEY="sk-xxxxx"
```

## Prepare Documents

Not needed — scripts download and unpack to `docs_corpus/`.

## Run Examples

### LlamaIndex
```bash
cd llamaindex
python naive_rag_llamaindex.py
```

### LangChain
```bash
cd langchain
python naive_rag_langchain.py
```

Both scripts will:
1. Download questions from API
2. Download and extract documents from API
3. Load PDFs and create vector index
4. Answer all questions with timing metrics
5. Create `submission.json`
6. Create `code_archive.zip` (starter-kit + examples)
7. Submit to evaluation API
8. Print submission UUID

## Example Output

```
Downloading questions...
Loaded 100 questions
Downloading documents...
Extracted documents to docs_corpus
Loading documents...
Loaded 50 documents
Creating index...
100%|██████████| 50/50 [00:10<00:00,  4.8it/s]

Answering questions...
[1/100] q1
[2/100] q2
...
[100/100] q100

Saved submission.json
Using code archive: code_archive.zip
Submitting...
{"uuid":"3fa85f64-5717-4562-b3fc-2c963f66afa6","status":"queued"}
```

## Code Archive

By default, the examples create `code_archive.zip` in the working directory.  
To use a custom archive path, set:
```bash
export CODE_ARCHIVE_PATH="path/to/code_archive.zip"
```

## Configuration

Both examples use:
- **LLM**: `gpt-4o-mini` via OpenRouter
- **Embeddings**: `text-embedding-ada-002` via OpenAI or OpenRouter
- **Chunk size**: 512 tokens
- **Chunk overlap**: 50 tokens
- **Top-k retrieval**: 3 documents

You can modify these parameters in the script source code.
