# Morag

RAG-система для локальных Markdown-файлов, Confluence и Jira с поддержкой локальных LLM.

## Возможности

- **Гибридный поиск** — sparse + dense векторы с RRF-fusion
- **Локальные LLM** — любой OpenAI-совместимый эндпойнт (Ollama, LM Studio, облако)
- **Умное чанкование** — семантический чанкер на эмбеддингах, цепочка сплиттеров, опциональный LLM-чанкер
- **Контекстуализация** — LLM-summary для каждого чанка + иерархическое doc_summary
- **Идемпотентность и full sync** — пропуск неизменённых документов, каскадное удаление устаревших
- **Daemon-режим** — cron-расписание, параллельная индексация, retry с backoff
- **Цитаты и ссылки** — URL источников в ответах
- **PDF и Vision LLM** — конвертация PDF через docling-serve, описание изображений
- **Русский язык** — FRIDA + GTE-multilingual

## Стек

| Компонент | Технология |
|---|---|
| Векторная БД | [Qdrant](https://qdrant.tech) |
| Dense embeddings | [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) (1536-dim) |
| Sparse embeddings | [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) |
| LLM | Любой OpenAI-совместимый |

## Быстрый старт

### Требования

- Python 3.12+, [Poetry](https://python-poetry.org)
- [Qdrant](https://qdrant.tech/documentation/quick-start/)
- LLM-сервер (опционален для индексации, обязателен для ретривала)

### Установка

```bash
git clone https://github.com/catonmoon/morag.git
cd morag
poetry install
cp config.example.yml config.yml
# отредактировать config.yml — см. config.example.yml для описания всех опций
```

### Ollama (рекомендуется)

```bash
brew install ollama          # macOS
ollama pull qwen3-coder:30b
ollama serve                 # http://localhost:11434
```

`config.example.yml` уже настроен на Ollama. Для vision (изображения Confluence) — раскомментировать `llm_vision`.

### Индексация

```bash
poetry run python -m cli.main index --config config.yml
poetry run python -m cli.main index --reset --config config.yml  # полная переиндексация
poetry run python -m cli.main serve --config config.yml          # daemon по cron
```

## Архитектура

### Пайплайн индексации

```
Source.get_metadata() → full sync → BFS по parent_doc_ids
  → per level: load_one() → DocumentProcessor → docs.upsert()
    → Chunker → ContextGenerator → ChunkProcessor → chunks.upsert()
```

Режимы чанкинга: `semantic` (default, на эмбеддингах), `passthrough`, `llm`. Контекст: `noop` или `llm`.

### Источники данных

| Источник | `source_type` |
|---|---|
| Markdown-файлы | `markdown` |
| PDF-файлы | `pdf` |
| Confluence | `confluence` |
| Confluence PDF-вложения | `attached_pdf` |
| Jira (по ссылкам в документах) | `attached_jira` |

### Ретривал

Open WebUI Pipeline (`services/pipeline/morag.py`), совместим с любым OpenAI-compatible клиентом.

```
extract_intent → hybrid_search (RRF) → expand_neighbors
  → merge_groups → reranker (LLM) → fetch_doc_summaries → stream_answer
```

## Docker

```bash
docker compose build && docker compose up -d
```

Один `docker-compose.yml`: qdrant, morag-indexer, embedder-frida, embedder-gte, pipelines, open-webui.

## Разработка

```bash
poetry run ruff check src
poetry run pytest -v --cov
```
