# Morag

RAG-система для локальных Markdown-файлов, Confluence и Jira с поддержкой локальных LLM.

## Возможности

- **Гибридный поиск** — sparse + dense векторы с RRF-fusion
- **Локальные LLM** — любой OpenAI-совместимый эндпойнт (Ollama, LM Studio, облако)
- **Умное чанкование** — структурный hybrid-чанкер (CommonMark AST, магнитные заголовки, per-type oversized стратегии с рекурсией). Опционально: семантический на эмбеддингах, LLM-чанкер
- **Точный подсчёт токенов** — FRIDA tokenizer для чанкинга (embedder-native), TikToken для LLM
- **Адаптивный контекст** — LLM-summary для каждого чанка, размер адаптируется к бюджету embedder (chunk_max_tokens − text − path)
- **Позиционирование** — char_offset каждого чанка в документе, pages для PDF (paged documents)
- **Идемпотентность и full sync** — пропуск неизменённых документов, каскадное удаление устаревших
- **Daemon-режим** — cron-расписание, параллельная индексация, retry с backoff
- **Цитаты и ссылки** — URL источников в ответах
- **PDF и Vision LLM** — конвертация PDF через Vision LLM или docling-serve, описание изображений
- **Русский язык** — FRIDA + GTE-multilingual, razdel для сегментации предложений

## Публикации
- [habr: Юридическое поле экспериментов для RAG](https://habr.com/ru/articles/1014690/)
- [linkded.in: A Legal Proving Ground for RAG Experiments](https://www.linkedin.com/pulse/legal-proving-ground-rag-experiments-ivan-komarov-lwocf/)

## Стек

| Компонент | Технология |
|---|---|
| Векторная БД | [Qdrant](https://qdrant.tech) |
| Dense embeddings | [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) (1536-dim, 512 tok) |
| Sparse embeddings | [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) |
| LLM | Любой OpenAI-совместимый |

## Быстрый старт

### Требования

- Python 3.12+, [Poetry](https://python-poetry.org)
- [Qdrant](https://qdrant.tech/documentation/quick-start/)
- LLM-сервер (опционален для индексации без контекста, обязателен для ретривала)

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
ollama pull qwen3.5:9b
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
  → per level: load_one() → DocTitleProcessor → DocSummaryProcessor → docs.upsert()
    → HybridChunker → ContextGenerator → ChunkProcessors → chunks.upsert()
```

### HybridChunker (default)

Структурный чанкер с тремя стадиями:

1. **Parse blocks** — CommonMark AST разбор на типизированные блоки (heading, paragraph, table, list, fence, diagram)
2. **Greedy fill** — жадное наполнение чанков блоками до max_tokens. Магнитные заголовки (heading всегда в начале чанка). Oversized блоки обрабатываются per-type стратегией
3. **Post-merge** — склейка мелких чанков (< min_tokens) с соседями

Oversized стратегии (настраиваются в конфиге для каждого типа блока):

| Стратегия | Описание |
|---|---|
| `asis` | Оставить как есть (один большой чанк) |
| `split` | Структурное разбиение (предложения / элементы / строки) |
| `embed` | SemanticChunker (embedding-based границы) |
| `transform` | Преобразовать формат + рекурсия (таблица → key-value h4) |
| `llm` | LLM преобразует/разобьёт |

Другие режимы: `semantic` (на эмбеддингах), `passthrough`, `llm`.

### Два токенизатора

- **FRIDA HuggingFace** — для чанкинга (точный подсчёт токенов embedder модели)
- **TikToken** — для LLM (context window, prompt overhead)

Для русского текста FRIDA считает на 43% меньше токенов чем TikToken → чанки точно заполняют embedder capacity.

### Источники данных

| Источник | `source_type` | `paged` |
|---|---|---|
| Markdown-файлы | `markdown` | нет |
| PDF-файлы | `pdf` | да |
| Confluence | `confluence` | нет |
| Confluence PDF-вложения | `attached_pdf` | да |
| Jira (по ссылкам в документах) | `attached_jira` | нет |

Для paged документов (PDF) маркеры `<!-- page:N -->` извлекаются до чанкинга, каждый чанк получает номера страниц.

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

### ADR (Architecture Decision Records)

| ADR | Описание |
|---|---|
| [ADR-0008](docs/adr/0008-hybrid-chunker.md) | HybridChunker как режим чанкинга по умолчанию |
| [ADR-0009](docs/adr/0009-dual-tokenizer.md) | Два токенизатора — FRIDA для чанкинга, TikToken для LLM |

### Исследования

Результаты экспериментов и сравнений — в `experiments/`.
