# Morag

RAG-система для локальных Markdown-файлов, Confluence и Jira с поддержкой локальных LLM.

## Возможности

- **Гибридный поиск** — sparse + dense векторы с RRF-fusion
- **Локальные LLM** — любой OpenAI-совместимый эндпойнт (Ollama, LM Studio, облако)
- **Умное чанкование** — цепочка сплиттеров по заголовкам, таблицам, семантике; опциональный LLM-чанкер
- **Контекстуализация** — LLM генерирует summary роли каждого чанка в документе
- **Идемпотентность** — повторная индексация пропускает неизменённые документы
- **Параллельная индексация** — настраиваемый параллелизм для одновременной загрузки документов
- **Цитаты и ссылки на источники** — URL Confluence-страниц и Jira-задач сохраняются при индексации; ответ содержит кликабельные ссылки на исходные документы с указанием пути и даты обновления
- **Jira-интеграция** — автоматическое обнаружение ссылок на задачи в документах и их индексация в контексте страницы
- **Пайплайн ретривала на базе Open WebUI** — реализован как Open WebUI Pipeline, но не зависит от него напрямую: совместим с любым OpenAI-compatible клиентом через стандартный `/v1/chat/completions`
- **Vision LLM для изображений** — опциональная multimodal модель для описания схем, скриншотов и изображений со страниц Confluence; описания индексируются наравне с текстом
- **Поддержка русского языка** — модели FRIDA и GTE-multilingual

## Стек

| Компонент | Технология                                                                                    |
|---|-----------------------------------------------------------------------------------------------|
| Векторная БД | [Qdrant](https://qdrant.tech)                                                                 |
| Dense embeddings | [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) (1536-dim, Cosine)                |
| Sparse embeddings | [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) |
| LLM | Любой OpenAI-совместимый (Ollama, LM Studio, OpenAI и пр.)                                    |

## Быстрый старт

### 1. Требования

- Python 3.12+
- [Poetry](https://python-poetry.org)
- Запущенный [Qdrant](https://qdrant.tech/documentation/quick-start/) (локально или удалённо)
- LLM-сервер – для индексации опционален (для LLM-чанкинга и контекстуализации), для ретривала обязателен.

### 2. Установка

```bash
git clone https://github.com/catonmoon/morag.git
cd morag
poetry install
```

### 3. Конфигурация

```bash
cp config.example.yml config.yml
```

Отредактируй `config.yml`:

```yaml
sources:
  markdown:
    path: /path/to/your/docs      # путь к директории с MD-файлами

qdrant:
  host: localhost
  port: 6333
  collection_docs: docs
  collection_chunks: chunks

llm:
  base_url: http://localhost:11434/v1   # Ollama / LM Studio / OpenAI
  model: qwen2.5:7b
  api_key: ollama

indexing:
  chunker: passthrough    # passthrough | llm
  context: noop           # noop | llm
  block_limit: 4000
  dense_embedder:
    model: ai-forever/FRIDA
    # base_url: http://localhost:8082   # HTTP-режим (embedder-frida из docker-compose)
    # dim: 1536                         # обязателен в HTTP-режиме
  sparse_embedder:
    model: Alibaba-NLP/gte-multilingual-base
    # base_url: http://localhost:8081   # HTTP-режим (embedder-gte из docker-compose)
  concurrency: 1          # параллельных воркеров (3-5 для Confluence с vision LLM)
```

### 4. Индексация

```bash
# Обычная индексация (идемпотентная: пропускает неизменённые документы)
poetry run python -m cli.main index --config config.yml

# Полная переиндексация: сначала удалить все коллекции, затем проиндексировать заново
poetry run python -m cli.main index --reset --config config.yml
```

## Архитектура

### Пайплайн индексации

Сначала загружаются метаданные всех документов и выполняется idempotency-проверка.
Затем документы, требующие переиндексации, обрабатываются конкурентно (до `concurrency` одновременно).

```
Source.get_metadata()              # метаданные всех документов (без контента)
  → Idempotency check              # updated_at + size + счётчик чанков; load_one не вызывается
  ┌─ [W1] Source.load_one() ───────────────────────────────────────────────────┐
  │    → DocumentProcessor chain   # обогащение метаданных                     │
  │    → docs.upsert()             # сохранить документ до чанкования          │
  │    → RecursiveSplitter         # разбивка на блоки                         │  ← concurrency
  │    → Chunker                   # LLM или Passthrough                       │    параллельных
  │    → ContextGenerator          # LLM-summary или Noop                      │    воркеров
  │    → ChunkProcessor chain      # dense + sparse векторы, payload           │
  │    → chunks.upsert()           #                                           │
  └────────────────────────────────────────────────────────────────────────────┘
```

### Режимы чанкинга

| `chunker` | `context` | Описание |
|---|---|---|
| `passthrough` | `noop` | Быстро, без LLM. Один блок = один чанк, без summary. |
| `passthrough` | `llm` | Чанки по блокам, но с LLM-контекстом каждого. |
| `llm` | `noop` | LLM делит блок на семантические чанки, без summary. |
| `llm` | `llm` | Максимальное качество: LLM-чанкинг + LLM-контекст. |

### Источники данных

| Источник | `source_type` | Описание |
|---|---|---|
| Markdown-файлы | `markdown` | Рекурсивный скан директории `*.md` |
| Confluence | `confluence` | Страницы через REST API, HTML → Markdown; опциональное описание изображений через vision LLM |
| Jira | `jira` | Задачи, обнаруженные по ссылкам в других документах; описание, комментарии, подзадачи |

**Jira-интеграция** работает в два шага. Сначала индексируются Markdown и Confluence. Затем в уже
проиндексированных текстах ищутся ссылки вида `{jira_url}/browse/PROJ-123` — и найденные задачи
индексируются автоматически.

### Коллекции Qdrant

| Коллекция | Содержимое |
|---|---|
| `docs` | Полный текст + метаданные документов (`id`, `path`, `source_type`, `url`, `updated_at`, `creator`, ...) |
| `chunks` | Чанки: текст, контекст, dense-вектор `full`, sparse-вектор `keywords`, payload (`source_type`, `url`, `creator`, ...) |

Поле `path` — список строк: один документ может иметь несколько путей (например, Jira-задача,
упомянутая на нескольких страницах). Поле `url` содержит абсолютную ссылку на источник.
В ретривале используется для отображения кликабельных ссылок.

### Пайплайн ретривала

Ретривал реализован как Open WebUI Pipeline (`services/pipeline/morag.py`), но **не зависит
от Open WebUI напрямую**: файл подключается как стандартный pipeline и совместим с любым
клиентом через OpenAI-compatible API (`/v1/chat/completions`). Open WebUI — удобный
front-end по умолчанию, но не обязательная зависимость.

```
extract_intent (LLM)
  → hybrid_search (Qdrant RRF: sparse + dense) → N чанков
  → expand_neighbors (±NEIGHBOR_WINDOW соседних чанков по doc_id + order)
  → merge_into_groups (контигуальные соседи объединяются в один merged-чанк)
  → reranker (LLM бинарный фильтр, один вызов на merged-чанк)
  → sort by (updated_at desc, doc_id, order) — свежие документы первыми
  → stream_answer (LLM, SSE) с цитатами и ссылками на источники
```

**Цитаты и ссылки на источники.** Каждый релевантный чанк содержит метаданные из момента
индексации: `path` (иерархический путь документа), `url` (прямая ссылка на Confluence-страницу
или Jira-задачу), `updated_at`. В финальном ответе LLM получает эти данные и формирует
раздел со ссылками на источники. Пользователь видит, из каких документов взят ответ, и
может перейти к первоисточнику одним кликом.

**Слияние соседей перед реранкингом.** После расширения соседями, контигуальные
последовательности чанков одного документа объединяются в один merged-чанк. Метаданные
(path, context, updated_at и др.) берутся из центрального чанка — того у кого наибольший
RRF-score (оригинал из поиска). Соседи имеют score=0.0. Это сокращает число LLM-вызовов
реранкера с ~3×N до ≤N (не более чем исходное число результатов поиска).

**Сортировка по свежести.** После реранкинга результаты сортируются по `updated_at desc`,
сохраняя последовательный порядок чанков внутри каждого документа (`order asc`). LLM
получает наиболее актуальные источники первыми.

**Ключевые параметры** (env vars в `docker-compose.yml`):

| Параметр | По умолчанию | Описание |
|---|---|---|
| `QDRANT_NUM_RESULTS` | `30` | Число чанков из RRF-поиска |
| `NEIGHBOR_WINDOW` | `1` | Окно соседей (±N по order) |
| `FILTER_MAX_TOKENS` | `50` | Лимит токенов для LLM-реранкера |

> **После изменения `services/pipeline/morag.py`** нужно пересобрать образ:
> `docker compose build pipelines && docker compose up -d pipelines`

## Разработка

```bash
# Проверка кода
poetry run ruff check src

# Тесты
poetry run pytest -v --cov --cov-report=html:coverage_html

# Один тест
poetry run pytest tests/indexing/test_embedder.py -v
```

## Docker

```bash
docker compose build
docker compose up -d
docker compose logs | grep -i -E '(warning|error|exception)'
```

## Структура проекта

```
morag/
├── config.example.yml
├── cli/main.py                    # CLI: команда index
├── scripts/
│   └── jira_preview.py            # Превью Jira-задачи в консоль (для отладки)
└── src/morag/
    ├── config.py                  # Pydantic-модели конфига
    ├── sources/
    │   ├── markdown.py            # MarkdownSource
    │   ├── confluence.py          # ConfluenceSource
    │   ├── jira.py                # JiraSource
    │   └── jira_extractor.py      # JiraLinkExtractor — поиск ссылок в документах
    ├── indexing/                  # Пайплайн индексации
    │   ├── splitter.py            # Цепочка сплиттеров
    │   ├── chunker.py             # LLMChunker / PassthroughChunker
    │   ├── context.py             # LLMContextGenerator / NoopContextGenerator
    │   ├── embedder.py            # FridaEmbedder, GteSparseEmbedder, HttpFridaEmbedder, HttpGteSparseEmbedder
    │   ├── processors.py          # ChunkProcessor / DocumentProcessor
    │   └── pipeline.py            # Оркестратор
    ├── storage/                   # Qdrant: коллекции и репозитории
    └── llm/client.py              # OpenAI-совместимый клиент
```