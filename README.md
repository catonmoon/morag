# Morag

RAG-система для локальных Markdown-файлов, Confluence и Jira с поддержкой локальных LLM.

## Возможности

- **Гибридный поиск** — sparse + dense векторы с RRF-fusion
- **Локальные LLM** — любой OpenAI-совместимый эндпойнт (Ollama, LM Studio, облако)
- **Умное чанкование** — цепочка сплиттеров по заголовкам, таблицам, семантике; опциональный LLM-чанкер
- **Контекстуализация** — LLM генерирует summary роли каждого чанка в документе; опциональное иерархическое саммари документа (`doc_summary`) с учётом родительских страниц
- **Идемпотентность** — повторная индексация пропускает неизменённые документы
- **Полная синхронизация** — документы, удалённые из источника, автоматически удаляются из базы вместе с чанками
- **Автоматическая индексация по расписанию** — daemon-режим (`serve`) с cron-расписанием; повторный запуск при ещё работающей индексации пропускается
- **Параллельная индексация** — настраиваемый параллелизм для одновременной загрузки документов
- **Retry с экспоненциальным backoff** — настраиваемые повторные попытки для LLM и HTTP-эмбеддеров при сетевых сбоях
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
- LLM-сервер — для индексации опционален (нужен только для LLM-чанкинга, контекстуализации и doc_summary), для ретривала обязателен.

### 1а. Настройка Ollama (рекомендуется)

[Ollama](https://ollama.com) — простейший способ запустить локальную LLM. Рекомендуемая модель — **qwen3.5:9b**: поддерживает текст и vision (описание изображений Confluence).

**Установка Ollama:**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Загрузка модели:**

```bash
ollama pull qwen3.5:9b
```

**Запуск сервера:**

```bash
ollama serve   # по умолчанию слушает http://localhost:11434
```

После запуска Ollama готова к работе — `config.example.yml` уже настроен на неё по умолчанию.

> **Vision (изображения Confluence).** Та же модель `qwen3.5:9b` поддерживает multimodal-режим.
> Чтобы включить описание изображений, раскомментируй секцию `llm_vision` в `config.yml`.

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
  # retry:             # повторные попытки при ошибках LLM (по умолчанию: 3 повтора, backoff x2)
  #   max_retries: 3   # 0 — отключить retry
  #   delay: 1.0
  #   backoff: 2.0

indexing:
  chunker: passthrough    # passthrough | llm
  context: noop           # noop | llm
  block_limit: 4000
  dense_embedder:
    model: ai-forever/FRIDA
    # base_url: http://localhost:8082   # HTTP-режим (embedder-frida из docker-compose)
    # dim: 1536                         # обязателен в HTTP-режиме
    # retry: {max_retries: 3}           # retry только в HTTP-режиме
  sparse_embedder:
    model: Alibaba-NLP/gte-multilingual-base
    # base_url: http://localhost:8081   # HTTP-режим (embedder-gte из docker-compose)
    # retry: {max_retries: 3}           # retry только в HTTP-режиме
  concurrency: 1          # параллельных воркеров (3-5 для Confluence с vision LLM)
  schedule: "0 */6 * * *" # cron-расписание для serve-режима (опционально)
  # doc_summary:                  # генерировать иерархическое саммари документа (опционально)
  #   max_tokens: 128             # retry наследуется из llm.retry
```

### 4. Индексация

```bash
# Разовая индексация (идемпотентная: пропускает неизменённые документы, удаляет удалённые)
poetry run python -m cli.main index --config config.yml

# Полная переиндексация: сначала удалить все коллекции, затем проиндексировать заново
poetry run python -m cli.main index --reset --config config.yml

# Daemon-режим: запускает индексацию сразу и затем по расписанию из config.yml (indexing.schedule)
poetry run python -m cli.main serve --config config.yml
```

## Архитектура

### Пайплайн индексации

Документы обрабатываются в **BFS-порядке по иерархии** (`parent_doc_ids`): родители полностью
индексируются до начала обработки потомков. Это гарантирует, что при генерации `doc_summary`
для дочернего документа саммари родителя уже записано в базу. Внутри одного уровня документы
обрабатываются параллельно (до `concurrency` воркеров).

```
Source.get_metadata()              # метаданные всех документов (без контента)
  → Full sync                      # doc_id из source vs Qdrant → каскадно удалить устаревшие
  → BFS-разбивка по уровням        # Level 0: корни; Level 1: их дети; и т.д.

  ┌─ Level 0: корневые документы ──────────────────────────────────────────────┐
  │  [W1..Wn] параллельно:                                                     │
  │    Idempotency check → Source.load_one() → DocumentProcessor chain         │  ← concurrency
  │    (DocSummaryProcessor: parent summary = пусто)                           │    параллельных
  │    → docs.upsert() → Chunker → ContextGenerator → ChunkProcessor chain     │    воркеров
  │    → chunks.upsert()                                                       │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ Level 1: дочерние документы ──────────────────────────────────────────────┐
  │  [W1..Wn] параллельно:                                                     │
  │    ... (DocSummaryProcessor читает parent summary из doc_repo)             │
  └────────────────────────────────────────────────────────────────────────────┘
```

### Удаление устаревших документов (Full sync)

При каждом запуске `pipeline.run(source)` выполняется сверка: множество `doc_id`, возвращённых
`source.get_metadata()`, сравнивается с множеством `doc_id` того же `source_type` в Qdrant.
Документы, которые есть в базе, но отсутствуют у источника, удаляются каскадно — сначала все их
чанки из коллекции `chunks`, затем сам документ из `docs`.

Условия удаления по источникам:

| Источник | `doc_id` | Когда документ удаляется из базы |
|---|---|---|
| `markdown` | относительный путь файла (`docs/guide.md`) | файл удалён с диска или переименован |
| `confluence` | числовой page ID | страница удалена, или убрана из индексируемых spaces/ancestor_ids |
| `jira` | ключ задачи (`PROJ-123`) | ссылка на задачу удалена из всех проиндексированных документов |

**Особенность Jira.** Jira-задачи не хранятся в Jira-источнике напрямую — они обнаруживаются
по ссылкам в уже проиндексированных Markdown и Confluence страницах. Каждая задача хранит
`parent_doc_ids` — список `doc_id` страниц, где она упоминается. Это обеспечивает два независимых
механизма удаления:

**Механизм 1 — каскадное удаление через `parent_doc_ids`** (срабатывает автоматически):
- Confluence/Markdown страница удалена → Confluence/Markdown full sync вызывает `cascade_delete(page_id)`
- `cascade_delete` находит дочерние документы в Qdrant по полю `parent_doc_ids` — без сканирования текстов
- Если задача `PROJ-123` ссылалась только на эту страницу → она удаляется каскадно вместе со своими чанками

**Механизм 2 — Jira full sync** (срабатывает при изменении ссылок в живых страницах):
1. Сначала индексируются Markdown и Confluence (со своим full sync). После этого ссылочная база актуальна.
2. `JiraLinkExtractor` сканирует оставшиеся документы → строит актуальный `issue_map`.
   Если найдены ссылки — `pipeline.run(jira_source)` запускается, и full sync удаляет задачи,
   ссылки на которые исчезли из текстов страниц.

> **Известное ограничение.** Если ссылки на Jira убраны из **всех** документов одновременно,
> `JiraLinkExtractor` вернёт пустой `issue_map` и Jira-индексация будет пропущена целиком —
> full sync для `source_type='jira'` не запустится, старые задачи останутся в базе.
> Обходной путь: `python -m cli.main index --reset` для полной переиндексации.

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
  → fetch_doc_summaries (один батч-запрос к docs по уникальным doc_id)
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

**Саммари документов в контексте.** После реранкинга для каждого уникального документа из
результатов загружается `doc_summary` — один батч-запрос к коллекции `docs`. Если при индексации
было включено `indexing.doc_summary.max_tokens`, саммари добавляется в контекст каждого чанка
этого документа полем `Обзор документа:`. LLM получает высокоуровневое понимание документа
в дополнение к конкретному фрагменту. Если саммари не было сгенерировано — поле отсутствует,
поведение не меняется.

**Сортировка по свежести.** После реранкинга результаты сортируются по `updated_at desc`,
сохраняя последовательный порядок чанков внутри каждого документа (`order asc`). LLM
получает наиболее актуальные источники первыми.

**Ключевые параметры** (env vars в `docker-compose.yml`):

| Параметр | По умолчанию | Описание |
|---|---|---|
| `QDRANT_NUM_RESULTS` | `30` | Число чанков из RRF-поиска |
| `NEIGHBOR_WINDOW` | `1` | Окно соседей (±N по order) |
| `FILTER_MAX_TOKENS` | `50` | Лимит токенов для LLM-реранкера |
| `QDRANT_DOCS_COLLECTION` | `docs` | Коллекция документов для загрузки `doc_summary` |

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
├── cli/main.py                    # CLI: команды index, serve, query
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