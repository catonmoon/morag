# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**morag** — RAG-система для локальных файлов в Markdown-формате (и Confluence) с использованием локальной LLM. MVP реализован в папке `old/` и служит референсом для нового проекта.

## Commands

```bash
# Установка зависимостей
poetry install

# Проверка кода
ruff check src

# Запуск тестов
pytest -v --cov --cov-report=html:coverage_html

# Запуск одного теста
pytest tests/path/to/test_file.py::test_name -v

# Запуск сервера
uvicorn app.service:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker compose build
docker compose up -d
docker compose logs | grep -i -E '(warning|error|exception)'
```

## Structure

```
morag/
├── config.example.yml             # шаблон конфига (config.yml в .gitignore)
├── cli/
│   └── main.py                    # точки входа: index <path>, query <question>
│
├── src/morag/
│   ├── config.py                  # Pydantic-модели всего конфига
│   │
│   ├── sources/                   # источники данных
│   │   ├── base.py                # абстрактный Source + модели Document, Chunk
│   │   ├── markdown.py            # сканирование локальных MD-файлов
│   │   └── confluence.py          # (future)
│   │
│   ├── indexing/                  # пайплайн индексации
│   │   ├── token_counter.py       # TokenCounter интерфейс + TiktokenCounter
│   │   ├── splitter.py            # pre-split: BlockSplitter цепочка + RecursiveSplitter + pack_blocks
│   │   ├── chunker.py             # Chunker интерфейс: LLMChunker, PassthroughChunker
│   │   ├── context.py             # ContextGenerator интерфейс: LLMContextGenerator, NoopContextGenerator
│   │   ├── processors.py          # ChunkProcessor и DocumentProcessor цепочки
│   │   └── pipeline.py            # оркестратор: source → docs → chunks → qdrant
│   │
│   ├── storage/                   # работа с Qdrant
│   │   ├── collections.py         # создание коллекций docs и chunks
│   │   └── repository.py          # upsert, delete by filter, scroll
│   │
│   ├── retrieval/                 # поиск и фильтрация
│   │   ├── search.py              # гибридный поиск (sparse + dense, RRF)
│   │   └── reranker.py            # LLM-фильтр нерелевантных чанков
│   │
│   ├── llm/
│   │   └── client.py              # OpenAI-compatible клиент (облако / локальный сервер / localhost)
│   │
│   └── api/
│       └── app.py                 # FastAPI-приложение (опционально)
│
└── tests/
    ├── conftest.py
    ├── fixtures/docs/         # тестовые MD-файлы
    ├── sources/
    ├── indexing/
    ├── storage/
    └── retrieval/
```

## Architecture

### Indexing Pipeline

```
Source.load()
  → Document
  → [Idempotency check]          # сравнение updated_at + счётчик чанков
  → DocumentProcessor chain      # обогащение метаданных документа
  → docs.upsert()                # сохранить документ в Qdrant до чанкования
  → RecursiveSplitter + pack_blocks   # pre-split на блоки
  → Chunker                      # LLM или Passthrough
  → ContextGenerator             # LLM или Noop (независимый тумблер)
  → ChunkProcessor chain         # payload-метаданные + векторы
  → chunks.upsert()
```

#### Модели данных

**Document** (хранится в коллекции `docs`):
- `id` — уникальный идентификатор: относительный путь файла (от корня директории) или Confluence page ID
- `path` — путь для отображения (совпадает с id)
- `text` — полный текст документа в Markdown
- `updated_at` — дата изменения (mtime файла или last_modified из Confluence)
- `source_type` — `"markdown"` | `"confluence"`
- `size` — размер файла в байтах (из `stat.st_size`); используется в idempotency-проверке
- `indexed_at` — дата индексации; `None` до первого сохранения, выставляется репозиторием при `upsert`
- `payload` — dict, заполняется DocumentProcessor-ами

**Chunk** (хранится в коллекции `chunks`):
- `id` — UUID
- `doc_id` — ссылка на Document.id
- `path` — путь документа (для отображения и фильтрации)
- `order` — порядковый номер чанка в документе (0-based)
- `total` — общее количество чанков документа (используется для проверки полноты индексации)
- `text` — основной текст чанка
- `context` — LLM-суммари: краткое содержание документа + роль данного чанка в нём (пустая строка если NoopContextGenerator)
- `updated_at` — дата актуальности (наследуется от Document)
- `payload` — dict, заполняется ChunkProcessor-ами (автор, теги, и т.д.)
- `vectors` — dict[name → vector], заполняется embedding ChunkProcessor-ами:
  - dense-вектор: `list[float]` (например `'full'` → FRIDA, dim=1024)
  - sparse-вектор: `{'indices': list[int], 'values': list[float]}` (например `'keywords'` → GTE)

#### Idempotency

Перед индексацией документа:
1. Запросить `docs` по `doc_id`
2. Если найден и `updated_at` совпадает **и** `size` совпадает → запросить `chunks` по `doc_id`, сравнить `count` с `total` из любого чанка
   - `count == total` → документ актуален, пропустить
   - `count != total` → предыдущая индексация не завершилась, переиндексировать
3. Если не найден, или `updated_at` отличается, или `size` отличается → переиндексировать

При переиндексации: сначала `filter delete` чанков по `doc_id`, затем удалить документ из `docs`, затем полная индексация заново. Удаление документа всегда каскадно удаляет все его чанки.

Документ сохраняется в `docs` **до** начала чанкования — это позволяет при повторной обработке восстанавливать чанки из зафиксированной копии текста, не обращаясь к source.

### Chunking Design

#### Token Counter

```
TokenCounter          ← интерфейс: count(text) → int, fits(text, limit) → bool
    └── TiktokenCounter   ← реализация на tiktoken (cl100k_base)
```

#### Pre-split: цепочка сплиттеров

`RecursiveSplitter` рекурсивно применяет цепочку `BlockSplitter`-ов пока блоки не влезут в лимит. `FixedSizeSplitter` всегда последний — гарантирует завершение.

```
BlockSplitter
    ├── MarkdownHeaderSplitter   # по заголовкам Markdown (# ## ###...)
    ├── TableRowSplitter         # таблицы: N строк + дублирование шапки
    ├── SemanticSplitter         # по семантическим границам (cosine distance между предложениями)
    └── FixedSizeSplitter        # последний резерв: абзацы → предложения → слова → символы
```

После pre-split блоки жадно упаковываются в пачки (`pack_blocks`) до заполнения лимита токенов.

Значение лимита зависит от режима чанкинга:
- **LLM-режим**: `block_limit = context_window - prompt_overhead` (блок должен влезть в контекст модели)
- **Passthrough-режим**: `block_limit = желаемый_размер_чанка` (блок сразу является финальным чанком)

#### Chunker

Два независимых режима, выбираются в конфиге:

```
Chunker                  ← интерфейс: chunk(block: str) → list[str]
    ├── LLMChunker       # structured output: LLM разбивает блок на семантические чанки
    └── PassthroughChunker  # возвращает блок как есть (один блок = один чанк)
```

LLM для чанкинга может быть: облачная (OpenAI, Anthropic), локальная на сервере или на localhost (Ollama, LM Studio) — любая с OpenAI-совместимым интерфейсом.

#### ContextGenerator

Независимый от Chunker тумблер. Генерирует суммари для каждого чанка: краткое содержание всего документа + роль конкретного чанка в нём. Принимает полный текст документа и текст чанка.

```
ContextGenerator              ← интерфейс: generate(doc_text: str, chunk_text: str) → str
    ├── LLMContextGenerator   # вызов LLM с отдельным промптом
    └── NoopContextGenerator  # возвращает пустую строку
```

Chunker и ContextGenerator — два отдельных LLM-вызова. Это сделано намеренно: слабые локальные модели лучше справляются с одной конкретной задачей за раз.

#### Processor Chain

После сборки базового чанка применяется цепочка процессоров. Каждый процессор может дописывать поля в `payload` и/или добавлять именованные векторы в `vectors`.

```
ChunkProcessor                   ← интерфейс: process(chunk, document) → Chunk
    ├── DenseEmbeddingProcessor  # добавляет dense-вектор (например, для поля text или context)
    ├── SparseEmbeddingProcessor # добавляет sparse-вектор (keywords)
    └── ...                      # произвольные процессоры для payload (автор, теги, ACL и т.д.)

DocumentProcessor                ← интерфейс: process(document) → Document
    └── ...                      # обогащение метаданных документа перед сохранением
```

Состав процессоров определяет набор векторов в коллекции `chunks`. Схема коллекции строится из конфига — какие embedding-процессоры объявлены, такие именованные векторы регистрируются при создании коллекции.

### Qdrant Collections

| Коллекция | Содержимое |
|---|---|
| `docs` | Полный текст документов + метаданные |
| `chunks` | Чанки: текст, контекст, payload, все именованные векторы |

Коллекция `chunks` единственная — нет разделения на "простые" и "гибридные" чанки. Состав векторов определяется конфигом процессоров и может меняться без изменения архитектуры.

Поле `doc_id` в `chunks` индексировано в Qdrant payload-индексом — для эффективного `filter delete` при переиндексации.

### RAG Query Pipeline

```
Вопрос → extract_intent (LLM) → hybrid_search (Qdrant RRF) → reranker (LLM) → answer (LLM)
```

Hybrid search: sparse + dense с RRF-fusion. При ретривинге доступна выборка соседних чанков по полям `doc_id` + `order` (±N от найденного).

### Embedding Models

- **Dense**: `ai-forever/FRIDA` — русскоязычная модель, dim=1024, Cosine.
  Префикс `search_document:` при индексации, `search_query:` при поиске.

- **Sparse**: `Alibaba-NLP/gte-multilingual-base` — `AutoModelForTokenClassification`.
  Forward pass → ReLU → фильтр спецтокенов и нулевых весов → decode token_id → строку →
  дедупликация по `max` → MD5(слово) % 2^32 - 1 → индекс.
  Без префиксов. Без lowercase. Индекс хэша менять нельзя — сломает все сохранённые коллекции.

### Config

Конфиг читается из `config.yml` (на основе `config.example.yml`), валидируется через Pydantic в `src/morag/config.py`:
- `sources.markdown.path` — путь к директории с MD-файлами
- `qdrant.host`, `qdrant.port` — подключение к Qdrant
- `qdrant.collection_docs`, `qdrant.collection_chunks` — имена коллекций
- `llm.base_url`, `llm.model`, `llm.api_key` — OpenAI-совместимый эндпойнт (используется для chunker и context generator)
- `indexing.chunker` — `passthrough` | `llm`
- `indexing.context` — `noop` | `llm`
- `indexing.block_limit` — лимит токенов для pre-split блока
- `indexing.dense_model` — имя dense-модели (по умолчанию `ai-forever/FRIDA`)
- `indexing.sparse_model` — имя sparse-модели (по умолчанию `Alibaba-NLP/gte-multilingual-base`)

## Code Style

- Python 3.12+, Poetry для управления зависимостями
- Ruff для линтинга (line-length=100, single quotes)
- pytest с asyncio_mode="auto"
- Docstrings на русском языке (как в MVP)
- Логи на английском языке (`logger.info`, `logger.debug`, etc.)
