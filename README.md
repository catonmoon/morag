# morag

RAG-система для локальных Markdown-файлов (и Confluence) с поддержкой локальных LLM.

## Возможности

- **Гибридный поиск** — sparse + dense векторы с RRF-fusion
- **Локальные LLM** — любой OpenAI-совместимый эндпойнт (Ollama, LM Studio, облако)
- **Умное чанкование** — цепочка сплиттеров по заголовкам, таблицам, семантике; опциональный LLM-чанкер
- **Контекстуализация** — LLM генерирует суммари роли каждого чанка в документе
- **Идемпотентность** — повторная индексация пропускает неизменённые документы
- **Поддержка русского языка** — модели FRIDA и GTE-multilingual

## Стек

| Компонент | Технология |
|---|---|
| Векторная БД | [Qdrant](https://qdrant.tech) |
| Dense embeddings | [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) (1024-dim, Cosine) |
| Sparse embeddings | [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) |
| LLM | Любой OpenAI-совместимый (Ollama, LM Studio, OpenAI, Anthropic через прокси) |

## Быстрый старт

### 1. Требования

- Python 3.12+
- [Poetry](https://python-poetry.org)
- Запущенный [Qdrant](https://qdrant.tech/documentation/quick-start/) (локально или удалённо)
- LLM-сервер (опционально, нужен только для LLM-чанкинга и контекстуализации)

### 2. Установка

```bash
git clone https://github.com/your-org/morag.git
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
  block_limit: 32000
  dense_model: ai-forever/FRIDA
  sparse_model: Alibaba-NLP/gte-multilingual-base
```

### 4. Индексация

```bash
poetry run python -m cli.main index --config config.yml
```

## Архитектура

### Пайплайн индексации

```
Source.load()
  → Idempotency check          # updated_at + size + счётчик чанков
  → DocumentProcessor chain    # обогащение метаданных
  → docs.upsert()              # сохранить документ до чанкования
  → RecursiveSplitter          # разбивка на блоки (заголовки → таблицы → семантика → fixed)
  → pack_blocks                # жадная упаковка блоков до block_limit токенов
  → Chunker                    # LLM или Passthrough
  → ContextGenerator           # LLM-суммари или Noop
  → ChunkProcessor chain       # dense + sparse векторы, payload
  → chunks.upsert()
```

### Режимы чанкинга

| `chunker` | `context` | Описание |
|---|---|---|
| `passthrough` | `noop` | Быстро, без LLM. Один блок = один чанк, без суммари. |
| `passthrough` | `llm` | Чанки по блокам, но с LLM-контекстом каждого. |
| `llm` | `noop` | LLM делит блок на семантические чанки, без суммари. |
| `llm` | `llm` | Максимальное качество: LLM-чанкинг + LLM-контекст. |

### Коллекции Qdrant

| Коллекция | Содержимое |
|---|---|
| `docs` | Полный текст + метаданные документов |
| `chunks` | Чанки: текст, контекст, dense-вектор `full`, sparse-вектор `keywords` |

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
└── src/morag/
    ├── config.py                  # Pydantic-модели конфига
    ├── sources/                   # Источники данных (Markdown, Confluence)
    ├── indexing/                  # Пайплайн индексации
    │   ├── splitter.py            # Цепочка сплиттеров
    │   ├── chunker.py             # LLMChunker / PassthroughChunker
    │   ├── context.py             # LLMContextGenerator / NoopContextGenerator
    │   ├── embedder.py            # FridaEmbedder + GteSparseEmbedder
    │   ├── processors.py          # ChunkProcessor / DocumentProcessor
    │   └── pipeline.py            # Оркестратор
    ├── storage/                   # Qdrant: коллекции и репозитории
    ├── retrieval/                 # Гибридный поиск + reranker (в разработке)
    └── llm/client.py              # OpenAI-совместимый клиент
```