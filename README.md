# Morag

Агентская RAG-система для локальных Markdown-файлов, Confluence и Jira с поддержкой локальных LLM.

## Возможности

- **Агентский ретривал** — LLM сам ищет, фильтрует и уточняет запросы через function calling. Решает когда информации достаточно для ответа
- **Knowledge Map** — автоматическая карта документации, LLM ищет прицельно по разделам
- **Гибридный поиск** — семантический + лексический с multi-signal fusion
- **Локальные LLM** — любой OpenAI-совместимый эндпойнт (Ollama, LM Studio, облако)
- **Умное чанкование** — структурный чанкер на базе CommonMark AST, несколько стратегий обработки крупных блоков
- **Адаптивный контекст** — LLM-summary для каждого чанка, размер адаптируется к бюджету embedder
- **Идемпотентность** — пропуск неизменённых документов, каскадное удаление устаревших
- **Цитаты и ссылки** — URL источников в ответах, группировка по документам
- **PDF и Vision LLM** — конвертация PDF через Vision LLM или docling-serve
- **Русский язык** — нативная поддержка русского в embeddings, стемминге и сегментации

## Публикации
- [habr: Юридическое поле экспериментов для RAG](https://habr.com/ru/articles/1014690/)
- [linkedin: A Legal Proving Ground for RAG Experiments](https://www.linkedin.com/pulse/legal-proving-ground-rag-experiments-ivan-komarov-lwocf/)

## Быстрый старт

### Требования

- Docker и Docker Compose
- LLM-сервер: [Ollama](https://ollama.ai) или любой OpenAI-совместимый

```bash
git clone https://github.com/catonmoon/morag.git
cd morag

# 1. Ollama с LLM и эмбеддером на том же демоне
brew install ollama                  # macOS
ollama pull qwen3.5:9b               # LLM для агента
ollama pull qwen3-embedding:4b       # dense-эмбеддер (dim=2560, context=32K)
ollama serve

# 2. Конфиг
cp config.example.yml config.yml
# отредактировать config.yml — указать путь к документам в sources.local_documents.path

# 3. Положить документы в папку из config.yml (по умолчанию examples/)

# 4. Собрать и запустить
docker compose build
docker compose up -d
```

Откройте http://localhost:3000 и задавайте вопросы.

`docker compose up` поднимает: Qdrant, GTE sparse-эмбеддер, pipeline и запускает индексацию.

### Индексация

`morag-indexer` работает в daemon-режиме: при старте выполняет полную индексацию, затем переиндексирует по cron-расписанию из `config.yml` (секция `indexing.schedule`). Новые и изменённые документы подхватываются автоматически, удалённые — каскадно удаляются.

```bash
# Принудительная переиндексация с нуля
docker compose run morag-indexer index --reset
```

### Dense embedder

Используется **Qwen3-Embedding-4B** (dim=2560, context=32K токенов), можно запустить на Ollama.

```yaml
indexing:
  dense_embedder:
    model: qwen3-embedding:4b
    tokenizer: Qwen/Qwen3-Embedding-4B  # нативный токенизатор (HF) для точного подсчёта
    base_url: http://localhost:11434/v1  # Ollama OpenAI-compat (натив; Docker — host.docker.internal:11434/v1)
    dim: 2560
    timeout: 180
    document_template: '{text}'
    query_template: "Instruct: Given a user question, retrieve passages that answer the question\nQuery:{text}"
```

## Источники данных

| Источник | Описание |
|---|---|
| Markdown-файлы | Локальная директория с .md файлами |
| Confluence | Пространства и страницы (включая вложенные PDF) |
| Jira | Задачи по ссылкам из документов |
| PDF | Конвертация через Vision LLM или docling-serve |

Настраиваются в `config.yml` — см. `config.example.yml` для описания всех опций.

## Стек

| Компонент | Технология |
|---|---|
| Векторная БД | [Qdrant](https://qdrant.tech) |
| Dense embeddings | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)  |
| Sparse embeddings | [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) |
| LLM | Любой OpenAI-совместимый |
| UI | [Open WebUI](https://openwebui.com) (опционально) |

## Разработка

```bash
poetry install
poetry run ruff check src
poetry run pytest -v --cov
```

