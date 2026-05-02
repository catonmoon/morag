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
- **Console UI** — web-интерфейс для настройки индексации RAG

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
touch config.local.yml             # overlay для секретов и UI-правок
# минимальный config.yml — только sources.local_documents.path, остальное настроишь через console

# 3. Положить документы в папку из config.yml (по умолчанию data/)

# 4. Собрать и запустить
docker compose build
docker compose up -d
```

Дальше — два URL:
- **http://localhost:8000** — Console UI: конфигурация провайдеров, запуск индексации, статус, Knowledge Map.
- **http://localhost:3000** — Open WebUI: задавать вопросы (после того как что-то проиндексировано).

`docker compose up` поднимает: Qdrant, GTE sparse-embedder, indexer (daemon с cron), console и pipeline.

### Индексация

`morag-indexer` работает в daemon-режиме: ждёт cron-триггер из `config.yml` (`indexing.schedule`) либо on-demand-вызов из Console. Initial run при старте контейнера НЕ запускается — конфиг сначала надо завершить через UI. Новые и изменённые документы подхватываются на следующем прогоне, удалённые — каскадно удаляются.

```bash
# Принудительная переиндексация с нуля (через CLI)
docker compose run morag-indexer index --reset
# либо в Console UI: Dashboard → "Reset & Start"
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

## Console UI

Web-интерфейс на http://localhost:8000 для настройки и управления.

- **Dashboard** — счётчики, ссылки на Qdrant и Open WebUI, кнопки Start/Stop индексации с прогресс-баром, превью Knowledge Map.
- **Setup** — пошаговая настройка LLM и embedder через готовые пресеты (Grok, OpenRouter, Ollama, custom). Test connection per provider.
- **Settings** — текущий конфиг (read-only) + редактируемый overlay в `config.local.yml`.

Типичный поток: открыл Setup → выбрал провайдер из пресетов → ввёл api_key → Save → перешёл на Dashboard → Start. Прогресс смотришь в реальном времени, Stop останавливает после завершения текущего документа, Force Stop — сразу.

Console и индексатор работают в изолированных контейнерах. Console доступен только на localhost (без auth, не выставляй наружу).

## Источники данных

| Источник | Описание |
|---|---|
| Markdown-файлы | Локальная директория с .md файлами |
| Confluence | Пространства и страницы (включая вложенные PDF), Cloud + on-premise |
| Jira | Задачи по ссылкам из документов (только on-premise) |
| PDF | Конвертация через Vision LLM или docling-serve |

Поддерживается **несколько инстансов одного типа** — например, два Confluence-сервера (корпоративный + подрядчика) или несколько Jira. Каждый со своим уникальным `name`. Настраиваются в `config.yml` — см. `config.example.yml` для описания всех опций.

## LLM

Несколько LLM в одном пуле, переиспользуются по ролям. Минимум — два инстанса (text + vision), при поддержке multimodal-модели — один:

```yaml
llms:
  - name: main
    base_url: http://host.docker.internal:11434/v1
    model: qwen3:4b
    api_key: ollama                 # capabilities=[text] (default)
  - name: vision
    base_url: ...
    model: qwen2.5-vl:7b
    api_key: ollama
    capabilities: [text, vision]    # multimodal — годится и для text-роли

indexing:
  llm: main                         # для DocTitle/DocSummary/Context/KM/Chunker
  vision: vision                    # для PDF + изображений Confluence
```

Можно использовать несколько LLM с разными ролями: дешёвую для context-generation, умную для doc_summary и KM. Подробности — в `config.example.yml`.

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

