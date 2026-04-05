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

### Docker (рекомендуется)

Самый быстрый способ — всё в одном:

```bash
git clone https://github.com/catonmoon/morag.git
cd morag
cp config.example.yml config.yml
# отредактировать config.yml — указать источники данных

docker compose build && docker compose up -d
```

Это поднимет: Qdrant, embedding-серверы, pipeline и Open WebUI. Откройте http://localhost:3000 и задавайте вопросы.

### Требования для Docker

- Docker и Docker Compose
- LLM-сервер: [Ollama](https://ollama.ai) (рекомендуется) или любой OpenAI-совместимый

```bash
brew install ollama          # macOS
ollama pull qwen3.5:9b
ollama serve                 # http://localhost:11434
```

`config.example.yml` уже настроен на Ollama.

### Индексация

```bash
# Разовая индексация (подхватит новые и изменённые документы)
docker compose exec morag-indexer python -m cli.main index

# Полная переиндексация с нуля
docker compose exec morag-indexer python -m cli.main index --reset

# Daemon-режим (индексация по cron-расписанию из config.yml)
docker compose exec morag-indexer python -m cli.main serve
```

### Apple Silicon (MPS ускорение)

На MacBook с Apple Silicon embedding-модели работают на GPU через MPS — значительно быстрее чем CPU в Docker.

```bash
poetry install
cp config.example.yml config.yml

# 1. Qdrant
docker compose up -d qdrant

# 2. Ollama
brew install ollama
ollama pull qwen3.5:9b
ollama serve

# 3. Embedding-серверы (MPS ускорение)
python services/embedder_frida/app_native.py --port 8092   # dense, FRIDA
python services/embedder_gte/app_native.py --port 8091     # sparse, GTE

# 4. Индексация
poetry run python -m cli.main index --config config.yml
```

В `config.yml` указать:
```yaml
dense_embedder:
  base_url: http://localhost:8092
sparse_embedder:
  base_url: http://localhost:8091
```

Для ретривала — запустить pipeline и Open WebUI через Docker:
```bash
docker compose up -d pipelines open-webui
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
| Dense embeddings | [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) |
| Sparse embeddings | [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) |
| LLM | Любой OpenAI-совместимый |
| UI | [Open WebUI](https://openwebui.com) (опционально) |

## Разработка

```bash
poetry install
poetry run ruff check src
poetry run pytest -v --cov
```

### Исследования

Результаты экспериментов и сравнений — в `experiments/`.
