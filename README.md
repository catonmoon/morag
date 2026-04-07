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

# 1. LLM
brew install ollama          # macOS
ollama pull qwen3.5:9b
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

`docker compose up` поднимает: Qdrant, embedding-серверы (FRIDA + GTE), pipeline и запускает индексацию. `config.example.yml` уже настроен на Ollama.

### Индексация

`morag-indexer` работает в daemon-режиме: при старте выполняет полную индексацию, затем переиндексирует по cron-расписанию из `config.yml` (секция `indexing.schedule`). Новые и изменённые документы подхватываются автоматически, удалённые — каскадно удаляются.

```bash
# Принудительная переиндексация с нуля
docker compose run morag-indexer index --reset
```

### Apple Silicon (MPS ускорение)

На Mac с Apple Silicon dense embedding на GPU через MPS значительно быстрее чем CPU в Docker.

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Запустить FRIDA нативно с MPS
python services/embedder_frida/app.py --port 8092
```

В `config.yml` изменить:
```yaml
indexing:
  dense_embedder:
    base_url: http://localhost:8092          # вместо http://embedder-frida:8082
    # или для индексации из Docker:
    # base_url: http://host.docker.internal:8092
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

