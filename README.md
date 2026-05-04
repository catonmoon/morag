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

# 1. Ollama с LLM и эмбеддером на том же демоне (для quickstart-сценария)
brew install ollama                  # macOS
ollama pull qwen3.5:9b               # multimodal LLM для агента и vision-задач
ollama pull qwen3-embedding:4b       # dense-эмбеддер (dim=2560, context=32K)
ollama serve

# 2. Положить документы в ./data/ (для пробы — примеры из репо).
#    Папка примонтирована в контейнер как /app/data — единственный путь для local-source.
cp examples/*.md data/

# 3. Собрать и запустить
docker compose build
docker compose up -d

# 4. Открыть консоль http://localhost:8000 и пройти настройку:
#    a) ⚡ Быстрая настройка для Ollama → "Добавить рекомендуемые"
#       (одним кликом ставятся qwen3.5:9b как LLM+vision и qwen3-embedding:4b)
#    b) Источники → Добавить → "Локальная папка" с произвольным именем
#    c) Готово — на Dashboard разблокируется "Запустить"
```

Дальше — два URL:
- **http://localhost:8000** — Console UI: конфигурация, запуск индексации, статус, Knowledge Map.
- **http://localhost:3000** — Open WebUI: задавать вопросы (после того как что-то проиндексировано).

`docker compose up` поднимает: Qdrant, GTE sparse-embedder, indexer (daemon с cron-планировщиком), console, pipeline и Open WebUI.

### Индексация

`morag-indexer` работает в daemon-режиме. Запускается двумя путями:

- **On-demand** — кнопка «Запустить» в Console UI (Dashboard).
- **По расписанию** — cron, который включается в Console UI (Setup → Расписание); по умолчанию выключен.

Initial run при старте контейнера НЕ запускается — конфигурацию сначала надо завершить через UI (источники, LLM, embedder, роли). Новые и изменённые документы подхватываются на следующем прогоне, удалённые — каскадно удаляются.

```bash
# Принудительная переиндексация с нуля (через CLI)
docker compose run morag-indexer index --reset
# либо в Console UI: Dashboard → "Сбросить и запустить"
```

### Dense embedder

Используется **Qwen3-Embedding-4B** (dim=2560, context=32K токенов), запускается на Ollama. Настройка через Console UI (Setup → Embedder); рекомендуемая конфигурация ставится одним кликом через ⚡ Быстрая настройка.

Под капотом (записывается в `config.local.yml`):

```yaml
indexing:
  dense_embedder:
    model: qwen3-embedding:4b
    tokenizer: Qwen/Qwen3-Embedding-4B   # нативный токенизатор (HF) для точного подсчёта
    base_url: http://host.docker.internal:11434/v1   # Docker; натив — http://localhost:11434/v1
    api_key: ollama
    dim: 2560
```

## Console UI

Web-интерфейс на http://localhost:8000 для настройки и управления.

- **Главная (Dashboard)** — счётчики, ссылки на Qdrant и Open WebUI, статус Qdrant + моделей текущего конфига (карточка «Окружение»), кнопки Запустить/Остановить с прогресс-баром, превью Knowledge Map.
- **Настройка (Setup)** — пять секций:
  - ⚡ **Быстрая настройка для Ollama** — одним кликом ставит рекомендуемый стек (qwen3.5:9b multimodal + qwen3-embedding:4b). Карточка показывается пока стек не собран.
  - **Источники** — Local folder (зашит на `/app/data`), Confluence, Jira. Add/Edit/Delete через UI.
  - **LLM** — пул именованных LLM. Пресеты: OpenAI-compatible (Grok, OpenRouter, vLLM, OpenAI) и Ollama (с автоматическим списком моделей). Test/Edit/Delete.
  - **Embedder** — один dense-эмбеддер (replace-only). Те же два пресета. Кнопка «Выяснить» автоматически определяет dim вектора.
  - **Роли** — какая LLM на text-задачи, какая на vision.
  - **Расписание** — cron для автоиндексации (по умолчанию выключен).
- **Параметры (Settings)** — эффективная конфигурация (read-only) + редактируемый overlay `config.local.yml` для тонкой настройки.

Типичный поток для Ollama-юзера: Setup → ⚡ Быстрая настройка → «Добавить рекомендуемые» → Источники → +Добавить → Готово. Для другого провайдера — настроить вручную через секции LLM/Embedder/Роли. Прогресс на Dashboard в реальном времени, «Остановить» завершает после текущего документа, «Прервать» — мгновенно.

Console и индексатор работают в изолированных контейнерах. Console доступен только на localhost (без auth, не следует выставлять наружу).

## Источники данных

| Источник | Описание |
|---|---|
| Markdown-файлы | Локальная директория `./data/` (примонтирована как `/app/data`) |
| Confluence | Пространства и страницы (включая вложенные PDF), Cloud + on-premise |
| Jira | Задачи по ссылкам из документов (только on-premise) |
| PDF | Конвертация через Vision LLM или docling-serve |

Поддерживается **несколько инстансов** Confluence/Jira (корпоративный + подрядчика и т.п.), каждый со своим уникальным `name`. Настраиваются в Console UI (Setup → Источники), сохраняются в `config.local.yml`. Все опции описаны в `config.yml` (комментариях) и в YAML overlay через раздел Параметры.

## LLM

Настраиваются через Console UI (Setup → LLM). Один пул именованных LLM, переиспользуются по ролям. Минимум — одна multimodal-модель на text + vision (рекомендуется `qwen3.5:9b` через Ollama; ставится одним кликом через ⚡ Быстрая настройка).

Под капотом — стандартный YAML (записывается в `config.local.yml`):

```yaml
llms:
  - name: main
    base_url: http://host.docker.internal:11434/v1
    model: qwen3.5:9b
    api_key: ollama
    capabilities: [text, vision]    # multimodal — годится и для text, и для vision

indexing:
  llm: main                         # text-роли: DocTitle/DocSummary/Context/KM/Chunker
  vision: main                      # vision-роль: PDF + изображения Confluence
```

Для разделения ролей (например, дешёвая text-only для рутинных задач + dedicated VL для PDF) — добавьте второй LLM в пул через UI и переназначьте роль на него.

## Стек

| Компонент | Технология |
|---|---|
| Векторная БД | [Qdrant](https://qdrant.tech) |
| Dense embeddings | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)  |
| Sparse embeddings | [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) |
| LLM | Любой OpenAI-совместимый с поддержкой tools, контекст не менее 32K. Для PDF и изображений — нужна vision-модель. Минимум: `qwen3.5:9b` (multimodal, покрывает обе роли) |
| UI | [Open WebUI](https://openwebui.com) (опционально) |

## Разработка

```bash
poetry install
poetry run ruff check src
poetry run pytest -v --cov
```

