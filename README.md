# Morag

Агентская RAG-система для локальных Markdown-файлов, Confluence и Jira с поддержкой локальных LLM.

**Живое демо:** [morag.catonmoon.com](https://morag.catonmoon.com) — вопросы к подкасту
«Капитанский мостик» (58 выпусков, ~76 часов разговоров про ИИ).

## Возможности

- **Агентский ретривал** — LLM сам ищет, фильтрует и уточняет запросы через function calling. Решает когда информации достаточно для ответа
- **Knowledge Map** — автоматическая карта документации, LLM ищет прицельно по разделам
- **Гибридный поиск** — семантический + лексический с multi-signal fusion
- **Локальные LLM** — любой OpenAI-совместимый эндпойнт (Ollama, LM Studio, облако)
- **Умное чанкование** — структурный чанкер на базе CommonMark AST, несколько стратегий обработки крупных блоков
- **Адаптивный контекст** — LLM-summary для каждого чанка, размер адаптируется к бюджету embedder
- **Идемпотентность + плавная переиндексация** — пропуск неизменённых документов, каскадное удаление устаревших; во время прогона корпус остаётся запрашиваемым
- **Цитаты и ссылки** — URL источников в ответах, группировка по документам
- **PDF и Vision LLM** — конвертация PDF через Vision LLM или docling-serve
- **Аудио и видео** — транскрипция записей (диаризация, имена спикеров по голосу, правка терминов) и ответы **со ссылкой на секунду звука** — см. [examples/audio-rag](examples/audio-rag/)
- **Русский язык** — нативная поддержка русского в embeddings, стемминге и сегментации
- **Console UI** — web-интерфейс для настройки индексации RAG

## RAG по аудио и видео

Полный рецепт «набор записей → RAG с цитатами-моментами» — [examples/audio-rag/README.md](examples/audio-rag/README.md):
транскрипция через [services/asr-adaptor](services/asr-adaptor/) (двухпроходный Whisper с LLM-глоссарием,
кросс-эпизодный реестр голосов), тематические чанки с таймкодами, разговорный режим ответа, цитата =
кликабельная секунда записи. Решения и измерения — ADR [0015](docs/adr/0015-audio-moment-citations.md)–[0018](docs/adr/0018-cross-episode-voice-registry.md);
доклад с живым демо — [docs/talks/kapmost-demo](docs/talks/kapmost-demo/).

Как это выглядит вживую — **[morag.catonmoon.com](https://morag.catonmoon.com)**: вопросы к подкасту
«Капитанский мостик» [Дмитрия Колодезева](https://kolodezev.ru/) и Валентина Малых. Каждая цитата в
ответе — кликабельный момент записи: плеер встаёт на нужную секунду, расшифровка идёт караоке по
словам, видно каким запросом агент нашёл фрагмент и какое утверждение ответа он подпирает. Движок —
этот репозиторий; интерфейс, бренд и корпус расшифровок задаются конфигом инстанса и живут отдельно.

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
#       (qwen3.5:9b как LLM+vision и qwen3-embedding:4b — одним кликом)
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

### Производительность и выбор провайдера

**Локальная Ollama** — простой старт без секретов и API-ключей, но узкое место — **сериализация запросов** (`OLLAMA_NUM_PARALLEL=1` по умолчанию). В этом случае LLM и embedder работают по одному запросу за раз, что негативно сказывается на производительности. Например, на наборе из `examples/` (~10 markdown-файлов) индексация на M4 Pro занимает **~30 минут** — большую часть времени съедает context-генерация для каждого чанка и генерация эмбеддингов в один поток.

**Облачный провайдер (OpenRouter и аналоги)** — заметно быстрее за счёт реального параллелизма. Рекомендуемый минимум:
- **LLM**: модели уровня `qwen3.5:9b` или сильнее, с поддержкой tools и контекстом не менее 32K токенов. Для PDF и распознавания изображений в документации — нужна vision-модель (одна multimodal LLM закрывает обе роли).
- **Embedder**: семейство `qwen3-embedding` (или эквивалент). Два критичных параметра:
  - **Качество векторов на нужном вам языке** — для русского сверяйтесь с [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (фильтр Russian). Чисто английских моделей часто недостаточно для русского.
  - **Размер контекста входа embedder'а** — минимум 2-4K токенов, чтобы влез нормальный чанк с контекстом и навигационным path. Например, у FRIDA контекст всего 512 токенов — для section-чанкера и генерации контекста чанка это слишком мало (чанки урезаются, теряется смысл). У qwen3-embedding контекст 32K — с большим запасом.
- **`max_concurrent`** в Console UI (Setup → LLM → Edit) можно поднимать выше — при облачных провайдерах разумно начинать с 4-8 и смотреть на rate-limit и время отклика. Точное значение зависит от вашего тарифа и нагрузки провайдера.

В Console UI Setup → LLM пресет «OpenAI-compatible» подходит для всех таких провайдеров: задайте `base_url`, `model`, `api_key`.

### Dense embedder

Используется **Qwen3-Embedding-4B** (dim=2560, context=32K токенов). В самом недорогом случае запускается на Ollama; также доступен у разных облачных провайдеров. Настройка через Console UI (Setup → Embedder); для быстрого старта можно поставить через ⚡ Быстрая настройка.

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
- **Настройка (Setup)** — секции:
  - ⚡ **Быстрая настройка для Ollama** — одним кликом ставит рекомендуемый стек (qwen3.5:9b multimodal + qwen3-embedding:4b). Карточка показывается пока стек не собран.
  - **Источники** — Local folder (зашит на `/app/data`), Confluence, Jira. Add/Edit/Delete через UI.
  - **LLM** — пул именованных LLM. Пресеты: OpenAI-compatible (Grok, OpenRouter, vLLM, OpenAI) и Ollama (с автоматическим списком моделей). Test/Edit/Delete.
  - **Embedder** — один dense-эмбеддер (replace-only). Те же два пресета. Кнопка «Выяснить» автоматически определяет dim вектора.
  - **Роли** — какая LLM на text-задачи, какая на vision.
  - **Расписание** — cron для автоиндексации (по умолчанию выключен).
- **Параметры (Settings)** — эффективная конфигурация (read-only) + редактируемый overlay `config.local.yml` для тонкой настройки.

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

