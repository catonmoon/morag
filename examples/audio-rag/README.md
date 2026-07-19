# RAG по аудио/видео-записям: рецепт end-to-end

Из набора записей (mp3-ссылки, RSS-фид подкаста, локальные файлы, видео) получаем агентский RAG,
который отвечает на вопросы **со ссылкой на секунду звука** («момент»: запись · MM:SS · спикер,
клик — и слышно, как это было сказано). Референс-деплой: RAG по подкасту «Капитанский мостик»
(53 выпуска, 70 часов) — история решений в `docs/adr/0015-0018`.

## Что понадобится

- **Транскрайб-бэкенд** — `services/asr-adaptor` (диаризация + двухпроходный Whisper с
  LLM-глоссарием + имена спикеров по голосу; см. ADR-0017/0018). Аудио-модели Apple-Silicon-bound —
  primary-деплой нативный на Mac (см. `services/asr-adaptor/CLAUDE-less README` и `deploy/`);
  оркестратор без аудио можно в Docker, указав ему внешние аудио-URL.
- **LLM-ключ** (OpenAI-совместимый провайдер) — для глоссария/правки при транскрипции и для
  агента/реранкера при ответах.
- **Эмбеддинги**: dense (напр. Qwen3-Embedding через любого провайдера) + sparse GTE
  (`services/embedder_gte`, локальный сервис — публичных провайдеров нет).
- Qdrant, ffmpeg (для видео и длительностей), python 3.12.

## Шаг 1. Транскрипция корпуса

```bash
export ASR_BASE=https://your-host/asr                     # ваш asr-adaptor
export MP3_URL_TEMPLATE='https://site/episodes/ep{n}.mp3' # {n} — номер, {pfx} — сезонный префикс
export TITLE_TEMPLATE='Мой подкаст №{pfx}{n}'
export OUT_DIR=./transcripts/season1 CACHE_DIR=./media_cache/season1

services/asr-adaptor/client/run_corpus.sh 1 2 3 4 5      # последовательно! (реестр голосов)
# одиночная запись, прямой URL или локальный файл (видео → дорожка извлечётся ffmpeg'ом):
services/asr-adaptor/client/transcribe_one.sh 6 /path/to/lecture.mp4
```

На выходе — `epN.md` (front-matter + `[Имя] <!-- t:сек --> текст`) и `epN.json` (raw-сайдкар).
Реестр голосов стартует пустым: ведущих/гостей именует LLM из интро записи; узнанные голоса
дальше узнаются автоматически между записями (ADR-0018). ~10 минут на час записи.

## Шаг 2. Обогащение front-matter (опционально, если есть RSS/страницы выпусков)

```bash
export MP3_URL_TEMPLATE='https://site/episodes/ep{n}.mp3'
export RSS_URL='https://site/feed.xml'                    # даты публикации
export EPISODE_PAGE_TEMPLATE='https://site/ep-{n}.html'   # темы в title из meta description
export TITLE_TEMPLATE='Мой подкаст №{pfx}{n}'
export TRANSCRIPTS_DIR=./transcripts/season1 MEDIA_CACHE_DIR=./media_cache/season1
python tools/enrich_frontmatter.py 1 2 3 4 5
```

Появятся `date`, `duration_sec`, темы в title, `speakers` — на них опираются каталог выпусков
и фильтры retrieval.

## Шаг 3. Конфиг и индексация

`config.example.yml` рядом — скопируйте, впишите ключи и участников. Ключевое:
`chunker.mode: transcript` (тематические чанки с сохранением секунд и спикеров),
`retrieval.features.timestamp_citations: true` (цитаты-моменты, ADR-0015),
разговорный режим ответа в `prompts.section_overrides` (ADR-0016).

```bash
docker run -d --name qdrant -p 6333:6333 -v "$PWD/qdrant-data:/qdrant/storage" qdrant/qdrant
docker run -d --name gte -p 8081:8081 morag-embedder-gte   # или нативно, см. deploy-macos/
python -m cli.main index --config ./config.yml
```

## Шаг 4. Ответы

OWUI + pipelines (см. корневой docker-compose) либо нативный стек без Docker —
`deploy-macos/` содержит launchd-плисты нашего публичного деплоя (qdrant-бинарь + GTE +
pipelines + OWUI без авторизации, всё на loopback за реверс-прокси).

Обновление корпуса новой записью = Шаг 1 (один номер) → Шаг 2 → `cli.main index` без `--reset`
(инкрементально, старые записи скипаются) — ~15 минут на свежий выпуск.

## Грабли, собранные за вас

- Whisper-бэкенд обязан honor-ить `prompt` (стоковые сервер-обвязки игнорируют — потому свой).
- Python **3.12** для нативного стека: OWUI требует `<3.13`, пины GTE без cp313-wheels.
- OWUI не работает под URL-подпутём — публикуйте на поддомене.
- `WEBUI_AUTH=False` валиден только на чистой `data/` OWUI.
- Реестр голосов — биометрия: держите локально, в git не кладите.
- Транскрипты — производная вашего контента; чужие записи публикуйте только с разрешения.
