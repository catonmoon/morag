# asr-adaptor — транскрипция записей для аудио-RAG

Аудио-транскрайб-адаптер: стандартный OpenAI `POST /v1/audio/transcriptions` на входе,
на выходе — обогащённый транскрипт: **кто говорит** (диаризация + имена спикеров по голосу),
**что говорят** (двухпроходный Whisper с LLM-правкой терминов), **когда** (таймкоды реплик).
Готов к индексации в morag (`chunker.mode: transcript` → цитаты-моменты с точностью до секунды).

Рецепт использования end-to-end — [examples/audio-rag](../../examples/audio-rag/README.md).
Обоснования решений — ADR [0017](../../docs/adr/0017-two-pass-asr-acoustic-grounding.md)
(двухпроходный ASR) и [0018](../../docs/adr/0018-cross-episode-voice-registry.md) (реестр голосов).

## Конвейер

```
audio → диаризация (pyannote) → пасс-1 Whisper целым файлом (черновик)
      → глоссарий [LLM]: неканоничные термины → НАБОРЫ гипотез
      → пасс-2 Whisper по кускам ≤28с + каноники в initial_prompt → выбор ПО ЗВУКУ
      → склейка реплик → финал-раунд [LLM]: правка имён/терминов по контексту (raw-сайдкар цел)
      → Speaker_N (реестр голосов CAM++) → авто-наминг [LLM] из интро записи
      → markdown: front-matter + `[Имя] <!-- t:СЕК --> текст`
```

Принцип: **LLM = recall** (предлагает варианты) · **ASR = precision** (решает акустика) ·
**финал-LLM = арбитр по контексту**. LLM не может навязать замену, которой нет в звуке.

## API

- `POST /v1/audio/transcriptions` — multipart: `file`, `mode=async|sync`, `episode`, `title`, `url`.
  Ответ `verbose_json` + `x_enriched{markdown, turns, raw_sidecar, timing, speaker_map}`.
- `GET /v1/jobs/{id}` — статус/прогресс async-джобы (выпуск ~1 часа звука ≈ 10 мин).
- `GET /health` — включая пинг аудио-бэкендов. Один in-flight job (GPU — горло).

Клиенты прогона корпуса — [`client/`](client/): `run_corpus.sh` (последовательно — детерминизм
реестра голосов; идемпотентно/резюмируемо), `transcribe_one.sh` (URL-шаблон через env, локальные
файлы, видео → аудио-дорожка через ffmpeg).

## Аудио-бэкенды (Apple-Silicon-bound, дёргаются по HTTP)

| Бэкенд | Порт | Что |
|---|---|---|
| `backends/diarizer` | 8090 | pyannote → спаны «кто когда говорит» |
| `backends/transcribe` | 8123 | Whisper (mlx), **honor-ит `prompt`** — стоковые сервер-обвязки его игнорируют, поэтому свой |
| `backends/campp` | 8126 | CAM++ (3D-Speaker) — верификационные эмбеддинги голоса для реестра |

Модели Whisper: рекомендуем [bond005/whisper-podlodka-turbo](https://huggingface.co/bond005/whisper-podlodka-turbo)
для русской разговорной речи (конвертируется в MLX для Apple Silicon).

## Конфиг (env)

`ASR_LLM_BASE_URL` / `ASR_LLM_MODEL` / `OR_KEY` — LLM для глоссария/правки/наминга
(reasoning ОБЯЗАТЕЛЬНО off — на batch-structured зацикливается);
`ASR_DIARIZER_URL` / `ASR_BACKEND_URL` / `ASR_CAMPP_URL` — аудио-бэкенды;
`ASR_REGISTRY_PATH` — реестр голосов (локальный JSON; **биометрия — в git не класть**);
`ASR_MATCH_THRESHOLD` (0.55), `ASR_PROMPT_BUDGET` (≤224 токена подсказки Whisper),
`ASR_ENABLE_NAMING` (авто-наминг из интро, on).

## Деплой

- **Primary — нативно на macOS** (аудио-модели Apple-Silicon-bound): venv + launchd,
  плист-образец в [`deploy/`](deploy/); бэкенды — рядом, каждый со своим venv.
- **Docker** (`Dockerfile`, context — корень репо) — оркестратор без аудио: аудио-URL'ы
  указывают на внешнюю машину с бэкендами.
