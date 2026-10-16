"""Транскрайб-бэкенд для Mac — OpenAI-совместимый `/v1/audio/transcriptions`, который ЧЕСТНО
honor-ит `prompt` (initial_prompt) и отдаёт `avg_logprob` на сегмент.

Зачем: стоковый oMLX молча игнорит `prompt` (и `timestamp_granularities`) → biasing redecode не
работает по HTTP. Этот тонкий сервис оборачивает библиотеку `mlx_whisper.transcribe` (она honor-ит
initial_prompt) и закрывает ВСЮ mlx-специфику внутри бэкенда. Наш канонизирующий transcribe-прокси
говорит с ним стандартным API и про mlx ничего не знает (бэкенд свапается на OpenAI/Groq/whisper.cpp).

Запуск (Mac): см. start.sh (ffmpeg в PATH обязателен — mlx_whisper зовёт его для декода аудио).
Эндпоинт: POST /v1/audio/transcriptions  (multipart: file, model, language, prompt, temperature,
response_format=json|verbose_json). GET /v1/models, GET /health.
"""
from __future__ import annotations

import asyncio
import math
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# mlx_whisper.load_audio зовёт `ffmpeg` из PATH — в headless-запуске его может не быть
os.environ['PATH'] = '/opt/homebrew/bin:' + os.environ.get('PATH', '')

from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import mlx_whisper  # noqa: E402
from mlx_whisper.load_models import load_model as _load_model  # noqa: E402
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile  # noqa: E402

# ⚠️⚠️ Стек. 27.09 бэкенд упал SIGSEGV внутри MLX — `compile_dfs`, рекурсия по графу вычислений,
# переполнила стек (KERN_PROTECTION_FAILURE на охранной странице) на ~700-м запросе процесса: длинная
# запись, две модели попеременно (пасс-2 и второе ухо). Инференс шёл в главном потоке uvicorn — 8 МБ.
# Теперь он идёт в своём потоке с 1 ГБ стека (виртуальный резерв, не занятая память); заодно цикл
# событий не блокируется на время декодирования — `/health` отвечает и под нагрузкой. Один воркер:
# GPU всё равно один, а прежний блокирующий вызов и так выстраивал запросы в очередь.
# `TRANSCRIBE_DISABLE_COMPILE=1` — запасной ход, если стека не хватит: без `mx.compile` рекурсии нет,
# цена — скорость декодера (не мерилась; включать только по факту повторного падения).
threading.stack_size(1 << 30)
_INFER = ThreadPoolExecutor(max_workers=1, thread_name_prefix='infer')
_INFER.submit(lambda: None).result()          # поток создаётся сейчас, пока задан большой стек
threading.stack_size(0)
if os.environ.get('TRANSCRIBE_DISABLE_COMPILE', '').strip().lower() in ('1', 'true', 'yes', 'on'):
    import mlx.core as _mx  # noqa: E402
    _mx.disable_compile()

# ⚠️⚠️ Кэш моделей. `mlx_whisper.ModelHolder` держит ОДНУ модель и перечитывает веса при смене
# имени, а арбитраж чередует основную модель и второе ухо кусок за куском — каждый переход стоил
# загрузки весов с диска. Держим до `TRANSCRIBE_MODEL_CACHE` моделей (умолчание 2: основная и
# второе ухо; веса turbo ~1.6 ГБ, large ~3 ГБ единой памяти), вытесняем давнюю. 1 — прежнее
# поведение. Доступ — только из потока инференса (он один), поэтому без блокировок.
_CACHE_N = max(1, int(os.environ.get('TRANSCRIBE_MODEL_CACHE', '2') or 2))
_CACHE: dict = {}                                  # путь → модель, порядок = давность использования
STATS = {'requests': 0, 'loads': 0, 'load_s': 0.0, 'infer_s': 0.0, 'audio_s': 0.0}


def _get_model(cls, model_path, dtype):
    m = _CACHE.pop(model_path, None)
    if m is None:
        t0 = time.monotonic()
        m = _load_model(model_path, dtype=dtype)
        STATS['loads'] += 1
        STATS['load_s'] = round(STATS['load_s'] + time.monotonic() - t0, 2)
        while len(_CACHE) >= _CACHE_N:
            _CACHE.pop(next(iter(_CACHE)))
    _CACHE[model_path] = m
    return m


# `mlx_whisper.transcribe` — функция, заслоняющая одноимённый модуль; модуль — из sys.modules.
sys.modules['mlx_whisper.transcribe'].ModelHolder.get_model = classmethod(_get_model)

# Каталог с MLX-весами: env `TRANSCRIBE_MODELS_DIR`, иначе каталог моделей стека.
MODELS_DIR = Path(os.environ.get('TRANSCRIBE_MODELS_DIR')
                  or Path(os.environ.get('ASR_STACK_HOME') or (Path.home() / 'asr-stack')) / 'models')
# Модели ищем В КАТАЛОГЕ, а не перечисляем в коде: второе мнение другой моделью — это параметр
# запроса, и добавление модели не должно требовать правки движка. Каталог с весами (есть
# `weights.safetensors`) становится доступным именем сразу после скачивания.
# ⚠️ Имена из каталога, поэтому доменного тут не появится: это просто папки на диске.
def _discover() -> dict:
    found = {}
    if MODELS_DIR.is_dir():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir() and (d / 'weights.safetensors').is_file():
                found[d.name] = str(d)
            elif d.is_dir() and (d / 'engine.json').is_file():
                found[d.name] = str(d)     # другой движок (см. _OTHER_ENGINES), веса в той же папке
    # Дефолты остаются объявленными, даже если веса ещё не скачаны: так у клиента честная
    # ошибка «модели нет на диске», а не «такой модели не бывает».
    for name in ('whisper-podlodka-turbo', 'whisper-large-v3-turbo'):
        found.setdefault(name, str(MODELS_DIR / name))
    return found


MODELS = _discover()

# --- другие движки: модель другой ШКОЛЫ в роли второго уха (ADR-0030) ---------------------------
# Папка модели с `engine.json` вместо `weights.safetensors`: {"engine": "gigaam", "name": "v3_e2e_rnnt"}.
# Зачем: оба уха на whisper ошибаются заодно — декодер whisper продолжает текст как языковая модель
# и на неразборчивом месте досочиняет гладкое слово. GigaAM (Conformer, RNN-T/CTC, Сбер) выбирает
# слова по звуку покадрово; замер 2026-10-14 на 368 кусках «Мостика»: как второе ухо в правилах
# арбитража — то же качество, что whisper-large-v3-russian, при 25 с звука за ~1 с на CPU против ~10 с.
# torch и gigaam грузятся ЛЕНИВО, только когда такую модель спросили: установка только с whisper
# не меняется. Ставятся в этот же venv (`pip install gigaam torch`, см. install.sh).
_GIGA_MAX_S = 24.0          # без длинного режима GigaAM берёт до 25 с (длинный требует pyannote)
_LOADED: dict = {}


def _engine(path: str) -> dict | None:
    f = Path(path) / 'engine.json'
    if not f.is_file():
        return None
    import json
    return json.loads(f.read_text())


def _load_gigaam(model_dir: str, cfg: dict):
    import gigaam
    key = (model_dir, cfg['name'])
    if key not in _LOADED:
        # ⚠️ Потоки torch: декодер RNN-T упирается в питоновский цикл по токенам, и потоки ему МЕШАЮТ —
        # замерено 16.10 на M4 Pro (5 мин звука): 1–2 потока 1.32 с на минуту, 4 — 1.47, 8 — 1.95,
        # 12 — 2.51. Умолчание 2; задаётся до первого вызова, в потоке инференса. `GIGAAM_DEVICE`
        # перекрывает `engine.json` (cpu | mps | cuda — под другую машину, без правки папки модели).
        import torch
        torch.set_num_threads(max(1, int(os.environ.get('GIGAAM_THREADS', '2') or 2)))
        device = os.environ.get('GIGAAM_DEVICE', '').strip() or cfg.get('device') or 'cpu'
        _LOADED[key] = gigaam.load_model(cfg['name'], download_root=model_dir, device=device)
    return _LOADED[key]


def _count_cpu(t0: float, audio_s: float) -> None:
    # Счётчики своей полосы: один поток инференса на процесс, гонки нет; ключи отдельные от mlx —
    # `/stats` показывает, сколько процесс слушал каждым движком.
    STATS['requests_cpu'] = STATS.get('requests_cpu', 0) + 1
    STATS['infer_cpu_s'] = round(STATS.get('infer_cpu_s', 0.0) + time.monotonic() - t0, 2)
    STATS['audio_cpu_s'] = round(STATS.get('audio_cpu_s', 0.0) + audio_s, 1)


def _gigaam_transcribe(model_dir: str, cfg: dict, wav_path: str) -> str:
    """Текст куска моделью GigaAM. Кусок длиннее 24 с режется на равные части (концы — по сетке
    времени; это второе ухо, а не основной проход: слово на шве в худшем случае попадёт в спор)."""
    import soundfile as sf
    from gigaam.preprocess import SAMPLE_RATE, load_audio
    model = _load_gigaam(model_dir, cfg)
    t0 = time.monotonic()
    audio = load_audio(wav_path).numpy()
    parts = max(1, math.ceil(len(audio) / (_GIGA_MAX_S * SAMPLE_RATE)))
    step = math.ceil(len(audio) / parts)
    texts = []
    for k in range(parts):
        piece = audio[k * step:(k + 1) * step]
        if len(piece) < SAMPLE_RATE // 10:
            continue
        with tempfile.NamedTemporaryFile(suffix='.wav') as t:
            sf.write(t.name, piece, SAMPLE_RATE)
            texts.append(str(model.transcribe(t.name)).strip())
    _count_cpu(t0, len(audio) / SAMPLE_RATE)
    return ' '.join(x for x in texts if x)


def _gigaam_draft(model_dir: str, cfg: dict, wav_path: str, language: str) -> dict:
    """Черновик записи ЦЕЛИКОМ моделью GigaAM: части по самой тихой точке окна [20, 24] с, слова с
    временами (RNN-T, покадрово), сегменты по знакам конца предложения и паузам — `cutting.py`.
    Ответ той же формы, что у mlx-whisper с `word_timestamps`: разрез на куски читает `segments[].words`.
    Зовётся только по просьбе `word_timestamps=1` — её шлёт пасс-1 с `ASR_DRAFT_MODEL`; короткое ухо на
    кусках арбитража и корректора идёт прежним путём (сетка, без времён), байт в байт как до 16.10."""
    import soundfile as sf
    from gigaam.preprocess import SAMPLE_RATE, load_audio
    try:
        import cutting
    except ImportError:                      # запуск не из каталога бэкенда
        sys.path.insert(0, str(Path(__file__).parent))
        import cutting
    model = _load_gigaam(model_dir, cfg)
    t0 = time.monotonic()
    audio = load_audio(wav_path).numpy()
    words: list[dict] = []
    texts: list[str] = []
    for a, b in cutting.quiet_cuts(audio, SAMPLE_RATE, hi=_GIGA_MAX_S):
        piece = audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]
        if len(piece) < SAMPLE_RATE // 10:
            continue
        with tempfile.NamedTemporaryFile(suffix='.wav') as t:
            sf.write(t.name, piece, SAMPLE_RATE)
            r = model.transcribe(t.name, word_timestamps=True)
        texts.append(str(r.text).strip())
        words += [{'word': w.text, 'start': round(a + w.start, 3), 'end': round(a + w.end, 3)} for w in (r.words or [])]
    _count_cpu(t0, len(audio) / SAMPLE_RATE)
    return cutting.verbose_json(words, language, len(audio) / SAMPLE_RATE, text=' '.join(x for x in texts if x))
DEFAULT_MODEL = 'whisper-podlodka-turbo'
API_KEY = os.environ.get('TRANSCRIBE_API_KEY')  # опц. Bearer-защита

app = FastAPI(title='mlx-whisper transcribe backend')


def _clean(obj):
    """Рекурсивно заменяет нефинитные float (NaN/Inf) на None. mlx_whisper кладёт их в
    avg_logprob/no_speech_prob/compression_ratio на шумных/длинных сегментах, а starlette
    JSONResponse сериализует с allow_nan=False → 500 «Out of range float». Реальные таймстампы
    (start/end) финитны и не страдают; метрики качества адаптер не читает."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _check_auth(authorization: Optional[str]) -> None:
    if API_KEY and authorization != f'Bearer {API_KEY}':
        raise HTTPException(status_code=401, detail='bad api key')


def _temperatures(raw: str):
    """«» → дефолт библиотеки (лесенка), «0» → скаляр, «0,0.2,0.4» → своя лесенка.

    Возвращает None, когда параметр не задан: тогда в `transcribe` он не передаётся вовсе и
    работает дефолт mlx-whisper. Мусор в поле — тоже None: показ температуры не повод ронять
    расшифровку.
    """
    parts = [p.strip() for p in str(raw or '').split(',') if p.strip()]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else tuple(vals)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/stats')
def stats():
    """Счётчики процесса: запросы, загрузки весов, секунды загрузки и инференса, секунды звука.
    Стенд скорости читает их до и после прогона — перегрузка весов видна числом."""
    return {**STATS, 'cached': [Path(p).name for p in _CACHE], 'cache_size': _CACHE_N}


@app.get('/v1/models')
def models():
    return {'object': 'list',
            'data': [{'id': m, 'object': 'model', 'owned_by': (_engine(p) or {}).get('engine') or 'mlx-whisper'}
                     for m, p in MODELS.items()]}


@app.post('/v1/audio/transcriptions')
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: str = Form('ru'),
    prompt: str = Form(''),                       # ← honor-им (initial_prompt)
    # Температура принимает и ЛЕСЕНКУ: «0,0.2,0.4» — тогда у mlx-whisper включаются его штатные
    # датчики галлюцинации (`compression_ratio_threshold`, `logprob_threshold`): не понравился
    # ответ — передекодирует окно с температурой повыше.
    # ⚠️⚠️ Дефолт остаётся «0», то есть прежнее детерминированное поведение. Замерено 25.09 на 18
    # местах корпуса: полная лесенка (до 1.0) на невнятном звуке не спасает, а СОЧИНЯЕТ — три
    # ответа из восемнадцати пришли текстом на чужих языках там, где при «0» модель честно
    # возвращала пусто. Отступ хорош, когда есть куда отступать; на тишине его цена — выдумка.
    temperature: str = Form('0'),
    response_format: str = Form('verbose_json'),
    # Времена слов ОТ ДЕКОДЕРА, сразу в ответе: `segments[].words = [{word, start, end,
    # probability}]`. ⚠️ Приблизительные — это не выравнивание (MMS_FA в конце конвейера остаётся:
    # караоке и цитатам нужны точные и упорядоченные времена). Зато они есть В МОМЕНТ прослушивания
    # куска, а не в самом конце прогона: без них разбор внутри прогона не видит ни растянутых слов,
    # ни дыр, ни того, какое слово модель произнесла неуверенно (`probability`). Пусто — прежний
    # ответ байт в байт.
    word_timestamps: str = Form(''),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    # ⚠️ Неизвестное имя — отказ вслух, а не молчаливая модель по умолчанию: опечатка в имени
    # второго уха иначе давала ОДНУ модель в обоих ушах, и арбитраж «голосовал» сам с собой.
    # Пустое имя и OpenAI-совместимое `whisper-1` — модель по умолчанию, как раньше.
    if model and model != 'whisper-1' and model not in MODELS:
        raise HTTPException(400, f'модели {model!r} нет; есть: {", ".join(sorted(MODELS))}')
    repo = MODELS.get(model, MODELS[DEFAULT_MODEL])

    suffix = Path(file.filename or 'audio').suffix or '.wav'
    tmp = tempfile.mktemp(suffix=suffix)
    Path(tmp).write_bytes(await file.read())
    engine = _engine(repo)
    if engine is not None:
        want_words = word_timestamps.strip().lower() in ('1', 'true', 'yes', 'on')
        try:
            if engine.get('engine') != 'gigaam':
                raise HTTPException(400, f'движок {engine.get("engine")!r} модели {model!r} не поддержан')
            if want_words:
                # Черновик с временами слов (пасс-1 с ASR_DRAFT_MODEL) — см. `_gigaam_draft`.
                draft = await asyncio.get_running_loop().run_in_executor(
                    _INFER, lambda: _gigaam_draft(repo, engine, tmp, language))
                text = draft['text']
            else:
                text = await asyncio.get_running_loop().run_in_executor(
                    _INFER, lambda: _gigaam_transcribe(repo, engine, tmp))
        finally:
            Path(tmp).unlink(missing_ok=True)
        if response_format == 'text':
            return text
        if response_format == 'json':
            return {'text': text}
        if want_words:
            return draft
        # Сегментов у второго уха конвейер не читает (только `text`); отдаём один на весь кусок,
        # чтобы verbose_json оставался того же вида.
        return {'task': 'transcribe', 'language': language, 'duration': 0.0, 'text': text,
                'segments': [{'id': 0, 'start': 0.0, 'end': 0.0, 'text': text}] if text else []}
    try:
        kw = dict(path_or_hf_repo=repo, language=language)
        steps = _temperatures(temperature)
        if steps is not None:
            kw['temperature'] = steps
        if prompt:
            kw['initial_prompt'] = prompt
        if word_timestamps.strip().lower() in ('1', 'true', 'yes', 'on'):
            kw['word_timestamps'] = True
        def _run():
            t0 = time.monotonic()
            out = mlx_whisper.transcribe(tmp, **kw)
            STATS['requests'] += 1
            STATS['infer_s'] = round(STATS['infer_s'] + time.monotonic() - t0, 2)
            segs = out.get('segments') or []
            STATS['audio_s'] = round(STATS['audio_s'] + (segs[-1].get('end', 0.0) if segs else 0.0), 1)
            return out
        r = await asyncio.get_running_loop().run_in_executor(_INFER, _run)
    finally:
        Path(tmp).unlink(missing_ok=True)

    if response_format == 'text':
        return r['text'].strip()
    if response_format == 'json':
        return {'text': r['text'].strip()}
    # verbose_json: пробрасываем сегменты (avg_logprob уже внутри) — формат OpenAI-совм.
    # _clean: NaN/Inf из mlx_whisper → None, иначе starlette (allow_nan=False) даёт 500.
    return _clean({'task': 'transcribe', 'language': r.get('language', language),
                   'duration': r.get('segments', [{}])[-1].get('end', 0.0) if r.get('segments') else 0.0,
                   'text': r['text'].strip(), 'segments': r.get('segments', [])})
