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

import math
import os
import tempfile

# mlx_whisper.load_audio зовёт `ffmpeg` из PATH — в headless-запуске его может не быть
os.environ['PATH'] = '/opt/homebrew/bin:' + os.environ.get('PATH', '')

from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import mlx_whisper  # noqa: E402
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile  # noqa: E402

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
    # Дефолты остаются объявленными, даже если веса ещё не скачаны: так у клиента честная
    # ошибка «модели нет на диске», а не «такой модели не бывает».
    for name in ('whisper-podlodka-turbo', 'whisper-large-v3-turbo'):
        found.setdefault(name, str(MODELS_DIR / name))
    return found


MODELS = _discover()
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


@app.get('/v1/models')
def models():
    return {'object': 'list',
            'data': [{'id': m, 'object': 'model', 'owned_by': 'mlx-whisper'} for m in MODELS]}


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
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    repo = MODELS.get(model, MODELS[DEFAULT_MODEL])

    suffix = Path(file.filename or 'audio').suffix or '.wav'
    tmp = tempfile.mktemp(suffix=suffix)
    Path(tmp).write_bytes(await file.read())
    try:
        kw = dict(path_or_hf_repo=repo, language=language)
        steps = _temperatures(temperature)
        if steps is not None:
            kw['temperature'] = steps
        if prompt:
            kw['initial_prompt'] = prompt
        r = mlx_whisper.transcribe(tmp, **kw)
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
