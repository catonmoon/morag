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

MODELS = {
    'whisper-podlodka-turbo': str(Path.home() / 'llm-stack/models/whisper-podlodka-turbo'),
    'whisper-large-v3-turbo': str(Path.home() / 'llm-stack/models/whisper-large-v3-turbo'),
}
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
    temperature: float = Form(0.0),
    response_format: str = Form('verbose_json'),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    repo = MODELS.get(model, MODELS[DEFAULT_MODEL])

    suffix = Path(file.filename or 'audio').suffix or '.wav'
    tmp = tempfile.mktemp(suffix=suffix)
    Path(tmp).write_bytes(await file.read())
    try:
        kw = dict(path_or_hf_repo=repo, language=language, temperature=temperature)
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
