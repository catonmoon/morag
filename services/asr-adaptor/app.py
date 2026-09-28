"""asr-adaptor — OpenAI-совместимый `POST /v1/audio/transcriptions` адаптер (morag-сервис).

Аудио → обогащённый транскрипт (Speaker_N + тайминги + канонизация сущностей). Внутри — весь пайплайн
(diarize → пасс-1 → глоссарий → пасс-2 → финал-раунд → Speaker_N), аудио на Маке по HTTP, LLM (облако) =
Grok-4.3 на OpenRouter (reasoning off). См. CLAUDE.md.

Ответ: стандартный verbose_json (`text`/`segments`) + кастомный `x_enriched` (markdown, turns, raw, timing).
mode=async (дефолт) → 202 {job_id}, поллинг GET /v1/jobs/{id}. mode=sync — для коротких/smoke.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import audio_clients
import jobs
import prompts
from config import CFG
from pipeline import run_pipeline

# Без этого INFO-записи конвейера (сводка о покрытии) не доходят до лога: uvicorn настраивает
# свои логгеры, а корневой остаётся на WARNING.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

app = FastAPI(title='asr-adaptor', version='1.0')
_LLM = CFG.build_llm()
# Переопределения промптов стадий (`ASR_PROMPTS=<файл.toml>`) применяются ОДИН РАЗ при старте.
# Битый файл роняет старт, а не молчит: человек, тюнящий промпт, обязан узнать, что он не применился.
_PROMPTS = prompts.install(CFG.prompts_file)
if _PROMPTS:
    logging.getLogger('asr').info('промпты из %s: %s', CFG.prompts_file, ', '.join(_PROMPTS))


def _runner(mode: str):
    """`legacy` — `pipeline.run_pipeline`; `graph` — `graph.run.run_graph` (та же сигнатура)."""
    if mode == 'graph':
        from graph.run import run_graph  # noqa: PLC0415 — граф грузится, только если его выбрали
        return run_graph
    return run_pipeline


def _enriched(r: dict) -> dict:
    """Стандартный verbose_json + кастомный x_enriched.

    `segments` — НАСТОЯЩИЕ сегменты пасса-2 (раньше тут была заглушка `start == end == начало
    реплики`: реплика идёт до четырёх минут, и такое «время» бесполезно как якорь).
    """
    flat = [s for t in r['turns'] for s in t.get('segments') or []]
    return {
        'task': 'transcribe', 'language': 'ru', 'text': r['text'],
        'segments': [{'id': i, 'start': s['start'], 'end': s['end'], 'text': s['text']}
                     for i, s in enumerate(flat)],
        'x_enriched': {'format': 'morag-md-v1', 'markdown': r['markdown'], 'turns': r['turns'],
                       'raw_sidecar': r['raw_sidecar'], 'timing': r['timing'],
                       'speaker_map': r['speaker_map'],
                       'speaker_names': r.get('speaker_names', {}),
                       'name_conflicts': r.get('name_conflicts', []),
                       'coverage': r.get('coverage', {}),
                       'words': r.get('words'),
                       'glossary': r.get('glossary', []),
                       'doc_summary': r.get('doc_summary', ''),
                       # ⚠️ Отпечаток установки обязан дойти до АРТЕФАКТА, а не только до
                       # результата конвейера: `x_enriched` собирается по явному списку полей,
                       # и новое поле здесь легко забыть. Так и вышло — тест проверял результат
                       # `run_pipeline`, то есть не ту границу, и молчал.
                       'env': r.get('env', {}),
                       # Журнал переслушивания: где была порча, что услышало чистое ухо и что
                       # решили. Пусто — ключа нет вовсе, чтобы старые читатели не менялись.
                       **({'relisten': r['relisten']} if r.get('relisten') else {}),
                       # Журнал арбитража (ADR-0030) — тем же путём: x_enriched собирается явным
                       # перечнем, и поле, не названное здесь, до артефакта не доезжает — ловилось
                       # на первом живом прогоне: в логе 4 решения, в артефакте ноль.
                       **({'arbitration': r['arbitration']} if r.get('arbitration') else {}),
                       # Вердикты финал-раунда: что предложила модель, что принял код и почему.
                       **({'fixes': r['fixes']} if r.get('fixes') else {}),
                       # Граф (ASR_PIPELINE=graph): пройденные узлы, журнал инструментов, счётчики.
                       # У линейного конвейера ключа нет — прежний артефакт байт в байт.
                       **({'graph': r['graph']} if r.get('graph') else {})},
    }


@app.get('/health')
def health():
    return {'status': 'ok', 'downstream': audio_clients.health(), 'llm': CFG.llm_model}


@app.post('/warmup')
def warmup():
    """Заставить бэкенды ЗАГРУЗИТЬ МОДЕЛИ заранее — секундой тишины.

    ⚠️ Модели грузятся ПО ПЕРВОМУ ЗАПРОСУ, и первая стадия прогона молча стоит минуты: человек
    видит «идёт работа» и ничего больше (замерено живьём: три минуты тишины на первой записи).
    Греть надо пока человек заполняет поля, а не когда он уже ждёт результат.
    ⚠️ Ошибки НЕ роняют ответ: прогрев — удобство, а не условие работы; каждый бэкенд отчитывается
    сам за себя, и по этому ответу видно, кто из них не откликнулся.
    """
    import time
    import wave as wave_mod

    tmp = tempfile.mktemp(suffix='.wav')
    with wave_mod.open(tmp, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b'\x00' * 2 * 16000 * 2)      # две секунды тишины
    out = {}
    for name, call in (('diarizer', lambda: audio_clients.diarize(tmp, 1, 2)),
                       ('asr', lambda: audio_clients.asr(tmp)),
                       ('campp', lambda: audio_clients.campp(
                           tmp, [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]))):
        began = time.monotonic()
        try:
            call()
            out[name] = round(time.monotonic() - began, 1)
        except Exception as error:                     # noqa: BLE001 - прогрев не обязан удаться
            out[name] = f'err: {str(error)[:80]}'
    Path(tmp).unlink(missing_ok=True)
    return {'status': 'ok', 'warm': out}


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': 'asr-adaptor', 'object': 'model'}]}


@app.post('/v1/audio/transcriptions')
async def transcribe(file: UploadFile = File(...), model: str = Form('asr-adaptor'),
                     response_format: str = Form('verbose_json'), mode: str = Form(''),
                     episode: str = Form(''), title: str = Form(''), url: str = Form(''),
                     hints: str = Form(''), events: str = Form(''), pipeline: str = Form('')):
    # `pipeline=legacy|graph` — выбор конвейера на ОДНУ запись (A/B без перезапуска); пусто — конфиг.
    run = _runner(pipeline.strip().lower() if pipeline.strip() else CFG.pipeline)
    suffix = Path(file.filename or 'audio').suffix or '.mp3'
    tmp = tempfile.mktemp(suffix=suffix)
    Path(tmp).write_bytes(await file.read())

    # `hints` — единственный пер-джобовый канал, влияющий на КАЧЕСТВО: заведомо верные написания
    # этой записи (`{terms, names, about}`). Настройки корпуса живут в окружении и читаются один
    # раз при старте, подменить их на одну запись физически нельзя — а знание о записи у домена
    # обычно есть, и до сих пор оно пропадало. ⚠️ Битый JSON НЕ роняет расшифровку: без подсказок
    # запись выйдет ровно такой, какой выходила раньше.
    known = {}
    if hints.strip():
        try:
            known = json.loads(hints)
            if not isinstance(known, dict):
                raise ValueError('ожидался объект')
        except Exception as e:
            logging.getLogger('asr').warning('hints не разобраны (%s) — иду без них', e)
            known = {}

    async def job(progress):
        try:
            return _enriched(await run(
                tmp, _LLM, episode=episode, title=title, url=url, hints=known,
                progress=progress))
        finally:
            Path(tmp).unlink(missing_ok=True)

    if (mode or CFG.mode) == 'sync':
        return await job(lambda _: None)
    # `events=1` просит ленту стадий (см. jobs.py). Не попросили — всё как раньше, до байта.
    return {'job_id': jobs.submit(job, events=events not in ('', '0', 'false')), 'status': 'queued'}


@app.get('/v1/jobs/{job_id}')
def job_status(job_id: str, since: int | None = None):
    """Состояние задачи; с `?since=N` — ещё и лента событий новее курсора.

    ⚠️ БЕЗ `since` ответ обязан совпадать с прежним поле в поле: по нему живёт другой продукт.
    Поэтому ключи ленты добавляются только когда о ней спросили явно.
    """
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, 'job not found')
    out = {'job_id': job_id, 'status': j['status'], 'progress': j.get('progress', '')}
    if j['status'] == 'done':
        out['result'] = j['result']
    elif j['status'] == 'error':
        out['error'] = j.get('error')
    if since is not None:
        out['events'], out['cursor'], out['dropped'] = jobs.since(j, since)
    return out
