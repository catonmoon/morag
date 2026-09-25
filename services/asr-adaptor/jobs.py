"""Async job-store для длинных транскрайб-джоб (выпуск ~8-15 мин).

In-flight выпусков — `ASR_MAX_JOBS` (дефолт 1 = прежнее поведение). Параллель здесь — это
конвейеризация СТАДИЙ, а не одновременный трэшинг GPU: каждый аудио-ресурс (whisper, pyannote,
CAM++, MPS-выравнивание) отдельно гейтится семафором в pipeline, поэтому GPU-стадии выпуска B
идут, пока у A работает LLM. In-memory dict; progress-колбэк — для GET /v1/jobs/{id}.

**Лента событий (необязательная).** Тот же колбэк умеет принимать не строку, а СЛОВАРЬ — тогда
это событие стадии: что нашла диаризация, какой кусок распознан, какую замену предложила модель и
принял ли её сторож. События копятся в кольце задачи и выдаются по курсору — так клиент
показывает работу, а не полосу «примерно 30%».

⚠️ **Лента включается У ЗАДАЧИ, а не у ручки.** Не попросили — кольца нет, словари никто не шлёт,
ответ опроса не меняется ни одним ключом: на этом же адаптере живёт другой продукт, и он не
должен заметить ничего.
"""
from __future__ import annotations

import asyncio
import collections
import time
import uuid

from config import CFG

_JOBS: dict[str, dict] = {}
_SEM = asyncio.Semaphore(max(1, CFG.max_jobs))

# Сколько событий держим у ЖИВОЙ задачи и сколько оставляем после завершения. Кольцо ограничено
# намеренно: `_JOBS` и так растёт вечно (задачи не выселяются — на это поведение полагается чужой
# клиент), и неограниченный список внутри был бы ухудшением. Хвост нужен, чтобы последний опрос
# клиента забрал концовку прогона и не увидел «пропущено».
RING = 2000
TAIL = 200


def get(job_id: str):
    return _JOBS.get(job_id)


def _progress(job: dict):
    """Колбэк прогресса: строка — как раньше, словарь — событие в ленту.

    ⚠️ Проверка типа обязательна. Без неё одно событие попало бы в поле `progress`, которое чужой
    клиент читает как строку и печатает человеку.
    """
    def report(m):
        if isinstance(m, str):
            job['progress'] = m
            return
        ring = job.get('events')
        if ring is None:          # ленты не просили — ни аллокаций, ни работы
            return
        job['seq'] += 1
        # ⚠️ Поле конверта — `seq`, а не `i`: `i` у события уже занято СМЫСЛОМ (номер куска), и
        # порядковый номер молча затирал его. Ловилось тестом ленты: курсор шёл 1, 1, 1.
        m['seq'] = job['seq']
        if m.get('say'):          # событие может заодно обновить человеческую строку
            job['progress'] = m['say']
        ring.append(m)
    return report


async def _run(job_id: str, coro_factory):
    job = _JOBS[job_id]
    async with _SEM:
        job['status'] = 'running'
        job['started'] = time.time()
        try:
            job['result'] = await coro_factory(_progress(job))
            job['status'] = 'done'
        except Exception as e:
            job['status'] = 'error'
            job['error'] = str(e)[:500]
        job['finished'] = time.time()
        ring = job.get('events')
        if ring is not None and len(ring) > TAIL:
            job['events'] = collections.deque(list(ring)[-TAIL:], maxlen=RING)


def submit(coro_factory, *, events: bool = False) -> str:
    """coro_factory(progress_cb)->coroutine. Создаёт job, запускает в фоне, возвращает job_id."""
    job_id = uuid.uuid4().hex[:12]
    job: dict = {'status': 'queued', 'progress': '', 'created': time.time()}
    if events:
        job['events'] = collections.deque(maxlen=RING)
        job['seq'] = 0
    _JOBS[job_id] = job
    asyncio.create_task(_run(job_id, coro_factory))
    return job_id


def since(job: dict, cursor: int) -> tuple[list[dict], int, int]:
    """События новее курсора: (события, новый курсор, сколько пропущено).

    ⚠️ Пропуск СЧИТАЕМ и отдаём, а не замалчиваем: кольцо ограничено, и клиент, отставший на
    минуту, обязан узнать, что дыра есть, — иначе он покажет плавную картинку с провалом в
    середине, и никто не поймёт, почему у записи «потерялись» куски.
    """
    ring = job.get('events')
    if ring is None:
        return [], cursor, 0
    items = [e for e in ring if e['seq'] > cursor]
    dropped = ring[0]['seq'] - cursor - 1 if ring and ring[0]['seq'] > cursor + 1 else 0
    return items, (items[-1]['seq'] if items else cursor), dropped
