"""Async job-store для длинных транскрайб-джоб (выпуск ~8-15 мин).

In-flight выпусков — `ASR_MAX_JOBS` (дефолт 1 = прежнее поведение). Параллель здесь — это
конвейеризация СТАДИЙ, а не одновременный трэшинг GPU: каждый аудио-ресурс (whisper, pyannote,
CAM++, MPS-выравнивание) отдельно гейтится семафором в pipeline, поэтому GPU-стадии выпуска B
идут, пока у A работает LLM. In-memory dict; progress-колбэк — для GET /v1/jobs/{id}.
"""
from __future__ import annotations

import asyncio
import time
import uuid

from config import CFG

_JOBS: dict[str, dict] = {}
_SEM = asyncio.Semaphore(max(1, CFG.max_jobs))


def get(job_id: str):
    return _JOBS.get(job_id)


async def _run(job_id: str, coro_factory):
    job = _JOBS[job_id]
    async with _SEM:
        job['status'] = 'running'
        job['started'] = time.time()
        try:
            job['result'] = await coro_factory(lambda m: job.update(progress=m))
            job['status'] = 'done'
        except Exception as e:
            job['status'] = 'error'
            job['error'] = str(e)[:500]
        job['finished'] = time.time()


def submit(coro_factory) -> str:
    """coro_factory(progress_cb)->coroutine. Создаёт job, запускает в фоне, возвращает job_id."""
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {'status': 'queued', 'progress': '', 'created': time.time()}
    asyncio.create_task(_run(job_id, coro_factory))
    return job_id
