"""IndexerControlPlane — управление одной асинхронной задачей индексации в процессе.

Living внутри morag-indexer (см. cli/main.py::cmd_serve), доступен по HTTP на
внутреннем порту (default 9090). Console обращается через HTTP вместо спавна
subprocess'ов — это даёт чистое разделение ролей: console = control plane / UI,
indexer = runtime.

Принцип работы:
- Один indexing-task за раз. Защита через asyncio.Lock + проверка self._task.
- start_index() / start_rebuild_km() запускают cmd_index / cmd_rebuild_km
  как asyncio.Task — НЕ subprocess. Те же cancel_event и status_reporter
  что использует CLI-режим, переиспользуются.
- stop(grace) выставляет cancel_event и ждёт graceful завершения.
- При timeout — task.cancel() + abandon (state-file перезапишется при
  следующем /start).
- Cron-job из cmd_serve вызывает control_plane.start_index() — тот же lock,
  тот же state-file. On-demand из console и cron не конфликтуют.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from morag.indexing.status_reporter import FileStatusReporter
from morag.setup_gate import SetupIncomplete, is_setup_complete

logger = logging.getLogger(__name__)

# grace_seconds=None в stop() означает «ждать без таймаута» (graceful stop без
# принудительного прерывания). Эскалация на kill — только при явно заданном
# config.indexing.stop_grace_seconds или per-request grace_seconds.


@dataclass(frozen=True)
class RunInfo:
    started_at: str
    kind: Literal['index', 'rebuild_km']
    reset: bool = False
    scope: str | None = None  # реиндекс-эффорт: 'all' / имя источника; None = обычный прогон


class AlreadyRunning(RuntimeError):
    """start_*() вызван при ещё-живой задаче."""


class IndexerControlPlane:
    """Один в процессе. Шарится между cron'ом и HTTP-роутами cmd_serve.

    Пример (внутри cmd_serve):
        cp = IndexerControlPlane(
            config_path='/app/conf/config.yml',
            status_file_path='/app/conf/state/index_status.json',
            run_index=lambda **kw: cmd_index('/app/conf/config.yml', **kw),
            run_rebuild_km=lambda **kw: cmd_rebuild_km('/app/conf/config.yml', **kw),
        )
    """

    def __init__(
        self,
        config_path: str | Path,
        status_file_path: str | Path,
        run_index: Callable[..., Awaitable[None]],
        run_rebuild_km: Callable[..., Awaitable[None]],
        stop_grace_seconds: int | None = None,
    ) -> None:
        self._config_path = str(config_path)
        self._status_file_path = Path(status_file_path)
        self._run_index = run_index
        self._run_rebuild_km = run_rebuild_km
        self._stop_grace_seconds = stop_grace_seconds

        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._cancel_event: asyncio.Event | None = None
        self._current_run: RunInfo | None = None

    # -------- Public API --------

    async def start_index(
        self, reset: bool = False, reindex_scope: str | None = None,
    ) -> RunInfo:
        """Запустить индексацию. `reindex_scope` != None → плавный реиндекс-эффорт
        по scope ('all' или имя источника), см. ADR-0014."""
        async with self._lock:
            self._raise_if_running()
            self._raise_if_setup_incomplete()
            cancel_event = asyncio.Event()
            reporter = FileStatusReporter(self._status_file_path)

            async def _runner() -> None:
                try:
                    await self._run_index(
                        reset=reset,
                        reindex_scope=reindex_scope,
                        cancel_event=cancel_event,
                        status_reporter=reporter,
                    )
                except asyncio.CancelledError:
                    reporter.finish('cancelled')
                    raise
                except Exception as e:
                    logger.exception('Index task crashed')
                    reporter.finish('failed', error=f'{type(e).__name__}: {e}')

            info = RunInfo(
                started_at=_now_iso(),
                kind='index',
                reset=reset,
                scope=reindex_scope,
            )
            self._cancel_event = cancel_event
            self._task = asyncio.create_task(_runner(), name='indexer-index')
            self._current_run = info
            logger.info(
                'Indexer task started: kind=index reset=%s reindex_scope=%s',
                reset, reindex_scope,
            )
            return info

    async def start_rebuild_km(self) -> RunInfo:
        async with self._lock:
            self._raise_if_running()
            self._raise_if_setup_incomplete()
            cancel_event = asyncio.Event()
            reporter = FileStatusReporter(self._status_file_path)

            async def _runner() -> None:
                try:
                    await self._run_rebuild_km(
                        cancel_event=cancel_event,
                        status_reporter=reporter,
                    )
                except asyncio.CancelledError:
                    reporter.finish('cancelled')
                    raise
                except Exception as e:
                    logger.exception('Rebuild-km task crashed')
                    reporter.finish('failed', error=f'{type(e).__name__}: {e}')

            info = RunInfo(started_at=_now_iso(), kind='rebuild_km')
            self._cancel_event = cancel_event
            self._task = asyncio.create_task(_runner(), name='indexer-rebuild-km')
            self._current_run = info
            logger.info('Indexer task started: kind=rebuild_km')
            return info

    async def stop(self, grace_seconds: int | None = None) -> Literal['graceful', 'killed', 'not_running']:
        if not self.is_running():
            return 'not_running'
        assert self._task is not None and self._cancel_event is not None
        effective_grace = grace_seconds if grace_seconds is not None else self._stop_grace_seconds
        self._cancel_event.set()
        if effective_grace is None:
            # Ждём без таймаута — не прерываем принудительно. Для kill есть
            # отдельный endpoint /control/kill.
            logger.info('Stopping indexer task (no grace timeout, waiting until done)')
            await asyncio.shield(self._task)
            return 'graceful'
        logger.info('Stopping indexer task (grace=%ds)', effective_grace)
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=effective_grace)
            return 'graceful'
        except asyncio.TimeoutError:
            logger.warning('Grace timeout exceeded, cancelling task')
            self._task.cancel()
            # abandon — не ждём. Это редкий путь, плюс ждать может долго.
            return 'killed'

    async def kill(self) -> Literal['killed', 'not_running']:
        if not self.is_running():
            return 'not_running'
        assert self._task is not None
        self._task.cancel()
        return 'killed'

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        return {
            'is_running': self.is_running(),
            'run': {
                'started_at': self._current_run.started_at,
                'kind': self._current_run.kind,
                'reset': self._current_run.reset,
                'scope': self._current_run.scope,
            } if self._current_run else None,
            'progress': self._read_status_file(),
        }

    # -------- Internals --------

    def _raise_if_running(self) -> None:
        if self.is_running():
            raise AlreadyRunning(f'Task already running: {self._current_run}')

    def _raise_if_setup_incomplete(self) -> None:
        ok, blockers = is_setup_complete(self._config_path)
        if not ok:
            raise SetupIncomplete(blockers)

    def setup_status(self) -> dict[str, Any]:
        """For checklist endpoint — gate-check без побочек."""
        ok, blockers = is_setup_complete(self._config_path)
        return {'ok': ok, 'blockers': blockers}

    def _read_status_file(self) -> dict | None:
        if not self._status_file_path.exists():
            return None
        try:
            return json.loads(self._status_file_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')
