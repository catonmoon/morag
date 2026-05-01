"""StatusReporter — публикация прогресса индексации для внешних наблюдателей.

Используется консолью (services/console) для отображения прогресса и ETA.
В CLI-режиме без env MORAG_STATUS_FILE применяется NullStatusReporter — no-op.

State-file пишется атомарно (tmp + os.replace) — читатель никогда не видит
полузаписанный JSON.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

State = Literal['idle', 'running', 'completed', 'cancelled', 'failed']


class StatusReporter(Protocol):
    """Интерфейс публикации статуса индексации."""

    def start_phase(self, name: str, total: int) -> None:
        """Начать новую фазу с известным числом единиц работы."""
        ...

    def document_done(self, doc_id: str) -> None:
        """Отметить обработку одного документа."""
        ...

    def finish(self, state: State, error: str | None = None) -> None:
        """Завершить отчёт. state: completed | cancelled | failed."""
        ...


class NullStatusReporter:
    """Реализация-пустышка для CLI без подключённой консоли."""

    def start_phase(self, name: str, total: int) -> None:
        pass

    def document_done(self, doc_id: str) -> None:
        pass

    def finish(self, state: State, error: str | None = None) -> None:
        pass


class FileStatusReporter:
    """Пишет статус в JSON-файл с atomic rename.

    Поток событий:
      start_phase('indexing_local', 42) → state=running, processed=0, total=42
      document_done('id1') → processed=1
      ...
      start_phase('bm25_chunks', 1) → новая фаза, processed сбрасывается
      finish('completed') → state=completed
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._started_at = _now_iso()
        self._state: State = 'idle'
        self._phase = ''
        self._processed = 0
        self._total = 0
        self._current_doc_id: str | None = None
        self._error: str | None = None
        self._write()

    def start_phase(self, name: str, total: int) -> None:
        with self._lock:
            self._state = 'running'
            self._phase = name
            self._processed = 0
            self._total = total
            self._current_doc_id = None
            self._write()

    def document_done(self, doc_id: str) -> None:
        with self._lock:
            self._processed += 1
            self._current_doc_id = doc_id
            self._write()

    def finish(self, state: State, error: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._error = error
            self._write()

    def _write(self) -> None:
        payload = {
            'state': self._state,
            'phase': self._phase,
            'processed': self._processed,
            'total': self._total,
            'current_doc_id': self._current_doc_id,
            'started_at': self._started_at,
            'updated_at': _now_iso(),
            'error': self._error,
        }
        tmp = self._path.with_suffix(self._path.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        os.replace(tmp, self._path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')
