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
    """Интерфейс публикации статуса индексации.

    Жизненный цикл документа в активном пути:
        document_start(doc_id, title, url)   → попадает в current_docs (in-flight)
        document_set_chunks(doc_id, total)   → когда чанкер посчитал чанки
        document_chunk_done(doc_id) × N      → по мере генерации context per chunk
        document_done(doc_id)                → удаление из current_docs + processed += 1

    Skip-up-to-date / phase-counter callers вызывают только document_done — это
    просто инкремент processed без in-flight записи.
    """

    def start_phase(self, name: str, total: int) -> None:
        """Начать новую фазу с известным числом единиц работы."""
        ...

    def document_start(self, doc_id: str, title: str | None = None, url: str | None = None) -> None:
        """Начать обработку документа — добавить в in-flight список."""
        ...

    def document_set_chunks(self, doc_id: str, total: int) -> None:
        """Зафиксировать количество чанков документа (после чанкинга)."""
        ...

    def document_chunk_done(self, doc_id: str) -> None:
        """Инкремент chunks_done для документа в in-flight."""
        ...

    def document_done(self, doc_id: str) -> None:
        """Завершить обработку: убрать из in-flight (если был) + processed += 1."""
        ...

    def finish(self, state: State, error: str | None = None) -> None:
        """Завершить отчёт. state: completed | cancelled | failed."""
        ...


class NullStatusReporter:
    """Реализация-пустышка для CLI без подключённой консоли."""

    def start_phase(self, name: str, total: int) -> None:
        pass

    def document_start(self, doc_id: str, title: str | None = None, url: str | None = None) -> None:
        pass

    def document_set_chunks(self, doc_id: str, total: int) -> None:
        pass

    def document_chunk_done(self, doc_id: str) -> None:
        pass

    def document_done(self, doc_id: str) -> None:
        pass

    def finish(self, state: State, error: str | None = None) -> None:
        pass


class FileStatusReporter:
    """Пишет статус в JSON-файл с atomic rename.

    Поток событий:
      start_phase('indexing_local', 42)
      document_start('id1', title='Foo', url='http://...')   # in-flight: 1
      document_set_chunks('id1', 12)
      document_chunk_done('id1') × 12
      document_done('id1')                                   # processed=1, in-flight: 0
      ...
      start_phase('bm25_chunks', 1)
      document_done('bm25_chunks')                            # phase-counter, без start
      finish('completed')
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
        self._in_flight: dict[str, dict] = {}  # doc_id -> {title, url, started_at, chunks_done, chunks_total}
        self._error: str | None = None
        self._write()

    def start_phase(self, name: str, total: int) -> None:
        with self._lock:
            self._state = 'running'
            self._phase = name
            self._processed = 0
            self._total = total
            self._in_flight.clear()
            self._write()

    def document_start(self, doc_id: str, title: str | None = None, url: str | None = None) -> None:
        with self._lock:
            self._in_flight[doc_id] = {
                'doc_id': doc_id,
                'title': title,
                'url': url,
                'started_at': _now_iso(),
                'chunks_done': 0,
                'chunks_total': None,
            }
            self._write()

    def document_set_chunks(self, doc_id: str, total: int) -> None:
        with self._lock:
            entry = self._in_flight.get(doc_id)
            if entry is not None:
                entry['chunks_total'] = total
                self._write()

    def document_chunk_done(self, doc_id: str) -> None:
        with self._lock:
            entry = self._in_flight.get(doc_id)
            if entry is not None:
                entry['chunks_done'] = entry.get('chunks_done', 0) + 1
                self._write()

    def document_done(self, doc_id: str) -> None:
        with self._lock:
            self._in_flight.pop(doc_id, None)
            self._processed += 1
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
            'current_docs': list(self._in_flight.values()),
            'started_at': self._started_at,
            'updated_at': _now_iso(),
            'error': self._error,
        }
        tmp = self._path.with_suffix(self._path.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        os.replace(tmp, self._path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')