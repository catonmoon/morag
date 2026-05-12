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
from collections import deque
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

    def document_failed(self, doc_id: str, title: str | None, exc: BaseException) -> None:
        """Зафиксировать ошибку обработки документа — для UI-видимости.

        Вызывается из exception-handler'а в pipeline.py после rollback.
        Накапливает счётчик errors_count и список последних N в recent_errors.
        """
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

    def document_failed(self, doc_id: str, title: str | None, exc: BaseException) -> None:
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
        self._processed_real = 0
        self._total = 0
        self._in_flight: dict[str, dict] = {}  # doc_id -> {title, url, started_at, chunks_done, chunks_total}
        # Кольцевой буфер последних N completion-событий — UI считает rolling-rate
        # за окно (60с) для адаптивного ETA. kind='real' если документ прошёл
        # полный цикл (document_start был), 'skip' если только document_done
        # (idempotency пропуск). 300 хватает на ~5 минут плотного потока.
        self._recent_completions: deque[dict] = deque(maxlen=300)
        # Учёт ошибок обработки документов в текущем ране — нарастающий счётчик
        # + последние 20 для UI (показ списка по клику).
        self._errors_count: int = 0
        self._recent_errors: deque[dict] = deque(maxlen=20)
        self._error: str | None = None
        self._write()

    def start_phase(self, name: str, total: int) -> None:
        with self._lock:
            self._state = 'running'
            self._phase = name
            self._processed = 0
            self._processed_real = 0
            self._total = total
            self._in_flight.clear()
            self._recent_completions.clear()
            # errors_count и recent_errors НЕ сбрасываются — они per-RUN, не per-phase.
            # Один run = один FileStatusReporter instance (создан в cmd_index),
            # пробегает phases [stubs, indexing_*, bm25, knowledge_map] и собирает
            # ошибки за весь прогон.
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
            was_in_flight = self._in_flight.pop(doc_id, None) is not None
            self._processed += 1
            if was_in_flight:
                self._processed_real += 1
            self._recent_completions.append({
                'ts': _now_iso(),
                'kind': 'real' if was_in_flight else 'skip',
            })
            self._write()

    def document_failed(
        self, doc_id: str, title: str | None, exc: BaseException,
    ) -> None:
        """Зафиксировать ошибку обработки документа. Вызывается из exception-handler'а
        в `IndexingPipeline.run()` после rollback частично сохранённых чанков.
        """
        with self._lock:
            self._errors_count += 1
            self._recent_errors.append({
                'doc_id': doc_id,
                'title': title or '',
                'error_type': type(exc).__name__,
                'error_msg': str(exc)[:500],
                'ts': _now_iso(),
            })
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
            'processed_real': self._processed_real,
            'total': self._total,
            'current_docs': list(self._in_flight.values()),
            'recent_completions': list(self._recent_completions),
            'errors_count': self._errors_count,
            'recent_errors': list(self._recent_errors),
            'started_at': self._started_at,
            'updated_at': _now_iso(),
            'error': self._error,
        }
        tmp = self._path.with_suffix(self._path.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        os.replace(tmp, self._path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')