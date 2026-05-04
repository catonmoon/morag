"""RunContext — глобальный счётчик прогонов индексации.

При каждом вызове `cmd_index` / `cmd_rebuild_km` бампится `run_number` (один
на весь прогон, identical timestamp). Все upsert'ы документов и чанков этого
прогона помечаются `payload['run_number']` и `payload['indexed_at']`.

См. ADR-0012, секция «5. Run versioning».

Counter persistится в state-файле (`conf/state/run_counter.json`).
Если файл потерян — recovery через `max(run_number)` из существующих
документов в Qdrant (избегаем коллизий с историческими прогонами).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path('/app/conf/state/run_counter.json')


@dataclass(frozen=True)
class RunContext:
    """Контекст одного прогона индексации.

    Создаётся через `RunContext.begin()` в начале `cmd_index`/`cmd_rebuild_km`.
    Передаётся в `IndexingPipeline` — pipeline стампит run_number/indexed_at
    в payload каждого документа и чанка перед upsert.

    indexed_at заморожен в момент begin() — все точки одного прогона имеют
    одинаковый timestamp, не текущее время на момент upsert каждой точки.
    """

    run_number: int
    indexed_at: str  # ISO timestamp, frozen at begin()

    @classmethod
    def begin(
        cls,
        state_path: str | Path | None = None,
        recover_from_qdrant: 'callable | None' = None,
    ) -> 'RunContext':
        """Bump counter, freeze indexed_at, return new context.

        recover_from_qdrant: опциональный callable () → int, вызывается если
        state-файл отсутствует. Должен вернуть max(run_number) из docs collection.
        Если не передан — при отсутствии файла используется 0.
        """
        path = Path(state_path) if state_path else _resolve_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        current = _read_counter(path)
        if current is None:
            if recover_from_qdrant is not None:
                try:
                    current = recover_from_qdrant()
                    logger.warning(
                        'run_counter.json missing, recovered max from Qdrant: %d',
                        current,
                    )
                except Exception as e:
                    logger.warning(
                        'run_counter recovery failed: %s — starting from 0', e,
                    )
                    current = 0
            else:
                current = 0

        new_counter = current + 1
        indexed_at = _now_iso()
        _write_counter(path, new_counter, indexed_at)

        logger.info('Run #%d started (indexed_at=%s)', new_counter, indexed_at)
        return cls(run_number=new_counter, indexed_at=indexed_at)


def _resolve_state_path() -> Path:
    """Путь по умолчанию: env MORAG_RUN_COUNTER_FILE или /app/conf/state/."""
    env = os.environ.get('MORAG_RUN_COUNTER_FILE')
    if env:
        return Path(env)
    return DEFAULT_STATE_PATH


def _read_counter(path: Path) -> int | None:
    """None если файла нет / не читается / повреждён."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        value = data.get('current_run')
        if isinstance(value, int):
            return value
        logger.warning('run_counter.json: invalid current_run=%r, ignoring', value)
        return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning('run_counter.json read failed: %s', e)
        return None


def _write_counter(path: Path, counter: int, last_bumped_at: str) -> None:
    """Atomic write через tmp + rename. Не работает на bind-mounted single-files
    (см. config_io); но мы пишем в директорию-volume, не single-file mount —
    тут безопасно.
    """
    payload = {'current_run': counter, 'last_bumped_at': last_bumped_at}
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')
