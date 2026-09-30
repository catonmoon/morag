"""Лента событий конвейера — та же форма, что у прежнего линейного конвейера.

Окно загрузки записи читает `job.meta`, `diar.spans`, `draft.window`, `chunk.*`, `relisten.span`,
`arbitrate.swap`, `turn.*`, `spk.*`, `stage.*` — их форму держит золотой тест (снимок ленты в
`tests/asr_adaptor/golden/`). С 30.09 ещё `arbitrate.chunk` (счётчик и пометки читателя),
`arbitrate.swap.clean` (вариант третьего голоса) и события редактора `editor.page`,
`editor.listen`, `editor.done`, `turn.fix.witness`. Строка прогресса (`step`) — тоже прежняя:
по ней живёт чужой клиент, который ленту не просит.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger('asr')


class Emitter:
    """`step` — строка прогресса, `emit` — событие ленты, `stage`/`stage_done` — обёртки стадии.

    ⚠️ `emit` обёрнут в try/except намеренно: показ работы — украшение, и оно не имеет права
    уронить расшифровку. ⚠️⚠️ Первый аргумент `emit` — ПОЗИЦИОННЫЙ (`/`): у событий есть поле
    `kind`, и с именованным параметром прогон падал на «multiple values for argument 'kind'»
    после всей тяжёлой работы.
    """

    def __init__(self, progress, t0: float) -> None:
        self._progress = progress
        self._t0 = t0

    @property
    def enabled(self) -> bool:
        return bool(self._progress)

    def step(self, m: str) -> None:
        if self._progress:
            self._progress(m)

    def emit(self, kind, /, **f) -> None:
        if not self._progress:
            return
        try:
            self._progress({'t': kind, 'at': round(time.monotonic() - self._t0, 2), **f})
        except Exception:  # noqa: BLE001
            log.debug('событие %s не отправилось', kind, exc_info=True)

    def stage(self, name: str, **f) -> None:
        """Начало стадии: прежняя строка прогресса И событие — одним вызовом."""
        self.step(name)
        self.emit('stage.start', stage=name, **f)

    def stage_done(self, name: str, sec, **f) -> None:
        self.emit('stage.end', stage=name, sec=sec, **f)
