"""Состояние прогона: что узел читает и что пишет — одним объектом, без замыканий.

Поля-значения (звук, спаны, сегменты, реплики) отдельно от полей-накопителей (`tm`, `journal`,
`decisions`, `meter`, `done`). Поле с `metadata={'checkpoint': False}` в чекпойнт не пишется:
`cents` — центроиды голосов, биометрия живого человека, ей место только в реестре под 0600;
`counter` — токенизатор, его дешевле пересоздать, чем сериализовать.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class State:
    # --- вход -----------------------------------------------------------------------------------
    audio_path: str
    episode: str = ''
    title: str = ''
    url: str = ''
    hints: dict = field(default_factory=dict)
    t0: float = 0.0
    tmp: str = ''
    ticket: int | None = None
    # --- prepare --------------------------------------------------------------------------------
    wav: str = ''
    audio_sec: float = 0.0
    env: dict = field(default_factory=dict)
    # --- diarize / pass1 ------------------------------------------------------------------------
    spans: list = field(default_factory=list)
    segs: list = field(default_factory=list)
    full_text: str = ''
    p1_holes: list = field(default_factory=list)
    # --- glossary -------------------------------------------------------------------------------
    gloss: list = field(default_factory=list)
    seed: list = field(default_factory=list)
    hint_set: frozenset = frozenset()
    # --- chunking / pass2 -----------------------------------------------------------------------
    chunks: list = field(default_factory=list)
    counter: Any = field(default=None, metadata={'checkpoint': False})
    # --- relisten / arbitrate -------------------------------------------------------------------
    relisten_log: list = field(default_factory=list)
    arbitrate_log: list = field(default_factory=list)
    # --- turns / final-round --------------------------------------------------------------------
    turns: list = field(default_factory=list)
    dsum: str = ''
    protect: list = field(default_factory=list)
    known: tuple = ()
    raw_side: dict = field(default_factory=dict)
    round_log: list = field(default_factory=list)
    # --- speakers / naming ----------------------------------------------------------------------
    cents: dict = field(default_factory=dict, metadata={'checkpoint': False})   # биометрия
    air: dict = field(default_factory=dict)
    mapping: dict = field(default_factory=dict)
    name_map: dict = field(default_factory=dict)
    name_conflicts: list = field(default_factory=list)
    # --- assemble / align / coverage ------------------------------------------------------------
    markdown: str = ''
    text: str = ''
    out_turns: list = field(default_factory=list)
    words_doc: Any = None
    cov: dict = field(default_factory=dict)
    # --- сквозные накопители --------------------------------------------------------------------
    tm: dict = field(default_factory=dict)          # тайминги стадий — как у конвейера
    journal: list = field(default_factory=list)     # каждый вызов инструмента
    decisions: list = field(default_factory=list)   # решения по местам
    meter: dict = field(default_factory=dict)       # вызовы LLM, секунды звука, исчерпания бюджета
    done: list = field(default_factory=list)        # пройденные узлы — для возобновления

    @classmethod
    def checkpoint_fields(cls) -> list[str]:
        """Имена полей, которые едут в чекпойнт (биометрия и объекты — нет)."""
        return [f.name for f in fields(cls) if f.metadata.get('checkpoint', True)]
