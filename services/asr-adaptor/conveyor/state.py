"""Состояние прогона: что узел читает и что пишет — одним объектом, без замыканий.

Поля-значения (звук, спаны, сегменты, реплики) отдельно от полей-накопителей (`tm`, `journal`,
`decisions`, `meter`, `done`). Поле с `metadata={'checkpoint': False}` в чекпойнт не пишется:
`cents` — центроиды голосов, биометрия живого человека, ей место только в реестре под 0600;
`counter` — токенизатор, его дешевле пересоздать, чем сериализовать.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class State:
    # --- вход -----------------------------------------------------------------------------------
    audio_path: str
    episode: str = ''
    title: str = ''
    url: str = ''
    hints: dict = field(default_factory=dict)
    editor: bool | None = None                      # редактор на этот прогон: None — из конфига
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
    seam_log: list = field(default_factory=list)     # шов: где снят повтор соседа
    voice_log: dict = field(default_factory=dict)    # голоса: восстановление диалога и перерезка по словам
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

    # --- чекпойнт -------------------------------------------------------------------------------
    # ⚠️ Реплики держат ссылки на те же словари, что и куски (`turns[].chunks`, `segments`), а
    # JSON эти связи не знает: после загрузки правка куска не отразится в реплике. Узлы после
    # `turns` куски уже не трогают, поэтому это безопасно — но помнить стоит.

    def to_json(self) -> dict:
        out: dict = {}
        for name in self.checkpoint_fields():
            v = getattr(self, name)
            if isinstance(v, (frozenset, set, tuple)):
                v = sorted(v) if isinstance(v, (frozenset, set)) else list(v)
            out[name] = v
        return out

    @classmethod
    def from_json(cls, data: dict) -> 'State':
        st = cls(audio_path=str(data.get('audio_path') or ''))
        for name in cls.checkpoint_fields():
            if name in data:
                setattr(st, name, data[name])
        st.hint_set = frozenset(st.hint_set or ())
        st.known = tuple(st.known or ())
        return st

    def save(self, path: str | Path) -> None:
        """Атомарно и под 0600: в чекпойнте расшифровка целиком, ей на диске не место открытой."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(self.to_json(), fh, ensure_ascii=False, default=str)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> 'State':
        with open(path, encoding='utf-8') as fh:
            return cls.from_json(json.load(fh))
