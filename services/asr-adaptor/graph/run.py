"""Прогон графа: `run_graph` — та же сигнатура и тот же результат, что у `pipeline.run_pipeline`.

Узлы идут по `nodes.NODES`; условие узла читается из конфига. Билет реестра голосов берётся ДО
первого await и отпускается в `finally` — той же цепочкой `pipeline._take_ticket/_release_turn`,
что у линейного конвейера: нумерация `Speaker_N` параллельных выпусков одна на оба.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time

from graph import nodes
from graph.deps import Deps
from graph.events import Emitter
from graph.state import State

log = logging.getLogger('asr')


def result(st: State) -> dict:
    """Результат в форме `run_pipeline` плюс ключ `graph` (узлы, журнал, счётчики)."""
    return {'markdown': st.markdown, 'text': st.text, 'turns': st.out_turns,
            'raw_sidecar': st.raw_side, 'timing': st.tm, 'speaker_map': st.mapping,
            'speaker_names': st.name_map, 'name_conflicts': st.name_conflicts,
            'coverage': st.cov, 'words': st.words_doc,
            'env': st.env,
            'glossary': st.gloss, 'doc_summary': st.dsum,
            **({'relisten': st.relisten_log} if st.relisten_log else {}),
            **({'arbitration': st.arbitrate_log} if st.arbitrate_log else {}),
            **({'fixes': st.round_log} if st.round_log else {}),
            'graph': {'nodes': list(st.done),
                      **({'journal': st.journal} if st.journal else {}),
                      **({'decisions': st.decisions} if st.decisions else {}),
                      **({'meter': st.meter} if st.meter else {})}}


async def run_graph(audio_path: str, llm, *, episode: str = '', title: str = '',
                    url: str = '', hints: dict | None = None, progress=None) -> dict:
    d = Deps(llm)
    st = State(audio_path=audio_path, episode=episode, title=title, url=url, hints=hints or {},
               t0=time.monotonic(), tmp=tempfile.mkdtemp(prefix='asr_'))
    ev = Emitter(progress, st.t0)
    st.ticket = d._take_ticket()  # порядок реестра = порядок поступления выпусков
    try:
        for name, fn, when in nodes.NODES:
            if when is not None and not when(d.cfg):
                continue
            await fn(st, d, ev)
            st.done.append(name)
        st.tm['total_s'] = round(time.monotonic() - st.t0, 1)
        return result(st)
    finally:
        d._release_turn(st.ticket)  # идемпотентно: упавший выпуск не вешает очередь реестра
        shutil.rmtree(st.tmp, ignore_errors=True)
