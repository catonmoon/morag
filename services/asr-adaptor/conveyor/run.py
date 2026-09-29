"""Прогон конвейера: `run_conveyor` — узлы по `nodes.NODES`, условие узла читается из конфига.

Результат — в форме прежнего линейного конвейера (его заменил этот, 29.09) плюс ключ `conveyor`.
Билет реестра голосов берётся ДО первого await и отпускается в `finally` (`pipeline._take_ticket/
_release_turn`): нумерация `Speaker_N` параллельных выпусков идёт в порядке поступления.

Чекпойнты (`ASR_CHECKPOINTS=<каталог>`): состояние пишется после каждого узла;
`run_conveyor(resume=<файл>)` продолжает с первого непройденного узла. Это ответ на «402 на
финал-раунде после десяти минут GPU» и дешёвый способ настраивать промпты финал-раунда без звука.
⚠️ Звук при возобновлении нужен снова: временный каталог прежнего прогона удалён, узел `prepare`
проходится заново (дёшево — конвертация в wav).
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

from conveyor import nodes
from conveyor.deps import Deps
from conveyor.events import Emitter
from conveyor.state import State

log = logging.getLogger('asr')


def result(st: State) -> dict:
    """Результат прежней формы плюс ключ `conveyor` (узлы, журнал, решения, счётчики)."""
    return {'markdown': st.markdown, 'text': st.text, 'turns': st.out_turns,
            'raw_sidecar': st.raw_side, 'timing': st.tm, 'speaker_map': st.mapping,
            'speaker_names': st.name_map, 'name_conflicts': st.name_conflicts,
            'coverage': st.cov, 'words': st.words_doc,
            'env': st.env,
            'glossary': st.gloss, 'doc_summary': st.dsum,
            **({'relisten': st.relisten_log} if st.relisten_log else {}),
            **({'arbitration': st.arbitrate_log} if st.arbitrate_log else {}),
            **({'fixes': st.round_log} if st.round_log else {}),
            'conveyor': {'nodes': list(st.done),
                      **({'journal': st.journal} if st.journal else {}),
                      **({'decisions': st.decisions} if st.decisions else {}),
                      **({'meter': st.meter} if st.meter else {})}}


async def run_conveyor(audio_path: str, llm, *, episode: str = '', title: str = '',
                    url: str = '', hints: dict | None = None, progress=None,
                    resume: str | None = None, editor: bool | None = None) -> dict:
    """`editor` — редактор на этот прогон (поле формы `editor=`); None — как в конфиге."""
    d = Deps(llm)
    cfg = d.cfg
    if resume:
        st = State.load(resume)
        st.audio_path, st.t0, st.tmp = audio_path, time.monotonic(), tempfile.mkdtemp(prefix='asr_')
        st.done = [n for n in st.done if n != 'prepare']       # звук конвертируется заново
        if 'chunking' in st.done:
            st.counter = d.WhisperTokenCounter(cfg.whisper_tokenizer)
        st.tm['resumed_from'] = str(resume)
        log.info('конвейер: возобновление с чекпойнта %s, пройдено %s', resume, ', '.join(st.done))
    else:
        st = State(audio_path=audio_path, episode=episode, title=title, url=url, hints=hints or {},
                   editor=editor, t0=time.monotonic(), tmp=tempfile.mkdtemp(prefix='asr_'))
    if editor is not None:
        st.editor = editor
    ev = Emitter(progress, st.t0)
    ckpt = (Path(cfg.checkpoints) / f"{st.episode or 'adhoc'}-{int(time.time())}.json"
            if cfg.checkpoints else None)
    st.ticket = d._take_ticket()  # порядок реестра = порядок поступления выпусков
    try:
        for name, fn, when in nodes.NODES:
            if when is not None and not when(cfg, st):
                continue
            if name in st.done:
                continue
            await fn(st, d, ev)
            st.done.append(name)
            if cfg.node_events:
                ev.emit('conveyor.node', node=name, meter=dict(st.meter))
            if ckpt is not None:
                try:
                    st.save(ckpt)
                except Exception:  # noqa: BLE001 — чекпойнт не имеет права уронить прогон
                    log.warning('чекпойнт %s не записался', ckpt, exc_info=True)
        order = [n for n, _, _ in nodes.NODES]
        st.done.sort(key=order.index)          # после возобновления `prepare` шёл последним
        st.tm['total_s'] = round(time.monotonic() - st.t0, 1)
        if nodes.editor_on(cfg, st):
            st.tm['editor'] = True
        out = result(st)
        if ckpt is not None:
            out['conveyor']['checkpoint'] = str(ckpt)
        return out
    finally:
        d._release_turn(st.ticket)  # идемпотентно: упавший выпуск не вешает очередь реестра
        shutil.rmtree(st.tmp, ignore_errors=True)
