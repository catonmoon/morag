"""Инструменты, привязанные к МЕСТУ: кусок арбитража (шаг 2), реплика финал-раунда (шаг 3).

Инструмент не получает номер куска аргументом: политика работает над одним местом, и лишний
аргумент — лишний способ ошибиться. Вместо этого реестр собирается ЗАНОВО на каждое место
(это дёшево) над объектом `Place`, а журнал и счётчик — общие на прогон.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from graph.loop import Place
from graph.tools import Registry, Tool, ToolError

log_warning = logging.getLogger('asr').warning


class Slices:
    """Кэш вырезок звука одного места: второе и чистое ухо на одном окне режут ffmpeg один раз,
    как и в линейном конвейере. Файлы удаляются по окончании места."""

    def __init__(self, d, tmp: str, wav: str, tag: str) -> None:
        self._d, self._tmp, self._wav, self._tag = d, tmp, wav, tag
        self._files: dict[tuple[float, float], str] = {}

    async def get(self, a: float, b: float) -> str:
        key = (round(a, 2), round(b, 2))
        if key not in self._files:
            path = str(Path(self._tmp) / f'{self._tag}_{int(round(a * 100))}_{int(round(b * 100))}.wav')
            await asyncio.to_thread(self._d._slice, self._wav, a, b, path)
            self._files[key] = path
        return self._files[key]

    def cleanup(self) -> None:
        for path in self._files.values():
            Path(path).unlink(missing_ok=True)
        self._files.clear()


def arbitrate_tools(place: Place, d, slices: Slices, *, journal: list, meter: dict) -> Registry:
    """Инструменты арбитража одного куска (ADR-0030): ворота-читатель, уши, правила, свидетели."""
    cfg = d.cfg
    A = d.arbitrate_stage
    c = place.chunk

    async def reader_flags() -> dict:
        flags = await A.reader_flags(d.llm, c.get('raw') or '', place.gate_terms)
        return {'suspicious': flags}

    async def listen(ear: str, window: str = 'same') -> dict:
        if ear == 'second' and not cfg.second_model:
            raise ToolError('второе ухо не настроено (ASR_SECOND_MODEL пуст)', recoverable=False)
        model = cfg.second_model if ear == 'second' else ''
        if window == 'same':
            a, b = float(c['start']), float(c['end'])
        else:
            disputes = [x for x in place.decisions if x.get('by') == 'спорно']
            a, b = A.ear_window(c, disputes, window, place.audio_sec, place.neighbours)
        path = await slices.get(a, b)
        async with d._res('whisper', cfg.whisper_slots):
            # ⚠️ Модель передаётся ТОЛЬКО второму уху — чистое ухо зовёт бэкенд прежней формой:
            # заглушки тестов и чужие обёртки знают сигнатуру `asr(path, prompt)`.
            r = await asyncio.to_thread(lambda: d.audio_clients.asr(path, '', **({'model': model} if model else {})))
        out = {'ear': ear, 'text': r.get('text') or '', 'a': a, 'b': b}
        place.heard[ear] = out
        return out

    async def arbitrate_rules() -> dict:
        second = place.heard.get('second')
        if second is None:
            raise ToolError('нечего разбирать: сначала послушай вторым ухом', hint='listen(ear="second")')
        clean = (place.heard.get('clean') or {}).get('text')
        text, decisions = A.arbitrate(c.get('raw') or '', second['text'], clean, place.canon,
                                      ratio=cfg.arbitrate_ratio)
        place.decisions = decisions
        return {'text': text, 'decisions': decisions,
                'disputes': sum(1 for x in decisions if x.get('by') == 'спорно'),
                'taken': sum(1 for x in decisions if x.get('taken'))}

    async def apply_swaps() -> dict:
        second = place.heard.get('second')
        if second is None:
            raise ToolError('нечего применять: второе ухо не слушали', hint='listen(ear="second")')
        clean = (place.heard.get('clean') or {}).get('text')
        decisions = A.apply(c, second['text'], clean, place.canon, ratio=cfg.arbitrate_ratio)
        return {'decisions': decisions, 'taken': sum(1 for x in decisions if x.get('taken'))}

    async def in_canon(word: str) -> dict:
        return {'word': word, 'known': A.in_canon(word, place.canon)}

    async def frequency(word: str) -> dict:
        f = A.freq(word, 'ru')
        return {'word': word, 'frequency': f, 'ordinary': bool(f is not None and f >= A.COMMON)}

    async def finish(decision: str, why: str = '') -> dict:
        return {'decision': decision, 'why': why}

    tools = [
        Tool('reader_flags', 'Читатель: какие слова куска выглядят ошибкой распознавания. Ворота, не судья: '
             'пустой список — кусок чист, слушать дальше незачем.', reader_flags,
             cost=lambda a, r: {'llm_calls': 1}),
        Tool('listen', 'Послушать место ещё раз. ear=second — модель другой школы; ear=clean — та же модель '
             'без подсказки (третий голос для голосования). window=same — окно куска как есть; chunk / '
             'window30 / neighbours — по спорным местам последнего разбора. Никогда не слово: окно по слову '
             'даёт вдвое меньше попаданий.', listen,
             schema={'properties': {'ear': {'type': 'string', 'enum': ['second', 'clean']},
                                    'window': {'type': 'string', 'enum': ['same', 'chunk', 'window30', 'neighbours']}},
                     'required': ['ear']},
             cost=lambda a, r: {'audio_s': float(r['b'] - r['a'])}),
        Tool('arbitrate_rules', 'Разобрать расхождения куска с услышанным правилами (частота, канон, '
             'голосование под вето). Возвращает решения и число споров, которые никто не решил.',
             arbitrate_rules),
        Tool('apply_swaps', 'Применить решения правил к куску (текст, сегменты, слова декодера). '
             'Единственный инструмент, который меняет текст.', apply_swaps),
        Tool('in_canon', 'Знает ли канон записи это написание (по звучанию, с допуском на форму слова).',
             in_canon, schema={'properties': {'word': {'type': 'string'}}, 'required': ['word']}),
        Tool('frequency', 'Насколько слово обычно в русском языке (частотник); None — частотника нет.',
             frequency, schema={'properties': {'word': {'type': 'string'}}, 'required': ['word']}),
        Tool('finish', 'Закончить место: apply — применить решения правил, skip — оставить как есть.',
             finish, schema={'properties': {'decision': {'type': 'string', 'enum': ['apply', 'skip']},
                                            'why': {'type': 'string'}},
                             'required': ['decision']}),
    ]
    return Registry(tools, journal=journal, meter=meter, place=place.name)


def final_tools(place: Place, d, slices: Slices, *, st, journal: list, meter: dict) -> Registry:
    """Инструменты финал-раунда одной реплики: вспомнить сущности, предложить замены (модель
    ПРЕДЛАГАЕТ, применяет код с вето), для отвергнутых замен известных слов — рассудить звуком."""
    cfg = d.cfg
    t = place.item
    raw = t.get('raw') or ''
    canonicals = d.relevant(raw, st.gloss)

    async def recall() -> dict:
        place.recalled = await d.recall_entities(st.dsum, raw, d.llm)
        return {'recalled': place.recalled}

    async def correct_turn() -> dict:
        fixes: list[dict] = []
        place.final = await d.correct(raw, st.dsum, d._around(st.turns, place.index, cfg.context_turns),
                                      canonicals, d.llm, st.protect, place.recalled or '',
                                      corpus_desc=cfg.corpus_desc, fixes_out=fixes,
                                      **({'protect': st.known} if st.known else {}))
        place.fixes = fixes
        return {'text': place.final, 'fixes': fixes, 'changed': place.final != raw}

    def _at(word: str) -> float | None:
        """Где в реплике звучит слово: по временам слов декодера, иначе по доле слова."""
        segs = t.get('segments') or []
        if not segs:
            return None
        key = word.strip('.,!?;:«»"').lower()
        for sg in segs:
            for w in sg.get('words') or ():
                if (w.get('word') or '').strip().strip('.,!?;:«»"').lower() == key:
                    return float(w['start'])
        words = raw.split()
        pos = next((k for k, w in enumerate(words) if w.strip('.,!?;:«»"').lower() == key), None)
        if pos is None:
            return None
        a0, b0 = segs[0]['start'], segs[-1]['end']
        return a0 + (b0 - a0) * pos / max(1, len(words))

    async def listen(word: str) -> dict:
        """Чистое ухо на 30 с вокруг слова реплики — окно, не слово (ADR-0030). Слова в реплике
        нет или бэкенд не ответил — «не слышно», реплика от этого не падает (как в конвейере)."""
        at = _at(word)
        if at is None:
            place.heard[word] = None
            return {'word': word, 'found': False, 'text': ''}
        a, b = max(0.0, at - 15), min(place.audio_sec, at + 15)
        try:
            path = await slices.get(a, b)
            async with d._res('whisper', cfg.whisper_slots):
                got = await asyncio.to_thread(lambda: d.audio_clients.asr(path, ''))
            heard = got.get('text') or ''
        except Exception as error:      # noqa: BLE001 — ухо не роняет реплику
            log_warning('арбитраж звуком не вышел (%s): %s', word, error)
            place.heard[word] = None
            return {'word': word, 'found': False, 'text': '', 'failed': f'{type(error).__name__}: {str(error)[:80]}'}
        place.heard[word] = heard
        return {'word': word, 'found': True, 'text': heard, 'a': a, 'b': b}

    async def sound_prefers(was: str, now: str) -> dict:
        said = d._ear_prefers(place.heard.get(was), was, now)
        if said != 'now':
            for f in place.fixes:
                if f.get('why') == 'known_term' and f['was'] == was and f['now'] == now:
                    f['why'] = f'known_term (ear: {said})'
        return {'was': was, 'now': now, 'said': said}

    async def apply_fix(was: str, now: str) -> dict:
        place.final, n_ok, _ = d.apply_fixes(place.final, [{'was': was, 'now': now}], canonicals, st.protect)
        for f in place.fixes:
            if f.get('why') == 'known_term' and f['was'] == was and f['now'] == now:
                f.update(ok=bool(n_ok), why='known_term→ear:now' if n_ok else 'known_term→ear:now,not_found')
        return {'was': was, 'now': now, 'applied': bool(n_ok)}

    async def finish(decision: str, why: str = '') -> dict:
        return {'decision': decision, 'why': why}

    tools = [
        Tool('recall', 'Вспомнить, какие имена, компании, продукты и термины упомянуты в реплике и как они '
             'пишутся — по описанию записи. Шаг перед правкой: без него модель теряет полноту.', recall,
             cost=lambda a, r: {'llm_calls': 1}),
        Tool('correct_turn', 'Предложить замены неверно распознанных имён и терминов; код применяет их под '
             'вето (потеря слов, перевод, выдуманное имя, известное слово записи). Возвращает текст и '
             'вердикты по каждой замене (ok, why).', correct_turn, cost=lambda a, r: {'llm_calls': 1}),
        Tool('listen', 'Послушать чистым ухом 30 с вокруг слова реплики (окно, не слово). Для замен, '
             'которые сторож отверг как замену известного слова (why=known_term).', listen,
             schema={'properties': {'word': {'type': 'string'}}, 'required': ['word']},
             cost=lambda a, r: {'audio_s': float(r.get('b', 0) - r.get('a', 0))}),
        Tool('sound_prefers', 'За кого звук: now (за замену), was (за прежнее), tie, silent. Сравнение по '
             'звучанию с тем, что услышало чистое ухо (сначала listen(word)).', sound_prefers,
             schema={'properties': {'was': {'type': 'string'}, 'now': {'type': 'string'}},
                     'required': ['was', 'now']}),
        Tool('apply_fix', 'Применить одну замену к тексту реплики (только если звук за неё).', apply_fix,
             schema={'properties': {'was': {'type': 'string'}, 'now': {'type': 'string'}},
                     'required': ['was', 'now']}),
        Tool('finish', 'Закончить место: apply — принять текст с применёнными заменами, skip — оставить '
             'реплику сырой.', finish,
             schema={'properties': {'decision': {'type': 'string', 'enum': ['apply', 'skip']},
                                    'why': {'type': 'string'}},
                     'required': ['decision']}),
    ]
    return Registry(tools, journal=journal, meter=meter, place=place.name)
