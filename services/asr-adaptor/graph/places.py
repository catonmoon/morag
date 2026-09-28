"""Инструменты, привязанные к МЕСТУ: кусок арбитража (шаг 2), реплика финал-раунда (шаг 3).

Инструмент не получает номер куска аргументом: политика работает над одним местом, и лишний
аргумент — лишний способ ошибиться. Вместо этого реестр собирается ЗАНОВО на каждое место
(это дёшево) над объектом `Place`, а журнал и счётчик — общие на прогон.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from graph.loop import Place
from graph.tools import Registry, Tool, ToolError


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
