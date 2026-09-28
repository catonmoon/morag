"""Политика решения по месту: что звать и когда остановиться.

`RulePolicy` — сегодняшняя последовательность конвейера, записанная явными шагами: арбитраж —
ворота читателя → второе ухо → [чистое ухо] → правила → [третий голос по требованию] →
применить; финал-раунд — вспомнить сущности → предложить замены → [для отвергнутых замен
известных слов: послушать → рассудить звуком → применить]. Граф с ней обязан совпадать с
линейным конвейером байт в байт — это держит золотой тест.

`LLMPolicy` — оркестратор через function calling: видит место, историю своих вызовов и схемы
тех же инструментов; выбирает, что звать и когда остановиться. Один вызов модели — один шаг.

⚠️ Ни одна политика не пишет текст. Замена ложится только из правил и свидетелей (звук, канон)
под вето в коде — модель может ПРЕДЛАГАТЬ, подтверждать вправе только звук, канон и экран.
"""
from __future__ import annotations

import json
import logging
from typing import Protocol

from graph.loop import Action, Finish, Place

log = logging.getLogger('asr')


class Policy(Protocol):
    kind: str

    async def next(self, place: Place) -> Action | Finish: ...

    def fallback(self, place: Place) -> Finish: ...


# --- правила ------------------------------------------------------------------------------------

class RulePolicy:
    """Правила конвейера как явные шаги. Читает конфиг стадии, ничего не помнит между местами."""

    kind = 'rule'

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    async def next(self, place: Place) -> Action | Finish:
        if place.kind == 'arbitrate':
            return self._arbitrate(place)
        if place.kind == 'final':
            return self._final(place)
        return Finish('skip', f'политика правил не знает место «{place.kind}»')

    def fallback(self, place: Place) -> Finish:
        return Finish('skip', 'бюджет шагов исчерпан')

    def _arbitrate(self, place: Place) -> Action | Finish:
        cfg = self.cfg
        if cfg.arbitrate_gate == 'reader':
            gate = place.last('reader_flags')
            if gate is None:
                return Action('reader_flags')
            if not (gate or {}).get('suspicious'):
                return Finish('skip', 'читатель: кусок чист')
        if not place.ok('listen', ear='second'):
            return Action('listen', {'ear': 'second', 'window': 'same'})
        if cfg.clean_ear == 'always' and not place.ok('listen', ear='clean'):
            return Action('listen', {'ear': 'clean', 'window': 'same'})
        rules = place.last('arbitrate_rules')
        if rules is None:
            return Action('arbitrate_rules')
        if (cfg.clean_ear == 'demand' and (rules or {}).get('disputes')
                and not place.ok('listen', ear='clean')):
            # Третий голос по требованию: только если после второго уха остались споры, не
            # решённые ни правилом, ни свидетелем. Окном, не словом (ADR-0030).
            return Action('listen', {'ear': 'clean', 'window': cfg.clean_ear_window})
        return Finish('apply', 'правила')

    def _final(self, place: Place) -> Action | Finish:
        cfg = self.cfg
        if place.last('recall') is None:
            return Action('recall')
        if place.last('correct_turn') is None:
            return Action('correct_turn')
        if cfg.protect_known and cfg.final_ear:
            # Замену ИЗВЕСТНОГО слова сторож отверг — но финал-раунд иногда чинит прайминг (звук
            # за замену в 10 случаях из 104). Слушаем чистым ухом; звук ближе к замене — берём.
            for f in place.fixes:
                if f.get('why') != 'known_term':
                    continue
                was, now = f['was'], f['now']
                if place.last('listen', word=was) is None:
                    return Action('listen', {'word': was})
                said = place.last('sound_prefers', was=was, now=now)
                if said is None:
                    return Action('sound_prefers', {'was': was, 'now': now})
                if (said or {}).get('said') == 'now' and place.last('apply_fix', was=was, now=now) is None:
                    return Action('apply_fix', {'was': was, 'now': now})
        return Finish('apply', 'правила')


# --- LLM-оркестратор ----------------------------------------------------------------------------

# Промпт — в реестре `prompts.py` под именем `policy.orchestrator.system`: домен настраивает его
# файлом, движок держит только generic-текст. ⚠️ Схема важнее промпта: инструменты, их аргументы и
# перечни описаны в схемах function calling; здесь — цель, правила и когда остановиться.
ORCHESTRATOR_SYS = (
    'Ты — оркестратор проверки ОДНОГО места автоматической расшифровки русской речи. Ты не пишешь '
    'и не правишь текст: замены делают правила и свидетели (звук, канон), а ты решаешь, какие '
    'инструменты звать и когда остановиться. Каждый инструмент стоит времени: прослушивание — '
    'секунды звука, читатель и правка — вызов модели. Не зови инструмент повторно с теми же '
    'аргументами и не проси окно по слову — только куском или шире.\n'
    'Место «кусок»: если текст выглядит чистым — сразу finish(skip); если есть подозрительные слова '
    '(reader_flags), послушай другой моделью (listen ear=second), разбери расхождения правилами '
    '(arbitrate_rules); остались споры — послушай чистым ухом (listen ear=clean, window=chunk) и '
    'разбери ещё раз; затем finish(apply). Сомневаешься в слове — спроси in_canon или frequency.\n'
    'Место «реплика»: recall, затем correct_turn; для замен с причиной known_term — listen(word), '
    'sound_prefers, и apply_fix только если звук за замену; затем finish(apply).\n'
    'Всегда заканчивай вызовом finish. Аргументы — строго по схеме инструмента.'
)


class LLMPolicy:
    """Оркестратор: один вызов модели с историей места и схемами инструментов — одно действие."""

    kind = 'llm'
    RETRIES = 2            # битые аргументы: столько раз показать ошибку и спросить снова
    MAX_TOKENS = 300

    def __init__(self, cfg, llm) -> None:
        self.cfg = cfg
        self.llm = llm

    def fallback(self, place: Place) -> Finish:
        return Finish('skip', 'бюджет шагов исчерпан')

    @staticmethod
    def system_prompt() -> str:
        import graph.policy as me  # noqa: PLC0415 — константу могли переопределить из файла промптов
        return me.ORCHESTRATOR_SYS

    def describe(self, place: Place) -> str:
        it = place.item
        if place.kind == 'arbitrate':
            head = (f"Место: кусок записи {float(it.get('start') or 0):.1f}–{float(it.get('end') or 0):.1f} с.\n"
                    f"Текст куска: «{(it.get('raw') or '')[:1500]}»")
        else:
            head = (f"Место: реплика с {float(it.get('start') or 0):.1f} с.\n"
                    f"Текст реплики: «{(it.get('raw') or '')[:2500]}»")
        known = place.gate_terms[:40]
        return (f"{head}\nИзвестные написания этой записи: {', '.join(known) if known else '—'}"
                f"{' …' if len(place.gate_terms) > 40 else ''}\n"
                f"Бюджет: {place.budget} шагов, каждый — один инструмент.")

    async def next(self, place: Place) -> Action | Finish:
        ex = place.extra.setdefault('llm', {})
        if 'messages' not in ex:
            ex['messages'] = [{'role': 'system', 'content': self.system_prompt()},
                              {'role': 'user', 'content': self.describe(place)}]
            ex['synced'], ex['calls'] = 0, []
        # Результаты инструментов с прошлого шага — в переписку, под тем id вызова, что дала модель.
        for name, args, res in place.history[ex['synced']:]:
            call_id = ex['calls'].pop(0) if ex['calls'] else f'call_{len(ex["messages"])}'
            ex['messages'].append({'role': 'tool', 'tool_call_id': call_id,
                                   'content': json.dumps(res, ensure_ascii=False, default=str)[:4000]})
        ex['synced'] = len(place.history)
        schemas = place.registry.schemas() if place.registry else []
        meter = place.registry.meter if place.registry else {}
        for _ in range(self.RETRIES + 1):
            meter['orchestrator_calls'] = meter.get('orchestrator_calls', 0) + 1
            try:
                resp = await self.llm.complete_with_tools(ex['messages'], schemas, max_tokens=self.MAX_TOKENS)
            except Exception as e:                                   # noqa: BLE001 — место остаётся как есть
                log.warning('оркестратор %s: %s: %s', place.name, type(e).__name__, str(e)[:120])
                return Finish('skip', f'оркестратор: {type(e).__name__}')
            msg = ((resp or {}).get('choices') or [{}])[0].get('message') or {}
            calls = msg.get('tool_calls') or []
            if not calls:
                return Finish('skip', f'оркестратор закончил без finish: {(msg.get("content") or "")[:80]}')
            tc = calls[0]                                            # один шаг — один инструмент
            fn = tc.get('function') or {}
            name = fn.get('name') or ''
            raw = fn.get('arguments') or ''
            try:
                args = json.loads(raw) if raw.strip() else {}
                if not isinstance(args, dict):
                    raise ValueError('ожидался объект')
            except ValueError as e:
                # Битые аргументы — сказать модели и спросить снова, а не падать и не гадать.
                ex['messages'].append({'role': 'assistant', 'content': msg.get('content') or '', 'tool_calls': [tc]})
                ex['messages'].append({'role': 'tool', 'tool_call_id': tc.get('id') or 'call_bad',
                                       'content': json.dumps({'error': f'аргументы не JSON: {e}',
                                                              'hint': 'верни объект по схеме инструмента'},
                                                             ensure_ascii=False)})
                meter['orchestrator_bad_args'] = meter.get('orchestrator_bad_args', 0) + 1
                continue
            ex['messages'].append({'role': 'assistant', 'content': msg.get('content') or '', 'tool_calls': [tc]})
            ex['calls'].append(tc.get('id') or f'call_{len(ex["messages"])}')
            return Action(name, args)
        return Finish('skip', 'оркестратор: аргументы не разобрались')


def make_policy(cfg, llm=None, kind: str = '') -> Policy:
    """`kind` на прогон (поле формы), иначе `ASR_GRAPH_POLICY=rule|llm`. Без клиента LLM
    оркестратора не бывает — правила."""
    if (kind or getattr(cfg, 'graph_policy', 'rule')) == 'llm' and llm is not None:
        return LLMPolicy(cfg, llm)
    return RulePolicy(cfg)
