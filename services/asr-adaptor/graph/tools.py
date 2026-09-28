"""Инструменты графа: одно дело каждый, схема аргументов, журнал каждого вызова, учёт цены.

Практика агентостроения, взятая сюда как правила (ADR-0030, план графа):
- **схема важнее промпта** — `enum`, типы и обязательные поля соблюдаются структурно; просьбы в
  прозе модель под давлением нарушает, а слабая модель — тем более;
- **ошибка инструмента — структура**, не исключение в никуда: `{error, recoverable, hint}`
  возвращается политике как наблюдение, и LLM-оркестратор может поправиться;
- **каждый вызов в журнал** (место, инструмент, аргументы, сводка результата, мс) — ненадёжность
  агентов рождается в инструментах, и без журнала её не найти;
- **цена считается здесь**: вызовы LLM и секунды прослушанного звука — это и есть «скорость»
  для итогового сравнения конвейеров.

Валидатор схемы — свой минимальный (type / required / enum / без лишних полей): зависимость
ради двадцати строк в публичном движке с прибитыми версиями не нужна.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(Exception):
    """Отказ инструмента, о котором стоит сказать политике: что не так и что можно сделать."""

    def __init__(self, message: str, *, recoverable: bool = True, hint: str = '') -> None:
        super().__init__(message)
        self.recoverable = recoverable
        self.hint = hint

    def as_result(self) -> dict:
        return {'error': str(self), 'recoverable': self.recoverable,
                **({'hint': self.hint} if self.hint else {})}


_TYPES: dict[str, tuple] = {'string': (str,), 'number': (int, float), 'integer': (int,),
                            'boolean': (bool,), 'array': (list,), 'object': (dict,)}


def validate(schema: dict, args: dict) -> None:
    """Проверить аргументы по схеме объекта. Лишние поля — тоже ошибка: модель, выдумавшая
    аргумент, обязана об этом узнать, а не получить молча проигнорированный вызов."""
    props = schema.get('properties') or {}
    extra = sorted(set(args) - set(props))
    if extra:
        raise ToolError(f'лишние аргументы: {", ".join(extra)}', hint=f'допустимы: {", ".join(props) or "—"}')
    for k in schema.get('required') or ():
        if k not in args:
            raise ToolError(f'нет обязательного аргумента «{k}»')
    for k, v in args.items():
        p = props[k]
        t = p.get('type')
        if t in _TYPES and (not isinstance(v, _TYPES[t]) or (t in ('number', 'integer') and isinstance(v, bool))):
            raise ToolError(f'аргумент «{k}» должен быть {t}, а не {type(v).__name__}')
        if 'enum' in p and v not in p['enum']:
            raise ToolError(f'аргумент «{k}»: «{v}» не из перечня', hint=f'допустимы: {", ".join(map(str, p["enum"]))}')


def _trunc(v: Any, n: int = 120, items: int = 8) -> Any:
    """Сводка значения для журнала: строки режутся, списки укорачиваются, словари — по полям."""
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + '…'
    if isinstance(v, list):
        out = [_trunc(x, n, items) for x in v[:items]]
        return out + [f'… ещё {len(v) - items}'] if len(v) > items else out
    if isinstance(v, dict):
        return {k: _trunc(x, n, items) for k, x in v.items()}
    return v


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]                       # async def fn(**args) -> Any
    schema: dict = field(default_factory=lambda: {'type': 'object', 'properties': {}})
    # Приращения счётчика по факту вызова: {'llm_calls': 1}, {'audio_s': 12.3}. Считает инструмент,
    # потому что только он знает, сколько секунд отрезал и звал ли модель.
    cost: Callable[[dict, Any], dict] | None = None
    # Внутренний инструмент зовёт только КОД узла, не политика: его нет в схемах для модели, и вызов
    # из цикла по месту отвергается. Так устроено применение решений — политика выбирает, что
    # слушать, а менять текст вправе только правила и свидетели (ADR-0030).
    internal: bool = False

    def as_schema(self) -> dict:
        """Форма function calling (OpenAI): то, что уходит LLM-оркестратору."""
        return {'type': 'function', 'function': {'name': self.name, 'description': self.description,
                                                 'parameters': {'type': 'object', 'additionalProperties': False,
                                                                **self.schema}}}


class Registry:
    """Инструменты одного места. `call` валидирует, зовёт, журналирует и считает цену."""

    def __init__(self, tools: list[Tool], *, journal: list, meter: dict, place: str = '') -> None:
        self._tools = {t.name: t for t in tools}
        self.journal = journal
        self.meter = meter
        self.place = place

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.as_schema() for t in self._tools.values() if not t.internal]

    def _tick(self, key: str, by: float = 1) -> None:
        self.meter[key] = round(self.meter.get(key, 0) + by, 2) if isinstance(by, float) else self.meter.get(key, 0) + by

    async def call(self, name: str, *, _internal: bool = False, **args) -> Any:
        tool = self._tools.get(name)
        if tool is None or (tool.internal and not _internal):
            open_ = ', '.join(n for n, t in self._tools.items() if not t.internal)
            raise ToolError(f'нет инструмента «{name}»', hint=f'есть: {open_}')
        validate(tool.schema, args)
        row: dict = {'place': self.place, 'tool': name, 'args': _trunc(args, 80)}
        t0 = time.monotonic()
        try:
            res = await tool.fn(**args)
        except ToolError as e:
            row.update(error=str(e), ms=round((time.monotonic() - t0) * 1000))
            self.journal.append(row)
            self._tick('tool_errors')
            raise
        except Exception as e:                                   # noqa: BLE001 — журнал, потом наверх
            row.update(error=f'{type(e).__name__}: {str(e)[:120]}', ms=round((time.monotonic() - t0) * 1000))
            self.journal.append(row)
            self._tick('tool_failures')
            raise
        row.update(result=_trunc(res), ms=round((time.monotonic() - t0) * 1000))
        self.journal.append(row)
        self._tick('tool_calls')
        self._tick(f'tool.{name}')
        if tool.cost:
            for k, v in (tool.cost(args, res) or {}).items():
                self._tick(k, v)
        return res
