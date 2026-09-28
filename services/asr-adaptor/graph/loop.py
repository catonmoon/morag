"""Цикл по месту: политика выбирает действие, реестр исполняет, место наблюдает — до `finish`
или до исчерпания бюджета шагов.

Место — кусок пасса-2 (арбитраж) или реплика (финал-раунд). Политика (правила или LLM) видит
только это место: его текст, что уже услышано и решено, сколько шагов осталось. Бюджет кончился
— место остаётся КАК ЕСТЬ (`fallback` политики), и это записывается: агент, который «почти
решил», не имеет права оставить полурешение.

⚠️ Вето — в коде инструментов и узла, независимо от политики: потеря слов, известные слова,
вставка на краю. Политика может выбрать инструмент, но не может обойти правила применения.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graph.tools import Registry, ToolError


@dataclass
class Action:
    tool: str
    args: dict = field(default_factory=dict)


@dataclass
class Finish:
    decision: str          # 'apply' | 'skip'
    why: str = ''


class Place:
    """Контекст одного места: что дано, что сделано (история наблюдений), что услышано и решено."""

    def __init__(self, name: str, kind: str, *, item: dict, canon: set, gate_terms: list = (),
                 audio_sec: float = 0.0, neighbours: tuple[float, float] | None = None,
                 index: int = 0) -> None:
        self.name = name
        self.kind = kind                      # 'arbitrate' | 'final'
        self.item = item                      # кусок пасса-2 или реплика
        self.index = index
        self.canon = canon
        self.gate_terms = list(gate_terms)
        self.audio_sec = audio_sec
        self.neighbours = neighbours
        self.history: list[tuple[str, dict, Any]] = []   # (инструмент, аргументы, результат)
        self.heard: dict[str, Any] = {}                  # ухо / слово → что услышано
        self.decisions: list[dict] = []                  # последний разбор правилами (арбитраж)
        self.final: str = ''                             # текст реплики после правок (финал-раунд)
        self.fixes: list[dict] = []                      # вердикты правки (финал-раунд)
        self.recalled: str = ''
        self.extra: dict = {}                            # память политики (LLM: сообщения)
        self.registry: Registry | None = None            # ставит цикл — политике нужны схемы
        self.budget: int = 0
        self.exhausted = False

    @property
    def chunk(self) -> dict:
        return self.item

    def observe(self, action: Action, result: Any) -> None:
        self.history.append((action.tool, dict(action.args), result))

    def last(self, tool: str, **match) -> Any:
        """Результат последнего вызова инструмента (с такими аргументами) или None."""
        for name, args, res in reversed(self.history):
            if name == tool and all(args.get(k) == v for k, v in match.items()):
                return res
        return None

    def count(self, tool: str) -> int:
        return sum(1 for name, _, _ in self.history if name == tool)

    def ok(self, tool: str, **match) -> bool:
        res = self.last(tool, **match)
        return res is not None and not (isinstance(res, dict) and 'error' in res)


async def decide_place(place: Place, policy, registry: Registry, steps: int) -> Finish:
    """Гонять политику по месту, пока она не скажет `finish` или не кончится бюджет шагов."""
    place.registry = registry
    place.budget = max(1, steps)
    for _ in range(place.budget):
        act = await policy.next(place)
        if isinstance(act, Finish):
            return act
        try:
            res = await registry.call(act.tool, **act.args)
        except ToolError as e:
            # Отказ инструмента — наблюдение, а не падение: политика вправе поправиться.
            res = e.as_result()
        place.observe(act, res)
        if act.tool == 'finish' and isinstance(res, dict) and 'error' not in res:
            return Finish(res.get('decision', 'skip'), res.get('why', ''))
    place.exhausted = True
    registry.meter['budget_exhausted'] = registry.meter.get('budget_exhausted', 0) + 1
    return policy.fallback(place)
