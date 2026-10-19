"""«Кто спрашивает» и доступ на ретриве (ADR-0034).

Запрос несёт, КТО спрашивает (подписанное поле `asker` тела запроса); документ несёт, КОМУ его
можно (поле payload `access`, доезжает до каждого чанка само); движок пересекает одно с другим во
всех путях, которыми документ доходит до агента. Что человек «проходит» (свои документы, открытые
ему, подразделения, группы), знает внешний хозяин прав — движок спрашивает его по HTTP и сравнивает
строки, не зная ни иерархии, ни ролей.

Без секции `retrieval.access` (`enabled: false`) контекст — `OPEN`: фильтров нет, поведение
прежнее байт в байт.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx
from qdrant_client.models import (
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    PayloadField,
)

logger = logging.getLogger(__name__)

# Поле payload документа (и его чанков): список строк «кому можно». Нет поля или пустой список —
# документ открыт всем.
ACCESS_FIELD = 'access'


class AccessRefused(Exception):
    """Запрос нёс `asker`, но подпись не сошлась. Ответ — ошибка, а не тихое понижение прав:
    неверный секрет — это поломка конфигурации вызывающего, её надо увидеть."""


@dataclass(frozen=True)
class RequestContext:
    """Контекст одного запроса. Собирается в `pipe()` и едет явным параметром до инструментов и
    поисковых вызовов: генератор `pipe()` не сериализован, запросы чередуются, поэтому ни `self`,
    ни contextvar для него не годятся.

    `principals is None` — доступ выключен (фильтра нет). Пустое множество — аноним: видно
    только открытое всем."""
    asker: str | None = None
    principals: frozenset[str] | None = None
    scope: dict[str, list[str]] = field(default_factory=dict)
    # Непрозрачные сведения о человеке от доверенного вызывающего (имя, отдел…). Ядро полей не
    # знает; здесь — только точка расширения для персонализации и инструментов «про людей».
    user: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.principals is not None


OPEN = RequestContext()


class RightsOwner:
    """HTTP-клиент хозяина прав. Два вопроса:

    - `POST {url}/principals {"asker": id}` → `{"principals": [...]}` — строки, которые человек
      проходит. Кэш на `ttl` секунд: их могут быть сотни, а разговор — десяток вызовов.
    - `POST {url}/check {"asker": id, "doc_ids": [...]}` → `{"allowed": [...]}` — живая проверка
      перед выдачей агенту. Без кэша: копия прав в индексе может отставать, хозяин — нет.

    Хозяин недоступен — не падаем: `principals` → пусто (человек видит открытое и своё по `id`),
    `check` → никому ничего (закрыто, а не открыто)."""

    def __init__(self, url: str, key: str = '', ttl: float = 300.0, timeout: float = 10.0) -> None:
        self._url = url.rstrip('/')
        self._headers = {'Authorization': f'Bearer {key}'} if key else {}
        self._ttl = ttl
        self._timeout = timeout
        self._cache: dict[str, tuple[float, frozenset[str]]] = {}
        self._lock = threading.Lock()

    def _post(self, path: str, body: dict) -> dict:
        r = httpx.post(self._url + path, json=body, headers=self._headers, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def principals(self, asker: str) -> frozenset[str]:
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(asker)
            if hit and hit[0] > now:
                return hit[1]
        try:
            got = frozenset(str(p) for p in (self._post('/principals', {'asker': asker})
                                             .get('principals') or []))
        except Exception as exc:  # noqa: BLE001 — хозяин недоступен: только открытое и своё
            logger.warning('[access] principals(%s) failed: %s', asker, exc)
            return frozenset()
        with self._lock:
            self._cache[asker] = (now + self._ttl, got)
        return got

    def check(self, asker: str | None, doc_ids: Iterable[str]) -> set[str]:
        ids = sorted(set(doc_ids))
        if not ids:
            return set()
        try:
            allowed = self._post('/check', {'asker': asker, 'doc_ids': ids}).get('allowed') or []
        except Exception as exc:  # noqa: BLE001 — не знаем — значит нельзя
            logger.warning('[access] check(%s, %d docs) failed: %s', asker, len(ids), exc)
            return set()
        return set(ids) & {str(d) for d in allowed}


def parse_asker(
    body: dict | None,
    *,
    enabled: bool,
    secret: str,
    scope_fields: Iterable[str] = (),
    owner: RightsOwner | None = None,
) -> RequestContext:
    """`body['asker']` → `RequestContext`.

    - доступ выключен — `OPEN`, поле `asker` игнорируется (поведение прежнее);
    - поля нет — аноним: принципалов нет, видно только открытое всем;
    - подпись не сошлась (или секрет не настроен) — `AccessRefused`;
    - иначе принципалы = сам `id` ∪ ответ хозяина прав; `scope` — только известные поля."""
    if not enabled:
        return OPEN
    raw = (body or {}).get('asker')
    if not isinstance(raw, dict):
        return RequestContext(principals=frozenset())
    key = str(raw.get('key') or '')
    if not secret or not hmac.compare_digest(key.encode(), secret.encode()):
        raise AccessRefused('asker: подпись не сошлась')
    asker = str(raw.get('id') or '').strip()
    if not asker:
        raise AccessRefused('asker: нет id')
    principals = {asker}
    if owner is not None:
        principals |= owner.principals(asker)
    known = set(scope_fields)
    scope: dict[str, list[str]] = {}
    for name, values in (raw.get('scope') or {}).items():
        if name not in known:
            logger.warning('[access] scope field %r unknown — dropped', name)
            continue
        vals = [str(v) for v in (values if isinstance(values, list) else [values]) if str(v).strip()]
        if vals:
            scope[name] = vals
    user = raw.get('user') if isinstance(raw.get('user'), dict) else {}
    return RequestContext(asker=asker, principals=frozenset(principals), scope=scope, user=user)


def visible(payload: dict | None, principals: frozenset[str] | None) -> bool:
    """Питоновское зеркало условия поиска — для точечных выдач (документ по id)."""
    if principals is None:
        return True
    label = (payload or {}).get(ACCESS_FIELD)
    if not label:
        return True
    items = label if isinstance(label, list) else [label]
    return any(str(x) in principals for x in items)


def access_conditions(
    principals: frozenset[str] | None, known_labels: set[str] | None = None,
) -> list:
    """Условия для `must` Qdrant-фильтра: «метки нет ИЛИ метка ∩ принципалы ≠ ∅».

    `known_labels` — все строки, встречающиеся в метках индекса: в условие идёт только
    пересечение (у человека сотни групп, в метках — десятки). `None` — доступ выключен."""
    if principals is None:
        return []
    empty = IsEmptyCondition(is_empty=PayloadField(key=ACCESS_FIELD))
    wanted = sorted(principals & known_labels if known_labels is not None else principals)
    if not wanted:
        return [empty]
    return [Filter(should=[empty, FieldCondition(key=ACCESS_FIELD, match=MatchAny(any=wanted))])]


def scope_conditions(scope: dict[str, list[str]] | None) -> list:
    """Фильтры, выбранные человеком: обязательные; агент сужает только внутри них."""
    return [
        FieldCondition(key=name, match=MatchAny(any=[str(v) for v in values]))
        for name, values in (scope or {}).items() if values
    ]


def describe_scope(scope: dict[str, list[str]] | None) -> str:
    return '; '.join(f'{k}: {", ".join(v)}' for k, v in (scope or {}).items() if v)
