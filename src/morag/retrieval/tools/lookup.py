"""Тип `lookup` — точечное обращение к заранее указанным справочным страницам.

Семантика: «при триггере, описанном в описании, загляни в конкретные страницы
(doc_ids)» — реранкер выберет релевантные чанки. Глоссарий — один ИНСТАНС
этого типа (`trigger: abbreviations`); другие инстансы (страницы политик,
регламенты) задают своё описание.

ОДНО описание на инстанс = ПОЛНЫЙ текст, который видит агент (в списке тулов).
Пусто → дефолт по `trigger`. Несколько инстансов одного типа различаются `name`.
Прежний отдельный «блок в system-prompt» свёрнут сюда (см. обсуждение): протокол
аббревиатур теперь часть описания, формат-правило дополнительно держит
пост-вызовный nudge в pipeline.
"""
from __future__ import annotations

from morag.retrieval.tools import ToolSpec


def abbr_description(note: str = '') -> str:
    """Полное описание lookup-инстанса в режиме «глоссарий аббревиатур»:
    доменная шапка (note или дефолт) + протокол «зови первым + формат (АББР)»."""
    head = note.strip() or 'Глоссарий с расшифровками аббревиатур и специальных терминов корпуса.'
    if not head.endswith(('.', '!', '?', ':')):
        head += '.'
    return (
        f'{head} Вызывай ЭТОТ инструмент ПЕРВЫМ — до find_section/search — если в вопросе '
        'есть хотя бы одно сокращение из 2+ заглавных букв (например API, SSO, IAM) или '
        'необычный термин: иначе поиск по самой аббревиатуре даст шум вместо результата. '
        'После расшифровки в последующих find_section/search включай И полное название, '
        'И аббревиатуру в формате «Полное название (АББР)» — в документах термин '
        'встречается и так, и так. Если сокращений в вопросе нет — этот шаг пропусти.'
    )


def generic_description(note: str = '') -> str:
    """Полное описание lookup-инстанса в обычном режиме (триггер из описания)."""
    head = note.strip() or 'Точечное обращение к заранее указанным справочным страницам корпуса.'
    if not head.endswith(('.', '!', '?', ':')):
        head += '.'
    return (
        f'{head} Вызывай, когда вопрос затрагивает описанную область, ДО основного поиска — '
        'найденные формулировки и термины используй в последующих find_section/search. '
        'На один и тот же запрос повторно не вызывай.'
    )


def _describe(cfg: dict) -> str:
    if cfg.get('description'):
        return cfg['description']
    return abbr_description() if cfg.get('trigger') == 'abbreviations' else generic_description()


def _status(cfg: dict, args: dict, resolve_title) -> str:
    """Унифицированный статус: «уточняю в "Title1", "Title2"» — названия
    страниц инстанса, а не абстрактное «смотрю в глоссарии»."""
    titles = [resolve_title(did) for did in (cfg.get('doc_ids') or [])[:3]]
    pages = ', '.join(f'«{t}»' for t in titles if t) or 'справочных страницах'
    return f"[{args.get('query', '')}] уточняю в {pages}"


SPEC = ToolSpec(
    type='lookup',
    default_name='lookup',
    core=False,
    parameters={
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': (
                    'Термин/вопрос, который уточняешь по справочным страницам, '
                    'на русском (можно несколько через пробел).'
                ),
            },
        },
        'required': ['query'],
    },
    describe=_describe,
    prompt_section=lambda cfg: '',
    handler=lambda p, cfg, args: p._tool_lookup(cfg.get('doc_ids') or [], args['query']),
    status=_status,
    icon='📖',
)
