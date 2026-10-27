"""Тип `sections` — описания разделов базы знаний по запросу агента.

Альтернатива Карте знаний в системном промпте: та же информация (узлы Карты, сохранённые
по одному при её генерации), но агент берёт её сам и только нужную. `sections()` — перечень
верхнеуровневых разделов с кратким описанием и кодами; `sections(section_id)` — полный текст
узла (описание + подразделы с их кодами). Что не напечатано в Карте — тул не знает (паритет
с промптом; Карта строится только из открытых всем документов, ADR-0034).

ОДНО описание на инстанс = ПОЛНЫЙ текст, который видит агент (cfg.description перебивает
дефолт). В методику системного промпта идёт одна строка-подсказка (`hint`, по умолчанию
включена) — аналог шапки Карты «используй для навигации»: без неё агент тянется к тулу
только когда уже заблудился (замерено на smoke: 1 вызов на 5 вопросов, и тот седьмым шагом).
"""
from __future__ import annotations

from morag.retrieval.tools import ToolSpec


def sections_description(note: str = '') -> str:
    """Полное описание инстанса: доменная шапка (опц.) + что даёт тул и когда его звать."""
    base = (
        'Описания разделов базы знаний. Без аргументов — перечень верхнеуровневых разделов '
        'с кратким описанием и кодом каждого; с section_id — подробное описание раздела: '
        'о чём он и какие подразделы (с кодами) в нём есть. Вызывай в начале, чтобы понять, '
        'где может лежать ответ, и углубляйся в подходящий раздел; полученные коды передавай '
        'в search(section_ids=[…]) или find_section для сужения поиска. Сам текст документов '
        'тул не показывает — за содержанием иди в search/get_doc.'
    )
    note = note.strip()
    return f'{note} {base}' if note else base


def _describe(cfg: dict) -> str:
    return cfg.get('description') or sections_description()


def _status(cfg: dict, args: dict, resolve_title) -> str:
    sid = (args or {}).get('section_id')
    return f'описание раздела {sid}' if sid else 'перечень разделов базы'


def _prompt_section(cfg: dict) -> str:
    if not cfg.get('hint', True):
        return ''
    name = cfg.get('name') or 'sections'
    return (
        f'\n\nСтруктуру базы знаний (разделы, их описания и коды) даёт инструмент `{name}`: '
        f'вызови `{name}()` в начале, чтобы понять, где может лежать ответ, и '
        f'`{name}(section_id)` — чтобы раскрыть подразделы выбранного раздела.'
    )


SPEC = ToolSpec(
    type='sections',
    default_name='sections',
    core=False,
    parameters={
        'type': 'object',
        'properties': {
            'section_id': {
                'type': 'string',
                'description': (
                    'Код раздела из перечня (как напечатан, например D8CFGFG). '
                    'Не передавай — получишь перечень верхнеуровневых разделов.'
                ),
            },
        },
        'required': [],
    },
    describe=_describe,
    prompt_section=_prompt_section,
    handler=lambda p, cfg, args, ctx: p._tool_sections(args.get('section_id') or '', ctx),
    status=_status,
    icon='🗂️',
)
