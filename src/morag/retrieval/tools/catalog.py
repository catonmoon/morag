"""Тип `catalog` — полный структурный каталог корпуса (метаданные документов).

Закрывает дыру контентного RAG на запросах, требующих обойти ВЕСЬ корпус,
а не найти top-k: «перечисли всех X», «сколько документов с Y», «где не было Z»,
«самый частый W». Агент получает таблицу целиком (по строке на документ,
поля из cfg.fields) и сам считает/резолвит имена/группирует.

ОДНО описание на инстанс = ПОЛНЫЙ текст, который видит агент. Пусто → дефолт
(гайд агрегаций + перечень полей). Прежний отдельный system-prompt-блок свёрнут
сюда. Отдельный тип, НЕ сводится к lookup: отдаёт JSON-таблицу без реранка и
без чанков-цитат (lookup = get_doc → реранк → чанки).
"""
from __future__ import annotations

from morag.retrieval.tools import ToolSpec


def catalog_description(fields: list[str], note: str = '') -> str:
    """Полное описание catalog-инстанса: доменная шапка (note, опц.) + гайд
    агрегаций + перечень полей."""
    fields_str = ', '.join(fields or [])
    base = (
        f'Полный СТРУКТУРНЫЙ каталог всех документов корпуса — одна строка на документ, '
        f'поля: {fields_str}. Вызывай (НЕ find_section/search) для вопросов про СОСТАВ / '
        'ДАТЫ / КОЛИЧЕСТВО / ПЕРЕЧНИ / ЧАСТОТУ, требующих обойти ВЕСЬ корпус, а не найти '
        'top-k: «перечисли всех…», «сколько записей с…», «где не было…», «самый частый…», '
        '«последний с…». Ответ вычисли САМ по таблице: неточные имена из вопроса сопоставь '
        'с полными из каталога, посчитай строки, сравни даты. Для вопросов о СОДЕРЖАНИИ '
        '(что обсуждали, что кто сказал про тему) это НЕ каталог — иди find_section/search.'
    )
    note = note.strip()
    return f'{note} {base}' if note else base


def _describe(cfg: dict) -> str:
    return cfg.get('description') or catalog_description(cfg.get('fields') or [])


SPEC = ToolSpec(
    type='catalog',
    default_name='catalog',
    core=False,
    parameters={'type': 'object', 'properties': {}, 'required': []},
    describe=_describe,
    prompt_section=lambda cfg: '',
    handler=lambda p, cfg, args: p._tool_catalog(cfg.get('fields') or []),
    status=lambda cfg, args, _t: 'читаю каталог корпуса',
    icon='📋',
)
