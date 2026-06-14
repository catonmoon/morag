"""Финал-раунд: контекст-арбитр (порт pass2_final + final-round из pass2_full на morag LLMClient, async).

Контекст = doc-summary (1× на выпуск) + per-chunk summary (двухфазно). correct правит ТОЛЬКО сущности,
прозу — дословно. Гейт `has_entity_signal` (раунд только на репликах с сущностями). RAW сохраняет
вызывающая сторона (сайдкар). Промпты перенесены дословно (валидированы на eval_final).
"""
from __future__ import annotations

import re

from .glossary import relevant

_LAT = re.compile('[A-Za-z]')
_DIG = re.compile(r'\d')

_CORRECT_SYS = (
    'Ты редактируешь черновую ASR-расшифровку фрагмента русскоязычного подкаста про технологии. '
    'ЕДИНСТВЕННАЯ задача — исправить НЕВЕРНО РАСПОЗНАННЫЕ имена собственные, названия компаний, '
    'моделей, продуктов, технические термины и аббревиатуры, опираясь на контекст разговора и список '
    'известных корректных терминов. ВСЕ остальные слова, их порядок и пунктуацию оставляй ДОСЛОВНО — '
    'не перефразируй, не сокращай, не «улучшай» стиль, не дополняй. Если сомневаешься — НЕ трогай. '
    'Верни ТОЛЬКО исправленный текст фрагмента, без пояснений и без кавычек.'
)
_SUMMARY_SYS = (
    'По краткому описанию всего выпуска и одному фрагменту опиши в 1-2 предложениях, О ЧЁМ именно этот '
    'фрагмент и какие конкретные имена/компании/модели/термины в нём упоминаются (в правильном '
    'написании). Кратко, без воды.'
)
_DOC_SYS = (
    'Сожми выпуск подкаста в 4-6 предложений: темы + ключевые имена/компании/модели/термины в '
    'ПРАВИЛЬНОМ написании. Без вступлений и воды.'
)


def has_entity_signal(text: str, gloss) -> bool:
    """Реплика с сущностями (гоним раунд): латиница / цифры / glossary-match / заглавная не в начале."""
    if _LAT.search(text) or _DIG.search(text):
        return True
    if relevant(text, gloss):
        return True
    words = re.findall(r'\S+', text)
    for i, w in enumerate(words):
        if i > 0 and w[:1].isupper() and not words[i - 1].endswith(('.', '!', '?', ':', '…')):
            return True
    return False


async def doc_summary(full_text: str, llm) -> str:
    return await llm.complete(
        [{'role': 'system', 'content': _DOC_SYS}, {'role': 'user', 'content': full_text[:8000]}],
        max_tokens=400)


async def chunk_summary(doc_sum: str, chunk_text: str, llm) -> str:
    body = f'Описание выпуска: {doc_sum}\n\nФрагмент (черновой ASR):\n{chunk_text}'
    return await llm.complete(
        [{'role': 'system', 'content': _SUMMARY_SYS}, {'role': 'user', 'content': body}], max_tokens=300)


async def correct(text: str, doc_sum: str, chunk_sum: str, canonicals, llm) -> str:
    cand = ', '.join(canonicals[:200]) if canonicals else '(нет)'
    body = (f'Описание выпуска: {doc_sum}\nЧто в этом фрагменте: {chunk_sum}\n\n'
            f'Известные корректные термины и имена (каноники): {cand}\n\nЧерновой фрагмент:\n{text}')
    out = await llm.complete(
        [{'role': 'system', 'content': _CORRECT_SYS}, {'role': 'user', 'content': body}], max_tokens=2000)
    return (out or '').strip() or text
