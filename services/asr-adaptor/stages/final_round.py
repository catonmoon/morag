"""Финал-раунд: контекст-арбитр (порт pass2_final + final-round из pass2_full на morag LLMClient, async).

Контекст = doc-summary (1× на выпуск) + per-chunk summary (двухфазно). correct правит ТОЛЬКО сущности,
прозу — дословно. Гейт `has_entity_signal` (раунд только на репликах с сущностями). RAW сохраняет
вызывающая сторона (сайдкар). Промпты перенесены дословно (валидированы на eval_final).
"""
from __future__ import annotations

import asyncio
import logging
import re

from .glossary import _sentence_batches, is_common_ru, is_plain_english, relevant

log = logging.getLogger('asr')

_LAT = re.compile('[A-Za-z]')
_DIG = re.compile(r'\d')
_CYR = re.compile('[а-яёА-ЯЁ]')

MAX_WAS_WORDS = 3   # гарбл может разъехаться на пару слов («и Селедец»), абзац — не может
MAX_NOW_WORDS = 6

DEFAULT_CORPUS = 'русскоязычного подкаста про технологии'

# ⚠️ Подставляется .replace(), а НЕ .format(): ниже в промпте есть литеральные фигурные скобки
# ({"fixes":…}), и format() на них падает KeyError.
_CORPUS_SLOT = '@CORPUS@'

_CORRECT_SYS = (
    f'Ты вычитываешь черновую ASR-расшифровку фрагмента {_CORPUS_SLOT}. '
    'Найди НЕВЕРНО РАСПОЗНАННЫЕ имена собственные, названия компаний, моделей, продуктов, '
    'технические термины и аббревиатуры — и верни СПИСОК ЗАМЕН, а не исправленный текст.\n'
    'Для каждой замены: "was" — фраза ИЗ ФРАГМЕНТА ДОСЛОВНО, ровно как она там написана, от одного '
    'до трёх слов (бери соседнее слово, если гарбл разъехался: «и Селедец» → «Иван Оселедец»); '
    '"now" — правильная форма.\n'
    'Прозу, порядок слов и пунктуацию не трогай — замены применит код, остальной текст останется '
    'как есть. НЕ переводи русское на английский («агенты» — не «agents», «умные очки» — не «smart '
    'glasses»). Латиницей — только то, что и в русском тексте пишут латиницей (NVIDIA, ChatGPT, '
    'H200); есть обычная русская форма — оставляй русскую. Аббревиатуры НЕ разворачивай: сказано '
    '«MCP» — остаётся «MCP». Сомневаешься — НЕ включай замену.\n'
    'Исправлять нечего — верни пустой список. Верни СТРОГО JSON {"fixes":[{"was":..,"now":..}]}, '
    'без иного текста.'
)

_FIX_SCHEMA = {
    'type': 'object',
    'properties': {'fixes': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {'was': {'type': 'string'}, 'now': {'type': 'string'}},
        'required': ['was', 'now']}}},
    'required': ['fixes'],
}
_RECALL_SYS = (
    'По описанию выпуска и фрагменту черновой ASR-расшифровки перечисли, какие конкретные имена, '
    'компании, модели, продукты и термины в этом фрагменте упоминаются — в ПРАВИЛЬНОМ написании. '
    'Одной-двумя фразами, без воды. Ничего не выдумывай: только то, что в фрагменте есть.'
)
_DOC_MERGE_SYS = (
    'Тебе даны сводки последовательных частей одного выпуска подкаста. Слей их в 6-8 предложений: '
    'темы + ключевые имена/компании/модели/термины в ПРАВИЛЬНОМ написании. Ничего не выдумывай, '
    'повторы убери. Без вступлений и воды.'
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
    """Сводка по ВСЕМУ выпуску, а не по его началу.

    Раньше здесь стояло `full_text[:8000]` — это 12% текста, первые девять-десять минут из
    семидесяти пяти. Правка на четырнадцатой минуте получала «описание выпуска», которое той части
    разговора не видело, и достраивала сущности по правдоподобию: так «WeChat» стал «Alipay» —
    Alipay в доступном контексте был, WeChat не было. Батчи считаем разом, как в глоссарии.
    """
    batches = list(_sentence_batches(full_text, 8000))
    parts = await asyncio.gather(*(
        llm.complete([{'role': 'system', 'content': _DOC_SYS}, {'role': 'user', 'content': b}],
                     max_tokens=400) for b in batches))
    parts = [p for p in parts if (p or '').strip()]
    if len(parts) <= 1:
        return parts[0] if parts else ''
    merged = await llm.complete(
        [{'role': 'system', 'content': _DOC_MERGE_SYS}, {'role': 'user', 'content': '\n'.join(parts)}],
        max_tokens=600)
    return (merged or '').strip() or ' '.join(parts)


def _key(s: str) -> str:
    return re.sub(r'[^0-9a-zа-яё]', '', s.lower())


def _term_survives(term: str, before: str, after: str) -> bool:
    """Известный термин, стоявший в тексте, обязан уцелеть.

    Считаем ВХОЖДЕНИЯ значимых слов термина, а не сравниваем строки целиком: в тексте обычно одна
    фамилия без имени. Подстрока заодно снимает падежи даром — «Колодезева» содержит «Колодезев», а
    гарбл «Колодзева» не содержит. Стеммер тут был лишним: по корпусу склонений у защищаемых имён
    нет вовсе (53 именительных «Колодезев», а «формы» оказались гарблом), зато он приносил
    зависимость и три подобранных на глаз константы.
    """
    lo_before, lo_after = before.lower(), after.lower()
    for word in re.findall(r'[^\W_]{5,}', term.lower()):
        if lo_before.count(word) > lo_after.count(word):
            return False
    return True


def _is_excision(was: str, now: str) -> bool:
    """Замена ничего не принесла, только выкинула слова, — это редактирование речи, не канонизация.

    Канонизация всегда что-то ДАЁТ: правильное написание, латинское имя, расшифровку. Если новых
    слов нет, а старых стало меньше — из фразы просто изъяли кусок. Замерено на четырёх выпусках:
    19 таких случаев, восемь меняют смысл, и в четырёх исчезают названные вслух сущности —
    «Roosevelt и Churchill» → «Roosevelt», «Visa и Mastercard» → «Visa», «NVIDIA TSMC» → «NVIDIA».
    Отдельная закономерность: так модель выбрасывает то, чего не смогла починить («в Promptaf»,
    «на AMD-шном процессе»). Пусть лучше остаётся сырым — там гарбл хотя бы виден.
    """
    a = re.findall(r'[^\W_]+', was.lower())
    b = re.findall(r'[^\W_]+', now.lower())
    return bool(a) and set(b) < set(a) and len(b) < len(a)


def _changes_number(was: str, now: str) -> bool:
    """Замена меняет ЧИСЛО в обозначении — запрещено, число решает акустика.

    Зеркало правила глоссария «номера моделей: число НЕ меняй, варьируй серию». Поймано чтением
    прогона recall+1 на ep20: контекст соседних реплик, набитых H100/A100, «нормализовал» под них
    выбивающиеся обозначения — «H200»→«H100» (флагманский кейс ADR-0017, акустика выбрала H200
    через подсказку «аж двести») и «10.80»→«A100» (речь про GTX 1080). Обе стороны с цифрами и
    цифры разные — только этот случай: «аж двести»→H200 (слева цифр нет) остаётся законным,
    «10.80»→«10-80» и «АДМС-128»→«ADAS-128» (цифры те же) — тоже.
    """
    a, b = re.findall(r'\d', was), re.findall(r'\d', now)
    return bool(a) and bool(b) and a != b


def _is_translation(was: str, now: str, canon: set[str]) -> bool:
    """Замена «частотное русское слово → обычная английская фраза», не санкционированная глоссарием.

    Перевод — не канонизация: «агенты»→agents, «умные очки»→smart glasses. Имя от перевода отличаем
    по капитализации и частотности (см. glossary.is_plain_english): «сам Артман»→Sam Altman остаётся.
    Обозначения с цифрой (H200, GPT-4) исключены — их проговаривают числительными, источник частотный.
    """
    if not _LAT.search(now) or _CYR.search(now) or _DIG.search(now):
        return False
    if _key(now) in canon or any(_key(w) in canon for w in now.split()):
        return False
    return is_common_ru(was) and is_plain_english(now)


def apply_fixes(text: str, fixes, canonicals=(), always=()) -> tuple[str, int, int]:
    """Применить замены к тексту ДЕТЕРМИНИРОВАННО. Возвращает (текст, применено, отброшено).

    Смысл всей конструкции: у модели не берут прозу — только пары «было → стало», и каждая проверяется
    по отдельности. Поэтому правка физически не может удалить речь, переставить слова или перевести
    фразу; худшее, что бывает, — замена не применилась. Свободное переписывание такой гарантии не
    даёт в принципе: LLM недетерминирована, а проверять её вывод постфактум эвристиками (объём,
    смена письма) — гадание, что мы и наблюдали.

    Отбрасываем замену, если: `was` не найдено в тексте дословно · `was` длиннее трёх слов (это уже
    не сущность) · `now` пустое или неправдоподобно длинное · замена ничего не приносит, а только
    выкидывает слова · это перевод обычной речи · замена рушит известный термин корпуса.

    Заменяем ПО ГРАНИЦАМ СЛОВА, а не подстрокой: на живом прогоне замена «СМЛ»→«ASML» попала внутрь
    слова «АСМЛ» и дала «АASML». Внутри слова сущностей не бывает.

    Проверки на «объём реплики уехал» здесь НЕТ и не нужно: каждая замена ограничена по построению
    (`was` ≤ 3 слов, `now` ≤ 6 и обязано найтись в тексте), поэтому обрезать реплику нечем. Такая
    проверка осталась бы от прежней схемы со свободным переписыванием и только била бы по законным
    сжатиям: «си плюс плюс» → «C++» это сразу минус 40% на короткой реплике.
    """
    canon = {_key(c) for c in (canonicals or ())}
    applied = skipped = 0
    for fix in (fixes or ()):
        was, now = (fix.get('was') or '').strip(), (fix.get('now') or '').strip()
        if not was or not now or was == now:
            skipped += 1
            continue
        if len(was.split()) > MAX_WAS_WORDS or len(now.split()) > MAX_NOW_WORDS or len(now) > 60:
            log.warning('final-round: замена отброшена (слишком длинная): %r → %r', was, now)
            skipped += 1
            continue
        pattern = re.compile(rf'(?<!\w){re.escape(was)}(?!\w)')
        if not pattern.search(text):
            log.warning('final-round: замена отброшена (нет в тексте по границам слова): %r', was)
            skipped += 1
            continue
        if _changes_number(was, now):
            log.warning('final-round: замена отброшена (меняет число): %r → %r', was, now)
            skipped += 1
            continue
        if _is_excision(was, now):
            log.warning('final-round: замена отброшена (изъятие слов): %r → %r', was, now)
            skipped += 1
            continue
        if _is_translation(was, now, canon):
            log.warning('final-round: замена отброшена (перевод): %r → %r', was, now)
            skipped += 1
            continue
        candidate = pattern.sub(now.replace('\\', r'\\'), text)
        # Проверяем ПО РЕЗУЛЬТАТУ: термин «Дмитрий Колодезев», а замена приходит на одну фамилию —
        # «Колодезев»→«Колодзев». Строкового пересечения нет, и первая версия проверки такую
        # замену пропускала (поймано на ep13).
        broken = [t for t in (always or ()) if t and not _term_survives(t, text, candidate)]
        if broken:
            log.warning('final-round: замена отброшена (рушит %s): %r → %r', broken[0], was, now)
            skipped += 1
            continue
        text = candidate
        applied += 1
    return text, applied, skipped


async def recall_entities(doc_sum: str, text: str, llm) -> str:
    """Шаг «вспомни сущности» перед правкой: что в этом фрагменте упомянуто и как это пишется.

    Сначала я счёл его вырожденным пересказом и выбросил — и потерял полноту: на четырёх выпусках
    ушли `Midjourney`, `CUDA`, `Three Mile Island`, `Olkiluoto`, `Kaspersky`. Он не пересказывает
    текст, а заставляет модель СНАЧАЛА вспомнить сущности и лишь потом применять замены; контекст
    соседних реплик этого не заменяет — он добавляет осторожности, а не памяти.
    """
    body = f'Описание выпуска: {doc_sum}\n\nФрагмент (черновой ASR):\n{text}'
    return await llm.complete(
        [{'role': 'system', 'content': _RECALL_SYS}, {'role': 'user', 'content': body}], max_tokens=300)


async def correct(text: str, doc_sum: str, context: str, canonicals, llm, always=(),
                  recalled: str = '', corpus_desc: str = DEFAULT_CORPUS) -> str:
    """Правка сущностей: модель ПРЕДЛАГАЕТ замены, применяет их код.

    Прозу у модели не берём вовсе — только пары «было → стало» (их проверяет и применяет
    `apply_fixes`). Поэтому правка не может ни удалить речь, ни перевести её, ни переставить слова:
    текст вне замен остаётся байт-в-байт. Свободное переписывание такой гарантии не давало — модель
    недетерминирована, и её вывод приходилось ловить эвристиками постфактум: обрезанный ответ съел
    365 слов на ep11, а верно распознанный «Колодезев» вернулся гарблом. Здесь худший исход —
    замена не применилась.

    Исключение наружу не глушим: вызывающая сторона оставит реплику сырой (pipeline._final_round).
    """
    cand = ', '.join(canonicals[:200]) if canonicals else '(нет)'
    body = (f'Описание выпуска: {doc_sum}\n'
            f'Что упомянуто в этом фрагменте: {recalled or "(не выделено)"}\n\n'
            f'Разговор вокруг фрагмента (только для понимания, править его НЕ надо):\n{context}\n\n'
            f'Известные корректные термины и имена (каноники): {cand}\n\n'
            f'ФРАГМЕНТ, в котором ищем замены:\n{text}')
    res = await llm.complete_json(
        [{'role': 'system', 'content': _CORRECT_SYS.replace(_CORPUS_SLOT, corpus_desc or DEFAULT_CORPUS)},
         {'role': 'user', 'content': body}],
        schema=_FIX_SCHEMA, schema_name='fixes', max_tokens=1500)

    fixes = (res or {}).get('fixes') or []
    fixed, applied, skipped = apply_fixes(text, fixes, canonicals, always)
    if skipped:
        log.info('final-round: применено замен %d, отброшено %d', applied, skipped)
    return fixed
