"""Короткий идентификатор документа для агента: `D7QK2MV` вместо `local:corp:a/b/c/doc.md`.

Зачем. Идентификаторы агент передаёт в аргументах инструментов, а аргументы модель ПЕЧАТАЕТ — тем
же механизмом, которым пишет прозу. Замерено на корпусе расшифровок (ADR-0025): при
длине `doc_id` в 72 символа медианой агент роняет буквы при переписывании, и 8-9% вызовов `get_doc`
уходят в несуществующий документ. Заплатки в промпте исчерпаны — лечится только тем, что
переписывать больше нечего.

Форма: `<буква типа><тело><контрольный знак>`, например `D7QK2MV` (документ) и `S3M9XV1` (раздел).

⚠️ Алфавит из 31 знака, и это ПРОСТОЕ число — на нём держится вся проверка. Знаков-двойников нет:
выброшены `I`, `L`, `O`, `U` (путаются с `1` и `0`) и `Z` (чтобы размер стал простым). Одна форма
записи — только верхний регистр.

⚠️ `doc_id` берётся как НЕПРОЗРАЧНАЯ строка целиком. Разбирать его нельзя: у вложений Confluence он
четырёхсегментный (`confluence:<name>:att:<id>`), и любая схема с `split(':')` на этом ломается.
"""
from __future__ import annotations

import hashlib

# 31 знак: цифры и буквы без I, L, O, U, Z. Размер — простое число (см. `check_char`).
ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXY'
_MODULUS = len(ALPHABET)
_VALUE = {c: i for i, c in enumerate(ALPHABET)}

DOC = 'D'                # обычный документ → get_doc, search(doc_ids=…)
SECTION = 'S'            # структурный узел (раздел) → search(section_ids=…)
_KINDS = (DOC, SECTION)

DEFAULT_BODY_LEN = 5     # см. ADR-0025: на 247 объектах ни одного удлинения, на миллионе — 1.7%
_MAX_BODY_LEN = 24       # предел деривации из sha256; до него не доходит даже миллиард документов


def body_chars(doc_id: str, length: int = DEFAULT_BODY_LEN) -> str:
    """`length` знаков тела — детерминированно из `doc_id`, без состояния.

    Знаки берутся последовательно, поэтому тело длины N+1 продолжает тело длины N: удлинение при
    коллизии не переписывает уже выданное, а дописывает знак.
    """
    if not 1 <= length <= _MAX_BODY_LEN:
        raise ValueError(f'длина тела вне диапазона 1..{_MAX_BODY_LEN}: {length}')
    n = int.from_bytes(hashlib.sha256(doc_id.encode('utf-8')).digest(), 'big')
    out = []
    for _ in range(length):
        out.append(ALPHABET[n % _MODULUS])
        n //= _MODULUS
    return ''.join(out)


def check_char(payload: str) -> str:
    """Контрольный знак: взвешенная сумма по ПРОСТОМУ модулю 31.

    Считается по букве типа ВМЕСТЕ с телом, поэтому ловится и подмена `D`↔`S`.

    Почему это, а не алгоритм Дамма: гарантия та же, но выводится из арифметики, а не из таблицы.
    Модуль простой, значит при замене одного знака невязка равна `(i+1)·Δ ≢ 0`, а при перестановке
    соседних — `Δ ≢ 0`; ни то, ни другое не может обнулиться. Замерено на 50 000 испорченных кодах:
    100.00% и 100.00%. Дамму для того же понадобилась бы квазигруппа 32×32 — таблица, которую в
    generic-движке пришлось бы зашивать константой.
    """
    return ALPHABET[sum((i + 1) * _VALUE[c] for i, c in enumerate(payload)) % _MODULUS]


def make(doc_id: str, *, structural: bool = False, length: int = DEFAULT_BODY_LEN) -> str:
    """Код документа: буква типа + тело + контрольный знак."""
    kind = SECTION if structural else DOC
    body = body_chars(doc_id, length)
    return kind + body + check_char(kind + body)


def parse(code: str) -> tuple[str, str] | None:
    """`(буква типа, тело)` — либо `None`, если это не код: испорчен, выдуман или чужая строка.

    ⚠️ Проверяется ФОРМА, а не существование документа: код может быть безупречным и всё равно
    указывать в никуда. Это два разных отказа, и агенту на них нужно разное — «перепиши код» против
    «сделай search». Разводить их обязан вызывающий.
    """
    if not isinstance(code, str):
        return None
    code = code.strip()
    if len(code) < 3 or code[0] not in _KINDS:
        return None
    if any(c not in _VALUE for c in code):
        return None
    body = code[1:-1]
    if check_char(code[:-1]) != code[-1]:
        return None
    return code[0], body


def looks_like_code(value: str) -> bool:
    """Похоже ли на код ПО ФОРМЕ — без проверки контрольного знака.

    Нужно ровно в одном месте: отличить «агент передал испорченный код» от «агент передал длинный
    `doc_id`». Первому надо сказать «перепиши», второму — ничего.
    """
    if not isinstance(value, str):
        return False
    value = value.strip()
    return (
        3 <= len(value) <= _MAX_BODY_LEN + 2
        and value[0] in _KINDS
        and all(c in _VALUE for c in value)
    )


def assign(
    doc_id: str,
    *,
    structural: bool = False,
    taken: dict[str, str] | None = None,
    length: int = DEFAULT_BODY_LEN,
) -> str:
    """Код для документа с разрешением коллизии удлинением тела.

    `taken` — уже занятые коды (`код → doc_id`); он же источник истины и он же обновляется
    вызывающим. Свой прежний код документ получает обратно, а не новый.

    ⚠️ Уникальность обеспечивается ЗАПИСЬЮ, а не длиной. Совпало — берём на знак больше от того же
    хэша, и так до `_MAX_BODY_LEN`. Поэтому переполниться нечему: на миллионе документов удлиняются
    1.7%, максимум до семи знаков тела.
    ⚠️ Порядок назначения задаёт вызывающий, и он обязан быть детерминированным (у нас — по
    возрастанию `doc_id`), иначе полная пересборка корпуса выдаст другую таблицу.
    """
    taken = taken if taken is not None else {}
    for n in range(length, _MAX_BODY_LEN + 1):
        code = make(doc_id, structural=structural, length=n)
        owner = taken.get(code)
        if owner is None or owner == doc_id:
            return code
    raise RuntimeError(f'не удалось выдать код для {doc_id!r}: заняты все длины до {_MAX_BODY_LEN}')


def is_structural_doc_id(doc_id: str) -> bool:
    """Структурный узел (папка) — у него `doc_id` заканчивается слэшем.

    Признак дублируется полем `structural` в payload; здесь он нужен там, где payload'а нет под
    рукой (миграция, тесты).
    """
    return doc_id.endswith('/')
