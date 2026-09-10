"""Короткий идентификатор документа: форма, контрольный знак, разрешение коллизий.

Это фундамент правки ADR-0025: на нём стоят и печать агенту, и резолв аргументов инструментов, и
миграция payload. Поэтому проверяются не «функции работают», а свойства, из-за которых схема и
выбрана: опечатка обязана ловиться тождественно, уже выданный код — не меняться, а переполниться
не должно быть чему.
"""
from __future__ import annotations

import random

import pytest

from morag import shortid

DOC_A = 'local:demo:talks/2024/alpha.md'
DOC_B = 'local:demo:talks/2024/beta.md'
SECTION_A = 'local:demo:talks/2024/'
# ⚠️ У вложений Confluence doc_id ЧЕТЫРЁХсегментный — код обязан считаться от строки целиком.
ATTACHMENT = 'confluence:demo:att:12345'


# --- алфавит и форма --------------------------------------------------------


def test_alphabet_is_prime_sized_and_free_of_lookalikes():
    """⚠️ На простом размере алфавита держится СТОПРОЦЕНТНАЯ проверка (см. `check_char`).
    Уберут знак или добавят — гарантия молча станет вероятностной."""
    assert len(shortid.ALPHABET) == 31
    assert all(c not in shortid.ALPHABET for c in 'ILOUZ'), 'знаки-двойники и Z недопустимы'
    assert shortid.ALPHABET == shortid.ALPHABET.upper()
    assert len(set(shortid.ALPHABET)) == len(shortid.ALPHABET)


def test_code_shape():
    code = shortid.make(DOC_A)
    assert len(code) == 1 + shortid.DEFAULT_BODY_LEN + 1 == 7
    assert code[0] == shortid.DOC
    assert shortid.make(SECTION_A, structural=True)[0] == shortid.SECTION


def test_code_is_stable_and_distinct():
    assert shortid.make(DOC_A) == shortid.make(DOC_A), 'код обязан быть чистой функцией от doc_id'
    assert shortid.make(DOC_A) != shortid.make(DOC_B)


def test_longer_body_extends_shorter_one():
    """Удлинение при коллизии ДОПИСЫВАЕТ знак, а не пересчитывает код: иначе выданный код менялся
    бы от появления соседа."""
    short = shortid.body_chars(DOC_A, 5)
    assert shortid.body_chars(DOC_A, 7).startswith(short)


def test_composite_doc_id_is_opaque():
    """Четырёхсегментный doc_id вложения — обычная строка, и код от неё считается как от всякой."""
    assert shortid.parse(shortid.make(ATTACHMENT)) is not None


def test_body_length_is_bounded():
    with pytest.raises(ValueError):
        shortid.body_chars(DOC_A, 0)
    with pytest.raises(ValueError):
        shortid.body_chars(DOC_A, 99)


# --- контрольный знак: гарантия, а не вероятность ----------------------------


def _corrupt_one(code: str, rnd: random.Random) -> str:
    i = rnd.randrange(len(code))
    return code[:i] + rnd.choice([c for c in shortid.ALPHABET if c != code[i]]) + code[i + 1:]


def test_single_substitution_is_always_caught():
    """⚠️ ВСЕГДА, а не «почти всегда»: модуль простой, поэтому невязка `(i+1)·Δ` не обнуляется.
    Перебираем каждую позицию и каждый знак алфавита — исчерпывающе, без выборки."""
    for doc in (DOC_A, DOC_B, ATTACHMENT):
        code = shortid.make(doc)
        for i in range(len(code)):
            for ch in shortid.ALPHABET:
                if ch == code[i]:
                    continue
                broken = code[:i] + ch + code[i + 1:]
                assert shortid.parse(broken) is None, f'не поймана замена {code}→{broken}'


def test_adjacent_transposition_is_always_caught():
    """Вторая по частоте ошибка переписывания. Невязка равна `Δ` — тоже не обнуляется."""
    for doc in (DOC_A, DOC_B, ATTACHMENT, SECTION_A):
        code = shortid.make(doc, structural=doc.endswith('/'))
        for i in range(len(code) - 1):
            if code[i] == code[i + 1]:
                continue
            swapped = code[:i] + code[i + 1] + code[i] + code[i + 2:]
            assert shortid.parse(swapped) is None, f'не поймана перестановка {code}→{swapped}'


def test_kind_letter_is_covered_by_the_check():
    """Контрольный знак считается по букве типа вместе с телом, поэтому подмена `D`↔`S` — тоже
    порча, а не другой законный код. Иначе агент, перепутав тип, попал бы в существующий объект."""
    code = shortid.make(DOC_A)
    assert shortid.parse(shortid.SECTION + code[1:]) is None


def test_fabricated_codes_are_mostly_rejected_locally():
    """Выдуманный код почти всегда отсекается ДО обращения к базе; остаток — забота вызывающего.
    Порог 90% с запасом: теоретически проходит 1/31 ≈ 3.2%."""
    rnd = random.Random(20260910)
    made_up = [
        shortid.DOC + ''.join(rnd.choice(shortid.ALPHABET) for _ in range(6))
        for _ in range(3000)
    ]
    rejected = sum(1 for c in made_up if shortid.parse(c) is None)
    assert rejected / len(made_up) > 0.90


def test_parse_rejects_foreign_strings():
    for bad in ('', 'D', 'x', 'local:demo:talks/2024/alpha.md', 'D7QK2M!', 'd7qk2mv', None):
        assert shortid.parse(bad) is None, f'принято за код: {bad!r}'


def test_parse_returns_kind_and_body():
    code = shortid.make(DOC_A)
    kind, body = shortid.parse(code)
    assert kind == shortid.DOC
    assert body == shortid.body_chars(DOC_A, shortid.DEFAULT_BODY_LEN)


def test_looks_like_code_separates_a_typo_from_a_long_doc_id():
    """Форма без проверки знака: нужна, чтобы отличить «испорченный код» от «длинного doc_id» и
    сказать агенту разное."""
    assert shortid.looks_like_code(_corrupt_one(shortid.make(DOC_A), random.Random(1)))
    assert not shortid.looks_like_code(DOC_A)
    assert not shortid.looks_like_code('confluence:demo:att:12345')


# --- назначение и коллизии --------------------------------------------------


def test_assign_returns_the_same_code_for_the_same_document():
    taken: dict[str, str] = {}
    first = shortid.assign(DOC_A, taken=taken)
    taken[first] = DOC_A
    assert shortid.assign(DOC_A, taken=taken) == first, 'повторный проход не должен менять код'


def test_collision_lengthens_the_newcomer_not_the_holder():
    """⚠️ Занял — держит. Удлиняется только пришедший позже, иначе выданный код менялся бы от
    появления соседа, а на этом стоит и ссылка в диалоге, и подсказка в отказе.

    Настоящей коллизии на пяти знаках не подстроить (28.6 млн комбинаций), поэтому код документа B
    занимает документ A — ровно то же состояние, что даёт коллизия.
    """
    held = shortid.make(DOC_B)
    taken = {held: DOC_A}                        # A держит код, который вывелся бы у B
    newcomer = shortid.assign(DOC_B, taken=taken)
    assert newcomer != held, 'пришедший позже обязан получить другой код'
    assert len(newcomer) == len(held) + 1, 'разрешение коллизии — ровно один добавленный знак'
    assert taken[held] == DOC_A, 'у держателя код не отобран'
    assert newcomer not in taken


def test_collision_is_resolved_by_one_more_char():
    """Настоящей коллизии на пяти знаках не подстроить, поэтому занимаем код чужим владельцем и
    смотрим, что деривация продолжается тем же хэшем, а не начинается заново."""
    squatted = shortid.make(DOC_A)
    taken = {squatted: 'local:demo:someone/else.md'}
    code = shortid.assign(DOC_A, taken=taken)
    kind, body = shortid.parse(code)
    assert kind == shortid.DOC
    assert body == shortid.body_chars(DOC_A, shortid.DEFAULT_BODY_LEN + 1)
    assert body.startswith(shortid.body_chars(DOC_A, shortid.DEFAULT_BODY_LEN))


def test_assign_is_deterministic_on_a_whole_corpus():
    """Порядок назначения детерминирован вызывающим (по возрастанию doc_id) — значит полная
    пересборка корпуса воспроизводит ТУ ЖЕ таблицу. Иначе коды поехали бы после реиндекса."""
    ids = [f'local:demo:talks/2024/doc{i}.md' for i in range(400)]

    def build() -> dict[str, str]:
        taken: dict[str, str] = {}
        for doc in sorted(ids):
            taken[shortid.assign(doc, taken=taken)] = doc
        return taken

    assert build() == build()


def test_no_collisions_on_a_corpus_of_thousands():
    """Уникальность обеспечивается записью, поэтому дублей быть не может ПО ПОСТРОЕНИЮ — проверяем
    это на объёме, а не на трёх примерах."""
    taken: dict[str, str] = {}
    ids = [f'local:demo:talks/{i // 50}/doc{i}.md' for i in range(5000)]
    ids += [f'local:demo:talks/{i}/' for i in range(100)]
    for doc in sorted(ids):
        code = shortid.assign(doc, structural=shortid.is_structural_doc_id(doc), taken=taken)
        assert code not in taken
        taken[code] = doc
    assert len(taken) == len(ids)
    assert all(shortid.parse(c) is not None for c in taken)


def test_structural_ids_are_recognised_by_the_trailing_slash():
    assert shortid.is_structural_doc_id(SECTION_A)
    assert not shortid.is_structural_doc_id(DOC_A)
