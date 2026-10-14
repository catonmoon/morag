"""Страж латиницы: ухо, пишущее кириллицей, не вправе заменить латинское слово куска.

Поймано на стенде «Мостика» (2026-10-14) с GigaAM вторым ухом: правило частоты меняло «NovelAI»
на «вел» и «cloaked» на «Cloud» — латиницу кусок взял из подсказки, ухо её просто не пишет.
"""
from stages.arbitrate import arbitrate


RAW = 'мы взяли модель на NovelAI и посмотрели'
SECOND = 'мы взяли модель на вел и посмотрели'


def test_guard_keeps_the_latin_word():
    text, decisions = arbitrate(RAW, SECOND, None, set(), latin_guard=True)
    assert 'NovelAI' in text
    assert not [d for d in decisions if d.get('taken')]


def test_off_by_default_behaviour_is_unchanged():
    on, _ = arbitrate(RAW, SECOND, None, set(), latin_guard=True)
    off, _ = arbitrate(RAW, SECOND, None, set())
    # без стража правило решает как раньше (что бы оно ни решило), стража оно не видит
    assert off == arbitrate(RAW, SECOND, None, set())[0]
    assert 'NovelAI' in on


def test_cyrillic_disputes_are_untouched_by_the_guard():
    raw, second, clean = 'это материнские оплаты такого класса', 'это материнские платы такого класса', \
        'это материнские платы такого класса'
    a, _ = arbitrate(raw, second, clean, set(), latin_guard=True)
    b, _ = arbitrate(raw, second, clean, set())
    assert a == b
