"""Третий голос по требованию не зовётся, если все споры куска — о форме одного слова.

Замер 14.10.2026 на выпуске 2-5 «Капитанского мостика» с ухом другой школы: 39 из 196 спорных
кусков спорили только о форме («повышенные/повышенной»), третий голос слушал бы на 688 с меньше из
3341, цена — одно решение голосованием («тем»→«теми»). Флаг ASR_CLEAN_EAR_SKIP_INFLECTION, по
умолчанию выключен.
"""
from types import SimpleNamespace

from conveyor.loop import Action, Place
from conveyor.policy import RulePolicy, _inflection_only


def _cfg(skip: bool):
    return SimpleNamespace(arbitrate_gate='', clean_ear='demand', clean_ear_window='chunk',
                           clean_ear_skip_inflection=skip)


def _place(decisions):
    p = Place('chunk:1', 'arbitrate', item={'raw': 'x', 'start': 0.0, 'end': 10.0}, canon=set())
    p.observe(Action('listen', {'ear': 'second', 'window': 'same'}), {'ear': 'second', 'text': 'x'})
    p.observe(Action('arbitrate_rules', {}), {'disputes': len(decisions), 'decisions': decisions, 'taken': 0})
    p.decisions = decisions
    return p


FORM_ONLY = [{'i': 3, 'was': 'повышенные', 'now': 'повышенной', 'by': 'спорно', 'taken': False}]
REAL = [{'i': 3, 'was': 'оплаты', 'now': 'платы', 'by': 'спорно', 'taken': False}]


def test_inflection_only_detector():
    assert _inflection_only(FORM_ONLY)
    assert not _inflection_only(REAL)
    assert not _inflection_only(FORM_ONLY + REAL)
    assert not _inflection_only([])


async def test_skip_only_when_flag_is_on():
    assert isinstance(await RulePolicy(_cfg(False)).next(_place(FORM_ONLY)), Action)      # как раньше: слушаем
    fin = await RulePolicy(_cfg(True)).next(_place(FORM_ONLY))
    assert fin.decision == 'apply' and 'форме слова' in fin.why


async def test_real_dispute_still_calls_the_third_voice():
    act = await RulePolicy(_cfg(True)).next(_place(REAL))
    assert isinstance(act, Action) and act.tool == 'listen' and act.args['ear'] == 'clean'
