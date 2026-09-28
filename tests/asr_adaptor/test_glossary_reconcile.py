"""Сверка глоссария с известными написаниями: подмена по свидетелю, транслитерация без свидетеля — вон."""
from __future__ import annotations

import pytest

from stages import glossary as G


@pytest.fixture(autouse=True)
def fake_freq(monkeypatch):
    """Частотник в тесте — таблица: «flow», «rest» — обычные английские слова, прочее — нет."""
    monkeypatch.setattr(G, '_HAS_WF', True)
    monkeypatch.setattr(G, '_zipf', lambda w, lang: {'flow': 5.0, 'rest': 5.0, 'label': 4.4}.get(w, 0.0))


def test_garble_canonical_is_replaced_by_the_known_spelling():
    log = []
    out = G.reconcile([{'heard': 'постгресс', 'canonicals': ['Postgress']}], known=['Постгрес'], log=log)
    assert out == [{'heard': 'постгресс', 'canonicals': ['Постгрес']}]
    assert log[0]['why'] == 'известное написание'


def test_known_canonical_is_left_alone_even_if_another_known_form_sounds_the_same():
    out = G.reconcile([{'heard': 'Постгресе', 'canonicals': ['Постгрес']}], known=['Постгрес', 'Postgres'])
    assert out[0]['canonicals'] == ['Постгрес']


def test_fuzzy_substitution_needs_both_sides_and_a_rare_canonical():
    # «Flow» → «MLflow»: похоже по звучанию, но «flow» — обычное английское слово; «creds» → «Redis»:
    # каноник похож, а услышанное — нет. Оба остаются как были.
    out = G.reconcile([{'heard': 'флоу', 'canonicals': ['Flow']}, {'heard': 'кредами', 'canonicals': ['creds']}],
                      known=['MLflow', 'Redis'])
    assert [e['canonicals'] for e in out] == [['Flow'], ['creds']]
    out = G.reconcile([{'heard': 'Кубернетис', 'canonicals': ['Kubernetis']}], known=['Kubernetes'])
    assert out[0]['canonicals'] == ['Kubernetes']


def test_unknown_latin_canonical_is_left_alone():
    """Без свидетеля каноник не трогаем: правила «транслитерация — вон» здесь нет (решение владельца)."""
    out = G.reconcile([{'heard': 'Гитлаб', 'canonicals': ['GitLab']}, {'heard': 'Квен', 'canonicals': ['Qwen', 'Квен']}])
    assert [e['canonicals'] for e in out] == [['GitLab'], ['Qwen', 'Квен']]


def test_empty_input_is_fine():
    assert G.reconcile([]) == []
