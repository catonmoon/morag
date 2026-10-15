"""Регрессия корректора на настоящих местах «Капитанского мостика» с вердиктом владельца по слуху.

Случаи — `fixtures/proofread_regression.json`: свидетель (`heard_supports`), страж новой латиницы
(`new_latin`, `latin_known`) и род правки для маршрута (`edit_kind`). Каждый случай — реальная верная правка
или порча из разметки 13–15.10.2026; если поведение меняется сознательно — меняется и ожидание в файле.
"""
import json
from pathlib import Path

import pytest

from conveyor.editor import heard_supports
from conveyor.proofread import edit_kind, latin_known, new_latin
from stages import arbitrate as A

CASES = json.loads((Path(__file__).parent / "fixtures" / "proofread_regression.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES["witness"], ids=lambda c: f"{c['was']}→{c['now']}")
def test_witness(case):
    for ear, expected in case["supports"].items():
        got = heard_supports(A, case["ears"][ear], case["was"], case["now"])[0]
        assert got is expected, f"{ear}: {case['why']}"


@pytest.mark.parametrize("case", CASES["latin"], ids=lambda c: f"{c['was']}→{c['now']}")
def test_latin_guard(case):
    assert new_latin(case["was"], case["now"]) == case["new_latin"], case["why"]
    known = {w.lower() for t in case["known"] for w in t.replace("-", " ").split()}
    assert latin_known(case["now"], known) is case["known_ok"], case["why"]


@pytest.mark.parametrize("case", CASES["kind"], ids=lambda c: f"{c['was']}→{c['now']}")
def test_edit_kind(case):
    gloss = {w.lower() for t in case.get("gloss", []) for w in t.split()}
    assert edit_kind(case["was"], case["now"], gloss) == case["kind"], case.get("why", "")
