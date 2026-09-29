"""Редактор расшифровки: страницы, свидетели и вето, цикл страницы, прогон графа с редактором."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import pipeline
from fakes import HINTS
from conveyor import editor as E
from conveyor.deps import Deps
from conveyor.places import Slices
from conveyor.run import run_conveyor
from conveyor.tools import ToolError


def _turns():
    return [{'start': 0.0, 'end': 20.0, 'speaker': 'Speaker_1', 'raw': 'начало речь-20 конец',
             'final': 'начало речь-20 конец',
             'segments': [{'start': 0.0, 'end': 20.0, 'text': 'начало речь-20 конец'}]},
            {'start': 20.0, 'end': 50.0, 'speaker': 'Speaker_2', 'raw': 'дальше про Кавка и очередь',
             'final': 'дальше про Кавка и очередь',
             'segments': [{'start': 20.0, 'end': 50.0, 'text': 'дальше про Кавка и очередь'}]}]


def test_pages_cut_between_segments_and_show_who_and_when():
    turns = _turns()
    pages = E.make_pages(turns, page_s=15.0)
    assert [len(p) for p in pages] == [1, 1], 'сегмент не режется, страница — между сегментами'
    text = E.page_text(pages[1], turns)
    assert 'Speaker_2' in text and 'at=20.0' in text and 'Кавка' in text


def test_heard_supports_new_words_only_when_they_sound_in_the_window():
    A = pipeline.arbitrate_stage
    assert E.heard_supports(A, 'про Кафка и очередь', 'Кавка', 'Kafka')[0]
    ok, why = E.heard_supports(A, 'про что-то другое', 'Кавка', 'Kafka')
    assert not ok and 'Kafka' in why
    assert not E.heard_supports(A, '', 'Кавка', 'Kafka')[0]


def _tools(silence, known=('Kafka',)):
    st = SimpleNamespace(turns=_turns(), known=(), hints={'terms': list(known)}, gloss=[],
                         tmp=str(silence.parent), wav=str(silence))
    d = Deps(None, pipeline)
    A = pipeline.arbitrate_stage
    ps = E.PageState(0, E.make_pages(st.turns)[0], 90.0)
    canon = A.canon_from(st.hints, [])
    reg = E.editor_tools(ps, st, d, Slices(d, st.tmp, st.wav, 't'), list(known), canon, journal=[], meter={})
    return st, ps, reg


async def test_canon_alone_confirms_only_the_spelling_of_what_already_sounds_the_same(rich, silence):
    st, ps, reg = _tools(silence)
    res = await reg.call('propose', was='Кавка', now='Kafka', at=21.0, witness='canon')
    assert not res['accepted'] and 'звук' in res['why'], 'звучит иначе («в» и «ф») — нужен звук, канона мало'
    st.turns[1]['final'] = 'дальше про Кафка и очередь'
    res = await reg.call('propose', was='Кафка', now='Kafka', at=21.0, witness='canon')
    assert res['accepted'] and st.turns[1]['final'] == 'дальше про Kafka и очередь'
    res = await reg.call('propose', was='очередь', now='Redis', at=21.0, witness='canon')
    assert not res['accepted'] and 'канон' in res['why'], 'написания нет в каноне — отказ'


async def test_sound_witness_needs_a_listened_window_that_supports_the_words(rich, silence):
    st, ps, reg = _tools(silence, known=('Kafka', 'Postgres'))
    res = await reg.call('propose', was='речь-20', now='печь-30', at=0.0, witness='sound')
    assert not res['accepted'] and 'не переслушано' in res['why']
    heard = await reg.call('listen', t0=15.0, t1=45.0)                 # заглушка слышит «печь-30»
    assert heard['text'] == 'начало печь-30 конец'
    res = await reg.call('propose', was='дальше', now='печь-30', at=21.0, witness='sound')
    assert res['accepted']
    res = await reg.call('propose', was='очередь', now='Postgres', at=21.0, witness='sound')
    assert not res['accepted'] and 'не слышно' in res['why']


async def test_lost_speech_is_inserted_only_with_sound_and_past_the_six_word_limit(rich, silence):
    st, ps, reg = _tools(silence)
    long_now = 'дальше надо было всё перенести на новую схему очень быстро про'
    ps.heard.append((20.0, 50.0, 'clean', 'дальше надо было всё перенести на новую схему очень быстро про Кавка'))
    res = await reg.call('propose', was='дальше про', now=long_now, at=21.0, witness='sound')
    assert res['accepted'], res
    assert st.turns[1]['final'].startswith(long_now)


async def test_a_phrase_not_in_the_text_is_a_recoverable_tool_error(rich, silence):
    st, ps, reg = _tools(silence)
    with pytest.raises(ToolError, match='дословно') as e:
        await reg.call('propose', was='нет такого', now='X', at=21.0, witness='canon')
    assert e.value.recoverable


def _tc(name, args, cid):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': args}}


class Script:
    def __init__(self, script):
        self.script, self.seen = list(script), []

    async def complete_with_tools(self, messages, tools, **kw):
        self.seen.append([dict(m) for m in messages])
        item = self.script.pop(0) if self.script else {'role': 'assistant', 'content': '',
                                                         'tool_calls': [_tc('finish', '{}', 'f')]}
        return {'choices': [{'message': item}]}


async def test_page_loop_runs_several_tools_per_answer_and_keeps_broken_calls_out_of_history(rich, silence):
    st, ps, reg = _tools(silence)
    st.turns[1]['final'] = 'дальше про Кафка и очередь'
    llm = Script([
        {'role': 'assistant', 'content': '', 'tool_calls': [
            _tc('lookup', '{"word": "Кавка"}', 'a'),
            _tc('propose', '{"was": "Кафка", "now": "Kafka", "at": 21.0, "witness": "canon"}', 'b'),
            _tc('listen', '{"t0": 1', 'c')]},                                  # битые аргументы
        {'role': 'assistant', 'content': '', 'tool_calls': [_tc('finish', '{}', 'd')]},
    ])
    meter: dict = {}
    await E.edit_page(ps, reg, llm, turns=st.turns, system='sys', about='демо', known=['Kafka'],
                      prev_tail='', meter=meter)
    assert st.turns[1]['final'] == 'дальше про Kafka и очередь'
    assert meter['editor_calls'] == 2 and meter['editor_bad_args'] == 1
    second = llm.seen[1]
    ids = [c['id'] for m in second if m.get('tool_calls') for c in m['tool_calls']]
    assert ids == ['a', 'b'], 'битый вызов в историю не попал'
    assert [m['tool_call_id'] for m in second if m['role'] == 'tool'] == ['a', 'b']
    assert json.loads([m for m in second if m['role'] == 'tool'][0]['content'])['known_spellings'] == ['Kafka']


async def test_conveyor_with_editor_skips_the_final_round_and_runs_the_editor_after_speakers(rich, silence):
    r = await run_conveyor(str(silence), llm=Script([]), episode='ep1', hints=HINTS, editor=True)
    nodes = r['conveyor']['nodes']
    assert nodes.index('editor') > nodes.index('speakers') and nodes.index('editor') < nodes.index('assemble')
    assert r['timing']['n_editor_pages'] >= 1
    assert not any(f.get('why') == 'known_term' for f in r.get('fixes') or ()), 'финал-раунд не звался'
    assert r['conveyor']['meter']['editor_calls'] >= 1


# --- экран в подсказку пасса-2 ----------------------------------------------------------------------

async def test_screen_terms_go_into_the_prompt_of_the_chunk_they_were_shown_with(rich, silence, monkeypatch):
    seen: list[tuple] = []

    def build_prompt(terms, counter, budget, always=(), hinted=()):
        seen.append((list(terms), set(hinted)))
        return 'каноники'

    monkeypatch.setattr(pipeline, 'build_prompt', build_prompt)
    hints = dict(HINTS, screen=[{'t0': 0.0, 't1': 12.0, 'terms': ['orders.orders', 'pyspark']},
                                {'t0': 70.0, 't1': 90.0, 'terms': ['order_total']}])
    r = await run_conveyor(str(silence), llm=object(), episode='ep1', hints=hints)
    first, last = seen[0], seen[-1]
    assert first[0][:2] == ['orders.orders', 'pyspark'] and 'pyspark' in first[1], 'слова экрана — первыми и мимо фильтра'
    assert 'order_total' not in first[0] and 'order_total' in last[0], 'только показ рядом с куском'
    assert r['timing']['n_chunks_screen'] >= 2


async def test_without_screen_the_prompt_input_is_unchanged(rich, silence, monkeypatch):
    seen: list = []
    monkeypatch.setattr(pipeline, 'build_prompt',
                        lambda terms, counter, budget, always=(), hinted=(): seen.append(list(terms)) or 'каноники')
    r = await run_conveyor(str(silence), llm=object(), episode='ep1', hints=HINTS)
    assert all(t == [] for t in seen) and 'n_chunks_screen' not in r['timing']


async def test_spelling_needs_canon_and_canon_needs_a_term_that_sounds_alike(rich, silence):
    """Свидетели по родам правки (второй живой прогон 29.09: из 8 принятых верна 1, порча 3)."""
    st, ps, reg = _tools(silence, known=('Kafka', 'Kubernetes', 'очередью'))
    ps.heard.append((20.0, 50.0, 'clean', 'дальше про Кавка и очередь'))
    res = await reg.call('propose', was='Кавка', now='Kavka', at=21.0, witness='sound')
    assert not res['accepted'] and 'написания' in res['why'], 'латиницу пишет канон, ухо написаний не судит'
    res = await reg.call('propose', was='Кавка', now='Kubernetes', at=21.0, witness='canon')
    assert not res['accepted'] and 'звук' in res['why'], 'канон не подменяет слово, которого никто не слышал'
    res = await reg.call('propose', was='очередь', now='очередью', at=21.0, witness='canon')
    assert not res['accepted'] and 'формы' in res['why'], 'другая форма слова — только если ухо её сказало'
    got = await reg.call('lookup', word='Кафка')
    assert got['known_spellings'] == ['Kafka'], 'непохожее написание — приманка, а не подсказка'
    assert st.turns[1]['final'] == 'дальше про Кавка и очередь'


async def test_context_words_locate_the_edit_but_the_veto_judges_only_the_change(rich, silence):
    """Первый круг стенда 29.09: 9 верных правок из 24 отказов — «too_long» за слова-адрес вокруг."""
    st, ps, reg = _tools(silence)
    st.turns[1]['final'] = 'дальше про Кафка и очередь из пяти длинных слов'
    res = await reg.call('propose', was='дальше про Кафка и очередь', now='дальше про Kafka и очередь',
                         at=21.0, witness='canon')
    assert res['accepted'], res
    assert st.turns[1]['final'] == 'дальше про Kafka и очередь из пяти длинных слов'
