"""Автозагрузка ссылок из вопроса: бюджет — от свободного места в окне, документ режется, а не
выбрасывается.

Замерено 15.09 на корпусе расшифровок: 60 % окна вслепую + история диалога + запись в 78k
токенов → промпт за окном, vLLM 400 «maximum context length», агент отвечал «связь с LLM
сорвалась» на КАЖДОМ втором вопросе разговора о записи. Здесь: (а) бюджет уменьшается на
занятое историей, (б) не влезающий первый документ загружается по чанкам до бюджета и
помечается как частичный, а не пропадает.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services' / 'pipeline'))
from morag_pipeline import Pipeline  # noqa: E402


def _pipeline(window: int, doc_chunks: list[dict]):
    p = object.__new__(Pipeline)
    p._s = {'agent_context_window': window, 'agent_max_tokens': 4096}
    p._count_tokens = lambda text: len(text.split())          # токен = слово, для наглядности
    p._resolve_ref = lambda ref: ref if ref.startswith('local:demo:') else None
    p._fetch_page_live = lambda *a, **k: None
    p._live_titles = {}
    p._run = lambda coro: coro
    p._doc_repo = type('R', (), {'get_by_id': staticmethod(lambda doc_id: object())})()
    p._searcher = type('S', (), {
        'fetch_doc_chunks_lite': staticmethod(lambda doc_id: [{'order': c['order']} for c in doc_chunks]),
        'fetch_chunks_by_orders': staticmethod(lambda doc_id, orders: list(doc_chunks)),
    })()
    p._get_doc_title = lambda doc_id: 'Запись'
    p._render_chunks_block = lambda chunks, header=None: '\n'.join(c['text'] for c in chunks)
    return p


def _chunks(n: int, words: int) -> list[dict]:
    return [{'doc_id': 'local:demo:a.md', 'order': i, 'chunk_id': f'c{i}', 'text': ' '.join(['w'] * words)}
            for i in range(n)]


def test_budget_shrinks_with_history():
    p = _pipeline(window=20000, doc_chunks=[])
    empty = p._auto_fetch_budget([{'role': 'system', 'content': 'x'}])
    with_history = p._auto_fetch_budget([
        {'role': 'system', 'content': 'x'},
        {'role': 'user', 'content': ' '.join(['h'] * 5000)},
        {'role': 'assistant', 'content': ' '.join(['h'] * 5000)},
    ])
    assert empty == 12000, 'на пустой истории — потолок 60 % окна (свободного места больше)'
    assert with_history == 20000 - 10001 - 4096 - 3000, 'история обязана уменьшать бюджет автозагрузки'
    assert with_history < empty
    assert p._auto_fetch_budget([{'role': 'user', 'content': ' '.join(['h'] * 30000)}]) == 0


def test_oversized_first_document_is_truncated_not_dropped():
    chunks = _chunks(10, 100)  # 1000 слов, бюджет 350 → влезут 3 чанка
    p = _pipeline(window=20000, doc_chunks=chunks)
    out = p._fetch_refs(['local:demo:a.md'], budget=350)
    assert [c['order'] for c in out['chunks']] == [0, 1, 2], 'режем по порядку, с начала'
    assert out['partial'] == ['Запись'] and out['not_fit'] == []
    assert out['loaded'] and 'не целиком' in out['loaded'][0]


def test_second_document_that_does_not_fit_goes_to_not_fit():
    chunks = _chunks(2, 100)
    p = _pipeline(window=20000, doc_chunks=chunks)
    out = p._fetch_refs(['local:demo:a.md', 'local:demo:b.md'], budget=250)
    assert len(out['chunks']) == 2 and out['partial'] == []
    assert out['not_fit'] == ['Запись'], 'второй документ, не влезший после первого, — в not_fit как раньше'


def test_document_within_budget_loads_whole():
    chunks = _chunks(3, 50)
    p = _pipeline(window=20000, doc_chunks=chunks)
    out = p._fetch_refs(['local:demo:a.md'], budget=1000)
    assert len(out['chunks']) == 3 and out['partial'] == [] and out['loaded'] == ['Запись']
