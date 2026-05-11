"""Тесты HybridSearcher._swap_narratives_to_parents — замена narrative-чанков
на parent при выдаче поиска. См. ADR-0013.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from morag.retrieval.searcher import HybridSearcher


def _make_chunk(chunk_id: str, score: float, chunk_type: str | None = None,
                parent_chunk_id: str | None = None) -> dict:
    return {
        'chunk_id': chunk_id,
        'doc_id': 'doc1',
        'path': ['Path'],
        'order': 0,
        'total': 1,
        'text': f'text-{chunk_id}',
        'context': '',
        'updated_at': '',
        'creator': '',
        'url': None,
        'source_type': '',
        'score': score,
        'chunk_type': chunk_type,
        'parent_chunk_id': parent_chunk_id,
    }


def _make_searcher_with_parents(parent_chunks: dict[str, dict]) -> HybridSearcher:
    """Сделать минимальный HybridSearcher с замоканным fetch_chunks_by_ids."""
    s = HybridSearcher.__new__(HybridSearcher)
    s.fetch_chunks_by_ids = AsyncMock(return_value=parent_chunks)
    return s


@pytest.mark.asyncio
async def test_swap_narrative_to_parent_basic():
    """1 narrative → 1 parent с score от narrative."""
    parent = _make_chunk('parent-1', score=0.0)
    narrative = _make_chunk(
        'narr-1', score=0.9,
        chunk_type='table_row_narrative', parent_chunk_id='parent-1',
    )
    s = _make_searcher_with_parents({'parent-1': parent})

    result = await s._swap_narratives_to_parents([narrative])

    assert len(result) == 1
    assert result[0]['chunk_id'] == 'parent-1'
    assert result[0]['score'] == 0.9  # строго наследуем narrative.score


@pytest.mark.asyncio
async def test_swap_dedupes_multiple_narratives_same_parent():
    """3 narrative с одним parent → parent один раз, score от первого (с highest score)."""
    parent = _make_chunk('parent-1', score=0.0)
    n1 = _make_chunk('n1', 0.9, 'table_row_narrative', 'parent-1')
    n2 = _make_chunk('n2', 0.8, 'table_row_narrative', 'parent-1')
    n3 = _make_chunk('n3', 0.7, 'table_row_narrative', 'parent-1')
    s = _make_searcher_with_parents({'parent-1': parent})

    result = await s._swap_narratives_to_parents([n1, n2, n3])

    assert len(result) == 1
    assert result[0]['chunk_id'] == 'parent-1'
    assert result[0]['score'] == 0.9  # от n1 (первого по итерации = highest score)


@pytest.mark.asyncio
async def test_swap_keeps_parent_if_already_present_first():
    """Если parent уже в результатах ВЫШЕ narrative по score — narrative drop'нется."""
    parent = _make_chunk('parent-1', score=0.95)  # высокий score — выше narrative
    narrative = _make_chunk('n1', 0.8, 'table_row_narrative', 'parent-1')
    s = _make_searcher_with_parents({})  # fetch не вызовется (parent уже среди результатов)

    result = await s._swap_narratives_to_parents([parent, narrative])

    assert len(result) == 1
    assert result[0]['chunk_id'] == 'parent-1'
    assert result[0]['score'] == 0.95  # сохраняется parent's natural score


@pytest.mark.asyncio
async def test_swap_narrative_first_then_parent_below():
    """narrative выше → swap, потом regular parent тоже встречается — drop его."""
    parent_in_result = _make_chunk('parent-1', score=0.3)
    narrative = _make_chunk('n1', 0.9, 'table_row_narrative', 'parent-1')
    s = _make_searcher_with_parents({})  # parent уже среди regular → not fetched

    result = await s._swap_narratives_to_parents([narrative, parent_in_result])

    # narrative первая в итерации (score 0.9), но т.к. parent есть среди regular_ids,
    # мы НЕ fetch'им его → narrative drop'ается. parent появится со своим score 0.3.
    assert len(result) == 1
    assert result[0]['chunk_id'] == 'parent-1'
    assert result[0]['score'] == 0.3  # parent's own score


@pytest.mark.asyncio
async def test_regular_chunks_passthrough():
    """Обычные чанки (не narrative) проходят без изменений."""
    c1 = _make_chunk('c1', 0.9)
    c2 = _make_chunk('c2', 0.8)
    s = _make_searcher_with_parents({})

    result = await s._swap_narratives_to_parents([c1, c2])

    assert result == [c1, c2]
    s.fetch_chunks_by_ids.assert_called_once_with([])


@pytest.mark.asyncio
async def test_mixed_results():
    """Mix: regular + narrative с разными parent. Дедуп per-parent."""
    parent_a = _make_chunk('parent-a', 0.0)
    parent_b = _make_chunk('parent-b', 0.0)
    c1 = _make_chunk('c1', 0.95)
    n1 = _make_chunk('n1', 0.9, 'table_row_narrative', 'parent-a')
    n2 = _make_chunk('n2', 0.85, 'table_row_narrative', 'parent-a')  # тот же parent
    n3 = _make_chunk('n3', 0.7, 'table_row_narrative', 'parent-b')
    c2 = _make_chunk('c2', 0.5)
    s = _make_searcher_with_parents({'parent-a': parent_a, 'parent-b': parent_b})

    result = await s._swap_narratives_to_parents([c1, n1, n2, n3, c2])

    # Ожидаем: c1 (0.95), parent-a (0.9 от n1), [n2 dropped], parent-b (0.7 от n3), c2 (0.5)
    assert [r['chunk_id'] for r in result] == ['c1', 'parent-a', 'parent-b', 'c2']
    assert [r['score'] for r in result] == [0.95, 0.9, 0.7, 0.5]


@pytest.mark.asyncio
async def test_narrative_with_missing_parent_id():
    """Malformed narrative без parent_chunk_id — silent drop."""
    narrative = _make_chunk('n1', 0.9, 'table_row_narrative', None)
    s = _make_searcher_with_parents({})

    result = await s._swap_narratives_to_parents([narrative])
    assert result == []


@pytest.mark.asyncio
async def test_narrative_with_parent_not_found_in_qdrant():
    """Parent не найден в qdrant (например удалён) — narrative drop'ается."""
    narrative = _make_chunk('n1', 0.9, 'table_row_narrative', 'parent-missing')
    s = _make_searcher_with_parents({})  # empty — parent not in dict

    result = await s._swap_narratives_to_parents([narrative])
    assert result == []  # narrative drop, нечего вернуть


@pytest.mark.asyncio
async def test_empty_input():
    s = _make_searcher_with_parents({})
    result = await s._swap_narratives_to_parents([])
    assert result == []
