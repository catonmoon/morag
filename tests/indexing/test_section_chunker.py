"""Тесты для SectionChunker — иерархической упаковки по markdown-заголовкам."""
from __future__ import annotations

import pytest

from morag.indexing.chunker import SectionChunker
from morag.indexing.token_counter import TiktokenCounter

counter = TiktokenCounter()


def _make_chunker(min_tokens: int = 30, max_tokens: int = 100, **kwargs) -> SectionChunker:
    return SectionChunker(
        counter=counter,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Section = атом (если влезает)
# ---------------------------------------------------------------------------

class TestSectionAsAtom:
    @pytest.mark.asyncio
    async def test_small_document_is_single_chunk(self):
        """Документ меньше max_tokens → один чанк."""
        text = "# Заголовок\n\nКороткий параграф.\n\nЕщё один параграф."
        chunker = _make_chunker(min_tokens=10, max_tokens=200)
        chunks = await chunker.chunk(text)
        assert len(chunks) == 1
        assert '# Заголовок' in chunks[0]
        assert 'Короткий параграф' in chunks[0]
        assert 'Ещё один параграф' in chunks[0]

    @pytest.mark.asyncio
    async def test_h1_with_three_h2_all_fit(self):
        """H1 с тремя H2, суммарно вмещается в max → один чанк."""
        text = (
            "# Глава 1\n\n"
            "Вводный текст главы.\n\n"
            "## 1.1 Раздел A\n\nА-контент.\n\n"
            "## 1.2 Раздел B\n\nБ-контент.\n\n"
            "## 1.3 Раздел C\n\nС-контент."
        )
        chunker = _make_chunker(min_tokens=10, max_tokens=500)
        chunks = await chunker.chunk(text)
        assert len(chunks) == 1
        for fragment in ['# Глава 1', '## 1.1', '## 1.2', '## 1.3', 'А-контент', 'С-контент']:
            assert fragment in chunks[0]

    @pytest.mark.asyncio
    async def test_order_preserved(self):
        """Порядок текста в чанке соответствует порядку в документе."""
        text = "# H1\n\nA\n\nB\n\n## H2\n\nC\n\nD"
        chunker = _make_chunker(min_tokens=5, max_tokens=200)
        chunks = await chunker.chunk(text)
        assert len(chunks) == 1
        idx_a = chunks[0].index('A')
        idx_b = chunks[0].index('B')
        idx_h2 = chunks[0].index('## H2')
        idx_c = chunks[0].index('C')
        idx_d = chunks[0].index('D')
        assert idx_a < idx_b < idx_h2 < idx_c < idx_d


# ---------------------------------------------------------------------------
# Декомпозиция при превышении max_tokens
# ---------------------------------------------------------------------------

class TestHierarchicalDecomposition:
    @pytest.mark.asyncio
    async def test_h1_too_big_splits_into_h2(self):
        """H1 не вмещается, но каждый H2 по отдельности — да: разбиваем по H2."""
        long_content = ("Это длинный параграф с несколькими словами. " * 20).strip()
        text = (
            f"# Глава\n\n## Раздел A\n\n{long_content}\n\n## Раздел B\n\n{long_content}"
        )
        # max=400: каждый H2 с long влезает отдельно, но вместе — нет
        chunker = _make_chunker(min_tokens=10, max_tokens=400)
        chunks = await chunker.chunk(text)
        assert len(chunks) >= 2
        a_chunks = [c for c in chunks if 'Раздел A' in c]
        b_chunks = [c for c in chunks if 'Раздел B' in c]
        assert a_chunks, f'Раздел A не найден: {chunks}'
        assert b_chunks, f'Раздел B не найден: {chunks}'
        assert a_chunks[0] != b_chunks[0]

    @pytest.mark.asyncio
    async def test_three_level_descent(self):
        """H1 → H2 → H3 каскад при последовательном превышении max."""
        long = ("Абзац с некоторым содержанием. " * 20).strip()
        text = (
            f"# H1\n\n## H2a\n\n### H3a1\n\n{long}\n\n### H3a2\n\n{long}\n\n"
            f"## H2b\n\n{long}"
        )
        chunker = _make_chunker(min_tokens=10, max_tokens=150)
        chunks = await chunker.chunk(text)
        # Должны увидеть разбиение хотя бы на 3 чанка: H3a1, H3a2, H2b
        assert len(chunks) >= 3


# ---------------------------------------------------------------------------
# Преамбула H-уровня приклеивается к первому дочернему подразделу
# ---------------------------------------------------------------------------

class TestPreambleAttachment:
    @pytest.mark.asyncio
    async def test_preamble_attached_to_first_child(self):
        """Преамбула главы и её заголовок приклеиваются к первому H2-подразделу,
        если (# Глава + преамбула + первый H2-раздел) вмещаются в max_tokens.
        """
        # Каждый раздел ~90 токенов контента — оба вместе не влезут в 400,
        # но первый с преамбулой главы — вписывается.
        long = ("Наполнитель контента для раздела. " * 20).strip()
        text = (
            "# Глава\n\n"
            "Эта глава посвящена теме X. Она состоит из двух разделов.\n\n"
            f"## Раздел A\n\n{long}\n\n"
            f"## Раздел B\n\n{long}"
        )
        chunker = _make_chunker(min_tokens=10, max_tokens=400)
        chunks = await chunker.chunk(text)
        # Преамбула главы + её заголовок должны быть в первом чанке (вместе с Разделом A)
        first = chunks[0]
        assert '# Глава' in first
        assert 'Эта глава посвящена' in first
        assert '## Раздел A' in first
        # И НЕ должны появиться во втором чанке
        second = chunks[1]
        assert 'Эта глава посвящена' not in second
        assert '## Раздел B' in second
        # # Глава тоже не должна быть во втором чанке (она прилипла к первому)
        assert '# Глава' not in second


# ---------------------------------------------------------------------------
# Fallback на greedy при отсутствии дочерних заголовков
# ---------------------------------------------------------------------------

class TestGreedyFallback:
    @pytest.mark.asyncio
    async def test_single_section_with_many_paragraphs_splits(self):
        """H1 без подзаголовков, но с кучей параграфов → greedy разбиение."""
        paragraphs = [f"Параграф номер {i} — содержит немного текста для объёма." for i in range(20)]
        text = "# Заголовок\n\n" + "\n\n".join(paragraphs)
        chunker = _make_chunker(min_tokens=20, max_tokens=80)
        chunks = await chunker.chunk(text)
        assert len(chunks) >= 2
        # Первые чанки должны содержать контент
        joined = '\n\n'.join(chunks)
        for i in range(20):
            assert f"Параграф номер {i}" in joined


# ---------------------------------------------------------------------------
# Порядок блоков в корневом (безголовом) документе
# ---------------------------------------------------------------------------

class TestRootLevelDocument:
    @pytest.mark.asyncio
    async def test_no_headings_at_all(self):
        """Документ без заголовков — всё в одной секции уровня 0."""
        text = "Первый параграф.\n\nВторой параграф.\n\nТретий параграф."
        chunker = _make_chunker(min_tokens=5, max_tokens=200)
        chunks = await chunker.chunk(text)
        assert len(chunks) == 1
        for t in ['Первый', 'Второй', 'Третий']:
            assert t in chunks[0]


# ---------------------------------------------------------------------------
# Chunks имеют ChunkResult-метаданные
# ---------------------------------------------------------------------------

class TestMetadata:
    @pytest.mark.asyncio
    async def test_chunk_with_metadata_returns_chunk_results(self):
        text = "# Заголовок\n\nСодержимое раздела."
        chunker = _make_chunker(min_tokens=5, max_tokens=200)
        results = await chunker.chunk_with_metadata(text)
        assert len(results) == 1
        assert results[0].text
        assert hasattr(results[0], 'pages')
        assert hasattr(results[0], 'char_offset')

    @pytest.mark.asyncio
    async def test_char_offset_first_chunk_is_zero(self):
        """char_offset первого чанка соответствует началу первого блока."""
        text = "# Глава\n\nПараграф один.\n\n## Раздел\n\nПараграф два."
        chunker = _make_chunker(min_tokens=5, max_tokens=200)
        results = await chunker.chunk_with_metadata(text)
        assert len(results) == 1
        # Первый блок документа — '# Глава', позиция 0
        assert results[0].char_offset == 0
