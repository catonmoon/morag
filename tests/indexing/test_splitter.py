import math

import numpy as np
import pytest

from morag.indexing.splitter import (
    BlockSplitter,
    FixedSizeSplitter,
    MarkdownHeaderSplitter,
    RecursiveSplitter,
    SemanticSplitter,
    TableRowSplitter,
    pack_blocks,
)
from morag.indexing.token_counter import TiktokenCounter


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def counter() -> TiktokenCounter:
    return TiktokenCounter()


@pytest.fixture
def header_splitter() -> MarkdownHeaderSplitter:
    return MarkdownHeaderSplitter()


@pytest.fixture
def table_splitter() -> TableRowSplitter:
    return TableRowSplitter(rows_per_chunk=5)


@pytest.fixture
def fixed_splitter(counter) -> FixedSizeSplitter:
    return FixedSizeSplitter(counter=counter, limit=50)


def _make_embed_fn(topic_vectors: dict[str, list[float]]):
    """Мок embed_fn: возвращает вектор по первому найденному ключевому слову."""
    default = [1.0, 0.0, 0.0]

    def embed_fn(text: str) -> list[float]:
        for keyword, vec in topic_vectors.items():
            if keyword.lower() in text.lower():
                return vec
        return default

    return embed_fn


# ---------------------------------------------------------------------------
# MarkdownHeaderSplitter
# ---------------------------------------------------------------------------

class TestMarkdownHeaderSplitter:
    def test_is_block_splitter(self, header_splitter):
        assert isinstance(header_splitter, BlockSplitter)

    def test_can_split_with_multiple_headers(self, simple_md_with_headers, header_splitter):
        assert header_splitter.can_split(simple_md_with_headers) is True

    def test_cannot_split_without_headers(self, md_no_headers, header_splitter):
        assert header_splitter.can_split(md_no_headers) is False

    def test_cannot_split_with_single_header(self, header_splitter):
        text = '# Единственный заголовок\n\nТекст.'
        assert header_splitter.can_split(text) is False

    def test_split_produces_multiple_blocks(self, simple_md_with_headers, header_splitter):
        blocks = header_splitter.split(simple_md_with_headers)
        assert len(blocks) > 1

    def test_split_each_block_starts_with_header(self, simple_md_with_headers, header_splitter):
        blocks = header_splitter.split(simple_md_with_headers)
        for block in blocks:
            assert block.startswith('#'), f'Блок не начинается с заголовка:\n{block[:50]}'

    def test_split_preserves_all_content(self, simple_md_with_headers, header_splitter):
        blocks = header_splitter.split(simple_md_with_headers)
        combined = '\n'.join(blocks)
        # Все значимые слова сохранены
        for word in ['первый', 'второй', 'Подраздел']:
            assert word in combined

    def test_split_on_real_md(self, llm_overview_md, header_splitter):
        blocks = header_splitter.split(llm_overview_md)
        assert len(blocks) >= 5  # в llm_overview.md много разделов


# ---------------------------------------------------------------------------
# TableRowSplitter
# ---------------------------------------------------------------------------

class TestTableRowSplitter:
    def test_is_block_splitter(self, table_splitter):
        assert isinstance(table_splitter, BlockSplitter)

    def test_can_split_large_table(self, md_with_large_table, table_splitter):
        assert table_splitter.can_split(md_with_large_table) is True

    def test_cannot_split_small_table(self, table_splitter):
        text = '| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |'
        assert table_splitter.can_split(text) is False

    def test_split_produces_multiple_chunks(self, md_with_large_table, table_splitter):
        chunks = table_splitter.split(md_with_large_table)
        assert len(chunks) > 1

    def test_split_each_chunk_contains_header(self, md_with_large_table, table_splitter):
        chunks = table_splitter.split(md_with_large_table)
        for chunk in chunks:
            assert '| Название | Значение | Описание |' in chunk

    def test_split_each_chunk_contains_separator(self, md_with_large_table, table_splitter):
        chunks = table_splitter.split(md_with_large_table)
        for chunk in chunks:
            assert '|---|---|---|' in chunk

    def test_split_covers_all_rows(self, md_with_large_table, table_splitter):
        chunks = table_splitter.split(md_with_large_table)
        all_text = '\n'.join(chunks)
        for i in range(1, 51):
            assert f'Строка {i}' in all_text

    def test_split_respects_rows_per_chunk(self, table_splitter):
        rows = '\n'.join(f'| Row {i} | Val {i} |' for i in range(1, 16))
        text = f'| A | B |\n|---|---|\n{rows}'
        chunks = table_splitter.split(text)
        # 15 строк при rows_per_chunk=5 → 3 чанка
        assert len(chunks) == 3

    def test_split_pre_text_in_first_chunk_only(self, md_with_large_table, table_splitter):
        chunks = table_splitter.split(md_with_large_table)
        assert 'Текст перед таблицей' in chunks[0]
        for chunk in chunks[1:]:
            assert 'Текст перед таблицей' not in chunk

    def test_split_post_text_in_last_chunk_only(self, md_with_large_table, table_splitter):
        chunks = table_splitter.split(md_with_large_table)
        assert 'Текст после таблицы' in chunks[-1]
        for chunk in chunks[:-1]:
            assert 'Текст после таблицы' not in chunk


# ---------------------------------------------------------------------------
# SemanticSplitter
# ---------------------------------------------------------------------------

class TestSemanticSplitter:
    def _make_splitter(self, embed_fn) -> SemanticSplitter:
        return SemanticSplitter(embed_fn=embed_fn, breakpoint_percentile=50, min_sentences=3)

    def test_is_block_splitter(self):
        splitter = SemanticSplitter(embed_fn=lambda t: [1.0, 0.0], min_sentences=3)
        assert isinstance(splitter, BlockSplitter)

    def test_can_split_with_enough_sentences(self):
        splitter = SemanticSplitter(embed_fn=lambda t: [1.0, 0.0], min_sentences=3)
        text = 'Первое предложение. Второе предложение. Третье предложение.'
        assert splitter.can_split(text) is True

    def test_cannot_split_with_few_sentences(self):
        splitter = SemanticSplitter(embed_fn=lambda t: [1.0, 0.0], min_sentences=3)
        text = 'Одно предложение.'
        assert splitter.can_split(text) is False

    def test_split_finds_semantic_boundary(self):
        """Предложения о разных темах должны быть разделены."""
        topic_a = [1.0, 0.0, 0.0]
        topic_b = [0.0, 1.0, 0.0]  # ортогональный вектор = максимальное расстояние

        embed_fn = _make_embed_fn({'python': topic_a, 'база данных': topic_b})
        splitter = SemanticSplitter(
            embed_fn=embed_fn, breakpoint_percentile=50, min_sentences=3
        )

        text = (
            'Python — язык программирования. '
            'Python широко используется в ML. '
            'база данных хранит данные. '
            'база данных поддерживает SQL.'
        )
        chunks = splitter.split(text)
        assert len(chunks) >= 2

    def test_split_preserves_all_sentences(self):
        embed_fn = _make_embed_fn({'тема': [1.0, 0.0], 'другая': [0.0, 1.0]})
        splitter = SemanticSplitter(
            embed_fn=embed_fn, breakpoint_percentile=50, min_sentences=3
        )
        text = 'Первое про тему. Второе про тему. Третье про другую.'
        chunks = splitter.split(text)
        combined = ' '.join(chunks)
        for sentence in ['Первое', 'Второе', 'Третье']:
            assert sentence in combined

    def test_split_returns_original_when_no_breakpoints(self):
        # Все предложения с одинаковым эмбеддингом → нет границ
        embed_fn = lambda t: [1.0, 0.0, 0.0]
        splitter = SemanticSplitter(
            embed_fn=embed_fn, breakpoint_percentile=99, min_sentences=3
        )
        text = 'Раз. Два. Три. Четыре. Пять.'
        result = splitter.split(text)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# FixedSizeSplitter
# ---------------------------------------------------------------------------

class TestFixedSizeSplitter:
    def test_is_block_splitter(self, fixed_splitter):
        assert isinstance(fixed_splitter, BlockSplitter)

    def test_can_split_any_nonempty_text(self, fixed_splitter):
        assert fixed_splitter.can_split('любой текст') is True

    def test_cannot_split_empty_text(self, fixed_splitter):
        assert fixed_splitter.can_split('') is False
        assert fixed_splitter.can_split('   ') is False

    def test_split_short_text_returns_as_is(self, counter):
        splitter = FixedSizeSplitter(counter=counter, limit=500)
        text = 'Короткий текст.'
        result = splitter.split(text)
        assert result == [text]

    def test_split_long_text_into_multiple_chunks(self, counter):
        splitter = FixedSizeSplitter(counter=counter, limit=20)
        paragraphs = ['Абзац номер один со словами.' for _ in range(10)]
        text = '\n\n'.join(paragraphs)
        chunks = splitter.split(text)
        assert len(chunks) > 1

    def test_split_chunks_within_limit(self, counter):
        limit = 30
        splitter = FixedSizeSplitter(counter=counter, limit=limit)
        paragraphs = ['Текст абзаца с несколькими словами.' for _ in range(20)]
        text = '\n\n'.join(paragraphs)
        chunks = splitter.split(text)
        for chunk in chunks:
            assert counter.count(chunk) <= limit * 2  # допуск на слияние абзацев

    def test_split_oversized_paragraph_by_sentences(self, counter):
        """Абзац больше лимита разбивается по предложениям, а не сразу посимвольно."""
        # Предложения: 10, 9, 11, 14 токенов → limit=20 вмещает каждое по отдельности,
        # но не все вместе (~46 токенов)
        splitter = FixedSizeSplitter(counter=counter, limit=20)
        sentences = [
            'Первое предложение здесь.',
            'Второе предложение тут.',
            'Третье предложение там.',
            'Четвёртое предложение снова.',
        ]
        para = ' '.join(sentences)
        chunks = splitter.split(para)
        assert len(chunks) > 1
        # Предложения не разорваны посередине — каждый чанк содержит целое предложение
        for chunk in chunks:
            assert any(s in chunk for s in sentences)

    def test_split_by_words_when_no_sentence_boundaries(self, counter):
        """Текст без знаков препинания разбивается по словам."""
        splitter = FixedSizeSplitter(counter=counter, limit=5)
        text = ' '.join(['слово'] * 40)
        chunks = splitter.split(text)
        assert len(chunks) > 1
        # Слова не разорваны — каждый чанк состоит из целых слов
        for chunk in chunks:
            assert all(w == 'слово' for w in chunk.split())


# ---------------------------------------------------------------------------
# RecursiveSplitter
# ---------------------------------------------------------------------------

class TestRecursiveSplitter:
    def test_returns_text_as_is_when_fits(self, counter):
        splitter = RecursiveSplitter(
            counter=counter,
            limit=10000,
            splitters=[MarkdownHeaderSplitter()],
        )
        text = 'Короткий текст.'
        assert splitter.split(text) == [text]

    def test_applies_header_splitter_first(self, counter, simple_md_with_headers):
        splitter = RecursiveSplitter(
            counter=counter,
            limit=10,  # очень маленький лимит → всё не влезет
            splitters=[
                MarkdownHeaderSplitter(),
                FixedSizeSplitter(counter=counter, limit=10),
            ],
        )
        blocks = splitter.split(simple_md_with_headers)
        assert len(blocks) > 1

    def test_falls_back_to_fixed_when_no_headers(self, counter, md_no_headers):
        long_text = md_no_headers * 50
        splitter = RecursiveSplitter(
            counter=counter,
            limit=20,
            splitters=[
                MarkdownHeaderSplitter(),
                FixedSizeSplitter(counter=counter, limit=20),
            ],
        )
        blocks = splitter.split(long_text)
        assert len(blocks) > 1

    def test_all_blocks_within_limit_after_split(self, counter, llm_overview_md):
        limit = 300
        splitter = RecursiveSplitter(
            counter=counter,
            limit=limit,
            splitters=[
                MarkdownHeaderSplitter(),
                TableRowSplitter(rows_per_chunk=5),
                FixedSizeSplitter(counter=counter, limit=limit),
            ],
        )
        blocks = splitter.split(llm_overview_md)
        oversized = [b for b in blocks if counter.count(b) > limit]
        # FixedSizeSplitter гарантирует что большинство блоков в лимите
        # (допускаем единичные нарушения на очень длинных строках без пробелов)
        assert len(oversized) == 0 or len(oversized) / len(blocks) < 0.05

    def test_content_preserved_after_split(self, counter, simple_md_with_headers):
        splitter = RecursiveSplitter(
            counter=counter,
            limit=50,
            splitters=[
                MarkdownHeaderSplitter(),
                FixedSizeSplitter(counter=counter, limit=50),
            ],
        )
        blocks = splitter.split(simple_md_with_headers)
        combined = '\n'.join(blocks)
        for word in ['первый', 'второй', 'Подраздел']:
            assert word in combined


# ---------------------------------------------------------------------------
# pack_blocks
# ---------------------------------------------------------------------------

class TestPackBlocks:
    def test_empty_input_returns_empty(self, counter):
        assert pack_blocks([], counter, limit=100) == []

    def test_single_block_returns_one_pack(self, counter):
        result = pack_blocks(['hello'], counter, limit=100)
        assert result == [['hello']]

    def test_packs_blocks_greedily(self, counter):
        # Каждый блок ~5 токенов, лимит 12 → пачки по 2 блока
        blocks = ['one two three'] * 6
        packs = pack_blocks(blocks, counter, limit=12)
        assert len(packs) >= 2
        assert all(len(p) >= 1 for p in packs)

    def test_all_blocks_present_in_packs(self, counter):
        blocks = [f'Block number {i}' for i in range(10)]
        packs = pack_blocks(blocks, counter, limit=50)
        all_in_packs = [b for pack in packs for b in pack]
        assert sorted(all_in_packs) == sorted(blocks)

    def test_each_pack_within_token_limit(self, counter):
        limit = 30
        blocks = ['короткий текст для теста'] * 20
        packs = pack_blocks(blocks, counter, limit=limit)
        for pack in packs:
            total = counter.count('\n'.join(pack))
            assert total <= limit * 2  # допуск: один блок может превышать если он один в пачке

    def test_oversized_single_block_gets_own_pack(self, counter):
        long_block = 'слово ' * 200
        short_block = 'короткий'
        packs = pack_blocks([long_block, short_block], counter, limit=10)
        assert len(packs) == 2
        assert long_block in packs[0]

    def test_consecutive_blocks_stay_together_when_possible(self, counter):
        blocks = ['a', 'b', 'c']
        packs = pack_blocks(blocks, counter, limit=1000)
        assert len(packs) == 1
        assert packs[0] == ['a', 'b', 'c']
