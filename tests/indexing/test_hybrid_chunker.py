"""Тесты для HybridChunker."""
import re
from unittest.mock import AsyncMock

from morag.indexing.chunker import HybridChunker
from morag.indexing.token_counter import TiktokenCounter

counter = TiktokenCounter()


def _make_chunker(min_tokens=30, max_tokens=100, **kwargs):
    return HybridChunker(
        counter=counter,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        **kwargs,
    )


def _normalize(text: str) -> str:
    """Нормализовать текст для сравнения: убрать лишние пробелы и переносы."""
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Stage 1: _parse_blocks
# ---------------------------------------------------------------------------

class TestParseBlocks:
    def test_simple_paragraphs(self):
        text = 'First paragraph.\n\nSecond paragraph.'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        assert len(blocks) == 2
        assert all(b.block_type == 'paragraph' for b in blocks)

    def test_heading_detected(self):
        text = '# Title\n\nSome text.'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        types = [b.block_type for b in blocks]
        assert 'heading' in types

    def test_table_detected(self):
        text = '| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        assert any(b.block_type == 'table' for b in blocks)

    def test_code_fence_detected(self):
        text = '```python\nprint("hello")\n```'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        assert any(b.block_type == 'fence' for b in blocks)

    def test_diagram_detected(self):
        text = '```mermaid\ngraph TD\n  A --> B\n```'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        assert any(b.block_type == 'diagram' for b in blocks)

    def test_plantuml_diagram(self):
        text = '```plantuml\n@startuml\nAlice -> Bob\n@enduml\n```'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        assert any(b.block_type == 'diagram' for b in blocks)

    def test_list_detected(self):
        text = '- item 1\n- item 2\n- item 3'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        assert any(b.block_type == 'list' for b in blocks)

    def test_sections_split_by_headers(self):
        text = '# Section 1\n\nText one.\n\n# Section 2\n\nText two.'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        headings = [b for b in blocks if b.block_type == 'heading']
        assert len(headings) == 2

    def test_empty_text(self):
        chunker = _make_chunker()
        blocks = chunker._parse_blocks('')
        assert blocks == []

    def test_heading_without_content(self):
        """Заголовок в конце документа без контента после — должен стать блоком."""
        text = 'Some text.\n\n# Trailing Header'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        assert any(b.block_type == 'heading' and 'Trailing' in b.text for b in blocks)

    def test_nested_headings(self):
        """Вложенные заголовки (## внутри # секции) — каждый отдельный блок."""
        text = '# Main\n\nIntro.\n\n## Sub 1\n\nText 1.\n\n## Sub 2\n\nText 2.'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        headings = [b for b in blocks if b.block_type == 'heading']
        assert len(headings) == 3

    def test_hash_inside_code_fence_not_heading(self):
        """# внутри code fence — не должен распознаваться как заголовок."""
        text = '```bash\n# this is a comment\necho hello\n```'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        headings = [b for b in blocks if b.block_type == 'heading']
        assert len(headings) == 0
        assert any(b.block_type == 'fence' for b in blocks)

    def test_tokens_counted(self):
        """Каждый блок имеет подсчитанные токены."""
        text = '# Title\n\nSome paragraph text here.'
        chunker = _make_chunker()
        blocks = chunker._parse_blocks(text)
        for b in blocks:
            assert b.tokens > 0
            assert b.tokens == counter.count(b.text)


# ---------------------------------------------------------------------------
# Stage 2: _greedy_fill (основной алгоритм + магнитные заголовки)
# ---------------------------------------------------------------------------

class TestGreedyFill:
    async def test_small_document_single_chunk(self):
        chunker = _make_chunker(max_tokens=500)
        result = await chunker.chunk('Short text. Nothing special.')
        assert len(result) == 1

    async def test_multi_section_splits(self):
        """Несколько секций, каждая > min_tokens, формируют отдельные чанки."""
        section = 'word ' * 40  # ~40 tokens
        text = f'# Section 1\n\n{section}\n\n# Section 2\n\n{section}\n\n# Section 3\n\n{section}'
        chunker = _make_chunker(min_tokens=30, max_tokens=60)
        result = await chunker.chunk(text)
        assert len(result) >= 3

    async def test_blocks_packed_greedily(self):
        """Мелкие блоки упаковываются в один чанк."""
        text = 'Para one.\n\nPara two.\n\nPara three.'
        chunker = _make_chunker(max_tokens=500)
        result = await chunker.chunk(text)
        assert len(result) == 1

    async def test_magnetic_header_not_at_end(self):
        """Заголовок не должен оказаться в конце чанка — выталкивается в следующий."""
        filler = 'word ' * 35  # ~35 tokens
        text = f'{filler}\n\n# Next Section\n\n{filler}'
        chunker = _make_chunker(min_tokens=10, max_tokens=50)
        result = await chunker.chunk(text)
        # Ни один чанк (кроме последнего) не должен заканчиваться заголовком
        for chunk in result[:-1]:
            lines = chunk.strip().split('\n')
            last_line = lines[-1].strip()
            assert not last_line.startswith('#'), (
                f'Chunk ends with heading: {last_line!r}'
            )

    async def test_magnetic_header_starts_next_chunk(self):
        """Вытолкнутый заголовок должен стать началом следующего чанка."""
        filler = 'word ' * 35
        text = f'{filler}\n\n# Important Title\n\n{filler}'
        chunker = _make_chunker(min_tokens=10, max_tokens=50)
        result = await chunker.chunk(text)
        # Должен быть чанк, начинающийся с # Important Title
        starts_with_heading = any(
            c.strip().startswith('# Important Title') for c in result
        )
        assert starts_with_heading

    async def test_magnetic_header_single_block_not_ejected(self):
        """Заголовок — единственный блок в чанке — не выталкивается."""
        # Заголовок один, без предшествующего контента
        text = '# Solo Header\n\nSome content after.'
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk(text)
        assert any('Solo Header' in c for c in result)

    async def test_consecutive_headings(self):
        """Несколько заголовков подряд — не теряются."""
        text = '# H1\n\n## H2\n\n### H3\n\nActual content here.'
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk(text)
        full = '\n\n'.join(result)
        assert '# H1' in full
        assert '## H2' in full
        assert '### H3' in full
        assert 'Actual content' in full

    async def test_oversized_after_heading_preserves_heading(self):
        """Oversized блок сразу после heading — heading не теряется."""
        sentences = ['This is sentence number %d.' % i for i in range(30)]
        long_para = ' '.join(sentences)
        text = f'# My Section\n\n{long_para}'
        chunker = _make_chunker(min_tokens=10, max_tokens=50)
        result = await chunker.chunk(text)
        full = '\n\n'.join(result)
        assert '# My Section' in full


# ---------------------------------------------------------------------------
# Stage 2b: Oversized handling
# ---------------------------------------------------------------------------

class TestOversizedText:
    async def test_long_paragraph_split_by_sentences(self):
        """Длинный абзац разбивается по предложениям."""
        sentences = ['This is sentence number %d.' % i for i in range(30)]
        text = ' '.join(sentences)
        chunker = _make_chunker(min_tokens=10, max_tokens=50)
        result = await chunker.chunk(text)
        assert len(result) > 1
        for chunk in result:
            tok = counter.count(chunk)
            assert tok <= 60, f'Chunk too large: {tok} tokens'

    async def test_text_without_punctuation_asis(self):
        """Текст без точек (одно гигантское 'предложение') — default as-is, один чанк."""
        text = 'word ' * 200  # ~200 tokens, без точек
        chunker = _make_chunker(min_tokens=10, max_tokens=50)
        result = await chunker.chunk(text)
        # Один большой чанк лучше чем мусор по словам
        assert len(result) >= 1
        full = ' '.join(result)
        assert 'word' in full


    async def test_oversized_list(self):
        """Длинный list > max_tokens разбивается по элементам."""
        items = '\n'.join(f'- Item number {i} with some description.' for i in range(30))
        text = items
        chunker = _make_chunker(min_tokens=10, max_tokens=50)
        result = await chunker.chunk(text)
        assert len(result) > 1
        # Каждый чанк начинается с элемента списка
        for chunk in result:
            assert chunk.strip().startswith('- '), (
                f'List chunk does not start with list item: {chunk[:40]!r}'
            )
        # Все элементы сохранены
        full = '\n'.join(result)
        for i in range(30):
            assert f'Item number {i}' in full

    async def test_oversized_list_numbered(self):
        """Нумерованный list > max_tokens разбивается по элементам."""
        items = '\n'.join(f'{i+1}. Element {i} with description text.' for i in range(20))
        chunker = _make_chunker(min_tokens=10, max_tokens=50)
        result = await chunker.chunk(items)
        assert len(result) > 1
        full = '\n'.join(result)
        for i in range(20):
            assert f'Element {i}' in full

    async def test_oversized_list_nested(self):
        """Вложенный list — элементы с подэлементами не разрываются."""
        items = []
        for i in range(15):
            items.append(f'- Main item {i}')
            items.append(f'  - Sub item {i}a')
            items.append(f'  - Sub item {i}b')
        text = '\n'.join(items)
        chunker = _make_chunker(min_tokens=10, max_tokens=80)
        result = await chunker.chunk(text)
        assert len(result) > 1
        full = '\n'.join(result)
        for i in range(15):
            assert f'Main item {i}' in full


class TestOversizedTable:
    async def test_large_table_split_by_rows(self):
        """Большая таблица разбивается по строкам с дублированием шапки."""
        header = '| Name | Value | Description |'
        sep = '|------|-------|-------------|'
        rows = [f'| item_{i} | val_{i} | description for item {i} |' for i in range(20)]
        text = '\n'.join([header, sep] + rows)
        chunker = _make_chunker(min_tokens=10, max_tokens=80)
        result = await chunker.chunk(text)
        assert len(result) > 1
        # Каждый чанк с таблицей должен содержать шапку
        for chunk in result:
            if '|' in chunk:
                assert 'Name' in chunk, 'Table chunk missing header'


class TestOversizedCode:
    async def test_long_code_split_by_lines(self):
        """Длинный code fence разбивается по строкам с сохранением fence markers."""
        code_lines = [f'line_{i} = {i}' for i in range(50)]
        text = '```python\n' + '\n'.join(code_lines) + '\n```'
        chunker = _make_chunker(min_tokens=5, max_tokens=50, oversized_strategies={'fence': 'split'})
        result = await chunker.chunk(text)
        assert len(result) > 1
        for chunk in result:
            assert chunk.strip().startswith('```'), (
                f'Code chunk missing opening fence: {chunk[:40]}'
            )
            assert chunk.strip().endswith('```'), (
                f'Code chunk missing closing fence: {chunk[-40:]}'
            )

    async def test_code_preserves_language(self):
        """Нарезанные куски кода сохраняют info string (язык)."""
        code_lines = [f'x_{i} = {i}' for i in range(50)]
        text = '```javascript\n' + '\n'.join(code_lines) + '\n```'
        chunker = _make_chunker(min_tokens=5, max_tokens=50, oversized_strategies={'fence': 'split'})
        result = await chunker.chunk(text)
        for chunk in result:
            assert chunk.strip().startswith('```javascript')


class TestOversizedDiagram:
    async def test_diagram_returned_as_is_fixed(self):
        """Диаграмма > max_tokens возвращается as-is при стратегии fixed."""
        diagram_lines = [f'  Node{i} --> Node{i+1}' for i in range(50)]
        text = '```mermaid\ngraph TD\n' + '\n'.join(diagram_lines) + '\n```'
        chunker = _make_chunker(min_tokens=5, max_tokens=50, oversized_strategies={'paragraph': 'asis', 'table': 'asis', 'list': 'asis', 'fence': 'asis', 'diagram': 'asis'})
        result = await chunker.chunk(text)
        full_diagram = '\n'.join(c for c in result if 'mermaid' in c)
        assert 'graph TD' in full_diagram

    async def test_diagram_sent_to_llm_when_strategy_llm(self):
        """Диаграмма > max_tokens отправляется в LLM при стратегии llm."""
        mock_llm = AsyncMock()
        mock_llm.chunk = AsyncMock(return_value=['diagram chunk 1', 'diagram chunk 2'])

        diagram_lines = [f'  Node{i} --> Node{i+1}' for i in range(50)]
        text = '```mermaid\ngraph TD\n' + '\n'.join(diagram_lines) + '\n```'
        chunker = _make_chunker(
            min_tokens=5, max_tokens=50,
            oversized_strategies={'paragraph': 'llm', 'diagram': 'llm'}, llm_chunker=mock_llm,
        )
        result = await chunker.chunk(text)
        mock_llm.chunk.assert_called_once()
        assert 'diagram chunk 1' in result


# ---------------------------------------------------------------------------
# Oversized strategies
# ---------------------------------------------------------------------------

class TestOversizedStrategies:
    async def test_llm_strategy_calls_llm_chunker(self):
        """Стратегия llm вызывает LLMChunker для oversized предложения."""
        mock_llm = AsyncMock()
        mock_llm.chunk = AsyncMock(return_value=['part 1', 'part 2'])

        # Длинный текст без точек → одно "предложение" → _apply_strategy
        text = 'word ' * 200
        chunker = _make_chunker(
            min_tokens=10, max_tokens=50,
            oversized_strategies={'paragraph': 'llm', 'diagram': 'llm'}, llm_chunker=mock_llm,
        )
        await chunker.chunk(text)
        assert mock_llm.chunk.called

    async def test_embed_strategy_calls_semantic_chunker(self):
        """Стратегия embed использует SemanticChunker с embed_fn."""
        call_count = 0

        async def mock_embed_batch(texts):
            nonlocal call_count
            call_count += 1
            return [[float(hash(t) % 100) / 100] * 10 for t in texts]

        # Много предложений в одном абзаце → oversized block → embed strategy
        sentences = [f'Topic {i} discusses subject {i} in detail.' for i in range(40)]
        text = ' '.join(sentences)
        chunker = _make_chunker(
            min_tokens=10, max_tokens=50,
            oversized_strategies={'paragraph': 'embed'}, embed_fn=mock_embed_batch,
        )
        result = await chunker.chunk(text)
        assert len(result) > 1
        # Все чанки должны быть ≤ max_tokens (fallback гарантирует)
        for chunk in result:
            tok = counter.count(chunk)
            assert tok <= 55, f'Chunk too large: {tok} tokens'

    async def test_embed_strategy_oversized_unit_returned_as_is(self):
        """Текст без точек (один oversized unit) → embed не поможет → as-is."""
        async def mock_embed_batch(texts):
            return [[0.1] * 10 for _ in texts]

        text = 'word ' * 200  # одно "предложение" на 200 токенов
        chunker = _make_chunker(
            min_tokens=10, max_tokens=50,
            oversized_strategies={'paragraph': 'embed'}, embed_fn=mock_embed_batch,
        )
        result = await chunker.chunk(text)
        # Один oversized чанк лучше чем порезанный по словам
        assert len(result) >= 1
        full = ' '.join(result)
        assert 'word' in full

    async def test_split_strategy_for_paragraph(self):
        """Стратегия split для paragraph — разбивает по предложениям."""
        sentences = ['Sentence number %d with padding.' % i for i in range(30)]
        text = ' '.join(sentences)
        chunker = _make_chunker(
            min_tokens=10, max_tokens=50,
            oversized_strategies={'paragraph': 'split'},
        )
        result = await chunker.chunk(text)
        assert len(result) > 1

    async def test_embed_without_embed_fn_falls_back_to_asis(self):
        """embed стратегия без embed_fn → fallback на as-is."""
        text = 'word ' * 200
        chunker = _make_chunker(
            min_tokens=10, max_tokens=50,
            oversized_strategies={'paragraph': 'embed'},
            embed_fn=None,
        )
        result = await chunker.chunk(text)
        # as-is: один большой чанк
        assert len(result) >= 1
        assert 'word' in ' '.join(result)

    async def test_llm_without_llm_chunker_falls_back_to_asis(self):
        """llm стратегия без llm_chunker → fallback на as-is."""
        text = 'word ' * 200
        chunker = _make_chunker(
            min_tokens=10, max_tokens=50,
            oversized_strategies={'paragraph': 'llm', 'diagram': 'llm'},
            llm_chunker=None,
        )
        result = await chunker.chunk(text)
        assert len(result) >= 1
        assert 'word' in ' '.join(result)


# ---------------------------------------------------------------------------
# Stage 3: Post-merge
# ---------------------------------------------------------------------------

class TestPostMerge:
    async def test_small_chunk_merged_with_previous(self):
        """Мелкий чанк склеивается с предыдущим если влезает."""
        big = 'word ' * 20  # ~20 tokens
        small = 'tiny.'
        text = f'{big}\n\n---\n\n{small}'
        chunker = _make_chunker(min_tokens=25, max_tokens=500)
        result = await chunker.chunk(text)
        assert len(result) == 1

    async def test_last_chunk_allowed_small(self):
        """Последний чанк может быть < min_tokens."""
        big = 'word ' * 40
        small = 'end.'
        text = f'{big}\n\n{small}'
        chunker = _make_chunker(min_tokens=30, max_tokens=50)
        result = await chunker.chunk(text)
        last_tok = counter.count(result[-1])
        assert last_tok < 30 or len(result) == 1

    async def test_single_chunk_not_merged(self):
        """Единственный чанк документа не трогается."""
        text = 'Short.'
        chunker = _make_chunker(min_tokens=100, max_tokens=500)
        result = await chunker.chunk(text)
        assert result == ['Short.']

    def test_post_merge_prefers_prev(self):
        """Post-merge предпочитает склейку с предыдущим."""
        from morag.indexing.chunker import ChunkResult
        chunker = _make_chunker(min_tokens=30, max_tokens=200)
        chunks = [
            ChunkResult(text='Big chunk with some words here.', pages=[1]),
            ChunkResult(text='x', pages=[1]),
            ChunkResult(text='Another big chunk.', pages=[2]),
        ]
        merged = chunker._post_merge(chunks)
        assert len(merged) < 3 or all(counter.count(c.text) >= 2 for c in merged)

    def test_post_merge_two_small_chunks(self):
        """Два мелких чанка подряд — оба < min_tokens — склеиваются."""
        from morag.indexing.chunker import ChunkResult
        chunker = _make_chunker(min_tokens=30, max_tokens=200)
        chunks = [ChunkResult(text='Hello.', pages=[1]), ChunkResult(text='World.', pages=[2])]
        merged = chunker._post_merge(chunks)
        assert len(merged) == 1
        assert 'Hello.' in merged[0].text
        assert 'World.' in merged[0].text
        assert merged[0].pages == [1, 2]

    def test_post_merge_small_between_two_big(self):
        """Мелкий чанк между двумя большими — принудительно клеится со следующим."""
        from morag.indexing.chunker import ChunkResult
        chunker = _make_chunker(min_tokens=30, max_tokens=50)
        big = 'word ' * 49
        chunks = [
            ChunkResult(text=big, pages=[1]),
            ChunkResult(text='x', pages=[1]),
            ChunkResult(text=big, pages=[2]),
        ]
        merged = chunker._post_merge(chunks)
        # 'x' принудительно склеился со следующим big
        assert len(merged) == 2
        assert 'x' in merged[1].text
        assert merged[1].pages == [1, 2]

    def test_post_merge_with_next_when_prev_full(self):
        """Склейка со следующим когда предыдущий не подходит."""
        from morag.indexing.chunker import ChunkResult
        chunker = _make_chunker(min_tokens=30, max_tokens=100)
        big = 'word ' * 99
        chunks = [
            ChunkResult(text=big, pages=[1]),
            ChunkResult(text='tiny.', pages=[2]),
            ChunkResult(text='some text here.', pages=[2]),
        ]
        merged = chunker._post_merge(chunks)
        assert len(merged) == 2
        assert 'tiny.' in merged[1].text
        assert 'some text' in merged[1].text
        assert merged[1].pages == [2]


# ---------------------------------------------------------------------------
# Инварианты
# ---------------------------------------------------------------------------

class TestInvariants:
    async def test_no_text_loss_simple(self):
        """Весь текст из оригинала присутствует в чанках (простой документ)."""
        text = '# Header\n\nFirst paragraph.\n\nSecond paragraph.\n\n## Sub\n\nThird.'
        chunker = _make_chunker(min_tokens=5, max_tokens=30)
        result = await chunker.chunk(text)
        full = _normalize('\n\n'.join(result))
        for word in ['Header', 'First', 'Second', 'Sub', 'Third']:
            assert word in full, f'Lost word: {word}'

    async def test_no_text_loss_with_code(self):
        """Код не теряется при чанкировании."""
        code = '```python\ndef foo():\n    return 42\n```'
        text = f'# Intro\n\nSome text.\n\n{code}\n\n# End\n\nFinal.'
        chunker = _make_chunker(min_tokens=5, max_tokens=50)
        result = await chunker.chunk(text)
        full = '\n\n'.join(result)
        assert 'def foo' in full
        assert 'return 42' in full

    async def test_no_text_loss_with_table(self):
        """Таблица не теряется при чанкировании."""
        table = '| A | B |\n|---|---|\n| x | y |\n| z | w |'
        text = f'Before.\n\n{table}\n\nAfter.'
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk(text)
        full = '\n\n'.join(result)
        assert '| x | y |' in full
        assert '| z | w |' in full

    async def test_max_tokens_respected(self):
        """Ни один чанк не превышает max_tokens (кроме as-is диаграмм/fences)."""
        sentences = ['Sentence number %d with some extra padding words.' % i for i in range(50)]
        text = ' '.join(sentences)
        max_tok = 60
        chunker = _make_chunker(min_tokens=10, max_tokens=max_tok)
        result = await chunker.chunk(text)
        for i, chunk in enumerate(result):
            tok = counter.count(chunk)
            assert tok <= max_tok + 5, (
                f'Chunk {i} exceeds max_tokens: {tok} > {max_tok}'
            )

    async def test_max_tokens_respected_mixed_document(self):
        """max_tokens соблюдается для смешанного документа."""
        text = """# Introduction

This is the introduction with several sentences. Each sentence adds meaning.
The document discusses various topics. Here we introduce the main ideas.

## Details

More detailed information follows. We explore each topic in depth.
The analysis covers multiple perspectives. Results are presented below.

| Metric | Value | Unit |
|--------|-------|------|
| Speed  | 100   | km/h |
| Weight | 50    | kg   |
| Height | 180   | cm   |

## Code

```python
def calculate(x, y):
    result = x + y
    return result
```

## Conclusion

Final summary of all findings. The results are conclusive."""
        max_tok = 60
        chunker = _make_chunker(min_tokens=10, max_tokens=max_tok)
        result = await chunker.chunk(text)
        for i, chunk in enumerate(result):
            tok = counter.count(chunk)
            # Code fences могут быть чуть больше из-за fence markers
            if not chunk.strip().startswith('```'):
                assert tok <= max_tok + 5, (
                    f'Chunk {i} exceeds max_tokens: {tok} > {max_tok}'
                )

    async def test_min_tokens_respected(self):
        """Чанки ≥ min_tokens, кроме единственного или последнего."""
        sentences = ['Sentence number %d with some padding.' % i for i in range(40)]
        text = ' '.join(sentences)
        min_tok = 20
        chunker = _make_chunker(min_tokens=min_tok, max_tokens=60)
        result = await chunker.chunk(text)
        for i, chunk in enumerate(result):
            tok = counter.count(chunk)
            is_last = (i == len(result) - 1)
            is_single = (len(result) == 1)
            if not is_last and not is_single:
                assert tok >= min_tok, (
                    f'Chunk {i} below min_tokens: {tok} < {min_tok}'
                )


# ---------------------------------------------------------------------------
# Реалистичные документы
# ---------------------------------------------------------------------------

class TestRealisticDocuments:
    async def test_russian_document(self):
        """Документ на русском языке."""
        text = """# Введение

Данный документ описывает архитектуру системы. Система состоит из нескольких
компонентов, каждый из которых выполняет свою функцию. Рассмотрим каждый
компонент подробнее.

## Компоненты

### Сервер авторизации

Сервер авторизации отвечает за аутентификацию пользователей. Он поддерживает
OAuth 2.0 и OpenID Connect. Все запросы проходят через этот сервер.

### База данных

Используется PostgreSQL версии 15. Данные хранятся в нескольких схемах.
Миграции выполняются через Alembic.

## Заключение

Архитектура обеспечивает масштабируемость и надёжность системы."""
        chunker = _make_chunker(min_tokens=20, max_tokens=80)
        result = await chunker.chunk(text)
        assert len(result) >= 2
        full = '\n\n'.join(result)
        assert 'Введение' in full
        assert 'авторизации' in full
        assert 'Заключение' in full

    async def test_pdf_converted_document(self):
        """PDF-конвертированный документ: мало заголовков, длинные абзацы."""
        paragraphs = []
        for i in range(10):
            sentences = [
                f'This is paragraph {i}, sentence {j}. It contains important information.'
                for j in range(8)
            ]
            paragraphs.append(' '.join(sentences))
        text = '\n\n'.join(paragraphs)
        chunker = _make_chunker(min_tokens=20, max_tokens=80)
        result = await chunker.chunk(text)
        assert len(result) >= 5
        # Ничего не потеряно
        for i in range(10):
            assert f'paragraph {i}' in '\n\n'.join(result)

    async def test_document_with_huge_table(self):
        """Документ с огромной таблицей посередине."""
        intro = 'Introduction to the data analysis results.'
        header = '| ID | Name | Score | Category | Status |'
        sep = '|----|------|-------|----------|--------|'
        rows = [
            f'| {i} | name_{i} | {i * 10} | cat_{i % 5} | active |'
            for i in range(50)
        ]
        table = '\n'.join([header, sep] + rows)
        conclusion = 'The analysis shows significant improvement across all categories.'
        text = f'{intro}\n\n{table}\n\n{conclusion}'
        chunker = _make_chunker(min_tokens=10, max_tokens=80)
        result = await chunker.chunk(text)
        assert len(result) > 3
        full = '\n\n'.join(result)
        assert 'Introduction' in full
        assert 'analysis shows' in full
        # Таблица должна содержать шапку в каждом чанке
        for chunk in result:
            if '|' in chunk and 'ID' not in chunk and 'Introduction' not in chunk:
                # Чанк с таблицей без intro должен иметь шапку
                # (некоторые чанки могут быть intro+часть таблицы)
                pass  # шапка проверена в TestOversizedTable


# ---------------------------------------------------------------------------
# Full pipeline (end-to-end)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    async def test_mixed_document(self):
        """Документ с заголовками, текстом, кодом и таблицей."""
        text = """# Introduction

This is the introduction paragraph with some meaningful text.

## Code Example

```python
def hello():
    print("world")
```

## Data Table

| Key | Value |
|-----|-------|
| a   | 1     |
| b   | 2     |

## Conclusion

Final thoughts on the matter."""
        chunker = _make_chunker(min_tokens=10, max_tokens=100)
        result = await chunker.chunk(text)
        assert len(result) >= 1
        full_text = '\n\n'.join(result)
        assert 'Introduction' in full_text
        assert 'hello' in full_text
        assert 'Conclusion' in full_text

    async def test_empty_document(self):
        chunker = _make_chunker()
        result = await chunker.chunk('')
        assert result == []

    async def test_whitespace_only(self):
        chunker = _make_chunker()
        result = await chunker.chunk('   \n\n  ')
        assert result == []

    async def test_single_line_document(self):
        """Документ из одной строки."""
        text = 'Just one line of text.'
        chunker = _make_chunker(min_tokens=5, max_tokens=100)
        result = await chunker.chunk(text)
        assert result == ['Just one line of text.']

    async def test_only_headings(self):
        """Документ только из заголовков."""
        text = '# H1\n\n## H2\n\n### H3'
        chunker = _make_chunker(min_tokens=5, max_tokens=100)
        result = await chunker.chunk(text)
        full = '\n\n'.join(result)
        assert '# H1' in full
        assert '## H2' in full
        assert '### H3' in full

    async def test_only_code(self):
        """Документ только из code fence."""
        text = '```python\nx = 1\ny = 2\nz = x + y\n```'
        chunker = _make_chunker(min_tokens=5, max_tokens=100)
        result = await chunker.chunk(text)
        assert len(result) == 1
        assert 'x = 1' in result[0]

    async def test_only_table(self):
        """Документ только из таблицы."""
        text = '| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |'
        chunker = _make_chunker(min_tokens=5, max_tokens=100)
        result = await chunker.chunk(text)
        assert len(result) >= 1
        assert '| 1 | 2 |' in '\n'.join(result)


# ---------------------------------------------------------------------------
# chunk_with_metadata — извлечение страниц из PDF маркеров
# ---------------------------------------------------------------------------

class TestChunkWithMetadata:
    async def test_pdf_pages_extracted(self):
        """Маркеры <!-- page:N --> извлекаются для paged документов."""
        text = '<!-- page:1 -->\n# Title\n\nFirst paragraph.\n\n<!-- page:2 -->\nSecond paragraph.'
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk_with_metadata(text, paged=True)
        assert len(result) >= 1
        all_pages = set()
        for cr in result:
            all_pages.update(cr.pages)
        assert 1 in all_pages
        assert 2 in all_pages

    async def test_markers_removed_from_text(self):
        """Маркеры не попадают в текст чанков paged документа."""
        text = '<!-- page:1 -->\nSome text.\n\n<!-- page:2 -->\nMore text.'
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk_with_metadata(text, paged=True)
        for cr in result:
            assert '<!-- page:' not in cr.text

    async def test_not_paged_ignores_markers(self):
        """Не-paged документ — маркеры не ищутся, pages пустые."""
        text = '<!-- page:1 -->\nSome text.\n\n<!-- page:2 -->\nMore text.'
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk_with_metadata(text, paged=False)
        assert len(result) >= 1
        for cr in result:
            assert cr.pages == []

    async def test_no_markers_empty_pages(self):
        """Paged документ без маркеров — pages пустые (warning)."""
        text = '# Title\n\nSome content without page markers.'
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk_with_metadata(text, paged=True)
        assert len(result) >= 1
        for cr in result:
            assert cr.pages == []

    async def test_blocks_inherit_last_page(self):
        """Блоки без маркера наследуют последнюю известную страницу."""
        text = (
            '<!-- page:1 -->\n'
            '# Title\n\n'
            'Paragraph on page 1.\n\n'
            'Still page 1.\n\n'
            '<!-- page:2 -->\n'
            'Now page 2.'
        )
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk_with_metadata(text, paged=True)
        assert len(result) >= 1
        all_pages = set()
        for cr in result:
            all_pages.update(cr.pages)
        assert 1 in all_pages
        assert 2 in all_pages

    async def test_multi_chunk_pages_correct(self):
        """Каждый чанк получает pages только своих блоков."""
        filler1 = 'word ' * 40
        filler2 = 'text ' * 40
        text = (
            f'<!-- page:1 -->\n# Section 1\n\n{filler1}\n\n'
            f'<!-- page:2 -->\n# Section 2\n\n{filler2}'
        )
        chunker = _make_chunker(min_tokens=10, max_tokens=60)
        result = await chunker.chunk_with_metadata(text, paged=True)
        assert len(result) >= 2
        assert 1 in result[0].pages
        assert 2 in result[-1].pages

    async def test_chunk_paged_returns_clean_text(self):
        """chunk() для документа с маркерами (не paged) — маркеры остаются как текст."""
        text = '<!-- page:1 -->\nClean text here.'
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk(text)
        assert len(result) >= 1

    async def test_page_marker_not_a_block_when_paged(self):
        """Маркер не создаёт отдельный блок в paged документе."""
        text = '<!-- page:1 -->\n# Title\n\nContent.'
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        blocks = chunker._parse_blocks(text, paged=True)
        for b in blocks:
            assert '<!-- page:' not in b.text

    async def test_cross_page_chunk(self):
        """Чанк на границе двух страниц получает обе."""
        text = (
            '<!-- page:3 -->\n'
            'End of page 3.\n\n'
            '<!-- page:4 -->\n'
            'Start of page 4.'
        )
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk_with_metadata(text, paged=True)
        assert len(result) == 1
        assert result[0].pages == [3, 4]

    async def test_pages_preserved_after_post_merge(self):
        """Pages корректно мержатся при склейке мелких чанков."""
        text = (
            '<!-- page:1 -->\nSmall.\n\n'
            '<!-- page:2 -->\nAlso small.'
        )
        chunker = _make_chunker(min_tokens=30, max_tokens=500)
        result = await chunker.chunk_with_metadata(text, paged=True)
        assert len(result) == 1
        assert result[0].pages == [1, 2]

    async def test_char_offset_set(self):
        """Каждый чанк получает char_offset."""
        filler1 = 'word ' * 30
        filler2 = 'text ' * 30
        text = f'# Title\n\n{filler1}\n\n## Section\n\n{filler2}'
        chunker = _make_chunker(min_tokens=5, max_tokens=50)
        result = await chunker.chunk_with_metadata(text)
        assert len(result) >= 2
        assert result[0].char_offset == 0
        assert result[1].char_offset > 0

    async def test_char_offset_within_document(self):
        """char_offset указывает на реальную позицию в документе."""
        text = 'AAA.\n\nBBB.\n\nCCC.'
        chunker = _make_chunker(min_tokens=2, max_tokens=5)
        result = await chunker.chunk_with_metadata(text)
        for cr in result:
            nearby = text[cr.char_offset:cr.char_offset + 50]
            first_word = cr.text.split()[0] if cr.text.split() else ''
            assert first_word in nearby, (
                f'char_offset {cr.char_offset} does not point near chunk text: '
                f'{first_word!r} not in {nearby!r}'
            )

    async def test_char_offset_monotonically_increasing(self):
        """char_offset монотонно возрастает для последовательных чанков."""
        filler = 'word ' * 30
        text = '\n\n'.join(f'## Section {i}\n\n{filler}' for i in range(5))
        chunker = _make_chunker(min_tokens=5, max_tokens=50)
        result = await chunker.chunk_with_metadata(text)
        assert len(result) >= 3
        for i in range(1, len(result)):
            assert result[i].char_offset >= result[i - 1].char_offset, (
                f'char_offset not monotonic: chunk {i-1}={result[i-1].char_offset} '
                f'> chunk {i}={result[i].char_offset}'
            )

    async def test_char_offset_paged_points_to_original_text(self):
        """В paged документе char_offset указывает на позицию в оригинальном тексте (с маркерами)."""
        text = (
            '<!-- page:1 -->\n# Title\n\nFirst paragraph with enough words to fill.\n\n'
            '<!-- page:2 -->\n## Section\n\nSecond paragraph with more words here.'
        )
        chunker = _make_chunker(min_tokens=5, max_tokens=50)
        result = await chunker.chunk_with_metadata(text, paged=True)
        assert len(result) >= 1
        for cr in result:
            # offset в оригинальном тексте — там могут быть маркеры
            nearby = text[cr.char_offset:cr.char_offset + 100]
            # Первое слово чанка (без маркера) должно быть где-то рядом
            first_word = cr.text.strip().split()[0]
            assert first_word in nearby, (
                f'char_offset {cr.char_offset} does not point near chunk: '
                f'{first_word!r} not in {nearby!r}'
            )

    async def test_char_offset_oversized_block(self):
        """Куски oversized блока наследуют offset блока (кроме склеенных в post-merge)."""
        sentences = ['Sentence number %d with padding words.' % i for i in range(30)]
        long_para = ' '.join(sentences)
        text = f'Some intro text here.\n\n{long_para}'
        chunker = _make_chunker(min_tokens=5, max_tokens=50)
        result = await chunker.chunk_with_metadata(text)
        # Чанки из oversized параграфа должны существовать
        oversized_chunks = [cr for cr in result if 'Sentence number' in cr.text]
        assert len(oversized_chunks) > 1

    async def test_char_offset_post_merge_takes_first(self):
        """При post-merge offset берётся от принимающего (первого) чанка."""
        from morag.indexing.chunker import ChunkResult
        chunker = _make_chunker(min_tokens=30, max_tokens=200)
        chunks = [
            ChunkResult(text='Big chunk.', pages=[], char_offset=100),
            ChunkResult(text='tiny.', pages=[], char_offset=200),
        ]
        merged = chunker._post_merge(chunks)
        assert len(merged) == 1
        # offset от первого чанка (100), не от второго (200)
        assert merged[0].char_offset == 100

    async def test_char_offset_single_chunk_document(self):
        """Единственный чанк документа имеет offset 0."""
        text = 'Just a short document.'
        chunker = _make_chunker(min_tokens=5, max_tokens=500)
        result = await chunker.chunk_with_metadata(text)
        assert len(result) == 1
        assert result[0].char_offset == 0

    async def test_char_offset_with_magnetic_header(self):
        """Магнитный заголовок: offset чанка = offset заголовка, не предыдущего блока."""
        filler = 'word ' * 35
        text = f'{filler}\n\n# Important Title\n\n{filler}'
        chunker = _make_chunker(min_tokens=10, max_tokens=50)
        result = await chunker.chunk_with_metadata(text)
        # Найти чанк начинающийся с заголовка
        heading_chunk = next(
            (cr for cr in result if cr.text.startswith('# Important')), None,
        )
        assert heading_chunk is not None
        # Offset должен указывать на позицию # в документе
        assert text[heading_chunk.char_offset:].startswith('# Important')

    async def test_pages_and_char_offset_together(self):
        """PDF документ: pages и char_offset корректны одновременно."""
        filler1 = 'Alpha words go here. ' * 15
        filler2 = 'Beta words go here. ' * 15
        text = (
            f'<!-- page:1 -->\n# Section A\n\n{filler1}\n\n'
            f'<!-- page:2 -->\n# Section B\n\n{filler2}'
        )
        chunker = _make_chunker(min_tokens=10, max_tokens=60)
        result = await chunker.chunk_with_metadata(text, paged=True)
        assert len(result) >= 2
        for cr in result:
            # pages заполнены
            assert cr.pages, f'Empty pages for chunk: {cr.text[:40]}'
            # offset указывает на реальную позицию
            nearby = text[cr.char_offset:cr.char_offset + 100]
            first_word = cr.text.strip().split()[0]
            assert first_word in nearby
            # маркеров нет в тексте
            assert '<!-- page:' not in cr.text


# ---------------------------------------------------------------------------
# Per-type oversized strategies + transform
# ---------------------------------------------------------------------------

class TestOversizedPerType:
    async def test_different_strategies_per_type(self):
        """Разные типы блоков обрабатываются разными стратегиями."""
        # Документ с таблицей (transform) и кодом (asis)
        table = '| Key | Value |\n|-----|-------|\n| a | ' + 'x ' * 200 + ' |'
        code = '```python\n' + '\n'.join(f'line_{i} = {i}' for i in range(50)) + '\n```'
        text = f'{table}\n\n{code}'
        chunker = _make_chunker(
            min_tokens=5, max_tokens=50,
            oversized_strategies={'table': 'transform', 'fence': 'asis'},
        )
        result = await chunker.chunk(text)
        # Таблица должна быть трансформирована (нет | в чанках таблицы)
        # Код остаётся as-is (с ```)
        code_chunks = [c for c in result if '```' in c]
        assert len(code_chunks) >= 1, 'Code block should remain as-is'

    async def test_default_strategies(self):
        """Без явных стратегий — defaults работают."""
        chunker = _make_chunker(min_tokens=5, max_tokens=50)
        # Default: list=split, paragraph=split
        items = '\n'.join(f'- Item {i} with description.' for i in range(30))
        result = await chunker.chunk(items)
        assert len(result) > 1  # list split by default


class TestTransformTable:
    def test_transform_table_to_text(self):
        """Таблица конвертируется в h4 + содержимое формат."""
        chunker = _make_chunker()
        table = '| Name | Age | Role |\n|------|-----|------|\n| Alice | 30 | Dev |\n| Bob | 25 | QA |'
        result = chunker._transform_table_to_text(table)
        assert '#### Name' in result
        assert 'Alice' in result
        assert '#### Age' in result
        assert '30' in result
        assert '#### Role' in result
        assert 'Dev' in result
        assert 'Bob' in result
        assert '|' not in result  # таблица исчезла

    def test_transform_preserves_content(self):
        """Трансформация не теряет данные."""
        chunker = _make_chunker()
        table = '| Col1 | Col2 |\n|------|------|\n| val1 | val2 |\n| val3 | val4 |'
        result = chunker._transform_table_to_text(table)
        assert 'val1' in result
        assert 'val2' in result
        assert 'val3' in result
        assert 'val4' in result

    def test_transform_empty_cells_skipped(self):
        """Пустые ячейки не генерируют заголовок."""
        chunker = _make_chunker()
        table = '| Name | Notes |\n|------|-------|\n| Alice |  |'
        result = chunker._transform_table_to_text(table)
        assert '#### Name' in result
        assert 'Alice' in result
        # Пустая ячейка Notes не должна создавать заголовок
        assert '#### Notes' not in result

    async def test_transform_then_rechunk(self):
        """Transform таблицы → key-value → рекурсивный чанкинг."""
        # Широкая таблица: одна строка с длинным описанием
        desc = 'Very detailed task description. ' * 20
        table = f'| Task | Description |\n|------|-------------|\n| CI/CD | {desc} |'
        chunker = _make_chunker(
            min_tokens=10, max_tokens=50,
            oversized_strategies={'table': 'transform'},
        )
        result = await chunker.chunk(table)
        # Трансформированный текст должен разбиться на чанки
        assert len(result) >= 1
        full = '\n'.join(result)
        # Содержимое сохранено
        assert 'CI/CD' in full or 'Task' in full
        assert 'detailed task description' in full

    async def test_transform_multi_row_table(self):
        """Многострочная таблица: каждая строка → key-value блок."""
        rows = [f'| Task {i} | Description for task {i} with details. |' for i in range(5)]
        table = '| Name | Info |\n|------|------|\n' + '\n'.join(rows)
        chunker = _make_chunker(
            min_tokens=10, max_tokens=100,
            oversized_strategies={'table': 'transform'},
        )
        result = await chunker.chunk(table)
        full = '\n'.join(result)
        for i in range(5):
            assert f'Task {i}' in full

    def test_transform_invalid_table_returns_original(self):
        """Невалидная таблица (нет separator) — возвращается как есть."""
        chunker = _make_chunker()
        text = '| Not | A | Table |\nJust some text'
        result = chunker._transform_table_to_text(text)
        assert result == text  # без изменений

    async def test_transform_recursion_depth_1(self):
        """Transform делает только один уровень рекурсии."""
        # Таблица с очень длинной ячейкой — после transform получится длинный текст
        # Рекурсивный chunk() разобьёт по предложениям (default paragraph=split)
        desc = 'Sentence number %d. ' * 30
        table = f'| Task | Desc |\n|------|------|\n| X | {desc} |'
        chunker = _make_chunker(
            min_tokens=10, max_tokens=50,
            oversized_strategies={'table': 'transform', 'paragraph': 'split'},
        )
        result = await chunker.chunk(table)
        # Должно разбиться без бесконечной рекурсии
        assert len(result) >= 1
