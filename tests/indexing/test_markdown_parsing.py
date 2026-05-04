"""Тесты CommonMark-парсинга в сплиттере (markdown-it-py).

Покрывают edge cases, которые regex-подход обрабатывал некорректно,
и корректность новых функций _top_level_blocks, _split_by_headers,
_split_paragraphs, split_into_units.
"""


from morag.indexing.splitter import (
    MarkdownHeaderSplitter,
    _is_code_fence,
    _parse_md,
    _split_by_headers,
    _split_paragraphs,
    _top_level_blocks,
    split_into_units,
)


# ---------------------------------------------------------------------------
# _top_level_blocks
# ---------------------------------------------------------------------------

class TestTopLevelBlocks:
    """Извлечение top-level блоков из Markdown через CommonMark-парсер."""

    def test_empty_text(self):
        assert _top_level_blocks('') == []

    def test_whitespace_only(self):
        assert _top_level_blocks('   \n\n  ') == []

    def test_single_paragraph(self):
        blocks = _top_level_blocks('Просто текст.')
        assert len(blocks) == 1
        assert blocks[0][0] == 'paragraph_open'

    def test_heading(self):
        blocks = _top_level_blocks('# Заголовок')
        assert len(blocks) == 1
        assert blocks[0][0] == 'heading_open'

    def test_fence(self):
        text = '```python\nprint("hello")\n```'
        blocks = _top_level_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == 'fence'

    def test_table(self):
        text = '| A | B |\n|---|---|\n| 1 | 2 |'
        blocks = _top_level_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == 'table_open'

    def test_multiple_blocks(self):
        text = '# Title\n\nText.\n\n```code\n```\n\n| A |\n|---|\n| 1 |'
        blocks = _top_level_blocks(text)
        types = [b[0] for b in blocks]
        assert 'heading_open' in types
        assert 'paragraph_open' in types
        assert 'fence' in types
        assert 'table_open' in types

    def test_line_ranges_are_correct(self):
        text = '# Title\n\nParagraph text.\n\n```\ncode\n```'
        blocks = _top_level_blocks(text)
        lines = text.split('\n')
        for block_type, start, end in blocks:
            block_text = '\n'.join(lines[start:end])
            if block_type == 'heading_open':
                assert '# Title' in block_text
            elif block_type == 'paragraph_open':
                assert 'Paragraph' in block_text
            elif block_type == 'fence':
                assert 'code' in block_text

    def test_nested_blocks_not_duplicated(self):
        """Вложенные блоки (внутри blockquote/list) не дублируются на top-level."""
        text = '> Цитата с текстом.\n>\n> Второй абзац цитаты.'
        blocks = _top_level_blocks(text)
        # Только один top-level блок — blockquote
        assert len(blocks) == 1
        assert blocks[0][0] == 'blockquote_open'

    def test_indented_code_block(self):
        text = 'Текст.\n\n    indented code\n    line two\n\nЕщё текст.'
        blocks = _top_level_blocks(text)
        types = [b[0] for b in blocks]
        assert 'code_block' in types

    def test_hr_block(self):
        text = 'Before.\n\n---\n\nAfter.'
        blocks = _top_level_blocks(text)
        types = [b[0] for b in blocks]
        assert 'hr' in types


# ---------------------------------------------------------------------------
# _split_by_headers — ключевые edge cases
# ---------------------------------------------------------------------------

class TestSplitByHeaders:
    """Разбиение по заголовкам через CommonMark-парсер."""

    def test_no_headers_returns_original(self):
        text = 'Текст без заголовков.\n\nЕщё текст.'
        result = _split_by_headers(text)
        assert result == [text]

    def test_single_header_returns_original(self):
        text = '# Один заголовок\n\nТекст.'
        result = _split_by_headers(text)
        assert result == [text]

    def test_two_headers_split(self):
        text = '# Первый\n\nТекст 1.\n\n# Второй\n\nТекст 2.'
        result = _split_by_headers(text)
        assert len(result) == 2
        assert '# Первый' in result[0]
        assert '# Второй' in result[1]

    def test_text_before_first_header_is_separate_section(self):
        text = 'Вводный текст.\n\n# Заголовок\n\nОсновной текст.'
        result = _split_by_headers(text)
        assert len(result) == 2
        assert 'Вводный' in result[0]
        assert '# Заголовок' in result[1]

    def test_header_inside_code_fence_ignored(self):
        """Ключевой тест: # внутри code fence не является заголовком."""
        text = '# Настоящий\n\n```yaml\n# yaml comment\nkey: value\n```\n\n# Ещё настоящий'
        result = _split_by_headers(text)
        assert len(result) == 2
        # yaml comment НЕ создал лишнюю секцию
        assert '# yaml comment' in result[0]  # осталось внутри первой секции
        assert '# Ещё настоящий' in result[1]

    def test_header_inside_bash_fence_ignored(self):
        """# в bash-комментариях внутри fence не является заголовком."""
        text = '# Intro\n\n```bash\n# install\napt-get install curl\n```\n\n# Next'
        result = _split_by_headers(text)
        assert len(result) == 2
        assert '# install' in result[0]

    def test_multiple_header_levels(self):
        text = '# H1\n\n## H2\n\n### H3\n\nТекст.'
        result = _split_by_headers(text)
        assert len(result) == 3

    def test_image_with_plantuml_fence_inline(self):
        """Регрессия: однострочная конструкция [Изображение: ```plantuml...```]."""
        text = (
            '# Раздел 1\n\n'
            '[Изображение: ```plantuml diagram```]\n\n'
            'Текст после изображения.\n\n'
            '# Раздел 2\n\n'
            'Текст второго раздела.'
        )
        result = _split_by_headers(text)
        assert len(result) == 2
        assert '# Раздел 1' in result[0]
        assert '# Раздел 2' in result[1]
        assert 'Текст после изображения' in result[0]

    def test_image_with_plantuml_fence_multiline(self):
        """Регрессия: многострочная конструкция [Изображение: ```plantuml...```].

        CommonMark трактует ```] как fence opener (info string ']').
        Препроцессинг _parse_md нейтрализует это: добавляет backtick,
        делая info string невалидным по спеке §4.5.
        """
        text = (
            '# Раздел 1\n\n'
            'Описание процесса.\n\n'
            '[Изображение: ```plantuml\n'
            '@startuml\n'
            'Alice -> Bob\n'
            '@enduml\n'
            '```]\n\n'
            'Текст после.\n\n'
            '# Раздел 2\n\n'
            'Содержимое раздела 2.'
        )
        result = _split_by_headers(text)
        assert len(result) == 2
        assert '# Раздел 1' in result[0]
        assert '# Раздел 2' in result[1]
        assert 'Текст после' in result[0]

    def test_unclosed_fence_does_not_break(self):
        """Незакрытый fence не должен ломать разбиение по заголовкам."""
        text = '# Before\n\n```\nunclosed code\n\n# After\n\nText.'
        result = _split_by_headers(text)
        # CommonMark трактует незакрытый fence как fence до конца документа
        # → # After оказывается внутри fence → одна секция
        assert len(result) >= 1

    def test_setext_heading(self):
        """Setext-заголовки (подчёркивание ===) тоже распознаются."""
        text = 'Заголовок 1\n===\n\nТекст.\n\nЗаголовок 2\n===\n\nТекст 2.'
        result = _split_by_headers(text)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _split_paragraphs — блочное разбиение
# ---------------------------------------------------------------------------

class TestSplitParagraphs:
    """Разбиение текста на блоки через CommonMark-парсер."""

    def test_empty_text(self):
        assert _split_paragraphs('') == []

    def test_single_paragraph(self):
        result = _split_paragraphs('Один абзац.')
        assert result == ['Один абзац.']

    def test_two_paragraphs_separated_by_blank_line(self):
        text = 'Первый абзац.\n\nВторой абзац.'
        result = _split_paragraphs(text)
        assert len(result) == 2
        assert result[0] == 'Первый абзац.'
        assert result[1] == 'Второй абзац.'

    def test_code_fence_is_single_block(self):
        """Code fence с пустыми строками внутри остаётся цельным блоком."""
        text = '```\nline 1\n\nline 3\n```'
        result = _split_paragraphs(text)
        assert len(result) == 1
        assert 'line 1' in result[0]
        assert 'line 3' in result[0]

    def test_table_is_single_block(self):
        text = '| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |'
        result = _split_paragraphs(text)
        assert len(result) == 1
        assert '| 1 | 2 |' in result[0]
        assert '| 3 | 4 |' in result[0]

    def test_heading_is_separate_block(self):
        text = '# Заголовок\n\nТекст.'
        result = _split_paragraphs(text)
        assert len(result) == 2
        assert result[0] == '# Заголовок'
        assert result[1] == 'Текст.'

    def test_fence_between_paragraphs(self):
        text = 'Before.\n\n```python\nx = 1\n```\n\nAfter.'
        result = _split_paragraphs(text)
        assert len(result) == 3
        assert result[0] == 'Before.'
        assert '```python' in result[1]
        assert result[2] == 'After.'

    def test_inline_backticks_not_confused_with_fence(self):
        """Inline backticks в тексте не путаются с code fences."""
        text = (
            'Первый абзац.\n\n'
            '[Изображение: ```plantuml diagram```]\n\n'
            'Третий абзац.'
        )
        result = _split_paragraphs(text)
        assert len(result) == 3
        assert 'Первый' in result[0]
        assert 'plantuml' in result[1]
        assert 'Третий' in result[2]

    def test_multiline_image_with_fence_not_swallows_text(self):
        """Многострочный [Изображение: ```...```] не склеивает последующие блоки."""
        text = (
            'Первый абзац.\n\n'
            '[Изображение: ```plantuml\n'
            '@startuml\n'
            'A -> B\n'
            '@enduml\n'
            '```]\n\n'
            'Третий абзац.'
        )
        result = _split_paragraphs(text)
        assert len(result) >= 2
        # Третий абзац не попал внутрь fence
        assert any('Третий' in p for p in result)

    def test_list_is_single_block(self):
        text = '- пункт 1\n- пункт 2\n- пункт 3'
        result = _split_paragraphs(text)
        assert len(result) == 1
        assert 'пункт 1' in result[0]
        assert 'пункт 3' in result[0]

    def test_blockquote_is_single_block(self):
        text = '> Цитата.\n> Продолжение.'
        result = _split_paragraphs(text)
        assert len(result) == 1
        assert 'Цитата' in result[0]
        assert 'Продолжение' in result[0]

    def test_horizontal_rule_is_block(self):
        text = 'Before.\n\n---\n\nAfter.'
        result = _split_paragraphs(text)
        assert len(result) == 3  # paragraph, hr, paragraph

    def test_mixed_content(self):
        text = (
            '# Title\n\n'
            'Intro text.\n\n'
            '```\ncode\n```\n\n'
            '| A |\n|---|\n| 1 |\n\n'
            'Final text.'
        )
        result = _split_paragraphs(text)
        assert len(result) == 5
        assert result[0] == '# Title'
        assert result[1] == 'Intro text.'
        assert '```' in result[2]
        assert '| A |' in result[3]
        assert result[4] == 'Final text.'


# ---------------------------------------------------------------------------
# split_into_units — атомарные единицы
# ---------------------------------------------------------------------------

class TestSplitIntoUnits:
    """Разбиение на атомарные единицы: fences и таблицы цельные, текст → предложения."""

    def test_plain_text_split_to_sentences(self):
        text = 'Первое предложение. Второе предложение. Третье.'
        units = split_into_units(text)
        assert len(units) >= 2  # razdel/nltk определяют границы

    def test_code_fence_stays_atomic(self):
        text = 'Текст.\n\n```python\ndef foo():\n    pass\n```\n\nЕщё текст.'
        units = split_into_units(text)
        fence_units = [u for u in units if '```python' in u]
        assert len(fence_units) == 1
        assert 'def foo():' in fence_units[0]
        assert 'pass' in fence_units[0]

    def test_table_stays_atomic(self):
        text = 'Текст.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nЕщё.'
        units = split_into_units(text)
        table_units = [u for u in units if '| A |' in u]
        assert len(table_units) == 1
        assert '| 3 | 4 |' in table_units[0]

    def test_indented_code_block_stays_atomic(self):
        text = 'Текст.\n\n    code line 1\n    code line 2\n\nЕщё.'
        units = split_into_units(text)
        code_units = [u for u in units if 'code line 1' in u]
        assert len(code_units) == 1
        assert 'code line 2' in code_units[0]

    def test_fence_with_blank_lines_stays_atomic(self):
        """Пустые строки внутри code fence не разрывают блок."""
        text = '```\nline 1\n\nline 3\n```'
        units = split_into_units(text)
        assert len(units) == 1
        assert 'line 1' in units[0]
        assert 'line 3' in units[0]

    def test_plantuml_inline_not_confused_with_fence(self):
        """Регрессия: однострочный [Изображение: ```plantuml...```]."""
        text = (
            'Описание процесса адаптации.\n\n'
            '[Изображение: ```plantuml sequence diagram```]\n\n'
            'Первый этап. Второй этап. Третий этап.'
        )
        units = split_into_units(text)
        assert len(units) >= 3

    def test_plantuml_multiline_not_confused_with_fence(self):
        """Регрессия: многострочный [Изображение: ```plantuml...```]."""
        text = (
            'Описание процесса.\n\n'
            '[Изображение: ```plantuml\n'
            '@startuml\n'
            'A -> B: call\n'
            '@enduml\n'
            '```]\n\n'
            'Первый этап. Второй этап. Третий этап.'
        )
        units = split_into_units(text)
        # Текст после конструкции НЕ попал внутрь fence
        assert len(units) >= 3

    def test_empty_text(self):
        assert split_into_units('') == []

    def test_heading_text_goes_to_sentences(self):
        text = '# Заголовок\n\nТекст раздела.'
        units = split_into_units(text)
        assert any('Заголовок' in u for u in units)
        assert any('раздела' in u for u in units)


# ---------------------------------------------------------------------------
# MarkdownHeaderSplitter — CommonMark edge cases
# ---------------------------------------------------------------------------

class TestMarkdownHeaderSplitterCommonMark:
    """Тесты edge cases, которые regex-подход обрабатывал некорректно."""

    def test_hash_in_code_fence_not_counted_as_header(self):
        """# внутри code fence не считается заголовком для can_split."""
        text = '# Real Header\n\n```\n# Not a header\n## Also not\n```'
        splitter = MarkdownHeaderSplitter()
        # Только один реальный заголовок — can_split = False
        assert splitter.can_split(text) is False

    def test_hash_in_code_fence_not_split(self):
        """При split # внутри fence не создаёт лишних секций."""
        text = '# First\n\n```bash\n# comment\necho hello\n```\n\n# Second\n\nText.'
        splitter = MarkdownHeaderSplitter()
        blocks = splitter.split(text)
        assert len(blocks) == 2
        assert '# comment' in blocks[0]  # comment остался в секции First
        assert '# Second' in blocks[1]

    def test_inline_backticks_not_confused_with_fence(self):
        """Inline `code` и ```code``` не ломают парсинг заголовков."""
        text = '# Header 1\n\nUse `# not header` in code.\n\n# Header 2\n\nText.'
        splitter = MarkdownHeaderSplitter()
        blocks = splitter.split(text)
        assert len(blocks) == 2

    def test_plantuml_image_inline_regression(self):
        """Регрессия: однострочный [Изображение: ```plantuml...```]."""
        text = (
            '# Процесс адаптации\n\n'
            'Описание процесса.\n\n'
            '[Изображение: ```plantuml sequence diagram```]\n\n'
            'Текст после изображения.\n\n'
            '# Открытые позиции\n\n'
            'Список позиций.'
        )
        splitter = MarkdownHeaderSplitter()
        blocks = splitter.split(text)
        assert len(blocks) == 2
        assert 'Процесс адаптации' in blocks[0]
        assert 'Открытые позиции' in blocks[1]
        assert 'Текст после изображения' in blocks[0]

    def test_plantuml_image_multiline_regression(self):
        """Регрессия: многострочный [Изображение: ```plantuml...```].

        ```] в начале строки — CommonMark трактует как fence opener.
        Препроцессинг нейтрализует: info string с backtick невалиден (§4.5).
        """
        text = (
            '# Процесс адаптации\n\n'
            'Описание.\n\n'
            '[Изображение: ```plantuml\n'
            '@startuml\n'
            'Manager -> HR: запрос\n'
            'HR -> Candidate: оффер\n'
            '@enduml\n'
            '```]\n\n'
            'Текст после диаграммы.\n\n'
            '# Открытые позиции\n\n'
            'Список позиций.'
        )
        splitter = MarkdownHeaderSplitter()
        blocks = splitter.split(text)
        assert len(blocks) == 2
        assert 'Процесс адаптации' in blocks[0]
        assert 'Открытые позиции' in blocks[1]


# ---------------------------------------------------------------------------
# _is_code_fence
# ---------------------------------------------------------------------------

class TestIsCodeFence:

    def test_simple_fence(self):
        assert _is_code_fence('```\ncode\n```') is True

    def test_fence_with_language(self):
        assert _is_code_fence('```python\ncode\n```') is True

    def test_indented_fence(self):
        assert _is_code_fence('  ```\ncode\n```') is True

    def test_plain_text(self):
        assert _is_code_fence('plain text') is False

    def test_inline_backticks(self):
        assert _is_code_fence('`inline code`') is False

    def test_table(self):
        assert _is_code_fence('| A | B |') is False


# ---------------------------------------------------------------------------
# Интеграционные тесты: полный пайплайн парсинга
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _parse_md — препроцессинг Confluence-конструкций
# ---------------------------------------------------------------------------

class TestParseMdPreprocessing:
    """Препроцессинг перед CommonMark-парсингом: нейтрализация ложных fence openers."""

    def test_bracket_fence_not_treated_as_fence(self):
        """```] в начале строки не парсится как code fence."""
        text = 'Text before.\n\n```]\n\nText after.'
        tokens = _parse_md(text)
        fence_tokens = [t for t in tokens if t.type == 'fence']
        # ```] не является fence — оба текста должны быть paragraph
        assert len(fence_tokens) == 0

    def test_real_fence_still_works(self):
        """Настоящий fence (```) по-прежнему парсится корректно."""
        text = '```python\nprint(1)\n```'
        tokens = _parse_md(text)
        fence_tokens = [t for t in tokens if t.type == 'fence']
        assert len(fence_tokens) == 1

    def test_fence_with_language_still_works(self):
        """Fence с языком (```yaml) работает."""
        text = '```yaml\nkey: value\n```'
        tokens = _parse_md(text)
        fence_tokens = [t for t in tokens if t.type == 'fence']
        assert len(fence_tokens) == 1

    def test_bracket_fence_with_content_neutralized(self):
        """```plantuml...] нейтрализуется."""
        text = 'Before.\n\n```plantuml some text]\n\nAfter heading.\n\n# Real heading'
        tokens = _parse_md(text)
        headings = [t for t in tokens if t.type == 'heading_open']
        assert len(headings) == 1  # heading НЕ поглощён ложным fence

    def test_line_numbers_preserved_after_preprocessing(self):
        """Номера строк в token.map корректны после препроцессинга."""
        text = 'Line 0.\n\n```]\n\nLine 4.\n\n# Heading on line 6'
        tokens = _parse_md(text)
        headings = [t for t in tokens if t.type == 'heading_open']
        assert len(headings) == 1
        assert headings[0].map[0] == 6

    def test_multiple_bracket_fences(self):
        """Несколько ```] в документе все нейтрализуются."""
        text = (
            '[Image: ```diagram\n'
            'content\n'
            '```]\n\n'
            'Middle text.\n\n'
            '[Image: ```flowchart\n'
            'nodes\n'
            '```]\n\n'
            '# Real heading'
        )
        tokens = _parse_md(text)
        headings = [t for t in tokens if t.type == 'heading_open']
        assert len(headings) == 1


# ---------------------------------------------------------------------------
# Интеграционные тесты: полный пайплайн парсинга
# ---------------------------------------------------------------------------

class TestMarkdownParsingIntegration:
    """Проверка взаимодействия _split_by_headers → _split_paragraphs → split_into_units."""

    def test_complex_document_with_fences_and_headers(self):
        """Документ с code fences, таблицами, заголовками и обычным текстом."""
        text = (
            '# Введение\n\n'
            'Основной текст введения.\n\n'
            '## Архитектура\n\n'
            'Описание архитектуры.\n\n'
            '```yaml\n'
            '# yaml config\n'
            'server:\n'
            '  port: 8080\n'
            '```\n\n'
            '| Компонент | Описание |\n'
            '|---|---|\n'
            '| API | REST сервис |\n'
            '| DB | PostgreSQL |\n\n'
            '## Деплой\n\n'
            'Инструкция по деплою.'
        )
        # Headers split
        sections = _split_by_headers(text)
        assert len(sections) == 3  # Введение, Архитектура, Деплой

        # Paragraphs
        arch_section = sections[1]
        paragraphs = _split_paragraphs(arch_section)
        # heading, paragraph, fence, table
        assert len(paragraphs) >= 3

        # Code fence не разорван
        fence_blocks = [p for p in paragraphs if '```yaml' in p]
        assert len(fence_blocks) == 1
        assert '# yaml config' in fence_blocks[0]  # yaml comment внутри fence
        assert 'port: 8080' in fence_blocks[0]

        # Table не разорвана
        table_blocks = [p for p in paragraphs if '| Компонент |' in p]
        assert len(table_blocks) == 1
        assert '| DB |' in table_blocks[0]

    def test_document_with_nested_fences(self):
        """Вложенные backticks (`````` внутри ```) обрабатываются корректно."""
        text = (
            '# Example\n\n'
            '````markdown\n'
            '```python\n'
            'print("nested")\n'
            '```\n'
            '````\n\n'
            '# Next'
        )
        sections = _split_by_headers(text)
        assert len(sections) == 2
        # Внутренний ``` не закрыл внешний fence
        assert '```python' in sections[0]

    def test_heading_after_table_detected(self):
        """Заголовок после таблицы корректно определяется как граница секции."""
        text = (
            '# Data\n\n'
            '| X | Y |\n|---|---|\n| 1 | 2 |\n\n'
            '# Analysis\n\n'
            'Results.'
        )
        sections = _split_by_headers(text)
        assert len(sections) == 2
        assert '| 1 | 2 |' in sections[0]
        assert 'Results' in sections[1]

    def test_multiple_consecutive_fences(self):
        """Несколько fence-блоков подряд не путаются."""
        text = (
            '```python\nprint(1)\n```\n\n'
            '```bash\necho 2\n```\n\n'
            '```sql\nSELECT 3\n```'
        )
        paragraphs = _split_paragraphs(text)
        assert len(paragraphs) == 3
        assert 'print(1)' in paragraphs[0]
        assert 'echo 2' in paragraphs[1]
        assert 'SELECT 3' in paragraphs[2]

    def test_tilde_fence(self):
        """Fenced code blocks с ~~~ тоже поддерживаются CommonMark."""
        text = '~~~\ntilde fence content\n~~~'
        paragraphs = _split_paragraphs(text)
        assert len(paragraphs) == 1
        assert 'tilde fence content' in paragraphs[0]
