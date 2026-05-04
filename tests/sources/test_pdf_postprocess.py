from morag.sources.pdf_postprocess import CodeFencePostProcessor, DeduplicatePostProcessor


class TestCodeFencePreservesPageMarkers:
    def test_markers_not_stripped(self):
        text = '<!-- page:1 -->\n# Заголовок\n\n<!-- page:2 -->\nТекст.'
        result = CodeFencePostProcessor().process(text)
        assert '<!-- page:1 -->' in result
        assert '<!-- page:2 -->' in result

    def test_markers_survive_with_code_fences(self):
        text = (
            '```markdown\n'
            '<!-- page:1 -->\n# Заголовок\n'
            '```\n\n'
            '<!-- page:2 -->\nТекст.'
        )
        result = CodeFencePostProcessor().process(text)
        assert '<!-- page:1 -->' in result
        assert '<!-- page:2 -->' in result


class TestDeduplicatePreservesPageMarkers:
    def _dedup(self, text: str) -> str:
        return DeduplicatePostProcessor(
            threshold=0.7, window=5, min_phrase_len=20,
        ).process(text)

    def test_markers_preserved_in_similar_pages(self):
        """Маркеры не удаляются даже если текст страниц частично совпадает."""
        text = (
            '<!-- page:1 -->\n'
            'Это уникальный текст первой страницы документа с достаточной длиной.\n\n'
            '<!-- page:2 -->\n'
            'Это уникальный текст второй страницы документа с достаточной длиной.\n\n'
            '<!-- page:3 -->\n'
            'Это уникальный текст третьей страницы документа с достаточной длиной.'
        )
        result = self._dedup(text)
        assert '<!-- page:1 -->' in result
        assert '<!-- page:2 -->' in result
        assert '<!-- page:3 -->' in result

    def test_duplicate_pages_both_preserved(self):
        """Даже если контент страниц одинаков, обе сохраняются — номера страниц важны."""
        original = 'Длинный абзац который повторяется дважды на разных страницах документа.'
        text = (
            f'<!-- page:1 -->\n{original}\n\n'
            f'<!-- page:2 -->\n{original}\n\n'
            f'<!-- page:3 -->\nУникальный текст третьей страницы.'
        )
        result = self._dedup(text)
        assert '<!-- page:1 -->' in result
        assert '<!-- page:2 -->' in result
        assert '<!-- page:3 -->' in result

    def test_dedup_still_works_within_page(self):
        """Дубли без маркеров (внутри страницы) по-прежнему удаляются."""
        dup = 'Этот абзац повторяется внутри одной страницы и должен быть удалён.'
        text = (
            f'<!-- page:1 -->\nНачало страницы.\n\n'
            f'{dup}\n\n'
            f'{dup}\n\n'
            f'<!-- page:2 -->\nВторая страница.'
        )
        result = self._dedup(text)
        assert result.count(dup) == 1

    def test_all_markers_in_output_are_valid(self):
        """Все маркеры в результате имеют корректный формат."""
        import re
        text = '\n\n'.join(
            f'<!-- page:{i} -->\nСодержимое страницы номер {i} с уникальным текстом.'
            for i in range(1, 11)
        )
        result = self._dedup(text)
        markers = re.findall(r'<!-- page:(\d+) -->', result)
        assert len(markers) == 10
        assert [int(m) for m in markers] == list(range(1, 11))
