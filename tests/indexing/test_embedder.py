from morag.indexing.embedder import _word_to_index


# ---------------------------------------------------------------------------
# _word_to_index
# ---------------------------------------------------------------------------

class TestWordToIndex:
    def test_returns_int(self):
        assert isinstance(_word_to_index('hello'), int)

    def test_deterministic(self):
        assert _word_to_index('test') == _word_to_index('test')

    def test_different_words_different_indices(self):
        assert _word_to_index('hello') != _word_to_index('world')

    def test_result_within_bounds(self):
        idx = _word_to_index('любое слово')
        assert 0 <= idx < 4_294_967_295

    def test_known_value(self):
        """Стабильность хэша: конкретное значение не должно меняться никогда.

        Если этот тест падает — все сохранённые коллекции в Qdrant становятся
        несовместимыми с новым кодом. Изменять число ЗАПРЕЩЕНО.
        """
        assert _word_to_index('test') == 1085751994

    def test_case_sensitive(self):
        """Нет lowercase: 'Python' и 'python' — разные токены, разные индексы."""
        assert _word_to_index('Python') != _word_to_index('python')
