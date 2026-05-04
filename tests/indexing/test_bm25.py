"""Тесты для BM25 sparse vector builder."""
from __future__ import annotations


from morag.indexing.bm25 import tokenize, build_bm25_vectors, _word_to_index, _stem


class TestTokenize:
    def test_basic_words(self):
        assert tokenize('hello world') == ['hello', 'world']

    def test_english_stop_words_filtered(self):
        assert tokenize('the is a an') == []

    def test_russian_stop_words_filtered(self):
        assert tokenize('и в на не с что') == []

    def test_mixed_english(self):
        result = tokenize('the court ordered payment')
        assert 'court' in result
        assert 'order' in result  # 'ordered' стеммится в 'order'
        assert 'payment' in result
        assert 'the' not in result

    def test_lowercase(self):
        assert tokenize('COURT Order') == ['court', 'order']

    def test_numbers(self):
        result = tokenize('300000')
        assert '300000' in result

    def test_empty_string(self):
        assert tokenize('') == []

    def test_whitespace_only(self):
        assert tokenize('   \n\t  ') == []

    def test_punctuation_split(self):
        result = tokenize('court-ordered, payment.')
        assert 'court' in result
        assert 'order' in result  # stemmed
        assert 'payment' in result

    def test_page_markers(self):
        result = tokenize('<!-- page:1 --> important text')
        assert 'page' in result
        assert 'text' in result


class TestStemming:
    def test_russian_stemming(self):
        assert _stem('документов') == _stem('документы')
        assert _stem('документов') == _stem('документ')

    def test_english_stemming(self):
        assert _stem('documents') == _stem('document')
        assert _stem('courts') == _stem('court')

    def test_russian_verb_forms(self):
        assert _stem('решения') == _stem('решение')
        assert _stem('судебный') == _stem('судебная')

    def test_mixed_text_tokenize(self):
        result = tokenize('Судебные решения по документам')
        # 'по' — русское стоп-слово
        assert len(result) == 3
        assert _stem('судебные') in result
        assert _stem('решения') in result
        assert _stem('документам') in result

    def test_cyrillic_detection(self):
        # Кириллица → русский стеммер
        assert _stem('документов') == 'документ'
        # Латиница → английский стеммер
        assert _stem('documents') == 'document'

    def test_stemmed_tokens_produce_same_hash(self):
        idx1 = _word_to_index(_stem('документов'))
        idx2 = _word_to_index(_stem('документы'))
        idx3 = _word_to_index(_stem('документ'))
        assert idx1 == idx2 == idx3


class TestWordToIndex:
    def test_returns_int(self):
        assert isinstance(_word_to_index('hello'), int)

    def test_deterministic(self):
        assert _word_to_index('court') == _word_to_index('court')

    def test_different_words_different_index(self):
        assert _word_to_index('court') != _word_to_index('judge')

    def test_within_range(self):
        idx = _word_to_index('test')
        assert 0 <= idx < 4_294_967_295


class TestBuildBm25Vectors:
    def test_single_doc(self):
        vectors = build_bm25_vectors(['court order payment'])
        assert len(vectors) == 1
        assert len(vectors[0]['indices']) > 0
        assert len(vectors[0]['indices']) == len(vectors[0]['values'])

    def test_idf_rare_term_higher(self):
        texts = [
            'court order payment',
            'court decision ruling',
            'court judgment appeal',
            'rare_unique_term hello',
        ]
        vectors = build_bm25_vectors(texts)
        rare_idx = _word_to_index(_stem('rare_unique_term'))
        court_idx = _word_to_index(_stem('court'))

        rare_weight = 0
        court_weight = 0
        for idx, val in zip(vectors[3]['indices'], vectors[3]['values']):
            if idx == rare_idx:
                rare_weight = val
        for idx, val in zip(vectors[0]['indices'], vectors[0]['values']):
            if idx == court_idx:
                court_weight = val

        assert rare_weight > court_weight

    def test_empty_corpus(self):
        assert build_bm25_vectors([]) == []

    def test_empty_doc_in_corpus(self):
        vectors = build_bm25_vectors(['hello world', ''])
        assert len(vectors) == 2
        assert len(vectors[0]['indices']) > 0
        assert len(vectors[1]['indices']) == 0

    def test_stop_words_only(self):
        vectors = build_bm25_vectors(['the is a an', 'hello world'])
        assert len(vectors[0]['indices']) == 0
        assert len(vectors[1]['indices']) > 0

    def test_russian_stop_words_only(self):
        vectors = build_bm25_vectors(['и в на не с', 'документ решение'])
        assert len(vectors[0]['indices']) == 0
        assert len(vectors[1]['indices']) > 0

    def test_all_values_positive(self):
        vectors = build_bm25_vectors(['court order', 'judge ruling'])
        for v in vectors:
            for val in v['values']:
                assert val > 0

    def test_tf_increases_weight(self):
        vectors = build_bm25_vectors([
            'court court court',
            'court',
        ])
        court_idx = _word_to_index(_stem('court'))

        weight_many = 0
        weight_one = 0
        for idx, val in zip(vectors[0]['indices'], vectors[0]['values']):
            if idx == court_idx:
                weight_many = val
        for idx, val in zip(vectors[1]['indices'], vectors[1]['values']):
            if idx == court_idx:
                weight_one = val

        assert weight_many > weight_one

    def test_k1_b_parameters(self):
        texts = ['hello world', 'hello world extra words length variation']
        v1 = build_bm25_vectors(texts, k1=1.5, b=0.75)
        v2 = build_bm25_vectors(texts, k1=2.0, b=0.0)
        assert v1[1]['values'] != v2[1]['values']

    def test_word_forms_same_vector_index(self):
        """Разные формы слова → одинаковый индекс в BM25 векторе."""
        v1 = build_bm25_vectors(['документов'])
        v2 = build_bm25_vectors(['документы'])
        assert v1[0]['indices'] == v2[0]['indices']
