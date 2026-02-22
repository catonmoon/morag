from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from morag.indexing.embedder import Embedder, FridaEmbedder, GteSparseEmbedder, SparseEmbedder, _word_to_index


def make_frida(model_name: str = 'ai-forever/FRIDA') -> tuple[FridaEmbedder, MagicMock]:
    """Создать FridaEmbedder с заглушкой SentenceTransformer."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda text, **_: np.zeros(FridaEmbedder.DIM, dtype=np.float32)
    with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
        embedder = FridaEmbedder(model_name)
    return embedder, mock_model


class TestFridaEmbedder:
    def test_is_embedder(self):
        embedder, _ = make_frida()
        assert isinstance(embedder, Embedder)

    def test_dim_is_1024(self):
        embedder, _ = make_frida()
        assert embedder.dim == 1024

    def test_embed_returns_list_of_floats(self):
        embedder, _ = make_frida()
        result = embedder.embed('тестовый текст')
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embed_returns_correct_length(self):
        embedder, _ = make_frida()
        result = embedder.embed('тестовый текст')
        assert len(result) == FridaEmbedder.DIM

    def test_embed_query_returns_correct_length(self):
        embedder, _ = make_frida()
        result = embedder.embed_query('поисковый запрос')
        assert len(result) == FridaEmbedder.DIM

    def test_embed_uses_document_prefix(self):
        embedder, mock_model = make_frida()
        embedder.embed('мой текст')
        call_text = mock_model.encode.call_args[0][0]
        assert call_text.startswith('search_document: ')
        assert 'мой текст' in call_text

    def test_embed_query_uses_query_prefix(self):
        embedder, mock_model = make_frida()
        embedder.embed_query('мой запрос')
        call_text = mock_model.encode.call_args[0][0]
        assert call_text.startswith('search_query: ')
        assert 'мой запрос' in call_text

    def test_embed_and_embed_query_use_different_prefixes(self):
        embedder, mock_model = make_frida()
        embedder.embed('текст')
        doc_text = mock_model.encode.call_args[0][0]
        embedder.embed_query('текст')
        query_text = mock_model.encode.call_args[0][0]
        assert doc_text != query_text

    def test_loads_model_with_given_name(self):
        with patch('sentence_transformers.SentenceTransformer') as mock_cls:
            mock_cls.return_value = MagicMock()
            mock_cls.return_value.encode.return_value = np.zeros(FridaEmbedder.DIM, dtype=np.float32)
            FridaEmbedder('custom/model')
            mock_cls.assert_called_once_with('custom/model')

    def test_normalize_embeddings_is_false(self):
        """MVP использует ненормализованные эмбеддинги."""
        embedder, mock_model = make_frida()
        embedder.embed('текст')
        _, kwargs = mock_model.encode.call_args
        assert kwargs.get('normalize_embeddings') is False


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


# ---------------------------------------------------------------------------
# GteSparseEmbedder — тесты через моки (без загрузки реальной модели)
# ---------------------------------------------------------------------------

def make_gte(model_name: str = 'Alibaba-NLP/gte-multilingual-base'):
    """Создать GteSparseEmbedder с замоканными tokenizer и model."""
    import torch

    mock_tokenizer = MagicMock()
    mock_tokenizer.cls_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.pad_token_id = 2
    mock_tokenizer.unk_token_id = 3
    mock_tokenizer.decode.side_effect = lambda ids, **_: 'word' if ids[0] not in {0, 1, 2, 3} else ''

    # logits shape: [1, seq_len, 1] → squeeze → [seq_len]
    # симулируем 4 токена: [CLS]=0, tok1=4 (w=0.5), tok2=5 (w=0.0 — фильтруем), [PAD]=2
    def fake_tokenizer_call(text, **_):
        mock_enc = MagicMock()
        mock_enc.__getitem__ = lambda self, k: {
            'input_ids': torch.tensor([[0, 4, 5, 2]]),
        }[k]
        mock_enc.items.return_value = [
            ('input_ids', torch.tensor([[0, 4, 5, 2]])),
            ('attention_mask', torch.ones(1, 4, dtype=torch.long)),
        ]
        return mock_enc

    mock_tokenizer.side_effect = fake_tokenizer_call

    mock_model = MagicMock()
    mock_model.device = torch.device('cpu')
    logits = torch.tensor([[[0.5], [0.5], [0.0], [0.0]]])  # shape [1, 4, 1]
    mock_model_out = MagicMock()
    mock_model_out.logits = logits
    mock_model.return_value = mock_model_out

    with (
        patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer),
        patch('transformers.AutoModelForTokenClassification.from_pretrained', return_value=mock_model),
    ):
        embedder = GteSparseEmbedder(model_name)
    embedder._tokenizer = mock_tokenizer
    embedder._model = mock_model
    return embedder


class TestGteSparseEmbedder:
    def test_is_sparse_embedder(self):
        embedder = make_gte()
        assert isinstance(embedder, SparseEmbedder)

    def test_embed_returns_tuple(self):
        embedder = make_gte()
        result = embedder.embed('текст')
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_embed_returns_indices_and_values(self):
        embedder = make_gte()
        indices, values = embedder.embed('текст')
        assert isinstance(indices, list)
        assert isinstance(values, list)
        assert len(indices) == len(values)

    def test_embed_indices_are_ints(self):
        embedder = make_gte()
        indices, _ = embedder.embed('текст')
        assert all(isinstance(i, int) for i in indices)

    def test_embed_values_are_floats(self):
        embedder = make_gte()
        _, values = embedder.embed('текст')
        assert all(isinstance(v, float) for v in values)

    def test_embed_values_are_positive(self):
        """ReLU гарантирует неотрицательность, нулевые фильтруются."""
        embedder = make_gte()
        _, values = embedder.embed('текст')
        assert all(v > 0 for v in values)

    def test_embed_query_same_signature(self):
        embedder = make_gte()
        indices, values = embedder.embed_query('запрос')
        assert isinstance(indices, list)
        assert isinstance(values, list)

    def test_no_special_tokens_in_output(self):
        """Спец-токены (CLS, PAD, EOS, UNK) не попадают в результат.

        Mock: 4 токена — [CLS]=0, tok=4, zero_tok=5, [PAD]=2.
        CLS и PAD в unused_tokens, tok=5 отфильтрован по weight=0.
        Должен остаться ровно 1 токен.
        """
        embedder = make_gte()
        indices, values = embedder.embed('текст')
        assert len(values) == 1

    def test_deduplication_keeps_max_weight(self):
        """Если одно слово встречается дважды, берётся максимальный вес, а не сумма."""
        import torch

        # Два токена с одинаковым decode → одно слово, веса 0.3 и 0.9
        mock_tokenizer = MagicMock()
        mock_tokenizer.cls_token_id = 0
        mock_tokenizer.eos_token_id = None
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.unk_token_id = None
        mock_tokenizer.decode.return_value = 'word'  # оба токена → одно слово

        def fake_call(text, **_):
            enc = MagicMock()
            enc.items.return_value = [('input_ids', torch.tensor([[4, 5]]))]
            enc.__getitem__ = lambda s, k: torch.tensor([[4, 5]])
            return enc

        mock_tokenizer.side_effect = fake_call

        mock_model = MagicMock()
        mock_model.device = torch.device('cpu')
        out = MagicMock()
        out.logits = torch.tensor([[[0.3], [0.9]]])  # tok4=0.3, tok5=0.9
        mock_model.return_value = out

        with (
            patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer),
            patch('transformers.AutoModelForTokenClassification.from_pretrained', return_value=mock_model),
        ):
            embedder = GteSparseEmbedder()
        embedder._tokenizer = mock_tokenizer
        embedder._model = mock_model

        indices, values = embedder.embed('word word')

        assert len(indices) == 1          # одно уникальное слово
        assert len(values) == 1
        assert values[0] == pytest.approx(0.9)  # max, не сумма (1.2) и не первый (0.3)

    def test_loads_with_trust_remote_code(self):
        """trust_remote_code=True обязателен для GTE."""
        with (
            patch('transformers.AutoTokenizer.from_pretrained') as mock_tok,
            patch('transformers.AutoModelForTokenClassification.from_pretrained') as mock_model_cls,
        ):
            mock_tok.return_value = MagicMock(
                cls_token_id=0, eos_token_id=1, pad_token_id=2, unk_token_id=3,
            )
            mock_model_cls.return_value = MagicMock()
            GteSparseEmbedder('some/model')
            _, kwargs = mock_model_cls.call_args
            assert kwargs.get('trust_remote_code') is True
