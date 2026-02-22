from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from morag.indexing.embedder import Embedder, FridaEmbedder


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
