from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morag.indexing.embedder import (
    Embedder,
    HttpEmbedder,
    HttpGteSparseEmbedder,
    SparseEmbedder,
    _word_to_index,
)


# ---------------------------------------------------------------------------
# HttpEmbedder (generic, AsyncOpenAI SDK)
# ---------------------------------------------------------------------------

_FRIDA_DOC_TPL = 'search_document: {text}'
_FRIDA_QRY_TPL = 'search_query: {text}'


def make_mock_openai(embeddings: list[list[float]]) -> MagicMock:
    """Создать замоканный AsyncOpenAI с заданным ответом embeddings.create."""
    items = [
        MagicMock(index=i, embedding=emb)
        for i, emb in enumerate(embeddings)
    ]
    resp = MagicMock(data=items)
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=resp)
    return mock_client


def make_http_embedder(
    base_url: str = 'http://localhost:8082',
    model: str = 'qwen3-embedding:4b',
    dim: int = 2560,
    document_template: str = _FRIDA_DOC_TPL,
    query_template: str = _FRIDA_QRY_TPL,
) -> HttpEmbedder:
    """Создать HttpEmbedder с замоканным AsyncOpenAI."""
    with patch('openai.AsyncOpenAI'):
        return HttpEmbedder(
            base_url=base_url, model=model, dim=dim,
            document_template=document_template,
            query_template=query_template,
            timeout=10,
        )


class TestHttpEmbedder:
    def test_is_embedder(self):
        embedder = make_http_embedder()
        assert isinstance(embedder, Embedder)

    def test_dim_returns_configured_value(self):
        embedder = make_http_embedder(dim=2560)
        assert embedder.dim == 2560

    async def test_embed_returns_list_of_floats(self):
        embedder = make_http_embedder()
        embedder._client = make_mock_openai([[0.1] * 2560])

        result = await embedder.embed('тестовый текст')
        assert isinstance(result, list)
        assert len(result) == 2560
        assert all(isinstance(v, float) for v in result)

    async def test_embed_applies_document_template(self):
        embedder = make_http_embedder()
        embedder._client = make_mock_openai([[0.1] * 2560])

        await embedder.embed('мой текст')
        call = embedder._client.embeddings.create.call_args
        assert call.kwargs['input'] == 'search_document: мой текст'

    async def test_embed_query_applies_query_template(self):
        embedder = make_http_embedder()
        embedder._client = make_mock_openai([[0.1] * 2560])

        await embedder.embed_query('мой запрос')
        call = embedder._client.embeddings.create.call_args
        assert call.kwargs['input'] == 'search_query: мой запрос'

    async def test_qwen3_style_instruction_template(self):
        """Qwen3 использует многострочный template 'Instruct: ...\\nQuery:{text}'."""
        qwen_qry = 'Instruct: Given a question\nQuery:{text}'
        embedder = make_http_embedder(
            document_template='{text}',
            query_template=qwen_qry,
        )
        embedder._client = make_mock_openai([[0.1] * 2560])

        await embedder.embed('голый документ')
        assert embedder._client.embeddings.create.call_args.kwargs['input'] == 'голый документ'

        await embedder.embed_query('что такое X?')
        assert embedder._client.embeddings.create.call_args.kwargs['input'] == (
            'Instruct: Given a question\nQuery:что такое X?'
        )

    async def test_default_templates_pass_text_through(self):
        """Без template — отправляется голый text."""
        with patch('openai.AsyncOpenAI'):
            embedder = HttpEmbedder('http://localhost:8082', model='m', dim=4)
        embedder._client = make_mock_openai([[0.1, 0.2, 0.3, 0.4]])

        await embedder.embed('foo')
        assert embedder._client.embeddings.create.call_args.kwargs['input'] == 'foo'
        await embedder.embed_query('bar')
        assert embedder._client.embeddings.create.call_args.kwargs['input'] == 'bar'

    async def test_sends_model_in_body(self):
        with patch('openai.AsyncOpenAI'):
            embedder = HttpEmbedder('http://localhost:11434', model='qwen3-embedding:4b', dim=4)
        embedder._client = make_mock_openai([[0.1, 0.2, 0.3, 0.4]])

        await embedder.embed('foo')
        assert embedder._client.embeddings.create.call_args.kwargs['model'] == 'qwen3-embedding:4b'

    async def test_raises_on_openai_error(self):
        embedder = make_http_embedder()
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(side_effect=RuntimeError('boom'))
        embedder._client = mock_client

        with pytest.raises(RuntimeError, match='boom'):
            await embedder.embed('текст')

    async def test_embed_batch_sends_list_input(self):
        with patch('openai.AsyncOpenAI'):
            embedder = HttpEmbedder(
                'http://localhost:8082', model='m', dim=4,
                document_template=_FRIDA_DOC_TPL,
            )
        embedder._client = make_mock_openai([[0.1] * 4, [0.2] * 4])

        result = await embedder.embed_batch(['текст один', 'текст два'])
        assert len(result) == 2
        assert result[0] == [0.1] * 4
        assert result[1] == [0.2] * 4
        sent_input = embedder._client.embeddings.create.call_args.kwargs['input']
        assert isinstance(sent_input, list)
        assert len(sent_input) == 2
        assert all(t.startswith('search_document: ') for t in sent_input)

    async def test_embed_batch_sorts_by_index(self):
        """Результаты должны быть упорядочены по index, даже если сервер вернул не по порядку."""
        with patch('openai.AsyncOpenAI'):
            embedder = HttpEmbedder('http://localhost:8082', model='m', dim=1)
        # Симулируем ответ в обратном порядке
        items = [MagicMock(index=1, embedding=[0.2]), MagicMock(index=0, embedding=[0.1])]
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=MagicMock(data=items))
        embedder._client = mock_client

        result = await embedder.embed_batch(['а', 'б'])
        assert result == [[0.1], [0.2]]

    async def test_embed_batch_empty(self):
        embedder = make_http_embedder()
        assert await embedder.embed_batch([]) == []


class TestHttpEmbedderInflightCap:
    def test_no_semaphore_by_default(self):
        from morag.llm.client import _reset_shared_semaphores
        _reset_shared_semaphores()
        with patch('openai.AsyncOpenAI'):
            embedder = HttpEmbedder('http://emb', model='m', dim=4)
        assert embedder._semaphore is None

    def test_semaphore_created_with_max_concurrent(self):
        from morag.llm.client import _reset_shared_semaphores
        _reset_shared_semaphores()
        with patch('openai.AsyncOpenAI'):
            embedder = HttpEmbedder('http://emb', model='m1', dim=4, max_concurrent=2)
        assert embedder._semaphore is not None

    def test_shared_with_llm_client_on_same_key(self):
        """Embedder и LLMClient на одном (base_url, model) делят семафор."""
        from morag.llm.client import LLMClient, _reset_shared_semaphores
        _reset_shared_semaphores()
        with patch('openai.AsyncOpenAI'):
            llm = LLMClient(base_url='http://shared', model='m-x', max_concurrent=4)
            emb = HttpEmbedder('http://shared', model='m-x', dim=4, max_concurrent=4)
        assert llm._semaphore is emb._semaphore


# ---------------------------------------------------------------------------
# HttpGteSparseEmbedder (httpx.AsyncClient)
# ---------------------------------------------------------------------------

def _mock_encode_response(token_weights: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {'token_weights': [token_weights]}
    return resp


def make_sparse_embedder() -> HttpGteSparseEmbedder:
    with patch('httpx.AsyncClient'):
        return HttpGteSparseEmbedder('http://localhost:8081')


class TestHttpGteSparseEmbedder:
    def test_is_sparse_embedder(self):
        assert isinstance(make_sparse_embedder(), SparseEmbedder)

    async def test_embed_returns_tuple(self):
        embedder = make_sparse_embedder()
        embedder._client.post = AsyncMock(
            return_value=_mock_encode_response({'python': 0.8, 'код': 0.5}),
        )

        result = await embedder.embed('python код')
        assert isinstance(result, tuple)
        assert len(result) == 2

    async def test_embed_returns_indices_and_values(self):
        embedder = make_sparse_embedder()
        embedder._client.post = AsyncMock(
            return_value=_mock_encode_response({'слово': 0.7}),
        )

        indices, values = await embedder.embed('слово')
        assert isinstance(indices, list)
        assert isinstance(values, list)
        assert len(indices) == len(values) == 1

    async def test_embed_indices_are_ints(self):
        embedder = make_sparse_embedder()
        embedder._client.post = AsyncMock(
            return_value=_mock_encode_response({'hello': 0.9, 'world': 0.3}),
        )

        indices, _ = await embedder.embed('hello world')
        assert all(isinstance(i, int) for i in indices)

    async def test_embed_values_are_floats(self):
        embedder = make_sparse_embedder()
        embedder._client.post = AsyncMock(
            return_value=_mock_encode_response({'hello': 0.9}),
        )

        _, values = await embedder.embed('hello')
        assert all(isinstance(v, float) for v in values)

    async def test_indices_use_word_to_index_hash(self):
        """Индексы должны совпадать с результатами _word_to_index."""
        embedder = make_sparse_embedder()
        embedder._client.post = AsyncMock(
            return_value=_mock_encode_response({'python': 0.8, 'код': 0.5}),
        )

        indices, values = await embedder.embed('python код')
        word_index_map = dict(zip(indices, values))
        assert _word_to_index('python') in word_index_map
        assert _word_to_index('код') in word_index_map

    async def test_hash_collision_keeps_max_weight(self):
        """При коллизии хэшей двух разных слов берётся максимальный вес."""
        word_a = 'hello'
        word_b = 'world'
        with patch('morag.indexing.embedder._word_to_index', return_value=42):
            embedder = make_sparse_embedder()
            embedder._client.post = AsyncMock(
                return_value=_mock_encode_response({word_a: 0.3, word_b: 0.9}),
            )
            indices, values = await embedder.embed('hello world')

        assert indices == [42]
        assert values == [pytest.approx(0.9)]

    async def test_calls_encode_endpoint(self):
        embedder = make_sparse_embedder()
        embedder._client.post = AsyncMock(
            return_value=_mock_encode_response({'слово': 0.7}),
        )

        await embedder.embed('слово')
        path = embedder._client.post.call_args[0][0]
        assert path == '/encode'

    async def test_embed_query_same_as_embed(self):
        """Sparse-эмбеддер не различает документ и запрос."""
        embedder = make_sparse_embedder()
        embedder._client.post = AsyncMock(
            return_value=_mock_encode_response({'слово': 0.5}),
        )

        indices_doc, values_doc = await embedder.embed('слово')
        indices_query, values_query = await embedder.embed_query('слово')
        assert indices_doc == indices_query
        assert values_doc == values_query

    async def test_raises_on_http_error(self):
        import httpx

        embedder = make_sparse_embedder()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            'error', request=MagicMock(), response=MagicMock(),
        )
        embedder._client.post = AsyncMock(return_value=mock_resp)

        with pytest.raises(httpx.HTTPStatusError):
            await embedder.embed('текст')

    async def test_embed_batch_calls_encode_batch_endpoint(self):
        embedder = make_sparse_embedder()
        batch_resp = MagicMock()
        batch_resp.json.return_value = {
            'token_weights': [
                {'python': 0.8, 'код': 0.5},
                {'java': 0.7},
            ],
        }
        embedder._client.post = AsyncMock(return_value=batch_resp)

        result = await embedder.embed_batch(['python код', 'java'])
        assert len(result) == 2
        indices_0, _ = result[0]
        assert _word_to_index('python') in indices_0
        assert _word_to_index('код') in indices_0
        indices_1, _ = result[1]
        assert _word_to_index('java') in indices_1
        assert embedder._client.post.call_args[0][0] == '/encode_batch'
        assert embedder._client.post.call_args[1]['json']['texts'] == ['python код', 'java']

    async def test_embed_batch_empty(self):
        embedder = make_sparse_embedder()
        assert await embedder.embed_batch([]) == []


# ---------------------------------------------------------------------------
# DenseEmbedderConfig и SparseEmbedderConfig — валидация
# ---------------------------------------------------------------------------

class TestDenseEmbedderConfig:
    def test_default_model(self):
        from morag.config import DenseEmbedderConfig
        cfg = DenseEmbedderConfig()
        assert cfg.base_url is None
        assert cfg.model == 'qwen3-embedding:4b'

    def test_http_mode_requires_dim(self):
        from pydantic import ValidationError
        from morag.config import DenseEmbedderConfig
        with pytest.raises(ValidationError, match='dim'):
            DenseEmbedderConfig(base_url='http://localhost:8082')

    def test_http_mode_valid_with_dim(self):
        from morag.config import DenseEmbedderConfig
        cfg = DenseEmbedderConfig(base_url='http://localhost:8082', dim=1536)
        assert cfg.base_url == 'http://localhost:8082'
        assert cfg.dim == 1536

    def test_local_mode_dim_not_required(self):
        from morag.config import DenseEmbedderConfig
        cfg = DenseEmbedderConfig(model='ai-forever/FRIDA')
        assert cfg.dim is None

    def test_default_timeout(self):
        from morag.config import DenseEmbedderConfig
        cfg = DenseEmbedderConfig()
        assert cfg.timeout == 30


class TestSparseEmbedderConfig:
    def test_default_is_local_mode(self):
        from morag.config import SparseEmbedderConfig
        cfg = SparseEmbedderConfig()
        assert cfg.base_url is None
        assert cfg.model == 'Alibaba-NLP/gte-multilingual-base'

    def test_http_mode_no_dim_required(self):
        from morag.config import SparseEmbedderConfig
        cfg = SparseEmbedderConfig(base_url='http://localhost:8081')
        assert cfg.base_url == 'http://localhost:8081'

    def test_default_timeout(self):
        from morag.config import SparseEmbedderConfig
        cfg = SparseEmbedderConfig()
        assert cfg.timeout == 30
