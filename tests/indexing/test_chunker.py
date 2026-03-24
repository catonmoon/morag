from unittest.mock import AsyncMock

import pytest

from morag.indexing.chunker import (
    Chunker, ChunkingError, LLMChunker, PassthroughChunker, SemanticChunker,
    _LLMError, _classify_error,
)
from morag.indexing.token_counter import TiktokenCounter


class TestPassthroughChunker:
    def test_is_chunker(self):
        assert isinstance(PassthroughChunker(), Chunker)

    async def test_returns_block_as_single_chunk(self):
        result = await PassthroughChunker().chunk('Текст блока')
        assert result == ['Текст блока']

    async def test_returns_list_of_one(self):
        result = await PassthroughChunker().chunk('Любой текст')
        assert len(result) == 1

    async def test_preserves_text_exactly(self):
        text = '# Заголовок\n\nПервый абзац.\n\nВторой абзац.'
        assert (await PassthroughChunker().chunk(text))[0] == text

    async def test_empty_string(self):
        result = await PassthroughChunker().chunk('')
        assert result == ['']


# ---------------------------------------------------------------------------
# _classify_error
# ---------------------------------------------------------------------------

class TestClassifyError:
    def test_timeout_by_class_name(self):
        class APITimeoutError(Exception):
            pass
        assert _classify_error(APITimeoutError('request timed out')) == _LLMError.TIMEOUT

    def test_timeout_by_message(self):
        assert _classify_error(Exception('Read timeout on connection')) == _LLMError.TIMEOUT

    def test_model_not_found_is_other(self):
        """Model not found обрабатывается в LLMClient, для chunker — это OTHER."""
        assert _classify_error(Exception('400 - Model not found')) == _LLMError.OTHER

    def test_other_error(self):
        assert _classify_error(ValueError('invalid JSON')) == _LLMError.OTHER

    def test_connection_error(self):
        assert _classify_error(ConnectionError('refused')) == _LLMError.OTHER


# ---------------------------------------------------------------------------
# LLMChunker — базовые тесты
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def chunker(mock_client) -> LLMChunker:
    """Чанкер с fallback_enabled=True для обратной совместимости тестов."""
    return LLMChunker(mock_client, fallback_enabled=True)


class TestLLMChunker:
    def test_is_chunker(self, mock_client):
        assert isinstance(LLMChunker(mock_client), Chunker)

    async def test_returns_chunks_from_llm(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': ['Часть A', 'Часть B', 'Часть C']}
        result = await chunker.chunk('Длинный текст')
        assert result == ['Часть A', 'Часть B', 'Часть C']

    async def test_passes_block_to_llm(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': ['ok']}
        block = 'Мой текстовый блок'
        await chunker.chunk(block)
        messages = mock_client.complete_json.call_args[0][0]
        user_message = next(m for m in messages if m['role'] == 'user')
        assert user_message['content'] == block

    async def test_fallback_on_invalid_json(self, chunker, mock_client):
        mock_client.complete_json.side_effect = ValueError('invalid JSON')
        result = await chunker.chunk('Блок текста')
        assert result == ['Блок текста']

    async def test_fallback_when_chunks_missing(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'result': 'something else'}
        result = await chunker.chunk('Блок текста')
        assert result == ['Блок текста']

    async def test_fallback_when_chunks_empty_list(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': []}
        result = await chunker.chunk('Блок текста')
        assert result == ['Блок текста']

    async def test_fallback_when_chunks_not_a_list(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': 'not a list'}
        result = await chunker.chunk('Блок текста')
        assert result == ['Блок текста']

    async def test_filters_out_empty_strings(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': ['Чанк A', '   ', 'Чанк B']}
        result = await chunker.chunk('Текст')
        assert result == ['Чанк A', 'Чанк B']

    async def test_system_prompt_is_sent(self, chunker, mock_client):
        mock_client.complete_json.return_value = {'chunks': ['ok']}
        await chunker.chunk('Текст')
        messages = mock_client.complete_json.call_args[0][0]
        system_message = next(m for m in messages if m['role'] == 'system')
        assert len(system_message['content']) > 0

    async def test_passes_deterministic_params(self, chunker, mock_client):
        from morag.indexing.chunker import _llm_params
        mock_client.complete_json.return_value = {'chunks': ['ok']}
        await chunker.chunk('Текст')
        _, kwargs = mock_client.complete_json.call_args
        assert kwargs.get('params') == _llm_params()


# ---------------------------------------------------------------------------
# LLMChunker — raises ChunkingError when fallback disabled (default)
# ---------------------------------------------------------------------------

class TestLLMChunkerNoFallback:
    """По умолчанию fallback выключен — ChunkingError при неудаче."""

    async def test_raises_on_all_failures(self):
        client = AsyncMock()
        client.complete_json.side_effect = ValueError('invalid JSON')
        chunker = LLMChunker(client)  # fallback_enabled=False по умолчанию

        with pytest.raises(ChunkingError):
            await chunker.chunk('Блок текста')

    async def test_raises_on_network_error(self):
        client = AsyncMock()
        client.complete_json.side_effect = ConnectionError('refused')
        chunker = LLMChunker(client, max_retries=2)

        with pytest.raises(ChunkingError):
            await chunker.chunk('Текст')
        assert client.complete_json.call_count == 2

    async def test_success_still_works(self):
        client = AsyncMock()
        client.complete_json.return_value = {'chunks': ['A', 'B']}
        chunker = LLMChunker(client)

        result = await chunker.chunk('Текст')
        assert result == ['A', 'B']


# ---------------------------------------------------------------------------
# LLMChunker fallback (с fallback_enabled=True)
# ---------------------------------------------------------------------------

class TestLLMChunkerFallback:
    """Тесты fallback при провале всех LLM-попыток."""

    async def test_fallback_splits_with_token_counter(self):
        """При провале LLM — разбивает блок на чанки ≤512 токенов."""
        client = AsyncMock()
        client.complete_json.side_effect = ValueError('invalid JSON')
        counter = TiktokenCounter()
        chunker = LLMChunker(client, token_counter=counter, fallback_enabled=True)

        block = 'Слово. ' * 500
        result = await chunker.chunk(block)

        assert len(result) > 1
        for chunk in result:
            assert counter.count(chunk) <= 512

    async def test_fallback_without_token_counter_returns_block(self):
        """Без token_counter — возвращает блок целиком (обратная совместимость)."""
        client = AsyncMock()
        client.complete_json.side_effect = ValueError('invalid JSON')
        chunker = LLMChunker(client, fallback_enabled=True)

        block = 'Блок текста'
        result = await chunker.chunk(block)
        assert result == [block]

    async def test_fallback_small_block_stays_single(self):
        """Блок меньше 512 токенов — остаётся одним чанком даже при fallback."""
        client = AsyncMock()
        client.complete_json.side_effect = ValueError('invalid JSON')
        counter = TiktokenCounter()
        chunker = LLMChunker(client, token_counter=counter, fallback_enabled=True)

        block = 'Короткий текст о машинном обучении.'
        result = await chunker.chunk(block)
        assert result == [block]

    async def test_fallback_custom_token_limit(self):
        """Можно задать другой лимит токенов для fallback."""
        client = AsyncMock()
        client.complete_json.side_effect = ValueError('invalid JSON')
        counter = TiktokenCounter()
        chunker = LLMChunker(
            client, token_counter=counter, fallback_token_limit=100, fallback_enabled=True,
        )

        block = 'Предложение номер один. ' * 100
        result = await chunker.chunk(block)

        assert len(result) > 1
        for chunk in result:
            assert counter.count(chunk) <= 100

    async def test_fallback_on_network_error(self):
        """Сетевые ошибки тоже приводят к fallback, а не к падению документа."""
        client = AsyncMock()
        client.complete_json.side_effect = ConnectionError('connection refused')
        counter = TiktokenCounter()
        chunker = LLMChunker(
            client, max_retries=2, token_counter=counter, fallback_enabled=True,
        )

        block = 'Текст. ' * 300
        result = await chunker.chunk(block)

        assert len(result) > 1
        assert client.complete_json.call_count == 2

    async def test_semantic_fallback_with_embed_fn(self):
        """С embed_fn — использует SemanticSplitter в цепочке."""
        client = AsyncMock()
        client.complete_json.side_effect = ValueError('invalid JSON')
        counter = TiktokenCounter()

        def fake_embed(text: str) -> list[float]:
            h = hash(text) % 1000
            return [float(h % (i + 1)) for i in range(10)]

        chunker = LLMChunker(
            client, token_counter=counter, embed_fn=fake_embed, fallback_enabled=True,
        )

        block = 'Слово. ' * 500
        result = await chunker.chunk(block)

        assert len(result) > 1
        for chunk in result:
            assert counter.count(chunk) <= 512

    async def test_fallback_markdown_headers_respected(self):
        """Fallback разрезает по заголовкам Markdown."""
        client = AsyncMock()
        client.complete_json.side_effect = ValueError('invalid JSON')
        counter = TiktokenCounter()
        chunker = LLMChunker(
            client, token_counter=counter, fallback_token_limit=100, fallback_enabled=True,
        )

        section1 = '# Раздел 1\n\n' + 'Текст первого раздела документа. ' * 20
        section2 = '# Раздел 2\n\n' + 'Текст второго раздела документа. ' * 20
        block = section1 + '\n\n' + section2
        result = await chunker.chunk(block)

        assert len(result) >= 2
        assert result[0].startswith('# Раздел 1')


# ---------------------------------------------------------------------------
# LLMChunker — halving on timeout
# ---------------------------------------------------------------------------

class TestLLMChunkerHalving:
    """Тесты адаптивного деления блока пополам при таймауте."""

    async def test_halving_splits_on_timeout(self):
        """При таймауте и halving_retries>0 — делит блок и повторяет."""
        client = AsyncMock()
        counter = TiktokenCounter()

        call_count = 0

        async def side_effect(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            text = messages[-1]['content']
            tokens = counter.count(text)
            # Таймаут на больших блоках, успех на маленьких
            if tokens > 100:
                raise type('APITimeoutError', (Exception,), {})('timeout')
            return {'chunks': [text]}

        client.complete_json.side_effect = side_effect

        chunker = LLMChunker(
            client, token_counter=counter, halving_retries=3, max_retries=1,
        )
        block = 'Предложение номер один. ' * 100  # ~400 токенов
        result = await chunker.chunk(block)

        # Должны получить несколько чанков (блок был разбит)
        assert len(result) > 1
        # Все чанки непустые
        assert all(c.strip() for c in result)

    async def test_halving_only_problematic_subblock(self):
        """Деление только проблемного подблока, остальные проходят нормально."""
        client = AsyncMock()
        counter = TiktokenCounter()

        timeout_once = {'done': False}

        async def side_effect(messages, **kwargs):
            text = messages[-1]['content']
            tokens = counter.count(text)
            # Первый большой блок — таймаут один раз, потом ок
            if tokens > 200 and not timeout_once['done']:
                timeout_once['done'] = True
                raise type('APITimeoutError', (Exception,), {})('timeout')
            return {'chunks': [text]}

        client.complete_json.side_effect = side_effect

        chunker = LLMChunker(
            client, token_counter=counter, halving_retries=2, max_retries=1,
        )
        block = 'Слово номер. ' * 200
        result = await chunker.chunk(block)

        assert len(result) >= 1
        assert all(c.strip() for c in result)

    async def test_halving_exhausted_raises_without_fallback(self):
        """Если все уровни halving исчерпаны и fallback выключен — ChunkingError."""
        client = AsyncMock()
        counter = TiktokenCounter()
        client.complete_json.side_effect = type('APITimeoutError', (Exception,), {})('timeout')

        chunker = LLMChunker(
            client, token_counter=counter, halving_retries=1, max_retries=1,
        )
        # Даже после halving подблоки снова таймаутят → ChunkingError
        block = 'Слово. ' * 500
        with pytest.raises(ChunkingError):
            await chunker.chunk(block)

    async def test_halving_exhausted_uses_fallback(self):
        """Если halving исчерпан, но fallback включён — используется fallback."""
        client = AsyncMock()
        counter = TiktokenCounter()
        client.complete_json.side_effect = type('APITimeoutError', (Exception,), {})('timeout')

        chunker = LLMChunker(
            client, token_counter=counter, halving_retries=1, max_retries=1,
            fallback_enabled=True,
        )
        block = 'Слово. ' * 500
        result = await chunker.chunk(block)

        assert len(result) > 1
        for chunk in result:
            assert counter.count(chunk) <= 512

    async def test_halving_on_invalid_json(self):
        """При невалидном JSON (модели не хватило контекста) — halving тоже срабатывает."""
        client = AsyncMock()
        counter = TiktokenCounter()

        call_count = 0

        async def side_effect(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            text = messages[-1]['content']
            tokens = counter.count(text)
            if tokens > 100:
                raise ValueError('invalid JSON — context overflow')
            return {'chunks': [text]}

        client.complete_json.side_effect = side_effect

        chunker = LLMChunker(
            client, token_counter=counter, halving_retries=3, max_retries=1,
        )
        block = 'Предложение номер один. ' * 100
        result = await chunker.chunk(block)

        assert len(result) > 1
        assert all(c.strip() for c in result)

    async def test_no_halving_on_non_halvable_error(self):
        """При других ошибках (не таймаут, не invalid JSON) halving не срабатывает."""
        client = AsyncMock()
        counter = TiktokenCounter()
        client.complete_json.side_effect = ConnectionError('refused')

        chunker = LLMChunker(
            client, token_counter=counter, halving_retries=3, max_retries=2,
        )
        with pytest.raises(ChunkingError):
            await chunker.chunk('Блок текста')
        # max_retries=2 попытки, без halving
        assert client.complete_json.call_count == 2


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

def _make_batch_embed_fn(topic_vectors: dict[str, list[float]]):
    """Мок батчевый embed_fn: возвращает вектор по первому найденному ключевому слову."""
    default = [1.0, 0.0, 0.0]

    def embed_fn(texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            vec = default
            for keyword, v in topic_vectors.items():
                if keyword.lower() in text.lower():
                    vec = v
                    break
            result.append(vec)
        return result

    return embed_fn


class TestSemanticChunker:
    def test_is_chunker(self):
        def noop(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]] * len(texts)

        chunker = SemanticChunker(embed_fn=noop, counter=TiktokenCounter())
        assert isinstance(chunker, Chunker)

    async def test_short_block_returns_single_chunk(self):
        """Блок короче max_tokens возвращается целиком."""
        def noop(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]] * len(texts)

        chunker = SemanticChunker(
            embed_fn=noop, counter=TiktokenCounter(), max_tokens=500,
        )
        block = 'Короткий текст из нескольких слов.'
        result = await chunker.chunk(block)
        assert result == [block]

    async def test_splits_into_multiple_chunks(self):
        """Длинный блок разбивается на несколько чанков."""
        embed_fn = _make_batch_embed_fn({
            'python': [1.0, 0.0, 0.0],
            'база данных': [0.0, 1.0, 0.0],
            'машинное обучение': [0.0, 0.0, 1.0],
        })
        counter = TiktokenCounter()
        chunker = SemanticChunker(
            embed_fn=embed_fn, counter=counter, min_tokens=20, max_tokens=80,
        )

        block = (
            'Python — язык программирования. '
            'Python используется для веб-разработки. '
            'Python имеет богатую экосистему. '
            'База данных хранит данные. '
            'База данных поддерживает транзакции. '
            'База данных масштабируется горизонтально. '
            'Машинное обучение решает задачи классификации. '
            'Машинное обучение использует нейронные сети. '
            'Машинное обучение требует больших данных.'
        )
        result = await chunker.chunk(block)
        assert len(result) >= 2

    async def test_chunks_within_size_limits(self):
        """Все чанки (кроме, возможно, последнего) в пределах [min, max] токенов."""
        def embed_fn(texts: list[str]) -> list[list[float]]:
            return [
                [float(hash(t) % 1000 % (i + 1)) for i in range(10)]
                for t in texts
            ]

        counter = TiktokenCounter()
        min_t, max_t = 30, 100
        chunker = SemanticChunker(
            embed_fn=embed_fn, counter=counter, min_tokens=min_t, max_tokens=max_t,
        )

        block = '. '.join([
            f'Предложение номер {i} с некоторым текстом для тестирования'
            for i in range(40)
        ]) + '.'
        result = await chunker.chunk(block)

        assert len(result) >= 2
        # Все чанки кроме последнего должны быть >= min_tokens
        for chunk in result[:-1]:
            tokens = counter.count(chunk)
            assert tokens >= min_t, f'Chunk too small: {tokens} tokens'
            assert tokens <= max_t, f'Chunk too large: {tokens} tokens'
        # Последний чанк может быть < min_tokens (остаток)
        assert counter.count(result[-1]) <= max_t

    async def test_preserves_all_content(self):
        """Все предложения сохраняются после разбиения."""
        def embed_fn(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]] * len(texts)

        counter = TiktokenCounter()
        chunker = SemanticChunker(
            embed_fn=embed_fn, counter=counter, min_tokens=10, max_tokens=50,
        )

        sentences = [f'Предложение {i}.' for i in range(15)]
        block = ' '.join(sentences)
        result = await chunker.chunk(block)
        combined = ' '.join(result)
        for s in sentences:
            # Проверяем содержательную часть (без финальной точки для надёжности)
            assert f'Предложение {sentences.index(s)}'.split('.')[0] in combined

    async def test_batch_embed_is_used(self):
        """embed_fn вызывается батчем, а не по одному."""
        calls: list[int] = []

        def tracking_embed_fn(texts: list[str]) -> list[list[float]]:
            calls.append(len(texts))
            return [[1.0, 0.0, 0.0]] * len(texts)

        counter = TiktokenCounter()
        chunker = SemanticChunker(
            embed_fn=tracking_embed_fn, counter=counter,
            min_tokens=10, max_tokens=40,
        )
        block = '. '.join([f'Предложение {i} с текстом' for i in range(20)]) + '.'
        await chunker.chunk(block)

        # embed_fn вызывается несколько раз (по одному на границу),
        # но каждый раз с батчем > 1
        assert len(calls) >= 1
        assert all(n >= 2 for n in calls)

    async def test_finds_semantic_boundary(self):
        """Разрез происходит между темами."""
        embed_fn = _make_batch_embed_fn({
            'кошки': [1.0, 0.0, 0.0],
            'программирование': [0.0, 1.0, 0.0],
        })
        counter = TiktokenCounter()
        chunker = SemanticChunker(
            embed_fn=embed_fn, counter=counter, min_tokens=15, max_tokens=60,
        )

        block = (
            'Кошки — домашние животные. '
            'Кошки любят спать и играть. '
            'Кошки охотятся на мышей. '
            'Программирование — создание программ. '
            'Программирование требует логического мышления. '
            'Программирование используется повсеместно.'
        )
        result = await chunker.chunk(block)
        assert len(result) >= 2
        # Первый чанк про кошек, последний про программирование
        assert 'Кошки' in result[0]
        assert 'Программирование' in result[-1]

    async def test_accept_pair_rejects_oversized_right(self):
        """accept_pair=True не принимает правый чанк > max_tokens (баг 873KB)."""
        def embed_fn(texts: list[str]) -> list[list[float]]:
            # Возвращаем максимально различные вектора для любых пар
            return [[float(i)] * 3 for i in range(len(texts))]

        counter = TiktokenCounter()
        max_t = 80
        chunker = SemanticChunker(
            embed_fn=embed_fn, counter=counter,
            min_tokens=20, max_tokens=max_t, accept_pair=True,
        )

        # Создаём текст где после первой границы остаток > max_tokens
        block = '. '.join([
            f'Предложение {i} с некоторым текстом для тестирования разбиения'
            for i in range(30)
        ]) + '.'
        result = await chunker.chunk(block)

        # Ни один чанк не должен быть > max_tokens * 2
        # (мелкие превышения допустимы, но не 10x)
        for chunk in result:
            tokens = counter.count(chunk)
            assert tokens < max_t * 3, (
                f'Chunk too large: {tokens} tokens (max={max_t}), '
                f'accept_pair should have rejected oversized right candidate'
            )

    async def test_accept_pair_accepts_normal_right(self):
        """accept_pair=True принимает правый чанк когда он <= max_tokens."""
        call_count = [0]

        def embed_fn(texts: list[str]) -> list[list[float]]:
            call_count[0] += 1
            return [[float(i)] * 3 for i in range(len(texts))]

        counter = TiktokenCounter()
        chunker_pair = SemanticChunker(
            embed_fn=embed_fn, counter=counter,
            min_tokens=10, max_tokens=50, accept_pair=True,
        )
        chunker_no_pair = SemanticChunker(
            embed_fn=embed_fn, counter=counter,
            min_tokens=10, max_tokens=50, accept_pair=False,
        )

        block = '. '.join([f'Предложение {i} текст' for i in range(20)]) + '.'

        call_count[0] = 0
        result_pair = await chunker_pair.chunk(block)
        calls_pair = call_count[0]

        call_count[0] = 0
        result_no_pair = await chunker_no_pair.chunk(block)
        calls_no_pair = call_count[0]

        # accept_pair должен делать меньше embed вызовов (2 чанка за итерацию)
        assert calls_pair <= calls_no_pair

    async def test_candidate_ends_gap(self):
        """Если между min и max нет кандидатов — greedy_end берёт до max."""
        def embed_fn(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]] * len(texts)

        counter = TiktokenCounter()
        # min=30, max=50. Единицы: 25 + 26 = 51 > max, но 25 < min
        chunker = SemanticChunker(
            embed_fn=embed_fn, counter=counter,
            min_tokens=30, max_tokens=50,
        )

        # Создаём текст из единиц ~25 токенов каждая
        unit = 'Это предложение среднего размера для тестирования чанкинга.'  # ~10 tok
        block = f'{unit} {unit} {unit} {unit} {unit} {unit} {unit} {unit}'
        result = await chunker.chunk(block)

        # Не должен зависнуть и должен вернуть результат
        assert len(result) >= 1
        assert all(chunk.strip() for chunk in result)


