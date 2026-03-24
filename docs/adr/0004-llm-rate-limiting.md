# ADR-0004: LLM Rate Limiting

## Status

Accepted (2026-03-21)

## Context

При массовой индексации с concurrency=12-32 через OpenRouter возникает thundering herd:
множество параллельных LLM-запросов превышают пропускную способность провайдера (~100 req/min
для qwen3.5-9b), вызывая "model unavailable" ошибки и 300-секундные паузы.

Проблема не решается только настройкой concurrency документов, потому что количество
LLM-запросов зависит от числа чанков: document concurrency × chunks per doc.

### Требования

- Ограничить RPM на уровне LLM-клиента, а не pipeline
- Не менять логику pipeline — документы обрабатываются независимо
- Конфигурируемый лимит через config.yml
- Опциональность — если max_rpm не задан, rate limiting отключён

## Alternatives Considered

### 1. aiolimiter (leaky bucket) — выбрано

```python
from aiolimiter import AsyncLimiter
limiter = AsyncLimiter(max_rpm, 60)
async with limiter:
    await llm_call()
```

**Pros:**
- Самая популярная async rate limiter библиотека
- Чистый API через async context manager
- Zero dependencies (только asyncio)
- Leaky bucket — токены восполняются равномерно, не всплесками

**Cons:**
- Leaky bucket допускает кратковременные всплески (накопленные токены)
- Для RPM-масштаба это несущественно

### 2. asynciolimiter

**Pros:** три типа лимитеров (adaptive, strict, leaky bucket)
**Cons:** менее удобный API (.wait() вместо context manager), меньше community

### 3. Sliding window вручную (~25 строк)

**Pros:** точный контроль "ровно N запросов в любом 60s окне", без зависимостей
**Cons:** свой код для поддержки, нет battle-tested гарантий

### 4. Контроль на уровне pipeline (concurrency)

**Pros:** уже реализован
**Cons:** не учитывает количество чанков в документе, грубый инструмент

## Decision

Единый подход `max_rpm` для всех HTTP-сервисов: LLM и embedders.
Две реализации rate limiter под разные runtime:

### 1. Async: LLM clients (llm, llm_vision)

`aiolimiter.AsyncLimiter` (leaky bucket) на уровне `LLMClient._create()`:

```python
from aiolimiter import AsyncLimiter
self._rate_limiter = AsyncLimiter(max_rpm, 60) if max_rpm else None

async with self._rate_limit():
    response = await self._client.chat.completions.create(**kwargs)
```

Корутины ожидающие токен засыпают через `asyncio.sleep()`, event loop свободен
для других задач (embed, Qdrant upsert).

### 2. Sync: HTTP embedders (HttpFridaEmbedder, HttpGteSparseEmbedder)

`SyncRateLimiter` (sliding window) — собственная реализация на `threading.Lock` + `time.sleep`:

```python
class SyncRateLimiter:
    def __init__(self, max_rpm: int) -> None:
        self._max = max_rpm
        self._window = 60.0
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        # sliding window: ровно max_rpm запросов в любом 60s окне
```

HTTP embedders используют синхронный `httpx.Client`, поэтому async limiter не подходит.
`SyncRateLimiter` вызывается в `_do_call()` / `_do_call_batch()` перед HTTP-запросом.

### Конфигурация

```yaml
llm:
  max_rpm: 100          # async, aiolimiter

llm_vision:
  max_rpm: 60           # async, aiolimiter

indexing:
  dense_embedder:
    base_url: http://...  # HTTP mode
    max_rpm: 120          # sync, SyncRateLimiter

  sparse_embedder:
    base_url: http://...  # HTTP mode
    max_rpm: 120          # sync, SyncRateLimiter
```

`max_rpm` опционален во всех конфигах. Если не задан — rate limiting отключён.
Для локальных embedders (без `base_url`) `max_rpm` игнорируется.

### Scope

| Компонент | Файл | Limiter | Где применяется |
|-----------|------|---------|-----------------|
| `LLMClient` | `src/morag/llm/client.py` | `aiolimiter.AsyncLimiter` | `_create()` — все LLM-вызовы |
| `HttpFridaEmbedder` | `src/morag/indexing/embedder.py` | `SyncRateLimiter` | `_do_call()`, `_do_call_batch()` |
| `HttpGteSparseEmbedder` | `src/morag/indexing/embedder.py` | `SyncRateLimiter` | `_do_call()`, `_do_call_batch()` |
| Config | `src/morag/config.py` | — | `LLMConfig.max_rpm`, `DenseEmbedderConfig.max_rpm`, `SparseEmbedderConfig.max_rpm` |

## Consequences

- Thundering herd при массовой индексации через OpenRouter устранён
- Индексация становится предсказуемой: при max_rpm=40 и ~30 LLM-calls/doc
  максимум ~1.33 docs/min, ~80 docs/hour — без throttling ошибок
- Для локальных LLM (Ollama) и локальных embedders max_rpm не нужен — не задавать в конфиге
- `aiolimiter` добавлен как зависимость проекта
- `SyncRateLimiter` — собственная реализация (~20 строк), без внешних зависимостей,
  sliding window обеспечивает строгое соблюдение лимита
