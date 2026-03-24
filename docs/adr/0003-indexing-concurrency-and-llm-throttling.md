# ADR-0003: Indexing Concurrency and LLM Throttling

## Status

Accepted (2026-03-21)

## Context

При массовой индексации 300 юридических документов через OpenRouter (qwen/qwen3.5-9b)
с concurrency=32 обнаружена проблема thundering herd:

- OpenRouter для данной модели стабильно держит ~100 req/min
- При concurrency=32 пайплайн генерирует всплески до 170 req/min
- Модель отвечает "unavailable" → все 32 потока уходят в sleep(300s)
- Через 300с все 32 стреляют одновременно → повтор цикла
- ~40% wall-clock времени тратится на ожидание

### Измеренные показатели (concurrency=32, model_wait=300s)

| Metric | Value |
|--------|-------|
| Throughput | 27 docs/hour (0.45 docs/min) |
| Model unavailable events | 372 за 5.5 часов |
| Attempt 1 | 316 (85%) |
| Attempt 2 | 49 (13%) |
| Attempt 3 | 7 (2%) |
| GPU (FRIDA on MPS) | 100% active, 26.4W |
| CPU | ~5% user (not bottleneck) |
| Avg chunks/doc | 28 |
| LLM calls/doc | ~30 (title + summary + 28 contexts) |

### Pipeline per document

1. LLM: title (1 call)
2. LLM: structured legal summary (1 call)
3. SemanticChunker: split via FRIDA embeddings (GPU)
4. LLM: context generation (1 call per chunk, ~28 calls)
5. Dense embedding: FRIDA embed_batch (GPU)
6. Sparse embedding: GTE (CPU)

Context generation = ~93% всех LLM-вызовов.

## Decision

### 1. Снизить concurrency до 12

При 28 chunks/doc и concurrency=12 пиковая нагрузка:
~12 * (28 / avg_response_time) ≈ 80-100 req/min — в пределах стабильного throughput OpenRouter.

### 2. Уменьшить model_wait_seconds с 300 до 30

300с ожидание было рассчитано на перезагрузку локальной модели (Ollama).
Для OpenRouter модель обычно доступна через 5-10с после throttling.
30с с 5 retry = 150с суммарно — достаточно для восстановления.

### 3. Увеличить model_wait_retries с 3 до 5

Компенсация уменьшения wait_seconds — больше попыток с меньшим интервалом.

### Оценка эффекта

| Metric | Before | After (estimated) |
|--------|--------|-------------------|
| Concurrency | 32 | 12 |
| model_wait_seconds | 300 | 30 |
| model_wait_retries | 3 | 5 |
| Throughput | 27 docs/hour | ~90 docs/hour |
| 300 docs estimated | ~11 hours | ~3.5 hours |

## Future Considerations

### Random jitter для retry

Добавить random jitter к model_wait_seconds чтобы потоки не стреляли одновременно:
```python
await asyncio.sleep(wait_seconds + random.uniform(0, wait_seconds * 0.5))
```

### Батчевый context generation

Группировка 3-5 чанков в один LLM-запрос снизит количество вызовов в 3-5 раз.
Для документа с 28 чанками: 28 calls → 6-9 calls.

### Отключение context generation

`context.mode: none` убирает ~93% LLM-вызовов. 300 docs за ~1 час.
Trade-off: хуже качество retrieval для ambiguous queries.
Рассмотреть для быстрых итераций и соревнований.

### Адаптивный concurrency

Автоматическое снижение concurrency при обнаружении throttling
и повышение при стабильном throughput.

## Consequences

- Индексация больших корпусов через OpenRouter становится предсказуемой
- Конфиг `model_wait_seconds: 30` не подходит для локальных моделей (Ollama) —
  там модель действительно может загружаться 60-120с. Держать разные конфиги.
- При смене LLM-провайдера пересмотреть параметры concurrency и wait
