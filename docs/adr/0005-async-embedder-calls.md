# ADR-0005: Async Embedder Calls via run_in_executor

## Status

Accepted (2026-03-22)

## Context

При индексации с concurrency=12 SemanticChunker вызывает `embed_batch()` синхронно.
Это блокирует asyncio event loop и не даёт LLM-корутинам (context generation) работать
параллельно с chunking.

### Эволюция проблемы

1. **Локальный PyTorch (MPS)**: `model.encode()` держит GIL на время вычислений (CPU-bound).
   12 корутин по очереди ждут GPU — фактически последовательный chunking.
   Документ на 27K токенов блокировал event loop на 6 минут.

2. **Docker HTTP embedder**: MPS недоступен в Docker на Mac (виртуализация).
   CPU-режим: 42с на один текст — непригодно.

3. **Native HTTP embedder (MPS)**: отдельный процесс, MPS GPU, быстро (0.76с на батч из 28).
   Но `httpx.Client.post()` — sync вызов, блокирует event loop так же как и п.1.
   Event loop заморожен на время HTTP roundtrip (~0.5-5с на батч).
   LLM-корутины не могут отправить запросы к Grok.

### Ключевое наблюдение

HTTP I/O ≠ CPU-bound. При ожидании ответа от HTTP-сервера GIL свободен.
`ThreadPoolExecutor` + `run_in_executor` позволяет sync HTTP работать в потоке,
не блокируя event loop.

## Decision

Обернуть вызов `self._embed_fn(texts)` в `asyncio.loop.run_in_executor()`
в SemanticChunker. Sync HTTP-вызов уходит в поток, event loop свободен.

```python
# Было:
embeddings = self._embed_fn(unique_texts)

# Стало:
loop = asyncio.get_event_loop()
embeddings = await loop.run_in_executor(None, self._embed_fn, unique_texts)
```

### Почему ThreadPool, а не ProcessPool

- HTTP I/O отпускает GIL → потоки работают параллельно
- Не нужно сериализовать данные между процессами
- Не нужно загружать модель в каждом worker-е
- `None` = дефолтный ThreadPoolExecutor

### Scope

Только SemanticChunker — это единственное место, где sync embed_fn вызывается
внутри async context и блокирует event loop на значительное время.

DenseEmbeddingProcessor и SparseEmbeddingProcessor вызываются один раз после
всех контекстов, блокировка там некритична.

## Consequences

- Chunking и context generation работают параллельно для разных документов
- Нативный HTTP embedder (MPS) полностью раскрывает потенциал:
  отдельный процесс для GPU, event loop свободен для LLM I/O
- Изменение минимально — одна строка в SemanticChunker
- Для локального PyTorch embedder (без HTTP) эффекта не будет — GIL всё равно
  блокирует поток при CPU-bound `.encode()`. Но при использовании HTTP embedder
  (рекомендуемый режим при concurrency > 1) — значимый прирост

## Observed bottleneck shift

После решения проблемы event loop blocking (run_in_executor) и LLM throttling (rate limiter),
bottleneck сместился на **GPU chunking**:

- LLM RPM utilization: ~16% (9.5 из 60 req/min)
- GPU: 100% active, MPS 1578 MHz
- Rate limiter: 0 waits

SemanticChunker делает сотни embed_batch на больших документах (20K-44K токенов).
Все вызовы сериализуются на одном MPS device. LLM-корутины простаивают.

### Potential improvements

1. **DP-алгоритм** вместо жадного L→R — один embed_batch на весь документ
   (попарные distance между соседними units), затем глобальная оптимизация.
   Одно обращение к GPU вместо сотен.

2. **Dynamic batching** на стороне embedder-сервера — копить запросы 50-100мс
   и объединять в один GPU-батч. Лучшая утилизация GPU.

3. **Скользящее окно** — эмбеддить только K предложений слева/справа от границы,
   а не целые чанки-кандидаты. Меньше текста → меньше нагрузки на GPU.
