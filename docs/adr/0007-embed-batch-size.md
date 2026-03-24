# ADR-0007: Батчевый embed и upsert чанков

## Статус

Принято (2026-03-22)

## Контекст

После генерации контекстов все чанки документа обрабатываются процессорами
(DenseEmbeddingProcessor, SparseEmbeddingProcessor) и сохраняются в Qdrant
**одним батчем**. Для типичного документа (20-50 чанков) это работает нормально.

Проблема возникла на документе с **740 чанками** (DIFC Courts Rules, passthrough):
- Dense embed 740 текстов одним вызовом к MPS embedder-серверу
- Embedder-сервер не справился за timeout=180с
- `ReadTimeout` → документ failed
- Все 740 сгенерированных контекстов потеряны

## Решение

Разбить обработку чанков на батчи фиксированного размера:

```python
for batch_start in range(0, len(chunks), self._embed_batch_size):
    batch = chunks[batch_start:batch_start + self._embed_batch_size]
    for processor in self._chunk_processors:
        batch = processor.process_batch(batch, document)
    await self._chunk_repo.upsert_batch(batch)
```

### Конфигурация

```yaml
indexing:
  embed_batch_size: 64  # по умолчанию
```

`embed_batch_size` в `IndexingConfig`, пробрасывается в `IndexingPipeline`.

### Почему 64

- Типичный документ (20-50 чанков) — 1 батч, без изменений в поведении
- Большой документ (740 чанков) — 12 батчей по 64
- FRIDA на MPS: 64 текста × ~150 tok/текст ≈ 9600 токенов — укладывается в ~5-10с
- Upsert в Qdrant: 64 points — быстро

### Дополнительное преимущество

При батчевом upsert прогресс сохраняется инкрементально. Если процесс упадёт
на батче 8/12 — первые 7 батчей (448 чанков) уже в Qdrant. При idempotency-проверке
count (448) != total (740) → переиндексация, но без потери прогресса других документов.

## Последствия

- Документы с любым количеством чанков обрабатываются без timeout
- Embedder-сервер получает предсказуемую нагрузку (≤64 текста за вызов)
- Для малых документов (<64 чанков) — без изменений, 1 батч
- Лог показывает прогресс: `Batch 1-64/740 saved`, `Batch 65-128/740 saved`
