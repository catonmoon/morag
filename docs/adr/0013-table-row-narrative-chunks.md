# ADR-0013: Table-row-narrative-чанки (дублирующее покрытие markdown-таблиц)

## Статус

Принято (2026-05-11). Реализовано на main. Backwards-compatible (новые поля в payload, fallback на отсутствие — старые чанки работают как раньше).

## Контекст

Markdown-таблицы (особенно глоссарии и плотные спецификации) плохо работают с dense retrieval'ом. Конкретный кейс из virgo Confluence:

- **Глоссарий `confluence:virgo:1068775100`** содержит ~100 терминов в одной markdown-таблице (`| Термин | Определение |`).
- После `split_table_chunks(max_table_rows: 10)` он режется на 8-10 чанков по ~10 строк каждый.
- На прямой query «способ принятия решений основывающийся на доверии к сервису распознавания» (определение термина «Жёсткая типизация» из этого глоссария) — **ни один из его чанков не входит в top-30 dense-search**. Топ занимают страницы про CV-сервисы.

Причина: dense embedding 10-строчного chunk'а представляет «среднее» по 10 разнородным понятиям (ТС, УИ, Флоу, ЧД, Жёсткая типизация, ...). Конкретный термин теряется в семантическом шуме. Уменьшение `max_table_rows: 20 → 10` помогло мало — нужно куда более радикальное измельчение, но создавать chunks по 1 строке = терять контекст соседних строк (полезен агенту при ответе) и кратно увеличивать число чанков.

## Решение

**Дублирующее покрытие.** Для каждой markdown-таблицы создаём:

1. **Parent-чанки** (как и раньше) — оригинальные blocks (~10 строк), содержат полную таблицу. Используются как **возвращаемый агенту контент**.
2. **Narrative-чанки** (новые) — по одному на каждую data-строку таблицы. Содержат текст вида `Header1: val1\nHeader2: val2\n...`. Используются как **точечные search-keys** — при попадании в результат retrieval'а заменяются на parent (swap-to-parent).

Свойства:
- Narrative включается в индекс отдельной точкой Qdrant, имеет свой UUID, свои векторы (dense + sparse).
- `payload['chunk_type'] = 'table_row_narrative'` и `payload['parent_chunk_id'] = <UUID parent>` — search-side метки.
- Narrative **не возвращается агенту напрямую**, агент видит только parent (полную таблицу-фрагмент).
- Дедупликация: 3 narrative с одним parent → parent один раз с score первого (=highest score) narrative.

## Алгоритм

### Indexing-side: `add_table_narratives(chunks, min_rows)`

`src/morag/indexing/chunk_splitter.py::add_table_narratives` — запускается в `pipeline._chunk_document` после `split_table_chunks`, до ChunkProcessor chain:

```
chunker → context-gen → split_table_chunks (renumber order/total)
       → add_table_narratives (если narrate_tables.enabled)
       → stamp_payload (run_number/version) для новых narratives
       → ChunkProcessor chain (dense + sparse embed) → upsert
```

Логика:
1. Для каждого chunk в input списке: `_find_table(lines)` — детектит markdown-таблицу.
2. Если `data_rows < min_rows` → skip (status-таблицы, labels).
3. Парсим headers через `_parse_headers`.
4. Для каждой data-row: парсим cells (тот же `_parse_headers` — split('|') + strip).
5. Формируем `text = '\n'.join(f'{header}: {value}' for header, value in zip(headers, cells) if value)`. Пустые/прочерк ячейки пропускаются.
6. Если все ячейки пустые — narrative для этой строки не создаётся.
7. Narrative-чанк: `replace(parent, id=new_uuid, text=narrative_text, context='', order=-1, payload={...без content_kind/table_part, +chunk_type, +parent_chunk_id}, vectors={})`.
8. Narratives добавляются в конец списка `chunks` — parent позиции и нумерация не меняются.

### Retrieval-side: `_swap_narratives_to_parents(chunks)`

`src/morag/retrieval/searcher.py::HybridSearcher._swap_narratives_to_parents` — вызывается в `search_chunks` после RRF и `_point_to_chunk`:

```
RRF query → result.points → _point_to_chunk → _swap_narratives_to_parents → return
```

Логика (chunks приходят отсортированные по score):
1. Pre-scan: собрать `parent_ids` для всех narratives. Исключить те, что уже среди regular chunks.
2. Batch-fetch недостающие parent'ы через `fetch_chunks_by_ids` (Qdrant `retrieve` по UUID).
3. Iterate chunks в порядке score:
   - **narrative**: если parent уже в `seen_ids` — drop. Иначе подменяем на parent (score берём от narrative — **строгое наследование**, без max/mean).
   - **regular**: если `chunk_id` уже в `seen_ids` (свапнут narrative'ом) — drop. Иначе — passthrough.
4. Reranker и agent получают уже отсвапанный list — никаких narratives не видят.

`get_neighbors` также защищён: `fetch_chunk_by_order` фильтрует `must_not chunk_type='table_row_narrative'` — narratives с `order=-1` не лезут в neighbour walks.

### Эмбеддинг narrative

- **Dense** (`DenseEmbeddingProcessor._full_text`) = `path + text + context` = `parent.path + "Header: val\n..." + ""`. Path даёт doc-level bias.
- **Sparse** (`SparseEmbeddingProcessor._sparse_text`) — для narrative-чанков возвращает **только `chunk.text`**, игнорируя глобальные флаги `lexical_chunk_context`/`lexical_doc_summary`. Точечная правка по `chunk.payload['chunk_type'] == 'table_row_narrative'`.

**Не добавляем `doc_summary` в эмбеддер narrative** — он расфокусировал бы embedding между всеми narratives одного документа (общая часть = doc_summary). Per-row specificity критична для матча конкретного термина.

## Конфигурация

```yaml
indexing:
  chunker:
    narrate_tables:
      enabled: false           # off by default (новая фича, opt-in)
      min_rows: 5              # таблицы с <5 строк скипаем (status/labels)
```

Обоснование `min_rows: 5`:
- 1-2 строки = labels (`| Yes | No |`, `| key | value |`) — narrative не даёт ценности, parent сам компактный.
- 3-4 строки = сравнительные таблицы; agent и так нормально работает с parent целиком.
- 5+ строк = dataset (спецификация, чек-лист, глоссарий) — per-row narrative полезна.

## Стратегия наката

**Lazy через idempotency.** После включения `narrate_tables: true`:

- Существующие 25K чанков работают как раньше — `chunk_type/parent_chunk_id` отсутствуют → swap-логика их пропускает (filter False), регрессий нет.
- Документы, которые изменятся в источнике (Confluence/локальные), на следующем cycle переиндексируются с narratives.
- Точечно: ручное `delete` из Qdrant (как для глоссария `1068775100`) триггерит реиндексацию через idempotency `_is_up_to_date` (chunks_count=0 ≠ stored.total) → narrate подхватится.

Без миграции схемы Qdrant (payload schemaless). Опционально — создать payload-index на `chunk_type` (`keyword` type) для ускорения filter в `fetch_chunk_by_order`. Без индекса работает full-scan, на 25-100K чанков приемлемо (десятки ms).

## Альтернативы рассмотренные

- **Резать таблицу по 1 строке в chunker'е (max_table_rows: 1)** — отвергнуто: кратно растёт число chunks, каждый теряет соседей-row для контекста при возврате агенту. Narrative-подход даёт оба: точечный search + полный parent.
- **Full reindex после включения** — отвергнуто: дорого (часы для 7000 доков), и для большинства доков narratives не нужны (без таблиц).
- **Backfill-script для существующих чанков** — отложено как future option. Lazy достаточно для нашего темпа изменений.
- **LLM-классификация типа таблицы → адаптивная стратегия** — отложено как TODO. Нужны golden-set данные для валидации (без них непонятно какие типы реально есть в корпусе и какая стратегия per-type оптимальна). См. план в `/tmp/refactored-seeking-catmull.md` секция TODO.

## Последствия

**+ Точность retrieval на табличных терминах.** Для конкретных терминов (глоссарий, поля спецификации) narrative-чанк даёт «чистый» dense+sparse signal без шума соседей.

**+ Backwards-compatible.** Старые чанки работают. Schema не меняется. Возможен постепенный rollout.

**+ Контекст для агента сохраняется.** Narrative — только search-key, агенту приходит полный parent (~10 строк) с соседними терминами — полезно для cross-reference в ответе.

**− Рост числа points в Qdrant.** Для тяжёлых doc'ов с большими таблицами: +10-100% chunks. На корпусе virgo оценочно +10-20% общего volume (большинство doc'ов без таблиц / с маленькими таблицами скипаются по min_rows=5).

**− Дополнительные embed-вызовы при индексации.** Каждый narrative требует dense + sparse embed. Для глоссария 100 строк = 200 embed calls дополнительно. Linear со size, не критично.

**− Дополнительный fetch на retrieval-side.** `fetch_chunks_by_ids` (1 batch-request на search). Десятки ms. Кеширование возможно если станет горячо.

## Verification

- Unit: `tests/indexing/test_chunk_splitter_narratives.py` (7 тестов: basic, skip_small, skip_empty_cells, no_table, all_empty_row, min_rows_boundary, payload_inheritance).
- Unit: `tests/retrieval/test_searcher_swap.py` (9 тестов: basic, dedupe, parent-first, narrative-first, regular-passthrough, mixed, missing_parent_id, parent_not_found, empty).
- E2E на сервере (после деплоя):
  - Удалить глоссарий `confluence:virgo:1068775100` из chunks/docs.
  - Запустить index без reset.
  - Ожидать: docs=1, chunks ≈ 8 (parent) + ~100 (narratives) = ~108.
  - Manual dense search query про термин из глоссария — narrative-чанк должен попасть в топ-10.
  - Через OWUI задать вопрос про термин — ответ должен содержать термин с правильным определением.
- Регрессия: старые doc'и без таблиц не должны менять поведение, существующие тесты pass.
