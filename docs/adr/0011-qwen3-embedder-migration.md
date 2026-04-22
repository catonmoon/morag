# ADR-0011: Переход dense-эмбеддера с FRIDA на Qwen3-Embedding-4B

## Статус

Принято (2026-04-21). Миграция выполнена для коллекции `chunks_virgo_qwen` и
поставлена как дефолт в `config.virgo.qwen.yml`. FRIDA-путь (локальный
sentence-transformers + HTTP-микросервис `services/embedder_frida/`) остаётся в
коде для обратной совместимости со старыми коллекциями, но в новых
инсталляциях не рекомендуется.

## Контекст

До апреля 2026 проект использовал `ai-forever/FRIDA` (dim=1536, контекст 512
токенов, требует собственного HTTP-сервиса `services/embedder_frida/app.py` на
MPS/CUDA). Это создавало операционные проблемы:

- **Ручной запуск сервиса.** FRIDA нет в публичных реестрах (Ollama,
  OpenRouter). Docker-образ на macOS без CUDA не использует MPS — 42 сек на
  эмбеддинг. На проде Mac MPS нужно стартовать Python-скрипт вручную;
  docker-режим непригоден.
- **Контекст 512.** Заставляет нарезать чанки ≤256 токенов, что сильнее
  фрагментирует документы (для технической документации и ВКР оптимальнее
  1–2K токенов на чанк).
- **Dim 1536.** Средний по сравнению с конкурентами (Qwen3-Embedding-4B,
  BGE-M3) при сопоставимом или меньшем размере модели.
- **Сложный онбординг.** Новому разработчику нужно отдельно понимать
  HTTP-сервис, запускать его, следить за портом, отличать docker-ветку от
  нативной.

## Решение

Сделать **`qwen3-embedding:4b` через Ollama** дефолтным dense-эмбеддером
проекта.

### Характеристики модели

- **dim=2560** (native MRL, без truncate).
- **Контекст 32K токенов** — можно позволить чанки до 2K–4K токенов
  (`SectionChunker` с `max_tokens: 2000`).
- **Раздаётся через Ollama** — демон, запускается один раз, интерфейс
  OpenAI-совместимый (`/v1/embeddings`), без отдельного Python-сервиса.
- **Instruct-шаблон на query-side** обязателен для корректности:

  ```
  Instruct: Given a user question, retrieve passages that answer the question
  Query:{text}
  ```

  Document-side — без префикса.

- **Токенизатор.** Для `SectionChunker`/`HybridChunker` используем
  `tiktoken cl100k_base` (`tokenizer: tiktoken` в конфиге). Отличие от
  Qwen-нативного токенайзера ±40% на русском приемлемо при `max_tokens=2000` —
  модель всё равно в контексте 32K.

### Инфраструктура

Поддерживается уже реализованным `HttpEmbedder`
(`src/morag/indexing/embedder.py`) с шаблонами `document_template` /
`query_template` и отправкой `model` в body (Ollama требует). Тот же код
работает для FRIDA-микросервиса, vLLM, Ollama и OpenAI API — разница только в
конфигурации.

### Конфигурация

В `config.virgo.qwen.yml`:

```yaml
indexing:
  dense_embedder:
    model: qwen3-embedding:4b             # Ollama-нотация в /v1/embeddings body
    tokenizer: tiktoken                   # без HF-зависимости, ±40% на русском приемлемо
    base_url: http://localhost:11434      # Ollama локально
    dim: 2560                             # native Qwen3-4B
    timeout: 180
    document_template: '{text}'           # без префикса
    query_template: "Instruct: Given a user question, retrieve passages that answer the question\nQuery:{text}"
```

В `services/pipeline/morag.py` (valves для OWUI):

```
DENSE_EMBED_URL: http://host.docker.internal:11434
DENSE_EMBEDDER_MODEL: qwen3-embedding:4b
DENSE_ENCODING_FORMAT: base64             # 'float' если провайдер base64 не поддерживает
QUERY_TEMPLATE: "Instruct: Given a user question, retrieve passages that answer the question\nQuery:{text}"
```

### Архитектура коллекций

- `chunks_virgo_qwen` — новая коллекция, dim=2560.
- `chunks_frida` — продолжает существовать, dim=1536, не трогается.
- `chunks_qwen3_section`, `chunks_qwen3_section_ctxbm25` — исследовательские
  коллекции (5 ВКР, см. раздел «Исследования качества чанкинга»).
- `docs` / `knowledge_map` — могут шариться (контент одинаков, только векторы
  разные).

## Исследования качества чанкинга

Переход на Qwen3 сопровождался серией экспериментов на однородном корпусе из
5 академических документов (каждый 1600–3000 строк markdown). Полные отчёты
и артефакты лежат в `adventures/embedder_new/`.

### 1. Крупный чанк = законченная мысль

На образцовом документе 1644 строки MD:

| Метрика | FRIDA (`max=256`) | Qwen3 (`max=1024`) |
| --- | ---: | ---: |
| Всего чанков | 182 | 71 |
| Text chars mean | 796 | 2046 |
| Context chars mean | 458 | 164 |

Qwen3 пакует **в 2.6× меньше чанков** при **2.6× большем тексте** в каждом, и
LLM-context в 3× короче — крупный самодостаточный чанк меньше нуждается в
дизамбигуации. Соотношение устойчивое: на самом крупном документе корпуса
(2996 строк) — 2.85× (328 vs 115).

**Природа обрывов качественно разная:**

- FRIDA (`max=256`): **9.5% чанков — микро-огрызки** (<250 chars). Повторяющиеся
  markdown-структуры рвутся на атомы — одиночные легенды графиков, одиночные
  пункты из списка моделей, подписи к рисункам. Классический кейс — пункт
  списка (`- 7) Model X: F1=0.29`) вырванный из перечня 1..N и болтающийся
  без соседей.
- Qwen3 (`max=1024`): **0% tiny-чанков**. «Trunc-end» случаи (19.1%) — это
  физические концы таблиц и параграфов, мысль уже заключена.

### 2. Структурный SectionChunker как следующий шаг

HybridChunker даже с `max=1024` иногда режет логические единицы (описание
рисунка с пунктами 1–6 ломается посередине). `SectionChunker` — «раздел
markdown как атомарная логическая единица», рекурсивная упаковка по H-уровням
с пере-переносом magnetic heading'а в `prefix_blocks` первого дочернего
подраздела (см. ADR-0008 про HybridChunker, Section — его потомок).

Сравнение на 5-документном корпусе:

| Метрика | FRIDA hybrid (max 256) | Qwen3 hybrid (max 1024) | Qwen3 section (300/1024) |
| --- | ---: | ---: | ---: |
| Всего чанков | 972 | 355 | 411 |
| Tiny (<250 chars) | 58 (6.0%) | 2 (0.6%) | 3 (0.7%) |
| Trunc-end | 140 (14.4%) | 76 (21.4%) | 66 (16.1%) |
| Starts-with-heading | 19.7% | 14.1% | **41.8%** |
| Text chars mean / max | 736 / 2565 | 1978 / 6483 | 1709 / 4432 |

Главное: **41.8% section-чанков начинаются с заголовка** — в 3× чаще, чем у
hybrid. Каждый чанк сразу сигналит «это раздел X.Y.Z», а не «...продолжение
предыдущей мысли». Max size упал с 6483 до 4432 chars — section соблюдает
границы H-уровней и не пакует разнородные структуры подряд.

Побочный эффект: обнаружен и исправлен скрытый баг в
`HybridChunker._greedy_fill` — в ветке `block > max_tokens` терялся magnetic
heading. Теперь висящий заголовок префиксируется к первому oversized-подчанку.

### 3. Contextual BM25 (Anthropic-style) добавлен как опция

Флаг `indexing.lexical_chunk_context: true` подмешивает `chunk.context`
(LLM-summary чанка) в лексические векторы (keywords + bm25 + bm25_trigram).
Дополнительных LLM-вызовов ноль — context уже есть.

На тестовом корпусе эффект в пределах шума: mean semantic similarity ответов
0.5680 vs 0.5615 без ctxBM25 (+0.007). Anthropic заявляет −49% failure rate,
но это для их корпусов и chunk-level golden. Для мелкого кейса с doc-level
gold retrieval и так упирается в 99–100% потолок. Оставлено как опция через
флаг, включается осмысленно на крупных корпусах.

### 4. Разрешающая способность эксперимента

100 вопросов по 5 документам исчерпали метрики на уровне документа:
recall@5 = 99–100%, MRR@5 = 0.94–0.95 у всех четырёх Qwen3-конфигураций.
Semantic similarity ответов агента (cosine с golden) лежит в 0.56–0.57 —
различить < 1% разницы между конфигурациями на таком объёме невозможно.

Дальнейшие оптимизации чанкинга имеет смысл проверять на более серьёзных
golden — с chunk-level ground truth, на сотнях документов.

### 5. Итоговая конфигурация для продакшена

- **Чанкер:** `SectionChunker` (`chunker.mode: section`), `min_tokens: 200`,
  `max_tokens: 2000` (с Qwen3 32K контекстом и крупными доменными разделами).
- **Contextual BM25:** включён (`lexical_chunk_context: true`) — не ухудшает,
  может помочь на больших корпусах.
- **Токенизатор чанкера:** `tiktoken` (отклонение от Qwen-нативного в ±40% на
  русском компенсируется запасом до 32K).
- **Oversized per-type стратегии** (как в ADR-0008): `table: split`,
  `list: split`, `paragraph: split`, `fence: asis`, `diagram: asis`.

Эта связка — **основной выигрыш от миграции**. Qwen3 дал возможность крупных
чанков (32K контекст), SectionChunker превратил это в «один раздел = один
чанк» — качественно новый уровень целостности.

## Последствия

### Плюсы

- Нет необходимости запускать и поддерживать `services/embedder_frida/`. Один
  `ollama serve` на хосте для LLM и embeddings.
- Чанки крупнее (1–2K вместо ≤256) → меньше фрагментации, связные ответы от
  агента, больше контекста в одной цитате.
- Через Ollama легко менять модель (`qwen3:4b` → `qwen3:8b` → любая через
  `ollama pull`) без докера и редеплоев.
- `HttpEmbedder` generic — тот же путь работает для FRIDA HTTP, vLLM, Ollama,
  OpenAI API.
- Metal на Apple Silicon из коробки — без отдельных шагов для MPS.

### Минусы и риски

- **Ollama dim mismatch** ([issue #12368](https://github.com/ollama/ollama/issues/12368)):
  в некоторых комбинациях модели/версий Ollama возвращает урезанный dim.
  Mitigation: обязательная проверка `len(embedding) == expected_dim` перед
  созданием коллекции (уже есть в pipeline).
- **Ollama и `encoding_format: base64`.** В ряде версий игнорируется и
  возвращается `list[float]`. Mitigation: в `services/pipeline/morag.py` есть
  valve `DENSE_ENCODING_FORMAT` (`base64` | `float`). Для Qwen3/Ollama работает
  и то и другое.
- **Прогрев первого запроса Ollama** 30–60 сек. Mitigation: `timeout: 180`,
  `retry.max_retries: 3` в конфиге `HttpEmbedder`.
- **Tokenizer mismatch.** Мы считаем токены через tiktoken, а Qwen режет
  по-своему. На русском tiktoken переоценивает ~на 40%, но при
  `max_tokens=2000` остаётся большой запас до 32K модельного окна.
- **Существующие FRIDA-коллекции остаются.** Не нужно немедленно
  переиндексировать всё. `chunks_frida` продолжит работать, `chunks_virgo_qwen`
  — новый. При желании можно параллельно держать обе коллекции и сравнивать.

## Альтернативы, которые рассматривали

1. **Оставить FRIDA.** Лучшее знакомое поведение, но операционная боль не
   уходит. Rejected.
2. **BGE-M3 через Ollama / FlagEmbedding.** dim=1024, context=8K. Хорошая
   альтернатива, но уступает Qwen3-4B по retrieve-метрикам и не даёт прироста
   за счёт inexpensive scaling. Deferred.
3. **Qwen3-Embedding-8B.** dim=4096, ~2× качества ценой 2–3× времени/памяти.
   Для небольших корпусов (<10K документов) оправдано, но для Virgo с ~450
   страниц избыточно. Можно включить позже через тот же `HttpEmbedder` с другим
   `model` + `dim` в конфиге.
4. **OpenAI `text-embedding-3-large` через API.** Качество топ, но облачный,
   требует токенов, сложно для on-premise. Rejected как дефолт; доступно через
   `HttpEmbedder` при желании.

## Эволюция решения

1. Пайплайн жил на FRIDA с отдельным HTTP-сервисом. Онбординг болезненный,
   чанки по 256 токенов.
2. Переход на generic `HttpEmbedder` с template'ами (`document_template`,
   `query_template`). FRIDA продолжает работать, но путь к Ollama-моделям
   открыт.
3. Добавлена поддержка Qwen3 Instruct-шаблона на query-side и пересылка `model`
   в body (Ollama требует).
4. `SectionChunker` (ADR-0008) с `max_tokens: 2000` стал оптимальным
   спутником Qwen3 — 32K контекст позволяет.
