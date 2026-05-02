# ADR-0012: Refactor конфигурации — sources, LLMs, ID convention, run versioning

## Статус

Принято (2026-05-02). Реализация на ветке `refactor/sources-and-llms` от main.
Breaking change для config.yml schema и формата ID документов в Qdrant.

## Контекст

К весне 2026 в morag накопились структурные ограничения, проявившиеся при
попытке добавить console-UI и масштабировать систему на корпоративные сценарии:

### 1. Жёсткая schema источников

`SourcesConfig` имеет три именованных поля для трёх типов источников:
```python
class SourcesConfig(BaseModel):
    local_documents: LocalDocumentsConfig | None = None
    confluence:      ConfluenceConfig      | None = None
    jira:            JiraConfig            | None = None
```

Это значит:
- **Один инстанс на тип.** Невозможно настроить два Confluence-сервера или две
  Jira (типичный корпоративный кейс: «корпоративный Confluence + Confluence
  подрядчика»).
- **Добавление нового типа источника = правка top-level schema + код-обвязка
  per-type в `cmd_index`.** GitHub Issues, Notion, GitLab — каждое требует
  отдельной секции в конфиге и отдельной ветки `if` в orchestration.
- **UI Setup wizard повторяет structure** — три отдельных блока «Local /
  Confluence / Jira». Без natural way показать «N инстансов одного типа».

### 2. LLM-конфигурация: дублирование и неявные роли

```python
class Config(BaseModel):
    llm:        LLMConfig                # обязателен — для всего text-side
    llm_vision: LLMConfig | None = None  # для multimodal
```

Roles where `llm` используется (DocTitle, DocSummary, ContextGen, LLMChunker,
KnowledgeMapGenerator) — **неявны**. Чтобы понять «какие LLM-роли активны»,
нужно читать `cli/main.py` целиком. Hard-coded в коде, не в конфиге.

Для будущего сценария «cheap LLM для context generation, smart LLM для
агентского ответа» нет места в schema — пришлось бы дублировать всю
`LLMConfig`-секцию для каждой роли, со своими api_key, base_url, etc.

### 3. ID документов без namespace

```python
# Confluence page id 12345678 in space DOCS:
doc.id = '12345678'

# Та же страница в другом Confluence-сервере:
doc.id = '12345678'  # ← коллизия в Qdrant!
```

При single-instance-per-type (текущая schema) коллизий нет — system by design
не поддерживает multiple instances. Но это just-в-pre — как только мы хотим
multi-Confluence или multi-Jira, ID-collisions становятся блокером.

Дополнительная проблема: ID **не самодокументируем**. Глядя в логах на
`12345678` или `PROJ-1` — невозможно сказать «это какой Confluence/Jira».

### 4. Отсутствие run versioning

При cron-индексации каждые 6 часов невозможно ответить:
- «Когда этот документ последний раз попал в индекс?» (есть только `updated_at`
  — mtime источника, не indexing-time)
- «Сколько раз этот документ переиндексировался?» (нужно для отладки churn)
- «Какие документы попали в прогон от 2026-05-01 утра?» (нужно для аудита)

Нет ни `indexed_at`, ни счётчика прогонов, ни per-doc version'ов.

### 5. Embedder version drift

При смене dense embedder'а (например FRIDA → Qwen3-Embedding-4B per ADR-0011)
в идемпотентности `_is_up_to_date` сравнивается только `updated_at`. Если
источник не менялся — документ skip'ается, **векторы остаются от старого
embedder'а**. Silent staleness — проявляется как падение качества retrieval
без явной ошибки.

## Решение

Refactor конфига и core'а вокруг 5 связных идей.

### 1. Sources — discriminated union (`list[Source]`)

Источники описываются единым списком polymorphic-объектов с дискриминатором
`kind`:

```yaml
sources:
  - kind: local
    name: docs
    path: data/

  - kind: confluence
    name: corp
    url: https://corp.atlassian.net/
    username: integrator
    api_token: <secret>
    spaces: [DOCS, ENG]

  - kind: confluence       # ← multiple instances же тип бесплатно
    name: vendor
    url: https://vendor.atlassian.net/
    username: integrator
    api_token: <secret>

  - kind: jira             # ← multi-Jira аналогично
    name: internal
    url: https://jira.internal/
    username: integrator
    password: <secret>
```

Pydantic schema:

```python
class LocalSource(BaseModel):
    kind: Literal['local']
    name: str
    enabled: bool = True
    path: str

class ConfluenceSource(BaseModel):
    kind: Literal['confluence']
    name: str
    enabled: bool = True
    url: str
    username: str
    api_token: str | None = None
    password: str | None = None
    spaces: list[str] = []
    # ... остальные поля

class JiraSource(BaseModel):
    kind: Literal['jira']
    name: str
    enabled: bool = True
    url: str
    username: str
    password: str | None = None  # только on-prem (см. ниже)

Source = Annotated[
    Union[LocalSource, ConfluenceSource, JiraSource],
    Field(discriminator='kind'),
]

class Config(BaseModel):
    schema_version: Literal[1] = 1
    sources: list[Source] = Field(min_length=1)
    ...
```

`(kind, name)` — primary key источника. Уникальность валидируется в Pydantic
post-validator.

В `cmd_index` — registry-based dispatch вместо if-цепочек:

```python
SOURCE_BUILDERS = {
    'local': lambda cfg, deps: LocalDocumentSource(cfg, ...),
    'confluence': lambda cfg, deps: ConfluenceSource(cfg, ...),
    'jira': lambda cfg, deps: JiraSource(cfg, ...),
}

for src_cfg in config.sources:
    if not src_cfg.enabled:
        continue
    source = SOURCE_BUILDERS[src_cfg.kind](src_cfg, deps)
    await pipeline.run(source)
```

Добавление нового типа (`kind: github`) — новый Pydantic-класс + новая
запись в `SOURCE_BUILDERS`, без правок остальной schema.

### 2. LLMs — named pool + role mapping

Все LLM-инстансы описаны в одном пуле, роли ссылаются по имени:

```yaml
llms:
  - name: main
    base_url: https://api.x.ai/v1
    model: grok-4-1-fast-non-reasoning
    api_key: <secret>
    context_window: 256000

  - name: vision
    base_url: <vision-provider>/v1
    model: qwen2.5-vl-7b-instruct
    api_key: <secret>

  - name: smart           # опционально, для важных мест
    base_url: https://openrouter.ai/api/v1
    model: anthropic/claude-haiku-4.5
    api_key: <secret>

indexing:
  llm: main               # короткая форма: default для всех indexing text-ролей
  vision: vision

  # Расширенная форма с per-role overrides:
  # llm:
  #   default: main
  #   overrides:
  #     doc_summary: smart
  #     knowledge_map: smart
```

Pydantic:

```python
LLMCapability = Literal['text', 'vision']

class LLMInstance(BaseModel):
    name: str
    base_url: str
    model: str
    api_key: str
    capabilities: list[LLMCapability] = ['text']  # см. ниже
    context_window: int = 32768
    enable_thinking: bool | None = None
    max_concurrent: int | None = None
    # ... остальные поля

class LLMRoleMapping(BaseModel):
    """Поддерживает форматы:
       - 'main'  → default='main', overrides={}
       - {default: 'main', overrides: {doc_summary: 'smart'}}
    """
    default: str
    overrides: dict[str, str] = {}

    @model_validator(mode='before')
    def _normalize(cls, v):
        if isinstance(v, str):
            return {'default': v, 'overrides': {}}
        return v

class IndexingConfig(BaseModel):
    llm: LLMRoleMapping
    vision: str   # имя из llms-pool
    ...
```

В коде — resolver:

В `cmd_index` — простой dict-pool без отдельного Resolver-класса:

```python
clients = {
    llm.name: LLMClient(base_url=llm.base_url, model=llm.model, ...)
    for llm in config.llms
}

text_client = clients[config.indexing.llm.name_for('default')]
vision_client = clients[config.indexing.vision]
```

**`LLMInstance.capabilities`** — declarative объявление того что модель умеет.
Default = `['text']`. Multimodal — `[text, vision]`.

```yaml
llms:
  - name: grok                       # text-only (default)
    base_url: https://api.x.ai/v1
    model: grok-4-1-fast
    api_key: ...
    # capabilities не указано → ['text']

  - name: qwen-vl                    # multimodal — обе capability
    base_url: ...
    model: qwen2.5-vl-7b
    api_key: ...
    capabilities: [text, vision]
```

`Config.model_validator` проверяет что `indexing.vision` указывает на LLM с
`'vision'` в capabilities. Иначе `ValidationError` при `load_config()` — до
открытия Qdrant и любых API-вызовов. Нет «cryptic ошибок на 47-й странице PDF».

`text`-роли валидировать не нужно — `text` всегда default.

Multimodal-LLM можно использовать одновременно для `indexing.llm: qwen-vl` И
`indexing.vision: qwen-vl` — из пула возьмётся **один и тот же LLMClient**
(общий semaphore, общий HTTP-pool). Никакого дублирования в YAML.

`capabilities` — config-time guardrail, runtime ничего не делает (LLMClient,
VisionPdfConverter, ConfluenceSource._describe_image работают как раньше).

### 3. Document ID convention: `<kind>:<name>:<external-id>`

Все документы в Qdrant получают prefixed ID:

| Что | ID |
|---|---|
| Markdown файл `2024-VKR.md` в источнике `local:docs` | `local:docs:2024-VKR.md` |
| Confluence page 12345 в `corp` | `confluence:corp:12345` |
| Та же страница в `vendor` | `confluence:vendor:12345` |
| Jira issue PROJ-1 в `internal` | `jira:internal:PROJ-1` |
| Attached PDF (page 12345, attachment 99) в `corp` | `confluence:corp:att:12345:99` |

ID формирует source при `get_metadata()` / `load_one()`:

```python
def make_id(self, external_id: str) -> str:
    return f'{self.kind}:{self.name}:{external_id}'
```

`parent_doc_ids` тоже prefixed (внутри одного источника). `chunk.doc_id`
ссылается на prefixed-id родительского документа.

### 4. Stable payload-fields для UI / retrieval / debug

Помимо ID в payload каждого документа добавляются explicit fields:

```python
payload = {
    'source_type': 'confluence',  # = kind (как сейчас)
    'source_name': 'corp',         # NEW: имя инстанса
    'url': 'https://...',          # как сейчас, для citation render
    'title': '...',                # как сейчас
    'updated_at': '...',           # mtime источника, как сейчас
    # ... existing fields

    # NEW: tracking когда и в каком прогоне попало в индекс
    'indexed_at': '2026-05-02T10:30:15+00:00',
    'run_number': 42,              # см. п.5
    'version': 5,                  # см. п.5

    # NEW: detection embedder drift
    'embedder_fingerprint': 'sha256:abc...',  # см. ниже
}
```

`source_name` явно отделяет «к какому инстансу» — UI группирует по
`(source_type, source_name)`, retrieval может фильтровать «только из corp
Confluence», логи читаются.

**`embedder_fingerprint`** = `sha256(model + dim + base_url)`. При
индексации каждый документ помечается fingerprint'ом текущего embedder'а.
В `_is_up_to_date` — проверка fingerprint вместе с `updated_at`. Mismatch
→ документ переиндексируется (новый embedder → fresh vectors). Защита
от silent staleness при смене embedder'а.

### 5. Run versioning

Три новых поля в payload каждого документа и чанка:

```python
payload['indexed_at']   # ISO timestamp upsert'а
payload['run_number']   # глобальный счётчик cmd_index-вызовов
payload['version']      # per-doc счётчик переиндексаций
```

**`run_number`** — global monotonic counter, инкрементируется в начале
каждого `cmd_index` / `cmd_rebuild_km`. Persistent в `data/morag_state/run_counter.json`.
Cron, on-demand, успех, failure, cancel — всё одинаково: каждый вызов
получает свой номер.

**`version`** — per-doc, инкремент при каждом upsert. Новый док = 1.
Переиндексация = +1. Чанки наследуют версию документа (атомарность прогона).

**`indexed_at`** — момент upsert'а. Все точки одного прогона имеют одинаковый
timestamp (frozen в начале прогона).

**Recovery `run_counter` при потере state-файла:** читаем `max(run_number)`
из payload'ов docs collection (через scroll с `order_by`), используем как
starting point. Без recovery counter падал бы в 0 → коллизии с историческими
прогонами в audit-запросах.

**Use-cases:**
- `filter run_number == 42` — все документы конкретного прогона
- `filter indexed_at > yesterday` — что попало за последние сутки
- `sort by version DESC` — самые «churn'ящие» документы (что-то не так с
  идемпотентностью)
- В Console UI: «Last run: completed at HH:MM, processed N docs» (read
  recent run_number)

## Альтернативы которые рассмотрели

### Sources

- **Multi-instance через nested dict** (`confluence: {corp: {...}, vendor: {...}}`):
  отвергнуто — отдельные top-level fields per kind делают добавление новых
  типов источников по-прежнему дорогим. Discriminated union единообразен.
- **Backward compat** через Pydantic auto-wrap (`isinstance(v, dict): return [v]`):
  отвергнуто — пользователь явно сказал «без legacy, главное чистота».
  Один раз rewrite config → дальше чисто.

### LLMs

- **Inline LLMConfig per role** (вместо named pool):
  ```yaml
  indexing:
    context_generation:
      llm: {base_url: ..., model: ..., api_key: ...}
    doc_summary:
      llm: {base_url: ..., model: ..., api_key: ...}
  ```
  Отвергнуто — гигантское дублирование при использовании одной LLM в N ролях.
- **Discriminated union для LLMs** (`kind: openai|anthropic|ollama`):
  отвергнуто — все LLM-провайдеры выглядят одинаково через OpenAI-compat
  API (base_url + model + api_key). Различие — в роли использования, не в
  типе. Discriminated union здесь over-engineering.

### ID convention

- **Префикс только для multi-instance** (single instance оставляет ID без
  префикса для backward compat): отвергнуто — особый случай создаёт неявные
  exceptions, ломает invariants. Лучше всегда консистентно.
- **UUID-only IDs** (без человеко-читаемых компонентов): отвергнуто —
  `confluence:corp:12345` читается в логах, `7f8a...` нет.
- **Composite key через payload-only** (id = просто external, контекст в
  payload): отвергнуто — Qdrant point-id должен быть unique. Без префикса
  всё равно были бы коллизии.

### Run versioning

- **Bump counter только при успешном завершении**: отвергнуто — `run_number`
  нужен в payload AT INDEX TIME. Не успели закончить — payload уже записан.
  Bump-at-start даёт честное audit «попытка 42 сделала 2 документа из 5».
- **Хранить counter в Qdrant special-point вместо state-файла**: отвергнуто —
  state-file проще, recovery всё равно через scan Qdrant.
- **Обнулять counter при `--reset`**: отвергнуто — counter monotonic для
  системы, не для данных. Reset не должен ломать audit-историю предыдущих
  прогонов.

### Schema versioning

- **Без `schema_version`**: отвергнуто — рано или поздно schema снова поменяется,
  лучше иметь explicit маркер для migration scripts. Cheap insurance.

## Последствия

### Breaking changes

- **`config.yml` полностью переписан.** Старая структура
  (`sources.local_documents` / `confluence` / `jira`, `llm:` + `llm_vision:`
  на верхнем уровне) НЕ читается. Юзер вручную мигрирует или использует
  helper-script `scripts/migrate_config.py`.
- **Document IDs изменились формат.** Существующие точки в Qdrant имеют
  старые IDs без префикса. Нужно либо migration-script (read all → upsert with
  new id → delete old), либо `--reset`. У нас по факту планируется reset
  для смены embedder'а на Qwen3, поэтому migration-script не критичен.
- **CLAUDE.md и README обновляются** под новую schema.

### Не-breaking, но новые поведения

- При cron-no-op (всё up-to-date) `run_number` всё равно инкрементируется —
  «прогон 42 был, ничего не изменилось». Это feature, не bug.
- Embedder fingerprint автоматически re-индексирует документы при смене
  модели — раньше требовался `--reset` руками.

### Console UI обновляется

Текущий Setup wizard с разделами «LLM / Embedder / Documents» переделывается:
- **Sources** — единый список карточек с «+ Add source» (kind dropdown)
- **LLMs** — пул именованных LLM с «+ Add LLM»
- **Roles** — assignment UI для indexing.llm + indexing.vision
- Existing Settings (raw YAML) остаётся

### Что НЕ делается этим refactor'ом (явные non-goals)

- **Multi-tenant.** Console остаётся single-tenant, без auth, bind 127.0.0.1.
  Изменение этого потребует major refactor (per-user state, audit, isolation).
  Зафиксировано отдельно (см. CLAUDE.md, раздел Console + Control-plane).
- **Pipeline (retrieval) config унификация.** Pipeline-контейнер продолжает
  использовать env-vars + OWUI Valves. Не вводим `retrieval:` секцию в
  config.yml. См. CLAUDE.md, раздел Config.
- **Polymorphic LLMs** (`kind: openai|anthropic`). Текущий named pool
  достаточен — все провайдеры уже OpenAI-compatible.
- **Per-source Knowledge Map.** KM остаётся одним global tree'ом по всему
  корпусу. Multi-source merging работает естественно через `parent_doc_ids`.
  Если выяснится что нужны раздельные KM — отдельный ADR.

## Реализация: ориентировочный план этапов

1. **Pydantic schema** (`src/morag/config.py`) — Source discriminated union,
   LLMs pool, role mapping, schema_version, валидаторы уникальности `(kind, name)`.
2. **ID convention в Source-классах** — `make_id` helper, parent_doc_ids prefix.
3. **`cmd_index` orchestration** — registry-based loop, LLMResolver, RunContext.
4. **Run versioning infra** — `RunContext`, `data/morag_state/run_counter.json`,
   recovery, payload-индексы по `run_number` и `indexed_at`.
5. **Embedder fingerprint** — helper для подсчёта, проверка в `_is_up_to_date`,
   payload поле.
6. **Console UI** — Sources tab (list + add by kind), LLMs tab (pool + roles),
   обновлённый config_io.
7. **Migration helpers** — `scripts/migrate_config.py` (старый → новый
   формат), опционально `scripts/migrate_qdrant_ids.py` (если кому-то нужно
   без `--reset`).
8. **Тесты** — все новые компоненты + обновление существующих под новую schema.
9. **Документация** — README (user-facing) минимальные правки, CLAUDE.md
   подробное обновление, config.example.yml в новом формате.

Оценка: ~9-11ч.

## Future considerations

Что становится возможным благодаря этому refactor'у, но не входит в scope:

- **GitHub Issues / Notion / GitLab источники** — новый kind в discriminated
  union, builder в registry, форма в UI. Без правок остального.
- **Per-source rate limiting / quotas** — очевидное расширение через поле в
  Source-схеме.
- **A/B-testing prompt'ов через LLM-pool** — описать `name: experimental`,
  override на одну роль, сравнить.
- **Multi-tier Knowledge Map** (если single global перестанет масштабироваться):
  опции — KM per source-kind, KM per source-instance, KM с pruning по
  retrieval-relevance. Решать когда возникнет реальный pain.
- **Pipeline config унификация** через `retrieval:` секцию — когда придёт
  понимание что valves-UI недостаточен / надо synced с indexer-config.