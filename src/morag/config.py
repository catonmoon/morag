"""Pydantic-схема конфига morag.

См. ADR-0012 для обоснования архитектуры:
- Sources как list[Source] discriminated union (multi-instance support)
- LLMs как named pool + role mapping
- Document IDs префикс kind:name:
- Per-source enabled, schema_version, embedder fingerprint
- Run versioning через RunContext
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


# ============================================================================
# SOURCES — discriminated union по полю `kind`
# ============================================================================

SourceRole = Literal['primary', 'supplementary', 'hidden']


class _SourceBase(BaseModel):
    """Общие поля всех источников. Не используется напрямую (нет kind).

    `role` управляет тем, как retrieval включает источник в выдачу:
    - `primary` — основной поиск, всегда включён в дефолтную выдачу.
    - `supplementary` — opt-in: попадает только если агент явно запросил
      через `kinds=[...]` в search-tool ИЛИ через scope (section_ids/doc_ids:
      descendants раздела естественно тянут привязанные тикеты Jira).
    - `hidden` — admin kill-switch, никогда не ищется. Агент не может перебить.

    Дефолт `primary` — добавление нового источника не требует явного role.
    """
    name: str = Field(min_length=1, pattern=r'^[a-z0-9][a-z0-9_-]*$',
                      description='Уникальный id инстанса (lowercase, без пробелов)')
    enabled: bool = True
    role: SourceRole = 'primary'


class LocalSourceConfig(_SourceBase):
    """Источник локальных файлов (markdown, pdf и пр.)."""
    kind: Literal['local']
    path: str = Field(min_length=1, description='Путь к директории с файлами')


class AttachmentsConfig(BaseModel):
    enabled: bool = False                            # включить обработку вложений
    mime_types: list[str] = ['application/pdf']      # фильтр по MIME-типу; пока только PDF
    skip_ancestor_ids: list[str] = []                # пропускать вложения со страниц-потомков этих разделов
    url_mode: Literal['preview', 'download', 'parent_page'] = 'preview'


class ConfluenceSourceConfig(_SourceBase):
    """Confluence-инстанс (Cloud или on-premise)."""
    kind: Literal['confluence']
    url: str
    username: str
    password: str | None = None        # on-premise
    api_token: str | None = None       # Atlassian Cloud
    spaces: list[str] = []             # space keys; пусто — все доступные
    ancestor_ids: list[str] = []       # фильтр по ancestor page id; пусто — без фильтра
    skip_ancestor_ids: list[str] = []  # исключить страницы и всех их потомков
    min_image_size_bytes: int | None = None
    data_url_handling: Literal['skip', 'vision'] = 'skip'
    decorative_image_patterns: list[str] = [r'/images/icons/emoticons/']
    timeout: int = 180
    max_retries: int = 3
    attachments: AttachmentsConfig = AttachmentsConfig()

    @model_validator(mode='after')
    def _check_secret(self) -> 'ConfluenceSourceConfig':
        if not self.password and not self.api_token:
            raise ValueError(
                f'ConfluenceSourceConfig[{self.name}]: либо password (on-premise), '
                f'либо api_token (Cloud) должен быть задан'
            )
        return self


class JiraSourceConfig(_SourceBase):
    """Jira-инстанс. Только on-premise — Cloud-вариант не поддерживается
    (см. ADR-0012, обсуждение). Задачи находятся через ссылки в уже
    проиндексированных документах из других sources (Confluence, local).
    """
    kind: Literal['jira']
    url: str
    username: str
    password: str = Field(min_length=1)  # on-prem only
    timeout: int = 180
    max_retries: int = 3
    custom_fields: list[str] = []        # ручной список кастомных полей для индексации
    auto_custom_fields: bool = False     # авто-режим: тащить все поля экрана задачи (editmeta)


# Discriminated union: при загрузке Pydantic смотрит на поле `kind` и
# подбирает правильный класс. Добавление нового типа источника = новый
# Pydantic-класс с своим Literal['kind'] + расширение этого Union.
Source = Annotated[
    Union[LocalSourceConfig, ConfluenceSourceConfig, JiraSourceConfig],
    Field(discriminator='kind'),
]


# ============================================================================
# LLMs — named pool + role mapping
# ============================================================================

LLMCapability = Literal['text', 'vision']


class LLMInstance(BaseModel):
    """Один инстанс LLM в пуле. Уникальный по name.

    `capabilities` — declarative объявление что модель умеет.
    Используется только для config-time валидации (Config.model_validator
    проверяет что indexing.vision указывает на LLM с capability 'vision').
    На runtime ничего не меняет — обычный LLMClient. Default = ['text'].

    Multimodal-модель (Qwen2.5-VL, Claude Haiku) — `capabilities: [text, vision]`.
    Тогда её можно использовать одновременно для indexing.llm И indexing.vision —
    из пула возьмётся один и тот же LLMClient (общий semaphore, общий HTTP-pool).
    """
    name: str = Field(min_length=1, pattern=r'^[a-z0-9][a-z0-9_-]*$')
    base_url: str
    model: str
    api_key: str
    capabilities: list[LLMCapability] = Field(default_factory=lambda: ['text'])
    timeout: int = 180
    context_window: int = 32768
    max_tokens: int | None = None
    max_retries: int = 3
    model_wait_seconds: int = 0
    model_wait_retries: int = 0
    enable_thinking: bool | None = None
    max_concurrent: int | None = None

    @model_validator(mode='after')
    def _validate_capabilities_non_empty(self) -> 'LLMInstance':
        if not self.capabilities:
            raise ValueError(f'LLMInstance[{self.name}]: capabilities не может быть пустым')
        return self


class LLMRoleMapping(BaseModel):
    """Маппинг ролей на LLM из пула.

    Поддерживает две формы в YAML:
        llm: main                 # короткая: только default
        llm:                      # расширенная: default + overrides
          default: main
          overrides:
            doc_summary: smart
    """
    default: str
    overrides: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def _normalize(cls, v: Any) -> Any:
        # Короткая форма (строка) → разворачиваем в полную
        if isinstance(v, str):
            return {'default': v, 'overrides': {}}
        return v

    def name_for(self, role: str) -> str:
        """Имя LLM для заданной роли. Возвращает override или default."""
        return self.overrides.get(role, self.default)


# ============================================================================
# Qdrant
# ============================================================================

class QdrantConfig(BaseModel):
    host: str = 'localhost'
    port: int = 6333
    collection_docs: str = 'docs'
    collection_chunks: str = 'chunks'


# ============================================================================
# Indexing-side configs (без LLM-секций — те ушли в indexing.llm/.vision)
# ============================================================================

class DocTitleConfig(BaseModel):
    max_tokens: int | None = None
    scan_tokens: int = 32768
    scan_pages: int | None = None


class DocSummaryConfig(BaseModel):
    max_tokens: int | None = None
    mode: str = 'default'


class DocVectorConfig(BaseModel):
    max_tokens: int = 28672


class DenseEmbedderConfig(BaseModel):
    model: str
    tokenizer: str | None = None
    base_url: str | None = None
    api_key: str | None = None        # OpenAI-compatible providers; для Ollama не нужен
    dim: int | None = None
    document_template: str = '{text}'
    query_template: str = '{text}'
    timeout: int = 30
    max_retries: int = 3
    max_rpm: int | None = None
    max_concurrent: int | None = None

    @model_validator(mode='after')
    def _validate_http_dim(self) -> 'DenseEmbedderConfig':
        if self.base_url is not None and self.dim is None:
            raise ValueError('dense_embedder.dim is required when base_url is set')
        return self


class SparseEmbedderConfig(BaseModel):
    model: str = 'Alibaba-NLP/gte-multilingual-base'
    base_url: str | None = None
    timeout: int = 30
    max_retries: int = 3
    max_rpm: int | None = None
    # In-flight cap для shared semaphore: GTE CPU-bound и сериализует
    # запросы — burst > N → healthcheck timeout → autoheal restart →
    # in-flight rвутся. None = без cap'a, ограничение только через indexing.concurrency.
    max_concurrent: int | None = None


class OversizedConfig(BaseModel):
    table: str = 'transform'
    list: str = 'split'
    paragraph: str = 'split'
    fence: str = 'asis'
    diagram: str = 'asis'


class NarrateTablesConfig(BaseModel):
    """Дублирующее покрытие markdown-таблиц через per-row narrative-чанки.

    Для каждой строки таблицы (≥ min_rows строк в таблице) создаётся отдельный
    чанк вида `Header1: val1\\nHeader2: val2\\n...`. В retrieval такой чанк
    при попадании в результат заменяется на parent table-чанк (swap-to-parent),
    нужен только как точечный search-key. См. ADR-0013.
    """
    enabled: bool = False
    min_rows: int = 5            # таблицы с <5 data-строк не narrate'им (status/labels)


class ChunkerConfig(BaseModel):
    mode: str = 'hybrid'
    block_limit: int = 32000
    min_tokens: int = 50
    max_tokens: int = 250
    halving_retries: int = 0
    fallback: bool = False
    accept_pair: bool = False
    passthrough_threshold: int | None = None
    oversized: OversizedConfig = OversizedConfig()
    max_table_rows: int = 0
    narrate_tables: NarrateTablesConfig = NarrateTablesConfig()


class ContextConfig(BaseModel):
    mode: str = 'noop'
    max_tokens: int | None = None
    window_tokens: int | None = None
    chunk_max_tokens: int = 512


class KnowledgeMapConfig(BaseModel):
    enabled: bool = False
    collection: str = 'knowledge_map'
    # Алгоритм построения промпта:
    #   weighted    — adaptive: бюджет top-down пропорционально весам, per-child
    #                 решение «отдельный раздел или строка перечня» по бюджету.
    #                 Глубина зафиксирована = 2 (root → дети → стоп). Default.
    #   flat_topics — LLM-кластеризация плоского списка документов в темы.
    #                 Для источников без иерархии.
    prompt_strategy: str = 'weighted'
    # Целевой размер system prompt в токенах. Это «громкость» KM.
    prompt_budget: int = 8192
    # Порог big-vs-brief: ребёнок с пропорциональным бюджетом < node_min_tokens
    # уходит в строку перечня (`- Имя — хинт`), иначе — в отдельный раздел.
    node_min_tokens: int = 256
    exclude_source_types: list[str] = ['attached_jira', 'attached_pdf']
    flat_topics_target: int | None = None
    flat_topics_max_input_docs: int = 3000
    flat_topics_assign_batch: int = 5
    # Override параллелизма для KM-генерации. None = берём indexing.concurrency
    # (общий по индексации). Полезно когда у KM-фазы LLM-запросы тяжёлые
    # (длинный input при iterative_summarize для крупных секций) — снижаем
    # отдельно, не трогая doc-параллелизм.
    concurrency: int | None = None
    # Максимальная глубина subtree-markdown'a при сжатии (frontier-алгоритм).
    # Считается от корня раздела (= 0). Structural-документы не считаются в
    # счётчике (pass-through). 2 — оптимально для глубоких деревьев (Confluence):
    # render H2 (depth 0) + H3 (depth 1) + 1 уровень контекста (depth 2).
    # Больше = больше LLM-вызовов на глубинные узлы; меньше = LLM меньше видит.
    subtree_depth_limit: int = 2
    # Cap на размер intermediate-summary в LLM-вызове (safety против runaway,
    # не строгий target). None = без cap, LLM использует свой default.
    intermediate_max_tokens: int | None = 8192
    # Минимум токенов на элемент при expandable-рендере. Если
    # budget/(children+1) < этого порога — сразу collapse (иначе LLM не сможет
    # уложиться, brief-line хинты получаются бесполезные).
    per_element_min_tokens: int = 50


class IndexingConfig(BaseModel):
    """Индексация. Содержит ссылки на LLMs из пула через role mapping.

    `llm`/`vision`/`dense_embedder` опциональны на уровне Pydantic — при
    запуске cmd_index/cmd_serve проверяются как обязательные через setup_gate.
    Это позволяет хранить минимальный baseline config.yml в git без секретов
    и доконфигурировать через Console UI.
    """
    # LLM-роли:
    #   llm  — для всех text-задач (DocTitle, DocSummary, ContextGen, Chunker, KM)
    #   vision — для multimodal (PDF-страницы, Confluence images)
    llm: LLMRoleMapping | None = None
    vision: str | None = None  # имя LLM из пула; multimodal-задачи

    chunker: ChunkerConfig = ChunkerConfig()
    context: ContextConfig = ContextConfig()
    embed_batch_size: int = 64
    lexical_doc_summary: bool = False
    lexical_chunk_context: bool = False
    dense_embedder: DenseEmbedderConfig | None = None
    sparse_embedder: SparseEmbedderConfig = SparseEmbedderConfig()
    vision_max_tokens: int = 1024
    concurrency: int = 1
    schedule: str | None = None
    # Таймаут мягкой остановки индексации через control-plane. По истечении —
    # эскалация на kill. None (default) — ждать бесконечно, не прерывать
    # принудительно. Для жёсткой остановки используется отдельный /control/kill.
    stop_grace_seconds: int | None = None
    doc_title: DocTitleConfig = DocTitleConfig()
    doc_summary: DocSummaryConfig = DocSummaryConfig()
    doc_vector: DocVectorConfig = DocVectorConfig()
    knowledge_map: KnowledgeMapConfig = KnowledgeMapConfig()


# ============================================================================
# PDF
# ============================================================================

class PdfDoclingConfig(BaseModel):
    base_url: str = 'http://localhost:5001'
    timeout: int = 300


class PdfDeduplicateConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.7
    window: int = 5
    min_phrase_len: int = 20


class PdfPostProcessingConfig(BaseModel):
    strip_code_fences: bool = False
    dedup: PdfDeduplicateConfig = PdfDeduplicateConfig()


class PdfConfig(BaseModel):
    mode: str = 'docling'
    dpi: int = 144
    page_max_tokens: int = 4096
    concurrency: int = 1
    temperature: float = 0.0
    repetition_penalty: float | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    context_tail_lines: int = 0
    postprocessing: PdfPostProcessingConfig = PdfPostProcessingConfig()
    docling: PdfDoclingConfig = PdfDoclingConfig()


# ============================================================================
# Retrieval (агентский RAG-pipeline)
# ============================================================================

class _RetrievalRoleBase(BaseModel):
    """Базовый класс LLM-канала retrieval.

    `enable_thinking=None` — провайдер-флаги reasoning НЕ слать в extra_body.
    Полезно для xAI Grok, который реджектит unknown body fields. True/False
    шлёт `chat_template_kwargs.enable_thinking` + `reasoning_effort` +
    `reasoning.effort` (vLLM/Ollama/OpenRouter одновременно).

    Дефолты temperature/max_tokens разные у agent и reranker — отсюда
    специализированные подклассы.
    """
    llm: str = Field(min_length=1, description='Имя LLM из пула llms[]')


class RetrievalAgentConfig(_RetrievalRoleBase):
    """Agent — function-calling LLM. Открытый формат ответа, нужны токены."""
    enable_thinking: bool | None = None    # default «не слать флаги» (xAI compat)
    temperature: float = 0.3
    max_tokens: int = 4096
    # Sampling-параметры для борьбы с reasoning-петлями (Qwen3 9B при низкой
    # температуре часто зацикливается; presence_penalty + top_k/top_p — это
    # рекомендация QwenLM Issue #145). Default'ы консервативные — старые
    # конфиги без этих полей работают как раньше.
    top_p: float = 1.0
    top_k: int = 0                          # 0 = выключен (vLLM/Ollama convention)
    presence_penalty: float = 0.0


class RetrievalRerankerConfig(_RetrievalRoleBase):
    """Reranker — короткий structured вывод (номера в порядке релевантности)."""
    enable_thinking: bool | None = False   # default явное off — rerank быстрее без thinking
    temperature: float = 0.0               # детерминизм
    max_tokens: int = 100


# Алиас для обратной совместимости (тесты, внешние импорты)
RetrievalRoleConfig = RetrievalAgentConfig


class RetrievalFindSectionConfig(BaseModel):
    doc_pool: int = 20
    descent_threshold: float = 0.5
    top_docs: int = 3
    # Chunk-level peek: подтаскивает doc_id из топовых чанков (mediator для
    # глоссариев/таблиц, doc-level которых плохо ранжируется). См. ADR-0013.
    chunk_peek_limit: int = 10
    chunk_peek_docs: int = 3


class RetrievalGetDocConfig(BaseModel):
    """Параметры get_doc tool (см. DocReranker)."""
    # Override бюджета токенов на input одного rerank-батча.
    # 0 = auto: context_window − точные накладные (skeleton + chat-overhead + output_reserve).
    # >0 = принудительный потолок (для «иголка в стоге сена» — форсируем мелкие батчи).
    rerank_batch_max_tokens: int = 0


class RetrievalSearchConfig(BaseModel):
    # Максимум кандидатов из Qdrant RRF до rerank'а. Не «сколько отдать агенту» —
    # реранкер дальше обрежет по токен-бюджету и оставит сколько влезает в окно.
    # Большое значение увеличивает шанс что нужный чанк попадёт в кандидаты;
    # стоимость — больше работы Qdrant + больше токенов на rerank.
    limit: int = 100
    unique_docs_cap: int = 10
    sections_limit: int = 5
    max_iterations: int = 9
    # find_section-мандат: True (дефолт) — агент ОБЯЗАН звать find_section перед search
    # (иерархичный корпус, Confluence). False — find_section опционален (плоский корпус:
    # подкаст/собрания) → агент может идти в search/каталог напрямую.
    require_find_section: bool = True
    answer_max_tokens: int = 0
    # HNSW search-time `ef` для dense-канала. 0 = не переопределять (Qdrant default).
    # Поднимать при росте корпуса если ANN recall просел (релевантный чанк есть,
    # но dense top-N его не возвращает). См. CLAUDE.md TODO.
    hnsw_ef: int = 0
    # Override бюджета токенов на rerank-input. 0 = auto от
    # `llm.context_window - точные накладные`. >0 = ручной потолок (для
    # needle-mode — форсировать узкое окно чтобы LLM не пропускал редкие чанки).
    rerank_max_tokens: int = 0
    find_section: RetrievalFindSectionConfig = RetrievalFindSectionConfig()
    get_doc: RetrievalGetDocConfig = RetrievalGetDocConfig()


class RetrievalFeaturesConfig(BaseModel):
    enable_diversity_nudge: bool = True


# Дефолтная доменная «роль» агента в начале system prompt. Единый источник истины:
# pipeline вставляет её в _SYSTEM_PROMPT и заменяет на corpus_description (если задан),
# console показывает её в инпуте «Описание корпуса» как текущее значение.
DEFAULT_CORPUS_DESCRIPTION = 'Ты — ассистент по внутренней документации компании.'

# Дефолтный блок «Правила ответа» (стиль/дизайн ответа агента). Заменяется на
# retrieval.prompts.answer_style (если задан). Единый источник истины: pipeline
# подставляет в {answer_rules}, console показывает в инпуте «Правила/стиль ответа».
# Доменно-ориентирован (политики/свежесть/анти-конфабуляция) — для подкаста/юристов
# переопределяется. Текст обязан совпадать с блоком в morag_pipeline._SYSTEM_PROMPT.
DEFAULT_ANSWER_STYLE = (
    'Правила ответа:\n'
    '- Отвечай КРАТКО и по существу. Не пересказывай всё найденное — '
    'выбери только то, что прямо отвечает на вопрос.\n'
    '- Отвечай ТОЛЬКО на основе найденной информации из базы знаний. '
    'Не додумывай и не дополняй информацией из общих знаний.\n'
    '- ЗАПРЕЩЕНО делать выводы о политиках, правилах и разрешениях компании, '
    'если они НЕ прописаны явно в найденных документах. '
    'Наличие инструкции (например, «как настроить Mac») '
    'НЕ означает что это разрешено или рекомендовано. '
    'Если политика не описана явно — скажи что информации нет.\n'
    '- Если в базе нет ответа — честно сообщи об этом.\n'
    '- При наличии нескольких источников предпочитай более свежие документы '
    '(ориентируйся на поле «Обновлён» в результатах поиска). '
    'Если старый и новый документ противоречат — доверяй новому.\n'
)

# Дефолтные «Обязательные инструкции администратора» — добавляются в хвост промпта,
# если секция admin не переопределена. Единый источник (console преднаполняет
# редактор, pipeline применяет тот же дефолт в morag.retrieval.prompt).
DEFAULT_ADMIN_INSTRUCTIONS = (
    'Если информация не была найдена в конкретном разделе знаний '
    'или её недостаточно для полного ответа, ОБЯЗАТЕЛЬНО сделай '
    'дополнительный поиск без указания раздела (section_ids) — '
    'по всей базе знаний.'
)

# Заголовок секции «admin» в промпте — структурная обёртка вокруг текста инструкций.
# Живёт здесь (а не в prompt.py), чтобы back-compat-миграция legacy `admin_instructions`
# → section_overrides['admin'] могла обернуть raw-текст без циклического импорта.
ADMIN_HEADER = '\n\n## Обязательные инструкции администратора\n'


class RetrievalPromptsConfig(BaseModel):
    """Настройка системного промпта агента (WYSIWYG-модель).

    `section_overrides` — посекционные оверрайды промпта (id секции → полный текст
    секции). Источник истины для редактора. Ключи: role / intro / find_section_policy /
    tool_methodology / completeness / answer_rules / admin (см. `morag.retrieval.prompt`).
    Отсутствие ключа = использовать дефолт секции («сброс» = удалить ключ).

    Legacy-поля (`corpus_description` / `answer_style` / `admin_instructions`) —
    back-compat: мигрируются в `section_overrides` Pydantic-валидатором на load
    (старые деплои не ломаются). Новый ключ всегда перебивает legacy.
    """
    section_overrides: dict[str, str] = Field(default_factory=dict)
    # включать блок «### 3. ПРОВЕРКА ПОЛНОТЫ» (+ рантайм diversity-nudge). Для
    # юристов/подкаста, где ответ из одного места норма, — выключить (false).
    completeness_check: bool = True

    # --- legacy-поля (deprecated): только для миграции старых config.yml ---
    admin_instructions: str = ''
    corpus_description: str = ''
    answer_style: str = ''

    @model_validator(mode='after')
    def _migrate_legacy_prompt_fields(self):
        """Старые отдельные поля → generic section_overrides. Новый ключ важнее."""
        ov = dict(self.section_overrides)
        if self.corpus_description and 'role' not in ov:
            ov['role'] = self.corpus_description
        if self.answer_style and 'answer_rules' not in ov:
            ov['answer_rules'] = self.answer_style
        if self.admin_instructions and 'admin' not in ov:
            ov['admin'] = ADMIN_HEADER + self.admin_instructions
        self.section_overrides = ov
        return self


class RetrievalGlossaryConfig(BaseModel):
    """Подключаемый tool `lookup_glossary(query)` — обращение к одному или
    нескольким документам-глоссариям для расшифровки терминов/аббревиатур.

    `doc_ids` — список doc_id (если несколько — реранкер ранжирует чанки из
    всех вместе, агент видит top по релевантности независимо от источника).
    Старое поле `doc_id` (single) поддерживается для back-compat — при load
    конвертируется в `doc_ids=[doc_id]`.

    `enabled=True` без doc_ids — pipeline ругается при init и tool не добавляется.
    `description` — короткая фраза для system prompt.
    """
    enabled: bool = False
    doc_id: str = ''           # deprecated: для back-compat со старыми конфигами
    doc_ids: list[str] = Field(default_factory=list)
    description: str = ''

    @model_validator(mode='after')
    def _migrate_doc_id(self):
        """Если задан `doc_id` (single) и `doc_ids` пуст — мигрируем в список."""
        if self.doc_id and not self.doc_ids:
            self.doc_ids = [self.doc_id]
        return self


class RetrievalCatalogConfig(BaseModel):
    """Подключаемый tool `catalog()` — полный СТРУКТУРНЫЙ каталог всех документов
    корпуса (по строке на документ с выбранными полями payload). Закрывает дыру
    контентного RAG на запросах, требующих обойти ВЕСЬ корпус, а не найти top-k:
    «перечисли всех X», «сколько документов с Y», «где не было Z», «самый частый W».

    Агент получает таблицу целиком и сам считает/резолвит имена/группирует — для
    небольших корпусов это надёжнее и проще фильтр-движка (см. ADR/обсуждение).

    `fields` — какие поля payload включать в строку (напр. season/episode/date/
    speakers/title). `description` — фраза в system prompt про смысл корпуса (кто
    ведущие vs гости и т.п.), по ней агент различает роли при агрегации.

    `enabled=True` без `fields` — tool не добавляется (нечего показывать).
    """
    enabled: bool = False
    fields: list[str] = Field(default_factory=list)
    description: str = ''
    in_prompt: bool = False  # эксперимент: каталог-таблица в system prompt вместо KM (тогда tool не нужен)


class _RetrievalToolBase(BaseModel):
    """Общие поля инстанса агентского тула (запись списка `retrieval.tools`)."""
    enabled: bool = True
    # Имя функции для агента; '' = дефолт типа. Нужно при ≥2 инстансах одного типа.
    name: str = ''
    # Доменное описание/триггер: вставляется в описание тула и секцию system prompt
    # (для core-тулов непустое — полная замена дефолтного описания).
    description: str = ''


class RetrievalCoreToolConfig(_RetrievalToolBase):
    """Core-тулы (search/find_section/get_doc): нельзя выключить/удалить,
    description тюнится. `required` — только для find_section (политика
    «обязателен перед search»); None = взять search.require_find_section."""
    type: Literal['search', 'find_section', 'get_doc']
    required: bool | None = None

    @model_validator(mode='after')
    def _core_always_enabled(self):
        self.enabled = True  # core нельзя выключить — игнорируем enabled из конфига
        return self


class RetrievalLookupToolConfig(_RetrievalToolBase):
    """`lookup` — точечное обращение к справочным страницам (бывш. glossary).
    `trigger: abbreviations` — выверенный протокол аббревиатур (как прежний
    глоссарий); '' — generic-режим, триггер задаёт description."""
    type: Literal['lookup'] = 'lookup'
    doc_ids: list[str] = Field(default_factory=list)
    trigger: Literal['', 'abbreviations'] = ''


class RetrievalCatalogToolConfig(_RetrievalToolBase):
    """`catalog` — структурный каталог корпуса (см. RetrievalCatalogConfig —
    legacy-блок; новый способ объявления — записью в `tools`)."""
    type: Literal['catalog'] = 'catalog'
    fields: list[str] = Field(default_factory=list)


RetrievalToolConfig = Annotated[
    Union[RetrievalCoreToolConfig, RetrievalLookupToolConfig, RetrievalCatalogToolConfig],
    Field(discriminator='type'),
]


class RetrievalConfig(BaseModel):
    """Настройки агентского RAG-pipeline. Используется services/pipeline.

    Pipeline читает этот блок при старте контейнера. Изменения требуют
    `docker compose restart pipelines`. OWUI Valves остаются как override:
    значение Valve != sentinel → перебивает config.

    `agent` и `reranker` опциональные — baseline `conf/config.yml` хранит
    только нейтральные дефолты (search/features/prompts/http_timeout) без
    LLM-привязок (т.к. `llms[]` в baseline пуст). Юзер задаёт agent/reranker
    через Console UI → пишутся в `config.local.yml`.

    Если `agent` не задан → pipeline fail-soft в env-only mode (или fall back
    на agent_url из Valves). Pipeline-проверка обязательности — на runtime,
    не в Pydantic.
    """
    agent: RetrievalAgentConfig | None = None
    reranker: RetrievalRerankerConfig | None = None
    search: RetrievalSearchConfig = RetrievalSearchConfig()
    features: RetrievalFeaturesConfig = RetrievalFeaturesConfig()
    prompts: RetrievalPromptsConfig = RetrievalPromptsConfig()
    # Единый список агентских тулов (core + optional). Источник истины для
    # pipeline; legacy-блоки glossary/catalog авто-мигрируются сюда валидатором.
    tools: list[RetrievalToolConfig] = Field(default_factory=list)
    # DEPRECATED: старая форма объявления глоссария/каталога. Читается только
    # миграцией в `tools` (back-compat для существующих деплоев).
    glossary: RetrievalGlossaryConfig = RetrievalGlossaryConfig()
    catalog: RetrievalCatalogConfig = RetrievalCatalogConfig()
    http_timeout: int = 300

    @model_validator(mode='after')
    def _migrate_legacy_tool_blocks(self):
        """Legacy `glossary:`/`catalog:` блоки → записи `tools` (если тип ещё
        не объявлен явно). Существующие конфиги (morag3 и пр.) продолжают
        работать без правок; имя мигрированного глоссария — прежнее
        `lookup_glossary` (промпт/поведение агента не меняются)."""
        from morag.retrieval.tools.catalog import catalog_description
        from morag.retrieval.tools.lookup import abbr_description
        present = {t.type for t in self.tools}
        if ('lookup' not in present and self.glossary.enabled and self.glossary.doc_ids):
            # ОДНО полное описание: протокол аббревиатур + доменная шапка из старого note.
            self.tools.append(RetrievalLookupToolConfig(
                name='lookup_glossary',
                description=abbr_description(self.glossary.description),
                doc_ids=list(self.glossary.doc_ids),
                trigger='abbreviations',
            ))
        if ('catalog' not in present and self.catalog.enabled and self.catalog.fields
                and not self.catalog.in_prompt):
            self.tools.append(RetrievalCatalogToolConfig(
                description=catalog_description(list(self.catalog.fields), self.catalog.description),
                fields=list(self.catalog.fields),
            ))
        return self

    @model_validator(mode='after')
    def _validate_tool_names(self):
        """Имена тулов уникальны; core-типы — максимум по одной записи."""
        seen: set[str] = set()
        core_seen: set[str] = set()
        for t in self.tools:
            name = t.name or t.type
            if name in seen:
                raise ValueError(
                    f'Duplicate tool name {name!r} в retrieval.tools — '
                    f'у инстансов одного типа задайте разные name.'
                )
            seen.add(name)
            if t.type in ('search', 'find_section', 'get_doc'):
                if t.type in core_seen:
                    raise ValueError(f'Core-тул {t.type!r} объявлен дважды в retrieval.tools.')
                core_seen.add(t.type)
        return self


# ============================================================================
# Top-level Config
# ============================================================================

class Config(BaseModel):
    """Корень конфига morag.

    schema_version используется для миграций. Сейчас всегда 1; при breaking
    изменениях в будущем — bump + migration script.
    """
    schema_version: Literal[1] = 1
    # sources / llms могут быть пустыми в baseline config.yml — наполняются
    # через Console UI. Запуск индексации блокирует setup_gate если они пусты.
    sources: list[Source] = Field(default_factory=list)
    llms: list[LLMInstance] = Field(default_factory=list)
    qdrant: QdrantConfig = QdrantConfig()
    pdf: PdfConfig | None = None
    indexing: IndexingConfig | None = None  # None для retrieval-only setup'ов
    retrieval: RetrievalConfig | None = None  # None — pipeline в env-only mode

    # ---------- Validators ----------

    @model_validator(mode='after')
    def _validate_unique_sources(self) -> 'Config':
        """Пара (kind, name) должна быть уникальной."""
        seen: set[tuple[str, str]] = set()
        for src in self.sources:
            key = (src.kind, src.name)
            if key in seen:
                raise ValueError(
                    f'Duplicate source: kind={src.kind!r} name={src.name!r}. '
                    f'Каждая пара (kind, name) должна быть уникальной.'
                )
            seen.add(key)
        return self

    @model_validator(mode='after')
    def _validate_unique_llms(self) -> 'Config':
        """name каждой LLM в пуле — уникален."""
        seen: set[str] = set()
        for llm in self.llms:
            if llm.name in seen:
                raise ValueError(
                    f'Duplicate llm name: {llm.name!r}. Каждый LLM-инстанс '
                    f'должен иметь уникальное имя.'
                )
            seen.add(llm.name)
        return self

    @model_validator(mode='after')
    def _validate_llm_references(self) -> 'Config':
        """indexing.llm.* и indexing.vision должны ссылаться на существующие LLMs.

        Если роли не заданы (None) — проверка пропускается. Это позволяет хранить
        baseline config.yml без llms/ролей; setup_gate блокирует запуск индексации
        пока они не заполнены через UI.
        """
        if self.indexing is None:
            return self
        pool = {llm.name for llm in self.llms}

        unknown = []
        if self.indexing.llm is not None:
            if self.indexing.llm.default not in pool:
                unknown.append(f'indexing.llm.default={self.indexing.llm.default!r}')
            for role, name in self.indexing.llm.overrides.items():
                if name not in pool:
                    unknown.append(f'indexing.llm.overrides.{role}={name!r}')

        if self.indexing.vision is not None and self.indexing.vision not in pool:
            unknown.append(f'indexing.vision={self.indexing.vision!r}')

        if unknown:
            available = sorted(pool)
            raise ValueError(
                f'LLM reference(s) not found in llms pool: {", ".join(unknown)}. '
                f'Available: {available}'
            )
        return self

    @model_validator(mode='after')
    def _validate_retrieval_references(self) -> 'Config':
        """retrieval.agent.llm и retrieval.reranker.llm должны быть в llms-pool.

        Роли опциональны: если agent или reranker не заданы — пропускаем проверку
        (baseline config.yml хранит только default'ы без LLM-привязок).
        """
        if self.retrieval is None:
            return self
        pool = {llm.name for llm in self.llms}
        unknown = []
        for role_name, role in (('agent', self.retrieval.agent), ('reranker', self.retrieval.reranker)):
            if role is not None and role.llm not in pool:
                unknown.append(f'retrieval.{role_name}.llm={role.llm!r}')
        if unknown:
            raise ValueError(
                f'LLM reference(s) not found in llms pool: {", ".join(unknown)}. '
                f'Available: {sorted(pool)}'
            )
        return self

    @model_validator(mode='after')
    def _validate_role_capabilities(self) -> 'Config':
        """Каждая роль должна указывать на LLM с подходящим capability.

        - indexing.vision → должен иметь 'vision' в capabilities
        - indexing.llm.* → text-роли, дефолт capability='text' покрывает (не требует валидации)

        Валидация делается перед любой работой с провайдером — misconfig
        ловится при load_config(), а не на 47-й странице PDF.
        """
        if self.indexing is None or self.indexing.vision is None:
            return self

        vision_llm = self.llm_by_name(self.indexing.vision)
        if 'vision' not in vision_llm.capabilities:
            raise ValueError(
                f'indexing.vision={self.indexing.vision!r} → model={vision_llm.model!r}: '
                f"этот LLM не объявляет capability 'vision'. "
                f"Добавь `capabilities: [text, vision]` в llms[name='{vision_llm.name}'], "
                f'либо укажи другой multimodal-инстанс. '
                f"Текущие capabilities: {vision_llm.capabilities}"
            )
        return self

    def llm_by_name(self, name: str) -> LLMInstance:
        """Lookup helper. KeyError если name не найдено."""
        for llm in self.llms:
            if llm.name == name:
                return llm
        raise KeyError(f'LLM {name!r} not found in pool')

    def sources_by_kind(self, kind: str) -> list[Source]:
        """Все включённые источники указанного kind."""
        return [s for s in self.sources if s.kind == kind and s.enabled]

    def source_names_by_role(self, role: SourceRole) -> set[str]:
        """Set имён enabled-источников указанной роли."""
        return {s.name for s in self.sources if s.role == role and s.enabled}

    def source_roles_map(self) -> dict[str, SourceRole]:
        """source_name → role для всех enabled источников. Snapshot для retrieval."""
        return {s.name: s.role for s in self.sources if s.enabled}

    def source_kinds_map(self) -> dict[str, str]:
        """source_name → kind для всех enabled источников. Snapshot для retrieval."""
        return {s.name: s.kind for s in self.sources if s.enabled}


# ============================================================================
# Loading + overlay
# ============================================================================

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Глубокий мёрж двух dict'ов: вложенные dict — рекурсивно, остальное —
    overlay перекрывает. Списки заменяются целиком (не конкатенируются).
    """
    merged = dict(base)
    for key, overlay_value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            merged[key] = _deep_merge(base_value, overlay_value)
        else:
            merged[key] = overlay_value
    return merged


def load_config(path: str | Path = 'config.yml') -> Config:
    """Загрузить и валидировать конфиг из YAML с overlay.

    Если рядом с основным конфигом лежит `config.local.yml` — он deep-мёржится поверх.
    """
    primary_path = Path(path)
    with open(primary_path, encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    local_path = primary_path.with_name('config.local.yml')
    if local_path.exists():
        with open(local_path, encoding='utf-8') as f:
            local_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, local_data)

    return Config.model_validate(data)


__all__ = [
    'AttachmentsConfig',
    'ChunkerConfig',
    'Config',
    'ConfluenceSourceConfig',
    'ContextConfig',
    'DenseEmbedderConfig',
    'DocSummaryConfig',
    'DocTitleConfig',
    'DocVectorConfig',
    'IndexingConfig',
    'JiraSourceConfig',
    'KnowledgeMapConfig',
    'LLMInstance',
    'LLMRoleMapping',
    'LocalSourceConfig',
    'OversizedConfig',
    'PdfConfig',
    'PdfDeduplicateConfig',
    'PdfDoclingConfig',
    'PdfPostProcessingConfig',
    'QdrantConfig',
    'Source',
    'SparseEmbedderConfig',
    'ValidationError',
    'load_config',
]
