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

class _SourceBase(BaseModel):
    """Общие поля всех источников. Не используется напрямую (нет kind)."""
    name: str = Field(min_length=1, pattern=r'^[a-z0-9][a-z0-9_-]*$',
                      description='Уникальный id инстанса (lowercase, без пробелов)')
    enabled: bool = True


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
    custom_fields: list[str] = []        # опциональные кастомные поля для индексации


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


class OversizedConfig(BaseModel):
    table: str = 'transform'
    list: str = 'split'
    paragraph: str = 'split'
    fence: str = 'asis'
    diagram: str = 'asis'


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


class ContextConfig(BaseModel):
    mode: str = 'noop'
    max_tokens: int | None = None
    window_tokens: int | None = None
    chunk_max_tokens: int = 512


class KnowledgeMapConfig(BaseModel):
    enabled: bool = False
    depth: int = 2
    max_depth: int | None = None
    collection: str = 'knowledge_map'
    prompt_strategy: str = 'fixed'
    node_max_tokens: int = 256
    node_min_tokens: int = 256
    prompt_budget: int = 8192
    exclude_source_types: list[str] = ['attached_jira', 'attached_pdf']
    depth1_section_ids: list[str] = []
    flat_topics_target: int | None = None
    flat_topics_max_input_docs: int = 3000
    flat_topics_assign_batch: int = 5


class IndexingConfig(BaseModel):
    """Индексация. Содержит ссылки на LLMs из пула через role mapping."""
    # LLM-роли:
    #   llm  — для всех text-задач (DocTitle, DocSummary, ContextGen, Chunker, KM)
    #   vision — для multimodal (PDF-страницы, Confluence images)
    llm: LLMRoleMapping
    vision: str  # имя LLM из пула; multimodal-задачи

    chunker: ChunkerConfig = ChunkerConfig()
    context: ContextConfig = ContextConfig()
    embed_batch_size: int = 64
    lexical_doc_summary: bool = False
    lexical_chunk_context: bool = False
    dense_embedder: DenseEmbedderConfig
    sparse_embedder: SparseEmbedderConfig = SparseEmbedderConfig()
    vision_max_tokens: int = 1024
    concurrency: int = 1
    schedule: str | None = None
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
# Top-level Config
# ============================================================================

class Config(BaseModel):
    """Корень конфига morag.

    schema_version используется для миграций. Сейчас всегда 1; при breaking
    изменениях в будущем — bump + migration script.
    """
    schema_version: Literal[1] = 1
    sources: list[Source] = Field(min_length=1, description='Минимум один источник')
    llms: list[LLMInstance] = Field(min_length=1, description='Минимум одна LLM')
    qdrant: QdrantConfig = QdrantConfig()
    pdf: PdfConfig | None = None
    indexing: IndexingConfig | None = None  # None для retrieval-only setup'ов

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
        """indexing.llm.* и indexing.vision должны ссылаться на существующие LLMs."""
        if self.indexing is None:
            return self
        pool = {llm.name for llm in self.llms}

        # indexing.llm.default + overrides
        unknown = []
        if self.indexing.llm.default not in pool:
            unknown.append(f'indexing.llm.default={self.indexing.llm.default!r}')
        for role, name in self.indexing.llm.overrides.items():
            if name not in pool:
                unknown.append(f'indexing.llm.overrides.{role}={name!r}')

        # indexing.vision
        if self.indexing.vision not in pool:
            unknown.append(f'indexing.vision={self.indexing.vision!r}')

        if unknown:
            available = sorted(pool)
            raise ValueError(
                f'LLM reference(s) not found in llms pool: {", ".join(unknown)}. '
                f'Available: {available}'
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
        if self.indexing is None:
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
