from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class LocalDocumentsConfig(BaseModel):
    path: str


class AttachmentsConfig(BaseModel):
    enabled: bool = False                                    # включить обработку вложений
    mime_types: list[str] = ['application/pdf']              # фильтр по MIME-типу; пока только PDF
    skip_ancestor_ids: list[str] = []                        # пропускать вложения со страниц-потомков этих разделов
    url_mode: str = 'preview'                                # preview | download | parent_page


class ConfluenceConfig(BaseModel):
    url: str
    username: str
    password: str | None = None        # on-premise
    api_token: str | None = None       # Atlassian Cloud
    spaces: list[str] = []             # список space key для индексации; пусто — все доступные
    ancestor_ids: list[str] = []       # фильтр по ancestor page id; пусто — без фильтра
    skip_ancestor_ids: list[str] = []  # исключить страницы и всех их потомков
    min_image_size_bytes: int | None = None  # пропускать изображения меньше этого размера (байт); None — без фильтрации
    timeout: int = 180  # таймаут HTTP-запросов к Confluence API и скачивания изображений (секунды)
    max_retries: int = 3  # количество повторных попыток при сетевых ошибках (urllib3 Retry); 0 = без retry
    attachments: AttachmentsConfig = AttachmentsConfig()     # обработка вложений (PDF и др.)


class JiraConfig(BaseModel):
    url: str
    username: str
    password: str | None = None        # on-premise
    api_token: str | None = None       # Atlassian Cloud
    timeout: int = 180                 # таймаут HTTP-запросов к Jira API (секунды)
    max_retries: int = 3              # количество повторных попыток при сетевых ошибках (urllib3 Retry); 0 = без retry
    custom_fields: list[str] = []     # список ID кастомных полей (например ['customfield_10100']); названия берутся из Jira API


class SourcesConfig(BaseModel):
    local_documents: LocalDocumentsConfig | None = None
    confluence: ConfluenceConfig | None = None
    jira: JiraConfig | None = None


class QdrantConfig(BaseModel):
    host: str = 'localhost'
    port: int = 6333
    collection_docs: str = 'docs'
    collection_chunks: str = 'chunks'


class RetryConfig(BaseModel):
    max_retries: int = 3    # количество повторных попыток (0 = без retry)


class DocTitleConfig(BaseModel):
    max_tokens: int | None = None  # лимит токенов ответа LLM; None — генерация названия отключена
    scan_tokens: int = 32768       # глубина просмотра документа (токены от начала)
    scan_pages: int | None = None  # альтернатива: взять первые N страниц (если есть маркеры)


class DocSummaryConfig(BaseModel):
    max_tokens: int | None = None  # лимит токенов ответа LLM; None — генерация саммари отключена
    mode: str = 'default'  # режим промпта: 'default' (универсальный) или 'legal' (юридические документы)


class LLMConfig(BaseModel):
    base_url: str = 'http://localhost:11434/v1'
    model: str = 'qwen2.5-coder:7b'
    api_key: str = 'ollama'
    timeout: int = 180  # таймаут HTTP-запросов к LLM (секунды)
    context_window: int = 32768   # контекстное окно модели (токенов)
    max_tokens: int | None = None  # лимит токенов ответа; None — без ограничения
    retry: RetryConfig = RetryConfig()
    model_wait_seconds: int = 0   # ожидание перезагрузки модели (сек); 0 = не ждать
    model_wait_retries: int = 0   # количество попыток ожидания модели
    enable_thinking: bool | None = None  # включить/выключить thinking; None = поведение модели по умолчанию
    max_rpm: int | None = None    # лимит запросов в минуту; None = без ограничения


class DenseEmbedderConfig(BaseModel):
    model: str = 'ai-forever/FRIDA'
    base_url: str | None = None   # если задан → HTTP-режим; иначе — локальная модель
    dim: int | None = None        # обязателен в HTTP-режиме; в local-режиме определяется автоматически
    timeout: int = 30             # таймаут HTTP-запросов (секунды; только в HTTP-режиме)
    retry: RetryConfig = RetryConfig()  # политика повторных попыток (только в HTTP-режиме)
    max_rpm: int | None = None    # лимит запросов в минуту (только в HTTP-режиме); None = без ограничения

    @model_validator(mode='after')
    def _validate_http_dim(self) -> 'DenseEmbedderConfig':
        if self.base_url is not None and self.dim is None:
            raise ValueError('dense_embedder.dim is required when base_url is set (HTTP mode)')
        return self


class SparseEmbedderConfig(BaseModel):
    model: str = 'Alibaba-NLP/gte-multilingual-base'
    device: str | None = None     # устройство для local-режима: 'cpu' | 'mps' | 'cuda' | None (авто)
    base_url: str | None = None   # если задан → HTTP-режим; иначе — локальная модель
    timeout: int = 30             # таймаут HTTP-запросов (секунды; только в HTTP-режиме)
    retry: RetryConfig = RetryConfig()  # политика повторных попыток (только в HTTP-режиме)
    max_rpm: int | None = None    # лимит запросов в минуту (только в HTTP-режиме); None = без ограничения


class OversizedConfig(BaseModel):
    """Стратегии обработки oversized блоков по типам (hybrid chunker).

    Каждый тип блока > max_tokens обрабатывается своей стратегией:
    - asis: оставить как есть (один большой чанк)
    - split: структурное разбиение (предложения / элементы / строки)
    - embed: SemanticChunker (embedding-based границы)
    - transform: преобразовать формат + рекурсия через чанкинг
    - llm: LLM преобразует/разобьёт + рекурсия
    """
    table: str = 'transform'        # строка → key-value текст → рекурсия
    list: str = 'split'             # по элементам списка
    paragraph: str = 'split'        # по предложениям
    fence: str = 'asis'             # код как есть
    diagram: str = 'asis'           # диаграмма как есть


class ChunkerConfig(BaseModel):
    mode: str = 'hybrid'                # 'hybrid' | 'semantic' | 'passthrough' | 'llm'
    block_limit: int = 32000             # лимит токенов для pre-split блока (llm/passthrough)
    min_tokens: int = 50                 # мин. размер чанка в токенах (semantic/hybrid)
    max_tokens: int = 250                # макс. размер чанка в токенах (semantic/hybrid)
    halving_retries: int = 0             # деления блока пополам при таймауте LLM; 0 = выключено
    fallback: bool = False               # семантический fallback; False = при неудаче документ пропускается
    accept_pair: bool = False            # принимать оба чанка пары (left+right) за одну итерацию (2x быстрее)
    passthrough_threshold: int | None = None  # если документ > N токенов → passthrough вместо semantic; None = отключено
    oversized: OversizedConfig = OversizedConfig()  # стратегии по типу блока (hybrid)


class ContextConfig(BaseModel):
    mode: str = 'noop'                   # 'noop' | 'llm'
    max_tokens: int | None = None        # лимит токенов в ответе; None — без ограничения (или адаптивно если embedder_max_tokens)
    window_tokens: int | None = None     # окно вокруг позиции чанка (токены); None = отправлять весь документ
    chunk_max_tokens: int = 512            # макс. токенов на чанк (text + context + path); адаптивный context = chunk_max - text - path_overhead


class KnowledgeMapConfig(BaseModel):
    enabled: bool = False                # генерация карты документации после индексации
    depth: int = 2                       # кол-во уровней в системном промпте
    max_depth: int | None = None         # макс. глубина обхода дерева; None = до самого дна
    collection: str = 'knowledge_map'    # коллекция Qdrant для карт
    prompt_strategy: str = 'fixed'       # 'fixed' (node_max_tokens на узел) | 'weighted' (prompt_budget по потомкам)
    node_max_tokens: int = 256           # для fixed: лимит токенов на описание каждого узла
    node_min_tokens: int = 256           # для weighted: минимальный бюджет на узел (защита от обрывов)
    prompt_budget: int = 8192            # для weighted: общий бюджет токенов на системный промпт
    enable_thinking: bool = False        # включить thinking-режим LLM; по умолчанию выключен


class IndexingConfig(BaseModel):
    chunker: ChunkerConfig = ChunkerConfig()
    context: ContextConfig = ContextConfig()
    embed_batch_size: int = 64            # размер батча для embed + upsert чанков
    dense_embedder: DenseEmbedderConfig = DenseEmbedderConfig()
    sparse_embedder: SparseEmbedderConfig = SparseEmbedderConfig()
    vision_max_tokens: int = 1024  # лимит токенов ответа Vision LLM (изображения, формулы)
    concurrency: int = 1  # количество документов, обрабатываемых параллельно
    schedule: str | None = None  # cron-выражение для serve-режима (например '0 */6 * * *')
    doc_title: DocTitleConfig = DocTitleConfig()  # генерация названия документа; max_tokens=None — отключено
    doc_summary: DocSummaryConfig = DocSummaryConfig()  # генерация саммари документов; max_tokens=None — отключено
    knowledge_map: KnowledgeMapConfig = KnowledgeMapConfig()  # карта документации (ADR-0010)


class PdfDoclingConfig(BaseModel):
    base_url: str = 'http://localhost:5001'  # URL docling-serve
    timeout: int = 300                       # таймаут конвертации документа (секунды)


class PdfDeduplicateConfig(BaseModel):
    enabled: bool = False                    # включить дедупликацию
    threshold: float = 0.7                   # порог fuzzy-сходства (0..1)
    window: int = 5                          # скользящее окно (предыдущих абзацев)
    min_phrase_len: int = 20                 # мин. длина фразы для дедупликации


class PdfPostProcessingConfig(BaseModel):
    strip_code_fences: bool = False              # удалять orphan code fences из ответов Vision LLM
    dedup: PdfDeduplicateConfig = PdfDeduplicateConfig()


class PdfConfig(BaseModel):
    mode: str = 'docling'                    # 'docling' | 'vision'
    dpi: int = 144                           # разрешение рендеринга страниц (vision-режим)
    page_max_tokens: int = 4096              # лимит токенов ответа LLM на страницу (vision-режим)
    concurrency: int = 1                     # параллельных запросов к Vision LLM (vision-режим)
    temperature: float = 0.0                 # температура генерации (0 = детерминированная)
    repetition_penalty: float | None = None  # штраф за повторы (>1.0); None = не передавать
    frequency_penalty: float = 0.0           # OpenAI-стандартный штраф за частоту токенов
    presence_penalty: float = 0.0            # OpenAI-стандартный штраф за наличие токенов
    context_tail_lines: int = 0              # sliding window: строк хвоста предыдущей страницы (0 = выкл)
    postprocessing: PdfPostProcessingConfig = PdfPostProcessingConfig()
    docling: PdfDoclingConfig = PdfDoclingConfig()  # настройки docling-serve (docling-режим)


class Config(BaseModel):
    sources: SourcesConfig
    qdrant: QdrantConfig = QdrantConfig()
    llm: LLMConfig = LLMConfig()
    llm_vision: LLMConfig | None = None  # multimodal LLM для распознавания изображений (опционально)
    pdf: PdfConfig | None = None         # конвертация PDF → Markdown (опционально; mode: docling | vision)
    indexing: IndexingConfig = IndexingConfig()


def load_config(path: str | Path = 'config.yml') -> Config:
    """Загрузить и валидировать конфиг из YAML-файла."""
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)
