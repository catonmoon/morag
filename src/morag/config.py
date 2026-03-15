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


class DocSummaryConfig(BaseModel):
    max_tokens: int | None = None  # лимит токенов ответа LLM; None — генерация саммари отключена


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


class DenseEmbedderConfig(BaseModel):
    model: str = 'ai-forever/FRIDA'
    base_url: str | None = None   # если задан → HTTP-режим; иначе — локальная модель
    dim: int | None = None        # обязателен в HTTP-режиме; в local-режиме определяется автоматически
    timeout: int = 30             # таймаут HTTP-запросов (секунды; только в HTTP-режиме)
    retry: RetryConfig = RetryConfig()  # политика повторных попыток (только в HTTP-режиме)

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


class ChunkerConfig(BaseModel):
    mode: str = 'semantic'               # 'semantic' | 'passthrough' | 'llm'
    block_limit: int = 32000             # лимит токенов для pre-split блока (llm/passthrough)
    min_tokens: int = 50                 # мин. размер чанка в токенах (semantic)
    max_tokens: int = 250                # макс. размер чанка в токенах (semantic)
    halving_retries: int = 0             # деления блока пополам при таймауте LLM; 0 = выключено
    fallback: bool = False               # семантический fallback; False = при неудаче документ пропускается


class ContextConfig(BaseModel):
    mode: str = 'noop'                   # 'noop' | 'llm'
    max_tokens: int | None = None        # лимит токенов в ответе LLMContextGenerator; None — без ограничения


class IndexingConfig(BaseModel):
    chunker: ChunkerConfig = ChunkerConfig()
    context: ContextConfig = ContextConfig()
    dense_embedder: DenseEmbedderConfig = DenseEmbedderConfig()
    sparse_embedder: SparseEmbedderConfig = SparseEmbedderConfig()
    vision_max_tokens: int = 1024  # лимит токенов ответа Vision LLM (изображения, формулы)
    concurrency: int = 1  # количество документов, обрабатываемых параллельно
    schedule: str | None = None  # cron-выражение для serve-режима (например '0 */6 * * *')
    doc_summary: DocSummaryConfig = DocSummaryConfig()  # генерация саммари документов; max_tokens=None — отключено


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
