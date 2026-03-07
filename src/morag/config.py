from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class LocalDocumentsConfig(BaseModel):
    path: str


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


class JiraConfig(BaseModel):
    url: str
    username: str
    password: str | None = None        # on-premise
    api_token: str | None = None       # Atlassian Cloud
    timeout: int = 180                 # таймаут HTTP-запросов к Jira API (секунды)


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
    delay: float = 1.0      # начальная задержка между попытками (секунды)
    backoff: float = 2.0    # множитель: delay → delay*backoff → delay*backoff² → ...


class DocSummaryConfig(BaseModel):
    max_tokens: int | None = None  # лимит токенов ответа LLM; None — генерация саммари отключена


class LLMConfig(BaseModel):
    base_url: str = 'http://localhost:11434/v1'
    model: str = 'qwen2.5-coder:7b'
    api_key: str = 'ollama'
    timeout: int = 180  # таймаут HTTP-запросов к LLM (секунды)
    retry: RetryConfig = RetryConfig()


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


class IndexingConfig(BaseModel):
    chunker: str = 'passthrough'         # 'passthrough' | 'llm'
    context: str = 'noop'               # 'noop' | 'llm'
    block_limit: int = 32000
    llm_context_window: int = 32768     # контекстное окно LLM (токенов); используется для расчёта безопасного лимита блока
    context_max_output_tokens: int | None = None  # лимит токенов в ответе LLMContextGenerator; None — без ограничения
    dense_embedder: DenseEmbedderConfig = DenseEmbedderConfig()
    sparse_embedder: SparseEmbedderConfig = SparseEmbedderConfig()
    concurrency: int = 1  # количество документов, обрабатываемых параллельно
    schedule: str | None = None  # cron-выражение для serve-режима (например '0 */6 * * *')
    doc_summary: DocSummaryConfig = DocSummaryConfig()  # генерация саммари документов; max_tokens=None — отключено


class Config(BaseModel):
    sources: SourcesConfig
    qdrant: QdrantConfig = QdrantConfig()
    llm: LLMConfig = LLMConfig()
    llm_vision: LLMConfig | None = None  # multimodal LLM для распознавания изображений (опционально)
    indexing: IndexingConfig = IndexingConfig()


def load_config(path: str | Path = 'config.yml') -> Config:
    """Загрузить и валидировать конфиг из YAML-файла."""
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)
