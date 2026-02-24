from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class MarkdownSourceConfig(BaseModel):
    path: str


class SourcesConfig(BaseModel):
    markdown: MarkdownSourceConfig


class QdrantConfig(BaseModel):
    host: str = 'localhost'
    port: int = 6333
    collection_docs: str = 'docs'
    collection_chunks: str = 'chunks'


class LLMConfig(BaseModel):
    base_url: str = 'http://localhost:11434/v1'
    model: str = 'qwen2.5-coder:7b'
    api_key: str = 'ollama'


class IndexingConfig(BaseModel):
    chunker: str = 'passthrough'         # 'passthrough' | 'llm'
    context: str = 'noop'               # 'noop' | 'llm'
    block_limit: int = 32000
    llm_context_window: int = 32768     # контекстное окно LLM (токенов); используется для расчёта безопасного лимита блока
    dense_model: str = 'ai-forever/FRIDA'  # модель для dense-эмбеддингов
    sparse_model: str = 'Alibaba-NLP/gte-multilingual-base'  # модель для sparse-эмбеддингов
    sparse_device: str | None = None  # устройство для sparse-модели: 'cpu' | 'mps' | 'cuda' | None (авто)


class Config(BaseModel):
    sources: SourcesConfig
    qdrant: QdrantConfig = QdrantConfig()
    llm: LLMConfig = LLMConfig()
    indexing: IndexingConfig = IndexingConfig()


def load_config(path: str | Path = 'config.yml') -> Config:
    """Загрузить и валидировать конфиг из YAML-файла."""
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)
