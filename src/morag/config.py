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


class Config(BaseModel):
    sources: SourcesConfig
    qdrant: QdrantConfig = QdrantConfig()


def load_config(path: str | Path = 'config.yml') -> Config:
    """Загрузить и валидировать конфиг из YAML-файла."""
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)
