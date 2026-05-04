"""Embedder fingerprint — детект stale-векторов при смене модели.

См. ADR-0012, секция «4. Stable payload-fields».

При индексации каждый документ помечается `payload['embedder_fingerprint']`.
В `_is_up_to_date` сравнивается с current fingerprint — mismatch → reindex.

Защита от silent staleness: если поменяли embedder (например, FRIDA → Qwen3),
старые документы автоматически переиндексируются в следующий прогон, без
ручного `--reset`.
"""
from __future__ import annotations

import hashlib


def compute_embedder_fingerprint(model: str, dim: int, base_url: str | None) -> str:
    """SHA-256 short fingerprint от ключевых атрибутов dense embedder'а.

    base_url включён потому что одна и та же model может работать через разные
    серверы (Ollama vs vLLM), генерируя разные векторы (разные веса/quantization).
    Изменение URL → mismatch → reindex.

    Возвращает 16-char hex prefix — достаточно уникально, читабельно в логах.
    """
    ingredients = f'{model}|{dim}|{base_url or ""}'
    digest = hashlib.sha256(ingredients.encode('utf-8')).hexdigest()
    return digest[:16]
