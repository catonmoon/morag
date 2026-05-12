"""Retrieval-слой: поиск, реранкер, section-level retrieval.

OWUI/pipelines-независимый. Всё async, зависит только от morag.indexing (bm25,
embedder) и morag.llm (client). Используется в retrieval-pipeline
(`services/pipeline/morag_pipeline.py`), CLI-скриптах, тестах.
"""

from morag.retrieval.doc_reranker import DocReranker
from morag.retrieval.reranker import LLMReranker
from morag.retrieval.searcher import HybridSearcher
from morag.retrieval.sections import (
    ExtraDoc,
    FindSectionConfig,
    SectionEntry,
    SectionResult,
    aggregate_to_sections,
    descend_section,
    find_section,
)

__all__ = [
    'LLMReranker',
    'DocReranker',
    'HybridSearcher',
    'FindSectionConfig',
    'SectionResult',
    'SectionEntry',
    'ExtraDoc',
    'find_section',
    'aggregate_to_sections',
    'descend_section',
]
