"""Retrieval-слой: поиск, реранкер, section-level retrieval.

OWUI/pipelines-независимый. Всё async, зависит только от morag.indexing (bm25,
embedder) и morag.llm (client). Используется в retrieval-pipeline
(`services/pipeline/morag_pipeline.py`), CLI-скриптах, тестах.

**Ленивый импорт (PEP 562 `__getattr__`).** Сам по себе `import morag.retrieval`
(в т.ч. транзитом при `import morag.retrieval.tools`) НЕ тянет рерэнкеры/searcher,
а с ними indexing-зависимости (tiktoken/numpy/nltk). Это позволяет
lightweight-консоли импортировать `morag.retrieval.tools` (ToolSpec REGISTRY) без
[indexing]-extras. Тяжёлые символы (`HybridSearcher` и пр.) подгружаются при
ПЕРВОМ обращении — pipeline/CLI получают их как раньше (`from morag.retrieval
import HybridSearcher`), консоль их просто не трогает.
"""
# имя символа → модуль, из которого его лениво импортировать.
_LAZY = {
    'DocReranker': 'morag.retrieval.doc_reranker',
    'LLMReranker': 'morag.retrieval.reranker',
    'HybridSearcher': 'morag.retrieval.searcher',
    'ExtraDoc': 'morag.retrieval.sections',
    'FindSectionConfig': 'morag.retrieval.sections',
    'SectionEntry': 'morag.retrieval.sections',
    'SectionResult': 'morag.retrieval.sections',
    'aggregate_to_sections': 'morag.retrieval.sections',
    'descend_section': 'morag.retrieval.sections',
    'find_section': 'morag.retrieval.sections',
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    """PEP 562: ленивый резолв публичных символов слоя (импорт модуля при первом
    обращении + кэш в globals, чтобы повторно не импортировать)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    import importlib

    obj = getattr(importlib.import_module(module), name)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted(__all__)
