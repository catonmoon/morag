"""Pipeline отдаёт подсказки границ и привязки из Document.annotations только чанкеру с признаком
`supports_boundaries` (ADR-0027); остальным — прежний вызов без лишних аргументов."""
from unittest.mock import AsyncMock, MagicMock

from morag.indexing.chunker import ChunkResult, Chunker
from morag.indexing.context import NoopContextGenerator
from morag.indexing.pipeline import IndexingPipeline
from morag.sources.base import Source
from morag.storage.repository import ChunkRepository, DocRepository
from tests.indexing.test_pipeline import make_document, setup_source

ANN = [{'kind': 'boundary', 'at': 84.0}, {'kind': 'boundary', 'at': 333.0},
       {'kind': 'screen', 't0': 84.0, 't1': 333.0, 'text': 'x'},
       {'kind': 'ref', 'at': 140.5, 'to': 84.0, 'quote': 'вот здесь'},
       {'kind': 'ref', 'at': 200.0, 'quote': 'без референта'},
       {'kind': 'boundary'}]


class Recording(Chunker):
    supports_boundaries = True

    def __init__(self):
        self.calls = []

    async def chunk(self, block):
        return [block]

    async def chunk_with_metadata(self, text, *, paged=False, **kw):
        self.calls.append(kw)
        return [ChunkResult(text=text)]


class Plain(Recording):
    supports_boundaries = False

    async def chunk_with_metadata(self, text, *, paged=False):  # лишний kwarg уронил бы вызов
        self.calls.append({})
        return [ChunkResult(text=text)]


def _repos():
    doc_repo = AsyncMock(spec=DocRepository)
    doc_repo.get_ids_by_source_instance.return_value = set()
    doc_repo.get_payloads_by_ids.return_value = {}
    doc_repo.get_by_id.return_value = None
    return doc_repo, AsyncMock(spec=ChunkRepository)


async def _run(chunker, annotations):
    doc_repo, chunk_repo = _repos()
    pipeline = IndexingPipeline(doc_repo, chunk_repo, chunker=chunker,
                                context_generator=NoopContextGenerator(), skip_presplit=True)
    source = MagicMock(spec=Source)
    setup_source(source, [make_document(text='[A] <!-- t:0.0 --> Текст.', annotations=annotations)])
    await pipeline.run(source)
    return chunker.calls


async def test_hints_reach_supporting_chunker():
    calls = await _run(Recording(), ANN)
    assert calls == [{'boundaries': [84.0, 333.0], 'pins': [(140.5, 84.0)]}]


async def test_no_annotations_no_kwargs():
    assert await _run(Recording(), []) == [{}]


async def test_plain_chunker_gets_plain_call():
    assert await _run(Plain(), ANN) == [{}]


async def test_annotations_without_boundaries_give_nothing():
    calls = await _run(Recording(), [{'kind': 'ref', 'at': 1.0, 'to': 0.0}])
    assert calls == [{}]
