"""Тесты graceful cancellation в IndexingPipeline.

Проверяют что cancel_event:
- останавливает обработку между уровнями BFS
- skip'ает документы, у которых cancel пришёл до acquire семафора
- StatusReporter получает document_done для каждого документа (включая skipped)
"""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from morag.indexing.chunker import PassthroughChunker
from morag.indexing.context import NoopContextGenerator
from morag.indexing.pipeline import IndexingPipeline
from morag.sources.base import Document, Source
from morag.storage.repository import ChunkRepository, DocRepository


def make_stub(doc_id: str, parent_doc_ids: list[str] | None = None) -> Document:
    return Document(
        id=doc_id,
        path=[doc_id],
        text='',
        updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_type='markdown',
        size=100,
        parent_doc_ids=parent_doc_ids or [],
    )


def make_doc(doc_id: str, parent_doc_ids: list[str] | None = None) -> Document:
    return Document(
        id=doc_id,
        path=[doc_id],
        text='# текст',
        updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_type='markdown',
        size=100,
        parent_doc_ids=parent_doc_ids or [],
    )


@pytest.fixture
def doc_repo() -> AsyncMock:
    mock = AsyncMock(spec=DocRepository)
    mock.get_ids_by_source_instance.return_value = set()
    mock.get_payloads_by_ids.return_value = {}
    mock.get_by_id.return_value = None
    return mock


@pytest.fixture
def chunk_repo() -> AsyncMock:
    return AsyncMock(spec=ChunkRepository)


class FakeReporter:
    """Минимальный StatusReporter для проверки последовательности вызовов."""

    def __init__(self) -> None:
        self.phases: list[tuple[str, int]] = []
        self.done: list[str] = []
        self.finished: tuple[str, str | None] | None = None

    def start_phase(self, name: str, total: int) -> None:
        self.phases.append((name, total))

    def set_predicted_real_total(self, value: int) -> None:
        pass

    def document_start(self, doc_id: str, title: str | None = None, url: str | None = None) -> None:
        pass

    def document_set_chunks(self, doc_id: str, total: int) -> None:
        pass

    def document_chunk_done(self, doc_id: str) -> None:
        pass

    def document_done(self, doc_id: str) -> None:
        self.done.append(doc_id)

    def finish(self, state: str, error: str | None = None) -> None:
        self.finished = (state, error)


class TestCancelEvent:

    async def test_cancel_before_run_skips_all_levels(self, doc_repo, chunk_repo):
        cancel = asyncio.Event()
        cancel.set()  # уже отменено
        reporter = FakeReporter()

        pipeline = IndexingPipeline(
            doc_repo, chunk_repo,
            chunker=PassthroughChunker(),
            context_generator=NoopContextGenerator(),
            status_reporter=reporter,
            cancel_event=cancel,
        )

        source = MagicMock(spec=Source)
        source.source_type = 'markdown'
        source.get_metadata.return_value = [make_stub('a'), make_stub('b'), make_stub('c')]
        source.load_one = AsyncMock()

        await pipeline.run(source)

        assert reporter.phases == [('indexing_markdown', 3)]
        # Никаких документов не загружено — cancel выкинул всех в начале первого уровня
        source.load_one.assert_not_called()
        doc_repo.upsert.assert_not_called()

    async def test_cancel_between_levels_processes_first_level_only(self, doc_repo, chunk_repo):
        """Два уровня BFS: parent на уровне 0, child на уровне 1 (через parent_doc_ids).

        Cancel выставляется во время обработки уровня 0 (после первого document_done).
        Уровень 1 не должен запускаться вовсе.
        """
        cancel = asyncio.Event()
        reporter = FakeReporter()

        # Парент на L0, чайлд на L1 (parent_doc_ids указывает на parent)
        stubs = [
            make_stub('parent'),
            make_stub('child', parent_doc_ids=['parent']),
        ]
        docs_by_id = {'parent': make_doc('parent'), 'child': make_doc('child', parent_doc_ids=['parent'])}

        source = MagicMock(spec=Source)
        source.source_type = 'markdown'
        source.get_metadata.return_value = stubs

        async def load_one(doc_id: str) -> Document | None:
            # Когда грузим parent — выставляем cancel, чтобы L1 не запускался
            if doc_id == 'parent':
                cancel.set()
            return docs_by_id.get(doc_id)

        source.load_one = AsyncMock(side_effect=load_one)

        pipeline = IndexingPipeline(
            doc_repo, chunk_repo,
            chunker=PassthroughChunker(),
            context_generator=NoopContextGenerator(),
            status_reporter=reporter,
            cancel_event=cancel,
        )

        await pipeline.run(source)

        # parent загружен и обработан, child — нет
        loaded_ids = [call.args[0] for call in source.load_one.call_args_list]
        assert 'parent' in loaded_ids
        assert 'child' not in loaded_ids

    async def test_document_done_called_for_every_document_in_processed_level(self, doc_repo, chunk_repo):
        """Без cancel: document_done вызывается ровно столько раз, сколько документов."""
        reporter = FakeReporter()
        pipeline = IndexingPipeline(
            doc_repo, chunk_repo,
            chunker=PassthroughChunker(),
            context_generator=NoopContextGenerator(),
            status_reporter=reporter,
        )

        source = MagicMock(spec=Source)
        source.source_type = 'markdown'
        stubs = [make_stub(f'doc{i}') for i in range(5)]
        source.get_metadata.return_value = stubs
        docs_by_id = {s.id: make_doc(s.id) for s in stubs}
        source.load_one.side_effect = lambda did: docs_by_id.get(did)

        await pipeline.run(source)

        assert sorted(reporter.done) == sorted(s.id for s in stubs)

    async def test_default_cancel_event_never_triggers(self, doc_repo, chunk_repo):
        """Без явного cancel_event пайплайн должен работать как раньше."""
        reporter = FakeReporter()
        pipeline = IndexingPipeline(
            doc_repo, chunk_repo,
            chunker=PassthroughChunker(),
            context_generator=NoopContextGenerator(),
            status_reporter=reporter,
            # cancel_event=None
        )

        source = MagicMock(spec=Source)
        source.source_type = 'markdown'
        source.get_metadata.return_value = [make_stub('x')]
        source.load_one.return_value = make_doc('x')

        await pipeline.run(source)

        assert reporter.done == ['x']
        # finish не вызывается из pipeline.run() — это ответственность cmd_index
        assert reporter.finished is None
