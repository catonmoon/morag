from pathlib import Path

import pytest

from morag.sources.base import Document, Source
from morag.sources.markdown import MarkdownSource

FIXTURES_DIR = Path(__file__).parent.parent / 'fixtures' / 'docs'


@pytest.fixture
def source() -> MarkdownSource:
    return MarkdownSource(FIXTURES_DIR)


class TestMarkdownSource:
    def test_is_source(self, source):
        assert isinstance(source, Source)

    async def test_load_returns_documents(self, source):
        docs = await source.load()
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    async def test_finds_all_md_files(self, source):
        docs = await source.load()
        ids = {d.id for d in docs}
        assert 'overview.md' in ids
        assert 'changelog.md' in ids

    async def test_finds_nested_files(self, source):
        docs = await source.load()
        ids = {d.id for d in docs}
        assert 'api/endpoints.md' in ids
        assert 'api/auth.md' in ids

    async def test_total_count(self, source):
        docs = await source.load()
        # 4 MD-файла (директории создаёт DirectorySource)
        assert len(docs) == 4

    async def test_id_is_relative_path(self, source):
        docs = await source.load()
        for doc in docs:
            assert not Path(doc.id).is_absolute()
            assert doc.path == [doc.id]

    async def test_source_type_is_markdown(self, source):
        docs = await source.load()
        for doc in docs:
            assert doc.source_type == 'markdown'

    async def test_text_is_not_empty(self, source):
        docs = await source.load()
        for doc in docs:
            assert len(doc.text) > 0

    async def test_updated_at_is_timezone_aware(self, source):
        docs = await source.load()
        for doc in docs:
            assert doc.updated_at.tzinfo is not None

    async def test_size_is_populated(self, source):
        docs = await source.load()
        for doc in docs:
            assert doc.size > 0

    async def test_size_matches_file_content(self, tmp_path):
        content = '# Заголовок\n\nТекст документа.'
        (tmp_path / 'test.md').write_text(content, encoding='utf-8')
        source = MarkdownSource(tmp_path)
        docs = await source.load()
        assert docs[0].size == len(content.encode('utf-8'))

    async def test_indexed_at_is_none_before_indexing(self, source):
        docs = await source.load()
        for doc in docs:
            assert doc.indexed_at is None

    async def test_payload_is_empty_by_default(self, source):
        docs = await source.load()
        for doc in docs:
            assert doc.payload == {}

    async def test_load_is_sorted(self, source):
        docs = await source.load()
        ids = [d.id for d in docs]
        assert ids == sorted(ids)

    async def test_empty_directory_returns_empty_list(self, tmp_path):
        source = MarkdownSource(tmp_path)
        assert await source.load() == []

    async def test_ignores_non_md_files(self, tmp_path):
        (tmp_path / 'readme.txt').write_text('ignore me')
        (tmp_path / 'data.json').write_text('{}')
        (tmp_path / 'doc.md').write_text('# Doc')
        source = MarkdownSource(tmp_path)
        docs = await source.load()
        assert len(docs) == 1
        assert docs[0].id == 'doc.md'

    async def test_text_content_matches_file(self, tmp_path):
        content = '# Заголовок\n\nТекст документа.'
        (tmp_path / 'test.md').write_text(content, encoding='utf-8')
        source = MarkdownSource(tmp_path)
        docs = await source.load()
        assert docs[0].text == content


class TestMarkdownSourceGetMetadata:
    async def test_returns_stubs_with_empty_text(self, source):
        stubs = await source.get_metadata()
        assert all(d.text == '' for d in stubs)

    async def test_same_count_as_load(self, source):
        stubs = await source.get_metadata()
        docs = await source.load()
        assert len(stubs) == len(docs)

    async def test_stubs_have_correct_ids(self, source):
        stubs = await source.get_metadata()
        ids = {s.id for s in stubs}
        assert 'overview.md' in ids
        assert 'changelog.md' in ids

    async def test_stubs_source_type_is_markdown(self, source):
        stubs = await source.get_metadata()
        assert all(s.source_type == 'markdown' for s in stubs)

    async def test_stubs_size_matches_load(self, source):
        stubs = await source.get_metadata()
        docs = await source.load()
        stubs_by_id = {s.id: s for s in stubs}
        for doc in docs:
            assert stubs_by_id[doc.id].size == doc.size

    async def test_stubs_updated_at_matches_load(self, source):
        stubs = await source.get_metadata()
        docs = await source.load()
        stubs_by_id = {s.id: s for s in stubs}
        for doc in docs:
            assert stubs_by_id[doc.id].updated_at == doc.updated_at

    async def test_empty_directory_returns_empty_list(self, tmp_path):
        source = MarkdownSource(tmp_path)
        assert await source.get_metadata() == []

    async def test_stubs_are_sorted(self, source):
        stubs = await source.get_metadata()
        ids = [s.id for s in stubs]
        assert ids == sorted(ids)


class TestMarkdownSourceLoadOne:
    async def test_returns_full_document(self, source):
        doc = await source.load_one('overview.md')
        assert doc is not None
        assert doc.id == 'overview.md'
        assert len(doc.text) > 0

    async def test_text_matches_file_content(self, tmp_path):
        content = '# Тест\n\nТекст файла.'
        (tmp_path / 'test.md').write_text(content, encoding='utf-8')
        source = MarkdownSource(tmp_path)
        doc = await source.load_one('test.md')
        assert doc is not None
        assert doc.text == content

    async def test_returns_none_for_nonexistent_id(self, source):
        doc = await source.load_one('nonexistent.md')
        assert doc is None

    async def test_nested_path(self, source):
        doc = await source.load_one('api/endpoints.md')
        assert doc is not None
        assert doc.id == 'api/endpoints.md'
