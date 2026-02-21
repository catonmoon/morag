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

    def test_load_returns_documents(self, source):
        docs = source.load()
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_finds_all_md_files(self, source):
        docs = source.load()
        ids = {d.id for d in docs}
        assert 'overview.md' in ids
        assert 'changelog.md' in ids

    def test_finds_nested_files(self, source):
        docs = source.load()
        ids = {d.id for d in docs}
        assert 'api/endpoints.md' in ids
        assert 'api/auth.md' in ids

    def test_total_count(self, source):
        docs = source.load()
        assert len(docs) == 4

    def test_id_is_relative_path(self, source):
        docs = source.load()
        for doc in docs:
            assert not Path(doc.id).is_absolute()
            assert doc.id == doc.path

    def test_source_type_is_markdown(self, source):
        docs = source.load()
        for doc in docs:
            assert doc.source_type == 'markdown'

    def test_text_is_not_empty(self, source):
        docs = source.load()
        for doc in docs:
            assert len(doc.text) > 0

    def test_updated_at_is_timezone_aware(self, source):
        docs = source.load()
        for doc in docs:
            assert doc.updated_at.tzinfo is not None

    def test_size_is_populated(self, source):
        docs = source.load()
        for doc in docs:
            assert doc.size > 0

    def test_size_matches_file_content(self, tmp_path):
        content = '# Заголовок\n\nТекст документа.'
        (tmp_path / 'test.md').write_text(content, encoding='utf-8')
        source = MarkdownSource(tmp_path)
        docs = source.load()
        assert docs[0].size == len(content.encode('utf-8'))

    def test_indexed_at_is_none_before_indexing(self, source):
        docs = source.load()
        for doc in docs:
            assert doc.indexed_at is None

    def test_payload_is_empty_by_default(self, source):
        docs = source.load()
        for doc in docs:
            assert doc.payload == {}

    def test_load_is_sorted(self, source):
        docs = source.load()
        ids = [d.id for d in docs]
        assert ids == sorted(ids)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        source = MarkdownSource(tmp_path)
        assert source.load() == []

    def test_ignores_non_md_files(self, tmp_path):
        (tmp_path / 'readme.txt').write_text('ignore me')
        (tmp_path / 'data.json').write_text('{}')
        (tmp_path / 'doc.md').write_text('# Doc')
        source = MarkdownSource(tmp_path)
        docs = source.load()
        assert len(docs) == 1
        assert docs[0].id == 'doc.md'

    def test_text_content_matches_file(self, tmp_path):
        content = '# Заголовок\n\nТекст документа.'
        (tmp_path / 'test.md').write_text(content, encoding='utf-8')
        source = MarkdownSource(tmp_path)
        docs = source.load()
        assert docs[0].text == content
