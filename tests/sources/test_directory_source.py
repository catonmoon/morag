from pathlib import Path

import pytest

from morag.sources.base import Source
from morag.sources.directory import DirectorySource

FIXTURES_DIR = Path(__file__).parent.parent / 'fixtures' / 'docs'


@pytest.fixture
def source() -> DirectorySource:
    return DirectorySource(FIXTURES_DIR)


class TestDirectorySource:
    def test_is_source(self, source):
        assert isinstance(source, Source)

    def test_source_type(self, source):
        assert source.source_type == 'directory'

    async def test_get_metadata_returns_only_directories(self, source):
        stubs = await source.get_metadata()
        assert len(stubs) > 0
        for stub in stubs:
            assert stub.id.endswith('/')
            assert stub.structural is True
            assert stub.source_type == 'directory'

    async def test_finds_api_directory(self, source):
        stubs = await source.get_metadata()
        ids = {s.id for s in stubs}
        assert 'local:default:api/' in ids

    async def test_stubs_have_empty_text(self, source):
        stubs = await source.get_metadata()
        for stub in stubs:
            assert stub.text == ''

    async def test_stubs_size_is_zero(self, source):
        stubs = await source.get_metadata()
        for stub in stubs:
            assert stub.size == 0

    async def test_load_one_returns_structural_doc(self, source):
        doc = await source.load_one('local:default:api/')
        assert doc is not None
        assert doc.id == 'local:default:api/'
        assert doc.structural is True
        assert doc.text == 'api'  # имя директории
        assert doc.source_type == 'directory'

    async def test_load_one_nonexistent(self, source):
        doc = await source.load_one('local:default:nonexistent/')
        assert doc is None

    async def test_parent_doc_ids_root_dirs(self, source):
        stubs = await source.get_metadata()
        # api/ — корневая директория, parent_doc_ids = []
        api_stub = next(s for s in stubs if s.id == 'local:default:api/')
        assert api_stub.parent_doc_ids == []

    async def test_parent_doc_ids_nested_dirs(self, tmp_path):
        """Вложенная директория ссылается на родительскую."""
        nested = tmp_path / 'a' / 'b'
        nested.mkdir(parents=True)
        (nested / 'doc.md').write_text('# test')
        source = DirectorySource(tmp_path)
        stubs = await source.get_metadata()
        ids = {s.id for s in stubs}
        assert 'local:default:a/' in ids
        assert 'local:default:a/b/' in ids
        b_stub = next(s for s in stubs if s.id == 'local:default:a/b/')
        assert b_stub.parent_doc_ids == ['local:default:a/']

    async def test_empty_directory(self, tmp_path):
        source = DirectorySource(tmp_path)
        stubs = await source.get_metadata()
        assert stubs == []

    async def test_directory_without_supported_files(self, tmp_path):
        """Директория с неподдерживаемыми файлами не создаёт structural doc."""
        sub = tmp_path / 'data'
        sub.mkdir()
        (sub / 'file.txt').write_text('text')
        source = DirectorySource(tmp_path)
        stubs = await source.get_metadata()
        assert stubs == []

    async def test_directory_with_pdf_files(self, tmp_path):
        """Директория с PDF-файлами создаёт structural doc."""
        sub = tmp_path / 'reports'
        sub.mkdir()
        (sub / 'report.pdf').write_bytes(b'%PDF-1.4 fake')
        source = DirectorySource(tmp_path)
        stubs = await source.get_metadata()
        ids = {s.id for s in stubs}
        assert 'local:default:reports/' in ids

    async def test_stubs_are_sorted(self, source):
        stubs = await source.get_metadata()
        ids = [s.id for s in stubs]
        assert ids == sorted(ids)

    async def test_updated_at_is_timezone_aware(self, source):
        stubs = await source.get_metadata()
        for stub in stubs:
            assert stub.updated_at.tzinfo is not None
