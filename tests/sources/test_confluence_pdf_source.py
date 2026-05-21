from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morag.config import AttachmentsConfig, ConfluenceSourceConfig
from morag.sources.base import Document, Source
from morag.sources.confluence_pdf import ConfluencePdfSource
from morag.sources.pdf_converter import PdfConverter


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> ConfluenceSourceConfig:
    defaults = dict(
        kind='confluence', name='test',
        url='https://confluence.example.com',
        username='user',
        password='pass',
        attachments=AttachmentsConfig(enabled=True, mime_types=['application/pdf']),
    )
    defaults.update(kwargs)
    return ConfluenceSourceConfig(**defaults)


def _make_attachment(
    att_id: str = '123',
    title: str = 'report.pdf',
    media_type: str = 'application/pdf',
    when: str = '2024-06-01T10:00:00.000+00:00',
    download_url: str = '/download/attachments/42/report.pdf',
) -> dict:
    """Сформировать вложение в формате Confluence API."""
    return {
        'id': f'att{att_id}',
        'title': title,
        'metadata': {'mediaType': media_type},
        'version': {'when': when},
        '_links': {'download': download_url},
    }


def _make_parent_doc(
    page_id: str = '42',
    path: list[str] | None = None,
) -> Document:
    """Создать мок родительской страницы."""
    return Document(
        id=page_id,
        path=path or ['Root/My Page'],
        text='# My Page\n\nContent.',
        updated_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        source_type='confluence',
        size=100,
    )


@pytest.fixture
def mock_converter():
    """Мок PdfConverter."""
    converter = AsyncMock(spec=PdfConverter)
    converter.convert.return_value = '# PDF Content\n\nConverted text.'
    return converter


@pytest.fixture
def mock_doc_repo():
    """Мок DocRepository."""
    repo = AsyncMock()
    repo.get_ids_by_source_instance.return_value = set()
    repo.get_by_id.return_value = _make_parent_doc()
    return repo


def _make_source(
    mock_converter,
    mock_doc_repo,
    attachments_response: list[dict] | None = None,
    page_ids: set[str] | None = None,
    **cfg_kwargs,
) -> ConfluencePdfSource:
    """Создать ConfluencePdfSource с замоканным Confluence клиентом."""
    if page_ids is not None:
        mock_doc_repo.get_ids_by_source_instance.return_value = page_ids

    with patch('morag.sources.confluence_pdf.Confluence') as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.get_attachments_from_content.return_value = {
            'results': attachments_response or [],
        }
        src = ConfluencePdfSource(
            config=_make_config(**cfg_kwargs),
            converter=mock_converter,
            doc_repo=mock_doc_repo,
        )
        src._client = mock_client
        return src


# ---------------------------------------------------------------------------
# ConfluencePdfSource.__init__
# ---------------------------------------------------------------------------

class TestConfluencePdfSourceInit:
    def test_is_source(self, mock_converter, mock_doc_repo):
        src = _make_source(mock_converter, mock_doc_repo)
        assert isinstance(src, Source)

    def test_source_type(self, mock_converter, mock_doc_repo):
        src = _make_source(mock_converter, mock_doc_repo)
        assert src.source_type == 'attached_pdf'

    def test_requires_credential(self, mock_converter, mock_doc_repo):
        # Validation moved to Pydantic level (ConfluenceSourceConfig._check_secret),
        # см. test_config.py::test_confluence_requires_secret. ConfluencePdfSource
        # больше не дублирует проверку — конфиг до него не дойдёт.
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='password.*api_token'):
            ConfluenceSourceConfig(
                kind='confluence', name='test',
                url='https://example.com', username='user',
            )


# ---------------------------------------------------------------------------
# ConfluencePdfSource.get_metadata
# ---------------------------------------------------------------------------

class TestConfluencePdfSourceGetMetadata:
    async def test_returns_empty_when_no_pages(self, mock_converter, mock_doc_repo):
        """Если нет страниц Confluence — пустой список."""
        mock_doc_repo.get_ids_by_source_instance.return_value = set()
        src = _make_source(mock_converter, mock_doc_repo)
        stubs = await src.get_metadata()
        assert stubs == []

    async def test_returns_stubs_for_pdf_attachments(self, mock_converter, mock_doc_repo):
        """Находит PDF-вложения на странице."""
        att = _make_attachment(att_id='123', title='report.pdf')
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
        )
        stubs = await src.get_metadata()
        assert len(stubs) == 1
        assert stubs[0].id == 'confluence:test:att:123'
        assert stubs[0].source_type == 'attached_pdf'
        assert stubs[0].text == ''

    async def test_filters_by_mime_type(self, mock_converter, mock_doc_repo):
        """Вложения с неподходящим MIME-типом пропускаются."""
        attachments = [
            _make_attachment(att_id='1', title='doc.pdf', media_type='application/pdf'),
            _make_attachment(att_id='2', title='img.png', media_type='image/png'),
            _make_attachment(att_id='3', title='doc.docx',
                            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
        ]
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=attachments,
            page_ids={'42'},
        )
        stubs = await src.get_metadata()
        assert len(stubs) == 1
        assert stubs[0].id == 'confluence:test:att:1'

    async def test_stub_parent_doc_ids(self, mock_converter, mock_doc_repo):
        """parent_doc_ids содержит page_id."""
        att = _make_attachment()
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
        )
        stubs = await src.get_metadata()
        assert stubs[0].parent_doc_ids == ['42']

    async def test_stub_path_includes_page_path(self, mock_converter, mock_doc_repo):
        """Путь вложения формируется как page_path/filename."""
        mock_doc_repo.get_by_id.return_value = _make_parent_doc(path=['Root/My Page'])
        att = _make_attachment(title='report.pdf')
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
        )
        stubs = await src.get_metadata()
        assert stubs[0].path == ['Root/My Page/report.pdf']

    async def test_stub_path_fallback_when_no_parent(self, mock_converter, mock_doc_repo):
        """Если родительская страница не найдена — используется page_id."""
        mock_doc_repo.get_by_id.return_value = None
        att = _make_attachment(title='report.pdf')
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
        )
        stubs = await src.get_metadata()
        assert stubs[0].path == ['42/report.pdf']

    async def test_stub_updated_at_from_parent_page(self, mock_converter, mock_doc_repo):
        """updated_at наследуется от родительской страницы."""
        parent_date = datetime(2024, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_doc_repo.get_by_id.return_value = _make_parent_doc(page_id='42')
        mock_doc_repo.get_by_id.return_value.updated_at = parent_date

        att = _make_attachment()
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
        )
        stubs = await src.get_metadata()
        assert stubs[0].updated_at == parent_date

    async def test_multiple_pages_scanned(self, mock_converter, mock_doc_repo):
        """Обход нескольких страниц."""
        att1 = _make_attachment(att_id='1', title='a.pdf')
        att2 = _make_attachment(att_id='2', title='b.pdf')
        mock_doc_repo.get_ids_by_source_instance.return_value = {'10', '20'}
        mock_doc_repo.get_by_id.side_effect = lambda pid: _make_parent_doc(
            page_id=pid, path=[f'Page {pid}'],
        )

        with patch('morag.sources.confluence_pdf.Confluence') as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            def get_att(page_id, start=0, limit=50, expand='version'):
                if page_id == '10':
                    return {'results': [att1]}
                elif page_id == '20':
                    return {'results': [att2]}
                return {'results': []}

            mock_client.get_attachments_from_content.side_effect = get_att
            src = ConfluencePdfSource(
                config=_make_config(),
                converter=mock_converter,
                doc_repo=mock_doc_repo,
            )
            src._client = mock_client
            stubs = await src.get_metadata()

        assert len(stubs) == 2
        ids = {s.id for s in stubs}
        assert 'confluence:test:att:1' in ids
        assert 'confluence:test:att:2' in ids


# ---------------------------------------------------------------------------
# ConfluencePdfSource.load_one
# ---------------------------------------------------------------------------

class TestConfluencePdfSourceLoadOne:
    async def test_load_one_returns_document(self, mock_converter, mock_doc_repo):
        """load_one скачивает PDF, конвертирует и возвращает Document."""
        att = _make_attachment(att_id='123', title='report.pdf')
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
        )
        await src.get_metadata()

        mock_response = MagicMock()
        mock_response.content = b'%PDF-1.4 content'
        mock_response.raise_for_status = MagicMock()
        src._client._session.get.return_value = mock_response

        doc = await src.load_one('confluence:test:att:123')

        assert doc is not None
        assert doc.id == 'confluence:test:att:123'
        assert doc.source_type == 'attached_pdf'
        assert doc.text == '# PDF Content\n\nConverted text.'
        assert doc.parent_doc_ids == ['42']
        assert doc.path == ['Root/My Page/report.pdf']
        assert doc.updated_at == _make_parent_doc().updated_at
        mock_converter.convert.assert_called_once()

    async def test_load_one_returns_none_when_no_metadata(self, mock_converter, mock_doc_repo):
        """Без предварительного get_metadata возвращает None."""
        src = _make_source(mock_converter, mock_doc_repo)
        doc = await src.load_one('confluence:test:att:999')
        assert doc is None

    async def test_load_one_returns_none_on_download_error(self, mock_converter, mock_doc_repo):
        """При ошибке скачивания возвращает None."""
        att = _make_attachment(att_id='123')
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
        )
        await src.get_metadata()

        src._client._session.get.side_effect = Exception('Network error')

        doc = await src.load_one('confluence:test:att:123')
        assert doc is None

    async def test_load_one_returns_none_on_converter_error(self, mock_converter, mock_doc_repo):
        """При ошибке конвертации возвращает None."""
        mock_converter.convert.return_value = None
        att = _make_attachment(att_id='123')
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
        )
        await src.get_metadata()

        mock_response = MagicMock()
        mock_response.content = b'%PDF-1.4'
        mock_response.raise_for_status = MagicMock()
        src._client._session.get.return_value = mock_response

        doc = await src.load_one('confluence:test:att:123')
        assert doc is None

    async def test_load_one_passes_bytes_to_converter(self, mock_converter, mock_doc_repo):
        """Содержимое PDF передаётся в converter.convert()."""
        att = _make_attachment(att_id='123', title='report.pdf')
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
        )
        await src.get_metadata()

        pdf_content = b'%PDF-1.4 real content'
        mock_response = MagicMock()
        mock_response.content = pdf_content
        mock_response.raise_for_status = MagicMock()
        src._client._session.get.return_value = mock_response

        await src.load_one('confluence:test:att:123')

        call_args = mock_converter.convert.call_args
        assert call_args[0][0] == pdf_content
        assert call_args[0][1] == 'report.pdf'


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestAttachmentsConfig:
    def test_default_disabled(self):
        config = ConfluenceSourceConfig(
            kind='confluence', name='test',
            url='https://example.com', username='user', password='pass',
        )
        assert config.attachments.enabled is False
        assert config.attachments.mime_types == ['application/pdf']
        assert config.attachments.skip_ancestor_ids == []

    def test_enabled_with_custom_mimetypes(self):
        config = ConfluenceSourceConfig(
            kind='confluence', name='test',
            url='https://example.com', username='user', password='pass',
            attachments={'enabled': True, 'mime_types': ['application/pdf', 'image/png']},
        )
        assert config.attachments.enabled is True
        assert 'image/png' in config.attachments.mime_types

    def test_skip_ancestor_ids(self):
        config = ConfluenceSourceConfig(
            kind='confluence', name='test',
            url='https://example.com', username='user', password='pass',
            attachments={'enabled': True, 'skip_ancestor_ids': ['100', '200']},
        )
        assert config.attachments.skip_ancestor_ids == ['100', '200']


# ---------------------------------------------------------------------------
# skip_ancestor_ids
# ---------------------------------------------------------------------------

class TestSkipAncestorIds:
    async def test_skips_page_in_skip_ancestor_ids(self, mock_converter, mock_doc_repo):
        """Страница из skip_ancestor_ids пропускается."""
        att = _make_attachment(att_id='1', title='book.pdf')
        # repo возвращает prefixed IDs (как в реальности)
        mock_doc_repo.get_ids_by_source_instance.return_value = {'confluence:test:50'}
        mock_doc_repo.find_children.return_value = []

        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'confluence:test:50'},
            attachments=AttachmentsConfig(
                enabled=True, skip_ancestor_ids=['50'],  # raw в config (источник)
            ),
        )
        stubs = await src.get_metadata()
        assert stubs == []

    async def test_skips_descendants_of_skip_ancestor(self, mock_converter, mock_doc_repo):
        """Потомки skip_ancestor_ids тоже пропускаются (BFS)."""
        # Иерархия: 50 → 60 → 70 (все IDs prefixed в repo)
        mock_doc_repo.get_ids_by_source_instance.return_value = {
            'confluence:test:50', 'confluence:test:60', 'confluence:test:70', 'confluence:test:10',
        }

        child_60 = _make_parent_doc(page_id='confluence:test:60', path=['Library/Chapter1'])
        child_70 = _make_parent_doc(page_id='confluence:test:70', path=['Library/Chapter1/Section'])

        async def find_children(parent_id):
            if parent_id == 'confluence:test:50':
                return [child_60]
            if parent_id == 'confluence:test:60':
                return [child_70]
            return []

        mock_doc_repo.find_children.side_effect = find_children

        att = _make_attachment(att_id='1', title='doc.pdf')

        with patch('morag.sources.confluence_pdf.Confluence') as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            # Только page_id=10 должен быть обработан
            def get_att(page_id, start=0, limit=50, expand='version'):
                if page_id == '10':
                    return {'results': [att]}
                return {'results': []}

            mock_client.get_attachments_from_content.side_effect = get_att

            src = ConfluencePdfSource(
                config=_make_config(
                    attachments=AttachmentsConfig(
                        enabled=True, skip_ancestor_ids=['50'],
                    ),
                ),
                converter=mock_converter,
                doc_repo=mock_doc_repo,
            )
            src._client = mock_client
            stubs = await src.get_metadata()

        # Только вложение со страницы 10
        assert len(stubs) == 1
        # Страницы 50, 60, 70 не обрабатывались
        processed_pages = [
            call.args[0]
            for call in mock_client.get_attachments_from_content.call_args_list
        ]
        assert '50' not in processed_pages
        assert '60' not in processed_pages
        assert '70' not in processed_pages
        assert '10' in processed_pages

    async def test_no_skip_when_empty(self, mock_converter, mock_doc_repo):
        """Без skip_ancestor_ids все страницы обрабатываются."""
        att = _make_attachment(att_id='1', title='doc.pdf')
        mock_doc_repo.find_children.return_value = []
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
        )
        stubs = await src.get_metadata()
        assert len(stubs) == 1


# ---------------------------------------------------------------------------
# url_mode
# ---------------------------------------------------------------------------

class TestUrlMode:
    async def _load_one_with_url_mode(self, url_mode, mock_converter, mock_doc_repo):
        """Хелпер: создать source с url_mode, get_metadata + load_one."""
        att = _make_attachment(
            att_id='123', title='report.pdf',
            download_url='/download/attachments/42/report.pdf?version=1',
        )
        mock_doc_repo.find_children.return_value = []
        src = _make_source(
            mock_converter, mock_doc_repo,
            attachments_response=[att],
            page_ids={'42'},
            attachments=AttachmentsConfig(enabled=True, url_mode=url_mode),
        )
        await src.get_metadata()

        mock_response = MagicMock()
        mock_response.content = b'%PDF-1.4'
        mock_response.raise_for_status = MagicMock()
        src._client._session.get.return_value = mock_response
        src._client.get_attachment_by_id.return_value = {
            'version': {'when': '2024-06-01T10:00:00.000+00:00'},
        }

        return await src.load_one('confluence:test:att:123')

    async def test_preview_url(self, mock_converter, mock_doc_repo):
        """preview — on-premise viewpageattachments URL."""
        doc = await self._load_one_with_url_mode('preview', mock_converter, mock_doc_repo)
        assert doc is not None
        assert doc.url == (
            'https://confluence.example.com/pages/viewpageattachments.action'
            '?pageId=42&preview=/42/report.pdf'
        )

    async def test_download_url(self, mock_converter, mock_doc_repo):
        """download — прямая ссылка на скачивание."""
        doc = await self._load_one_with_url_mode('download', mock_converter, mock_doc_repo)
        assert doc is not None
        assert doc.url == (
            'https://confluence.example.com/download/attachments/42/report.pdf?version=1'
        )

    async def test_parent_page_url(self, mock_converter, mock_doc_repo):
        """parent_page — URL родительской страницы."""
        mock_doc_repo.get_by_id.return_value = _make_parent_doc(page_id='42')
        # У _make_parent_doc нет url, добавим
        parent = _make_parent_doc(page_id='42')
        parent.url = 'https://confluence.example.com/display/ML/My+Page'
        mock_doc_repo.get_by_id.return_value = parent

        doc = await self._load_one_with_url_mode('parent_page', mock_converter, mock_doc_repo)
        assert doc is not None
        assert doc.url == 'https://confluence.example.com/display/ML/My+Page'

    async def test_parent_page_url_none_when_no_parent(self, mock_converter, mock_doc_repo):
        """parent_page без родителя в doc_repo — url=None."""
        mock_doc_repo.get_by_id.return_value = None
        doc = await self._load_one_with_url_mode('parent_page', mock_converter, mock_doc_repo)
        assert doc is not None
        assert doc.url is None

    async def test_default_url_mode_is_preview(self):
        """По умолчанию url_mode = preview."""
        config = AttachmentsConfig(enabled=True)
        assert config.url_mode == 'preview'
