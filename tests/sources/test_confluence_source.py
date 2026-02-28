from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from morag.config import ConfluenceConfig
from morag.sources.base import Document, Source
from morag.sources.confluence import (
    ConfluenceSource,
    _html_to_markdown,
    _parse_confluence_date,
)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> ConfluenceConfig:
    defaults = dict(url='https://confluence.example.com', username='user', password='pass')
    defaults.update(kwargs)
    return ConfluenceConfig(**defaults)


def _make_cql_page(page_id: str, title: str, space_key: str, html: str = '<p>text</p>',
                   when: str = '2024-06-01T10:00:00.000+00:00') -> dict:
    """Сформировать страницу в формате Confluence CQL-результата."""
    return {
        'content': {
            'id': page_id,
            'title': title,
            'space': {'key': space_key},
            'body': {'view': {'value': html}},
            'history': {'lastUpdated': {'when': when}},
        }
    }


# ---------------------------------------------------------------------------
# _html_to_markdown
# ---------------------------------------------------------------------------

class TestHtmlToMarkdown:
    def test_paragraph(self):
        assert _html_to_markdown('<p>Hello</p>') == 'Hello'

    def test_heading(self):
        result = _html_to_markdown('<h1>Title</h1>')
        assert '# Title' in result

    def test_bold(self):
        result = _html_to_markdown('<strong>bold</strong>')
        assert 'bold' in result

    def test_table(self):
        html = '<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>'
        result = _html_to_markdown(html)
        assert 'A' in result and 'B' in result

    def test_empty_html(self):
        assert _html_to_markdown('') == ''

    def test_strips_script(self):
        result = _html_to_markdown('<script>alert(1)</script><p>text</p>')
        assert 'alert' not in result
        assert 'text' in result

    def test_strips_style(self):
        result = _html_to_markdown('<style>body{}</style><p>text</p>')
        assert 'body{}' not in result
        assert 'text' in result

    def test_returns_stripped_string(self):
        result = _html_to_markdown('  <p>text</p>  ')
        assert result == result.strip()


# ---------------------------------------------------------------------------
# _parse_confluence_date
# ---------------------------------------------------------------------------

class TestParseConfluenceDate:
    def test_iso_utc(self):
        dt = _parse_confluence_date('2024-06-01T10:00:00.000+00:00')
        assert dt == datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_iso_with_offset(self):
        dt = _parse_confluence_date('2024-06-01T13:00:00.000+03:00')
        assert dt == datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_z_suffix(self):
        dt = _parse_confluence_date('2024-06-01T10:00:00Z')
        assert dt == datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_empty_string_returns_now(self):
        before = datetime.now(tz=timezone.utc)
        dt = _parse_confluence_date('')
        after = datetime.now(tz=timezone.utc)
        assert before <= dt <= after

    def test_invalid_string_returns_now(self):
        before = datetime.now(tz=timezone.utc)
        dt = _parse_confluence_date('not-a-date')
        after = datetime.now(tz=timezone.utc)
        assert before <= dt <= after

    def test_result_is_utc(self):
        dt = _parse_confluence_date('2024-06-01T13:00:00.000+03:00')
        assert dt.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# ConfluenceSource.__init__
# ---------------------------------------------------------------------------

class TestConfluenceSourceInit:
    def test_is_source(self):
        with patch('morag.sources.confluence.Confluence'):
            src = ConfluenceSource(_make_config())
            assert isinstance(src, Source)

    def test_requires_credential(self):
        with pytest.raises(ValueError, match='api_token or password'):
            ConfluenceSource(ConfluenceConfig(
                url='https://example.com', username='user',
            ))

    def test_cloud_flag_when_api_token(self):
        with patch('morag.sources.confluence.Confluence') as mock_cls:
            ConfluenceSource(_make_config(password=None, api_token='token123'))
            _, kwargs = mock_cls.call_args
            assert kwargs['cloud'] is True

    def test_no_cloud_flag_when_password(self):
        with patch('morag.sources.confluence.Confluence') as mock_cls:
            ConfluenceSource(_make_config(password='pass', api_token=None))
            _, kwargs = mock_cls.call_args
            assert kwargs['cloud'] is False


# ---------------------------------------------------------------------------
# _build_cql
# ---------------------------------------------------------------------------

class TestBuildCql:
    def _src(self, **kwargs) -> ConfluenceSource:
        with patch('morag.sources.confluence.Confluence'):
            return ConfluenceSource(_make_config(**kwargs))

    def test_no_filters(self):
        cql = self._src()._build_cql()
        assert cql == 'type = page ORDER BY lastmodified DESC'

    def test_spaces_filter(self):
        cql = self._src(spaces=['ML', 'DEV'])._build_cql()
        assert 'space IN ("ML", "DEV")' in cql
        assert 'type = page' in cql

    def test_ancestor_ids_filter(self):
        cql = self._src(ancestor_ids=['111', '222'])._build_cql()
        assert 'ancestor IN ("111", "222")' in cql
        assert 'id IN ("111", "222")' in cql

    def test_ancestor_ids_takes_priority_over_spaces(self):
        cql = self._src(spaces=['ML'], ancestor_ids=['111'])._build_cql()
        assert 'ancestor IN' in cql
        assert 'space IN' not in cql

    def test_ends_with_order_by(self):
        cql = self._src(spaces=['ML'])._build_cql()
        assert cql.endswith('ORDER BY lastmodified DESC')

    def test_skip_ancestor_ids(self):
        cql = self._src(skip_ancestor_ids=['333', '444'])._build_cql()
        assert 'ancestor NOT IN ("333", "444")' in cql
        assert 'id NOT IN ("333", "444")' in cql

    def test_skip_ancestor_ids_with_spaces(self):
        cql = self._src(spaces=['ML'], skip_ancestor_ids=['333'])._build_cql()
        assert 'space IN ("ML")' in cql
        assert 'ancestor NOT IN ("333")' in cql
        assert 'id NOT IN ("333")' in cql

    def test_skip_ancestor_ids_not_in_cql_when_empty(self):
        cql = self._src()._build_cql()
        assert 'NOT IN' not in cql


# ---------------------------------------------------------------------------
# ConfluenceSource.load
# ---------------------------------------------------------------------------

class TestConfluenceSourceLoad:
    def _src_with_pages(self, pages: list[dict], **cfg_kwargs) -> ConfluenceSource:
        with patch('morag.sources.confluence.Confluence') as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.cql.return_value = {'results': pages}
            src = ConfluenceSource(_make_config(**cfg_kwargs))
            src._client = mock_client
            return src

    async def test_returns_list_of_documents(self):
        pages = [_make_cql_page('1', 'Page One', 'ML')]
        src = self._src_with_pages(pages)
        docs = await src.load()
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    async def test_document_id_is_page_id(self):
        pages = [_make_cql_page('42', 'My Page', 'ML')]
        src = self._src_with_pages(pages)
        docs = await src.load()
        assert docs[0].id == '42'

    async def test_document_path_is_space_slash_title(self):
        pages = [_make_cql_page('1', 'My Page', 'ML')]
        src = self._src_with_pages(pages)
        docs = await src.load()
        assert docs[0].path == 'ML/My Page'

    async def test_document_source_type(self):
        pages = [_make_cql_page('1', 'Page', 'ML')]
        src = self._src_with_pages(pages)
        assert (await src.load())[0].source_type == 'confluence'

    async def test_document_text_starts_with_title(self):
        pages = [_make_cql_page('1', 'My Title', 'ML', html='<p>body</p>')]
        src = self._src_with_pages(pages)
        assert (await src.load())[0].text.startswith('# My Title')

    async def test_document_text_contains_body(self):
        pages = [_make_cql_page('1', 'T', 'ML', html='<p>important content</p>')]
        src = self._src_with_pages(pages)
        assert 'important content' in (await src.load())[0].text

    async def test_document_updated_at_is_utc(self):
        pages = [_make_cql_page('1', 'T', 'ML', when='2024-06-01T13:00:00.000+03:00')]
        src = self._src_with_pages(pages)
        doc = (await src.load())[0]
        assert doc.updated_at.tzinfo == timezone.utc
        assert doc.updated_at == datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    async def test_document_size_is_byte_length(self):
        pages = [_make_cql_page('1', 'T', 'ML', html='<p>hi</p>')]
        src = self._src_with_pages(pages)
        doc = (await src.load())[0]
        assert doc.size == len(doc.text.encode('utf-8'))

    async def test_empty_results(self):
        src = self._src_with_pages([])
        assert await src.load() == []

    async def test_multiple_pages(self):
        pages = [
            _make_cql_page('1', 'Alpha', 'ML'),
            _make_cql_page('2', 'Beta', 'ML'),
            _make_cql_page('3', 'Gamma', 'DEV'),
        ]
        src = self._src_with_pages(pages)
        docs = await src.load()
        assert len(docs) == 3

    async def test_skips_malformed_page(self):
        pages = [
            _make_cql_page('1', 'Good', 'ML'),
            {'content': {}},  # нет обязательных полей → ошибка
        ]
        src = self._src_with_pages(pages)
        docs = await src.load()
        # Хорошая страница сохраняется, плохая пропускается
        assert len(docs) == 1
        assert docs[0].id == '1'

    def test_pagination(self):
        """При полном батче должен быть второй запрос."""
        page = _make_cql_page('1', 'P', 'ML')
        batch_full = [page] * 200
        batch_last = [page] * 5

        with patch('morag.sources.confluence.Confluence') as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.cql.side_effect = [
                {'results': batch_full},
                {'results': batch_last},
            ]
            src = ConfluenceSource(_make_config())
            src._client = mock_client
            pages = src._fetch_pages()

        assert len(pages) == 205
        assert mock_client.cql.call_count == 2
