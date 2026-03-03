from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morag.config import ConfluenceConfig
from morag.sources.base import Document, Source
from morag.sources.confluence import (
    ConfluenceSource,
    _build_page_path,
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


def _make_cql_page(page_id: str, title: str, space_key: str,
                   when: str = '2024-06-01T10:00:00.000+00:00',
                   space_name: str | None = None,
                   ancestors: list[dict] | None = None) -> dict:
    """Сформировать страницу в формате Confluence CQL-результата (только метаданные)."""
    return {
        'content': {
            'id': page_id,
            'title': title,
            'space': {'key': space_key, 'name': space_name or space_key},
            'ancestors': ancestors or [],
            'history': {'lastUpdated': {'when': when}},
        }
    }


def _make_full_page(page_id: str, title: str, space_key: str,
                    html: str = '<p>text</p>',
                    when: str = '2024-06-01T10:00:00.000+00:00',
                    space_name: str | None = None,
                    ancestors: list[dict] | None = None) -> dict:
    """Сформировать страницу в формате get_page_by_id (без обёртки content)."""
    return {
        'id': page_id,
        'title': title,
        'space': {'key': space_key, 'name': space_name or space_key},
        'ancestors': ancestors or [],
        'body': {'view': {'value': html}},
        'history': {'lastUpdated': {'when': when}},
    }


def _src_with_pages(pages_cql: list[dict], pages_full: list[dict] | None = None,
                    **cfg_kwargs) -> ConfluenceSource:
    """Создать ConfluenceSource с замоканными CQL и get_page_by_id."""
    if pages_full is None:
        # Строим full pages из CQL pages с дефолтным html
        pages_full = [
            _make_full_page(
                p['content']['id'],
                p['content']['title'],
                p['content']['space']['key'],
            )
            for p in pages_cql
        ]
    full_by_id = {p['id']: p for p in pages_full}

    with patch('morag.sources.confluence.Confluence') as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.cql.return_value = {'results': pages_cql}
        mock_client.get_page_by_id.side_effect = lambda pid, expand=None: full_by_id.get(pid)
        src = ConfluenceSource(_make_config(**cfg_kwargs))
        src._client = mock_client
        return src


# ---------------------------------------------------------------------------
# _build_page_path
# ---------------------------------------------------------------------------

class TestBuildPagePath:
    def test_no_ancestors(self):
        assert _build_page_path('My Space', [], 'Page') == 'Page'

    def test_ancestors_included(self):
        assert _build_page_path('Space', ['Root', 'Parent'], 'Child') == 'Root/Parent/Child'

    def test_space_name_not_in_path(self):
        assert _build_page_path('My Space', ['My Space', 'Parent'], 'Child') == 'My Space/Parent/Child'


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
# ConfluenceSource.get_metadata
# ---------------------------------------------------------------------------

class TestConfluenceSourceGetMetadata:
    async def test_returns_stubs_with_empty_text(self):
        pages = [_make_cql_page('1', 'Page One', 'ML')]
        src = _src_with_pages(pages)
        stubs = await src.get_metadata()
        assert len(stubs) == 1
        assert stubs[0].text == ''

    async def test_stub_id_is_page_id(self):
        pages = [_make_cql_page('42', 'My Page', 'ML')]
        src = _src_with_pages(pages)
        stubs = await src.get_metadata()
        assert stubs[0].id == '42'

    async def test_stub_path_no_ancestors(self):
        pages = [_make_cql_page('1', 'My Page', 'ML', space_name='Machine Learning')]
        src = _src_with_pages(pages)
        stubs = await src.get_metadata()
        assert stubs[0].path == 'My Page'

    async def test_stub_path_with_ancestors(self):
        ancestors = [{'id': '10', 'title': 'Root'}, {'id': '11', 'title': 'Parent'}]
        pages = [_make_cql_page('1', 'Child', 'ML', space_name='ML Space', ancestors=ancestors)]
        src = _src_with_pages(pages)
        stubs = await src.get_metadata()
        assert stubs[0].path == 'Root/Parent/Child'

    async def test_stub_source_type_is_confluence(self):
        pages = [_make_cql_page('1', 'Page', 'ML')]
        src = _src_with_pages(pages)
        stubs = await src.get_metadata()
        assert stubs[0].source_type == 'confluence'

    async def test_stub_updated_at_is_utc(self):
        pages = [_make_cql_page('1', 'T', 'ML', when='2024-06-01T13:00:00.000+03:00')]
        src = _src_with_pages(pages)
        stubs = await src.get_metadata()
        assert stubs[0].updated_at == datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    async def test_empty_results(self):
        src = _src_with_pages([])
        assert await src.get_metadata() == []

    async def test_multiple_pages(self):
        pages = [
            _make_cql_page('1', 'Alpha', 'ML'),
            _make_cql_page('2', 'Beta', 'ML'),
        ]
        src = _src_with_pages(pages)
        stubs = await src.get_metadata()
        assert len(stubs) == 2

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
            pages = src._fetch_pages_metadata()

        assert len(pages) == 205
        assert mock_client.cql.call_count == 2


# ---------------------------------------------------------------------------
# ConfluenceSource.load_one
# ---------------------------------------------------------------------------

class TestConfluenceSourceLoadOne:
    async def test_returns_document(self):
        pages = [_make_cql_page('42', 'My Page', 'ML')]
        src = _src_with_pages(pages, [_make_full_page('42', 'My Page', 'ML', '<p>body</p>')])
        doc = await src.load_one('42')
        assert doc is not None
        assert isinstance(doc, Document)
        assert doc.id == '42'

    async def test_document_text_starts_with_title(self):
        pages = [_make_cql_page('1', 'My Title', 'ML')]
        src = _src_with_pages(pages, [_make_full_page('1', 'My Title', 'ML', '<p>body</p>')])
        doc = await src.load_one('1')
        assert doc.text.startswith('# My Title')

    async def test_document_text_contains_body(self):
        pages = [_make_cql_page('1', 'T', 'ML')]
        src = _src_with_pages(pages, [_make_full_page('1', 'T', 'ML', '<p>important content</p>')])
        doc = await src.load_one('1')
        assert 'important content' in doc.text

    async def test_returns_none_on_error(self):
        with patch('morag.sources.confluence.Confluence') as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_page_by_id.side_effect = Exception('API error')
            src = ConfluenceSource(_make_config())
            src._client = mock_client
            doc = await src.load_one('999')
        assert doc is None


# ---------------------------------------------------------------------------
# ConfluenceSource.load (интеграция get_metadata + load_one)
# ---------------------------------------------------------------------------

class TestConfluenceSourceLoad:
    async def test_returns_list_of_documents(self):
        pages = [_make_cql_page('1', 'Page One', 'ML')]
        src = _src_with_pages(pages, [_make_full_page('1', 'Page One', 'ML', '<p>text</p>')])
        docs = await src.load()
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    async def test_document_id_is_page_id(self):
        pages = [_make_cql_page('42', 'My Page', 'ML')]
        src = _src_with_pages(pages, [_make_full_page('42', 'My Page', 'ML')])
        docs = await src.load()
        assert docs[0].id == '42'

    async def test_document_path_no_ancestors(self):
        pages = [_make_cql_page('1', 'My Page', 'ML')]
        full = [_make_full_page('1', 'My Page', 'ML', space_name='Machine Learning')]
        src = _src_with_pages(pages, full)
        docs = await src.load()
        assert docs[0].path == 'My Page'

    async def test_document_path_with_ancestors(self):
        ancestors = [{'id': '10', 'title': 'Root'}, {'id': '11', 'title': 'Parent'}]
        pages = [_make_cql_page('1', 'Child', 'ML')]
        full = [_make_full_page('1', 'Child', 'ML', space_name='ML Space', ancestors=ancestors)]
        src = _src_with_pages(pages, full)
        docs = await src.load()
        assert docs[0].path == 'Root/Parent/Child'

    async def test_document_source_type(self):
        pages = [_make_cql_page('1', 'Page', 'ML')]
        src = _src_with_pages(pages, [_make_full_page('1', 'Page', 'ML')])
        assert (await src.load())[0].source_type == 'confluence'

    async def test_document_text_starts_with_title(self):
        pages = [_make_cql_page('1', 'My Title', 'ML')]
        src = _src_with_pages(pages, [_make_full_page('1', 'My Title', 'ML', '<p>body</p>')])
        assert (await src.load())[0].text.startswith('# My Title')

    async def test_document_text_contains_body(self):
        pages = [_make_cql_page('1', 'T', 'ML')]
        src = _src_with_pages(pages, [_make_full_page('1', 'T', 'ML', '<p>important content</p>')])
        assert 'important content' in (await src.load())[0].text

    async def test_document_updated_at_is_utc(self):
        when = '2024-06-01T13:00:00.000+03:00'
        pages = [_make_cql_page('1', 'T', 'ML', when=when)]
        src = _src_with_pages(pages, [_make_full_page('1', 'T', 'ML', when=when)])
        doc = (await src.load())[0]
        assert doc.updated_at.tzinfo == timezone.utc
        assert doc.updated_at == datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    async def test_document_size_is_byte_length(self):
        pages = [_make_cql_page('1', 'T', 'ML')]
        src = _src_with_pages(pages, [_make_full_page('1', 'T', 'ML', '<p>hi</p>')])
        doc = (await src.load())[0]
        assert doc.size == len(doc.text.encode('utf-8'))

    async def test_empty_results(self):
        src = _src_with_pages([])
        assert await src.load() == []

    async def test_multiple_pages(self):
        pages = [
            _make_cql_page('1', 'Alpha', 'ML'),
            _make_cql_page('2', 'Beta', 'ML'),
            _make_cql_page('3', 'Gamma', 'DEV'),
        ]
        full_pages = [
            _make_full_page('1', 'Alpha', 'ML'),
            _make_full_page('2', 'Beta', 'ML'),
            _make_full_page('3', 'Gamma', 'DEV'),
        ]
        src = _src_with_pages(pages, full_pages)
        docs = await src.load()
        assert len(docs) == 3

    async def test_skips_when_load_one_returns_none(self):
        pages = [
            _make_cql_page('1', 'Good', 'ML'),
            _make_cql_page('2', 'Bad', 'ML'),
        ]
        full_pages = [_make_full_page('1', 'Good', 'ML')]

        with patch('morag.sources.confluence.Confluence') as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.cql.return_value = {'results': pages}
            full_by_id = {'1': full_pages[0]}
            # Страница '2' — API ошибка
            mock_client.get_page_by_id.side_effect = lambda pid, expand=None: (
                full_by_id[pid] if pid in full_by_id else (_ for _ in ()).throw(Exception('not found'))
            )
            src = ConfluenceSource(_make_config())
            src._client = mock_client
            docs = await src.load()

        assert len(docs) == 1
        assert docs[0].id == '1'
