import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morag.sources.base import Source
from morag.sources.pdf import PdfSource
from morag.sources.pdf_converter import (
    DoclingPdfConverter,
    PdfConverter,
    _parse_elements,
)


@pytest.fixture
def pdf_dir(tmp_path):
    """Директория с фиктивным PDF-файлом."""
    pdf_file = tmp_path / 'report.pdf'
    pdf_file.write_bytes(b'%PDF-1.4 fake content for testing')
    sub = tmp_path / 'docs'
    sub.mkdir()
    (sub / 'nested.pdf').write_bytes(b'%PDF-1.4 nested')
    return tmp_path


@pytest.fixture
def mock_converter():
    """Мок PdfConverter."""
    converter = AsyncMock(spec=PdfConverter)
    converter.convert.return_value = '# Default content'
    return converter


@pytest.fixture
def source(pdf_dir, mock_converter) -> PdfSource:
    return PdfSource(pdf_dir, converter=mock_converter)


class TestPdfSource:
    def test_is_source(self, source):
        assert isinstance(source, Source)

    def test_source_type(self, source):
        assert source.source_type == 'pdf'

    async def test_get_metadata_finds_pdfs(self, source):
        stubs = await source.get_metadata()
        ids = {s.id for s in stubs}
        assert 'local:default:report.pdf' in ids
        assert 'local:default:docs/nested.pdf' in ids

    async def test_stubs_have_empty_text(self, source):
        stubs = await source.get_metadata()
        for stub in stubs:
            assert stub.text == ''

    async def test_stubs_source_type_is_pdf(self, source):
        stubs = await source.get_metadata()
        for stub in stubs:
            assert stub.source_type == 'pdf'

    async def test_stubs_size_is_positive(self, source):
        stubs = await source.get_metadata()
        for stub in stubs:
            assert stub.size > 0

    async def test_stubs_updated_at_timezone_aware(self, source):
        stubs = await source.get_metadata()
        for stub in stubs:
            assert stub.updated_at.tzinfo is not None

    async def test_stubs_are_sorted(self, source):
        stubs = await source.get_metadata()
        ids = [s.id for s in stubs]
        assert ids == sorted(ids)

    async def test_parent_doc_ids_root(self, source):
        stubs = await source.get_metadata()
        root_stub = next(s for s in stubs if s.id == 'local:default:report.pdf')
        assert root_stub.parent_doc_ids == []

    async def test_parent_doc_ids_nested(self, source):
        stubs = await source.get_metadata()
        nested_stub = next(s for s in stubs if s.id == 'local:default:docs/nested.pdf')
        assert nested_stub.parent_doc_ids == ['local:default:docs/']

    async def test_stubs_have_url(self, source):
        stubs = await source.get_metadata()
        for stub in stubs:
            assert stub.url is not None
            assert stub.url.startswith('file://')

    async def test_ignores_non_pdf_files(self, tmp_path, mock_converter):
        (tmp_path / 'readme.md').write_text('# Doc')
        (tmp_path / 'data.txt').write_text('text')
        (tmp_path / 'doc.pdf').write_bytes(b'%PDF-1.4')
        source = PdfSource(tmp_path, converter=mock_converter)
        stubs = await source.get_metadata()
        assert len(stubs) == 1
        assert stubs[0].id == 'local:default:doc.pdf'

    async def test_empty_directory(self, tmp_path, mock_converter):
        source = PdfSource(tmp_path, converter=mock_converter)
        stubs = await source.get_metadata()
        assert stubs == []


class TestPdfSourceLoadOne:
    async def test_load_one_calls_converter(self, pdf_dir, mock_converter):
        """load_one конвертирует PDF через converter и возвращает Document."""
        mock_converter.convert.return_value = '# Report\n\nSome content.'
        source = PdfSource(pdf_dir, converter=mock_converter)

        doc = await source.load_one('local:default:report.pdf')

        assert doc is not None
        assert doc.id == 'local:default:report.pdf'
        assert doc.source_type == 'pdf'
        assert doc.text == '# Report\n\nSome content.'
        assert doc.size > 0
        assert doc.updated_at.tzinfo is not None
        mock_converter.convert.assert_called_once()

    async def test_load_one_returns_none_for_nonexistent(self, pdf_dir, mock_converter):
        source = PdfSource(pdf_dir, converter=mock_converter)
        doc = await source.load_one('local:default:nonexistent.pdf')
        assert doc is None

    async def test_load_one_returns_none_on_converter_error(self, pdf_dir, mock_converter):
        """При ошибке конвертера возвращает None."""
        mock_converter.convert.return_value = None
        source = PdfSource(pdf_dir, converter=mock_converter)
        doc = await source.load_one('local:default:report.pdf')
        assert doc is None

    async def test_load_one_preserves_original_metadata(self, pdf_dir, mock_converter):
        """Document содержит метаданные оригинального PDF, а не сконвертированного md."""
        mock_converter.convert.return_value = '# Doc'
        source = PdfSource(pdf_dir, converter=mock_converter)
        pdf_path = pdf_dir / 'report.pdf'
        pdf_stat = pdf_path.stat()

        doc = await source.load_one('local:default:report.pdf')

        assert doc is not None
        assert doc.size == pdf_stat.st_size
        assert doc.url == pdf_path.as_uri()

    async def test_load_one_passes_bytes_to_converter(self, pdf_dir, mock_converter):
        """load_one передаёт содержимое PDF в converter.convert()."""
        mock_converter.convert.return_value = '# Doc'
        source = PdfSource(pdf_dir, converter=mock_converter)

        await source.load_one('local:default:report.pdf')

        call_args = mock_converter.convert.call_args
        pdf_bytes = call_args[0][0]
        filename = call_args[0][1]
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes == (pdf_dir / 'report.pdf').read_bytes()
        assert filename == 'report.pdf'


class TestParseElements:
    def test_parse_pictures_valid_json(self):
        """Парсинг корректного JSON с картинками."""
        json_content = {
            'pictures': [
                {
                    'label': 'picture',
                    'prov': [{'page_no': 5, 'bbox': {
                        'l': 100.0, 't': 700.0, 'r': 500.0, 'b': 400.0,
                        'coord_origin': 'BOTTOMLEFT',
                    }}],
                },
            ],
        }
        pics = _parse_elements(json_content, 'pictures', 'test.pdf')
        assert len(pics) == 1
        assert pics[0].page_no == 5
        assert pics[0].bbox_l == 100.0
        assert pics[0].coord_origin == 'BOTTOMLEFT'

    def test_parse_json_string(self):
        """Парсинг JSON в виде строки."""
        json_str = json.dumps({
            'pictures': [
                {
                    'prov': [{'page_no': 1, 'bbox': {
                        'l': 10, 't': 20, 'r': 30, 'b': 40,
                        'coord_origin': 'BOTTOMLEFT',
                    }}],
                },
            ],
        })
        pics = _parse_elements(json_str, 'pictures', 'test.pdf')
        assert len(pics) == 1

    def test_parse_none_returns_empty(self):
        """None json_content — пустой список."""
        assert _parse_elements(None, 'pictures', 'test.pdf') == []

    def test_parse_no_pictures_key(self):
        """JSON без ключа pictures — пустой список."""
        assert _parse_elements({'texts': []}, 'pictures', 'test.pdf') == []

    def test_parse_empty_prov(self):
        """Картинка без prov пропускается."""
        json_content = {
            'pictures': [{'label': 'picture', 'prov': []}],
        }
        assert _parse_elements(json_content, 'pictures', 'test.pdf') == []

    def test_parse_missing_bbox_fields(self):
        """Картинка с неполным bbox пропускается."""
        json_content = {
            'pictures': [{'prov': [{'page_no': 1, 'bbox': {'l': 10}}]}],
        }
        assert _parse_elements(json_content, 'pictures', 'test.pdf') == []

    def test_parse_formulas(self):
        """Парсинг формул из texts с label='formula'."""
        json_content = {
            'texts': [
                {
                    'label': 'formula',
                    'prov': [{'page_no': 3, 'bbox': {
                        'l': 50.0, 't': 400.0, 'r': 300.0, 'b': 350.0,
                        'coord_origin': 'BOTTOMLEFT',
                    }}],
                },
                {
                    'label': 'paragraph',
                    'prov': [{'page_no': 3, 'bbox': {
                        'l': 50.0, 't': 300.0, 'r': 300.0, 'b': 250.0,
                        'coord_origin': 'BOTTOMLEFT',
                    }}],
                },
            ],
        }
        formulas = _parse_elements(json_content, 'formulas', 'test.pdf')
        assert len(formulas) == 1
        assert formulas[0].page_no == 3

    def test_parse_formulas_empty_texts(self):
        """Нет формул в texts — пустой список."""
        json_content = {
            'texts': [
                {'label': 'paragraph', 'prov': [{'page_no': 1, 'bbox': {
                    'l': 10, 't': 20, 'r': 30, 'b': 40,
                    'coord_origin': 'BOTTOMLEFT',
                }}]},
            ],
        }
        assert _parse_elements(json_content, 'formulas', 'test.pdf') == []

    def test_parse_unknown_kind(self):
        """Неизвестный kind — пустой список."""
        assert _parse_elements({'pictures': []}, 'unknown', 'test.pdf') == []


class TestDoclingPdfConverterSync:
    def test_requests_json_when_vision_client(self, pdf_dir):
        """При наличии vision_client запрашивает md + json."""
        vision_client = AsyncMock()
        converter = DoclingPdfConverter(
            'http://localhost:5001', vision_client=vision_client,
        )

        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            'document': {
                'md_content': '# Doc',
                'json_content': json.dumps({'pictures': []}),
            },
        }

        with patch('morag.sources.pdf_converter.httpx.post', return_value=response) as mock_post:
            result = converter._convert_sync(b'%PDF-1.4', 'report.pdf')

        assert result is not None
        md, pics, formulas = result
        assert md == '# Doc'
        assert pics == []
        assert formulas == []

        call_args = mock_post.call_args
        payload = call_args.kwargs.get('json') or call_args[1].get('json')
        assert 'json' in payload['options']['to_formats']
        assert 'md' in payload['options']['to_formats']

    def test_requests_only_md_without_vision_client(self):
        """Без vision_client запрашивает только md."""
        converter = DoclingPdfConverter(
            'http://localhost:5001', vision_client=None,
        )

        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = {'document': {'md_content': '# Doc'}}

        with patch('morag.sources.pdf_converter.httpx.post', return_value=response) as mock_post:
            result = converter._convert_sync(b'%PDF-1.4', 'report.pdf')

        assert result is not None
        md, pics, formulas = result
        assert md == '# Doc'
        assert pics == []
        assert formulas == []

        call_args = mock_post.call_args
        payload = call_args.kwargs.get('json') or call_args[1].get('json')
        assert payload['options']['to_formats'] == ['md']
