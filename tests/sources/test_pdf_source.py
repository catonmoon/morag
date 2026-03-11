import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from morag.sources.base import Document, Source
from morag.sources.pdf import PdfSource, _PictureInfo


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
def source(pdf_dir) -> PdfSource:
    return PdfSource(
        pdf_dir,
        docling_base_url='http://localhost:5001',
        docling_timeout=30,
    )


def _mock_convert_result(
    md_content: str,
    pictures: list[_PictureInfo] | None = None,
    formulas: list[_PictureInfo] | None = None,
):
    """Создать мок результата _convert_pdf."""
    return (md_content, pictures or [], formulas or [])


def _patch_convert(
    source: PdfSource,
    md_content: str,
    pictures: list[_PictureInfo] | None = None,
    formulas: list[_PictureInfo] | None = None,
):
    """Патч _convert_pdf для PdfSource."""
    result = _mock_convert_result(md_content, pictures, formulas)
    return patch.object(source, '_convert_pdf', new_callable=AsyncMock, return_value=result)


def _patch_convert_none(source: PdfSource):
    """Патч _convert_pdf возвращающий None (ошибка конвертации)."""
    return patch.object(source, '_convert_pdf', new_callable=AsyncMock, return_value=None)


class TestPdfSource:
    def test_is_source(self, source):
        assert isinstance(source, Source)

    def test_source_type(self, source):
        assert source.source_type == 'pdf'

    async def test_get_metadata_finds_pdfs(self, source):
        stubs = await source.get_metadata()
        ids = {s.id for s in stubs}
        assert 'report.pdf' in ids
        assert 'docs/nested.pdf' in ids

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
        root_stub = next(s for s in stubs if s.id == 'report.pdf')
        assert root_stub.parent_doc_ids == []

    async def test_parent_doc_ids_nested(self, source):
        stubs = await source.get_metadata()
        nested_stub = next(s for s in stubs if s.id == 'docs/nested.pdf')
        assert nested_stub.parent_doc_ids == ['docs/']

    async def test_stubs_have_url(self, source):
        stubs = await source.get_metadata()
        for stub in stubs:
            assert stub.url is not None
            assert stub.url.startswith('file://')

    async def test_ignores_non_pdf_files(self, tmp_path):
        (tmp_path / 'readme.md').write_text('# Doc')
        (tmp_path / 'data.txt').write_text('text')
        (tmp_path / 'doc.pdf').write_bytes(b'%PDF-1.4')
        source = PdfSource(tmp_path, 'http://localhost:5001')
        stubs = await source.get_metadata()
        assert len(stubs) == 1
        assert stubs[0].id == 'doc.pdf'

    async def test_empty_directory(self, tmp_path):
        source = PdfSource(tmp_path, 'http://localhost:5001')
        stubs = await source.get_metadata()
        assert stubs == []


class TestPdfSourceLoadOne:
    async def test_load_one_calls_docling(self, pdf_dir):
        """load_one конвертирует PDF и возвращает Document."""
        source = PdfSource(pdf_dir, 'http://localhost:5001')

        with _patch_convert(source, '# Report\n\nSome content.'):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert doc.id == 'report.pdf'
        assert doc.source_type == 'pdf'
        assert doc.text == '# Report\n\nSome content.'
        assert doc.size > 0
        assert doc.updated_at.tzinfo is not None

    async def test_load_one_returns_none_for_nonexistent(self, pdf_dir):
        source = PdfSource(pdf_dir, 'http://localhost:5001')
        doc = await source.load_one('nonexistent.pdf')
        assert doc is None

    async def test_load_one_returns_none_on_docling_error(self, pdf_dir):
        """При ошибке docling-serve возвращает None."""
        source = PdfSource(pdf_dir, 'http://localhost:5001')

        with _patch_convert_none(source):
            doc = await source.load_one('report.pdf')

        assert doc is None

    async def test_load_one_preserves_original_metadata(self, pdf_dir):
        """Document содержит метаданные оригинального PDF, а не сконвертированного md."""
        source = PdfSource(pdf_dir, 'http://localhost:5001')
        pdf_path = pdf_dir / 'report.pdf'
        pdf_stat = pdf_path.stat()

        with _patch_convert(source, '# Doc'):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert doc.size == pdf_stat.st_size
        assert doc.url == pdf_path.as_uri()


class TestPdfSourceImageProcessing:
    async def test_placeholders_replaced_by_description(self, pdf_dir):
        """<!-- image --> плейсхолдеры заменяются описанием от Vision LLM."""
        vision_client = AsyncMock()
        vision_client.complete_vision.return_value = 'График роста продаж за 2024 год'

        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Report\n\n<!-- image -->\n\nSome text after.'
        pics = [_PictureInfo(page_no=1, bbox_l=100, bbox_t=700, bbox_r=500, bbox_b=400,
                             coord_origin='BOTTOMLEFT')]
        crop_b64 = 'iVBORw0KGgoAAAANS'

        with _patch_convert(source, md, pics), \
             patch.object(PdfSource, '_crop_image', return_value=crop_b64):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert '<!-- image -->' not in doc.text
        assert 'График роста продаж' in doc.text
        assert '**[Изображение 1]**' in doc.text
        vision_client.complete_vision.assert_called_once()

    async def test_no_vision_client_keeps_placeholders(self, pdf_dir):
        """Без vision_client плейсхолдеры остаются."""
        source = PdfSource(pdf_dir, 'http://localhost:5001', vision_client=None)

        md = '# Doc\n\n<!-- image -->\n'

        with _patch_convert(source, md, []):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert '<!-- image -->' in doc.text

    async def test_formula_placeholders_removed(self, pdf_dir):
        """Плейсхолдеры формул удаляются."""
        source = PdfSource(pdf_dir, 'http://localhost:5001')

        md = '# Math\n\n<!-- formula-not-decoded -->\n\nText.'

        with _patch_convert(source, md):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert 'formula-not-decoded' not in doc.text

    async def test_multiple_images(self, pdf_dir):
        """Несколько изображений обрабатываются корректно."""
        vision_client = AsyncMock()
        vision_client.complete_vision.side_effect = ['Описание 1', 'Описание 2']

        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Doc\n\n<!-- image -->\n\nText between.\n\n<!-- image -->\n'
        pics = [
            _PictureInfo(page_no=1, bbox_l=100, bbox_t=700, bbox_r=500, bbox_b=400,
                         coord_origin='BOTTOMLEFT'),
            _PictureInfo(page_no=2, bbox_l=100, bbox_t=600, bbox_r=500, bbox_b=300,
                         coord_origin='BOTTOMLEFT'),
        ]

        with _patch_convert(source, md, pics), \
             patch.object(PdfSource, '_crop_image', return_value='AAAA'):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert '<!-- image -->' not in doc.text
        assert '**[Изображение 1]**' in doc.text
        assert '**[Изображение 2]**' in doc.text
        assert 'Описание 1' in doc.text
        assert 'Описание 2' in doc.text
        assert vision_client.complete_vision.call_count == 2

    async def test_vision_error_graceful(self, pdf_dir):
        """При ошибке VLM изображение заменяется плейсхолдером без описания."""
        vision_client = AsyncMock()
        vision_client.complete_vision.side_effect = Exception('VLM timeout')

        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Doc\n\n<!-- image -->\n'
        pics = [_PictureInfo(page_no=1, bbox_l=100, bbox_t=700, bbox_r=500, bbox_b=400,
                             coord_origin='BOTTOMLEFT')]

        with _patch_convert(source, md, pics), \
             patch.object(PdfSource, '_crop_image', return_value='AAAA'):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert '<!-- image -->' not in doc.text
        assert '**[Изображение 1]**' in doc.text

    async def test_crop_failure_graceful(self, pdf_dir):
        """При ошибке вырезания изображения — плейсхолдер без описания."""
        vision_client = AsyncMock()
        vision_client.complete_vision.return_value = 'Should not be called'

        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Doc\n\n<!-- image -->\n'
        pics = [_PictureInfo(page_no=1, bbox_l=100, bbox_t=700, bbox_r=500, bbox_b=400,
                             coord_origin='BOTTOMLEFT')]

        with _patch_convert(source, md, pics), \
             patch.object(PdfSource, '_crop_image', return_value=None):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert '<!-- image -->' not in doc.text
        assert '**[Изображение 1]**' in doc.text
        vision_client.complete_vision.assert_not_called()

    async def test_placeholder_picture_count_mismatch(self, pdf_dir):
        """Когда плейсхолдеров больше чем картинок — лишние остаются."""
        vision_client = AsyncMock()
        vision_client.complete_vision.return_value = 'Описание'

        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Doc\n\n<!-- image -->\n\n<!-- image -->\n\n<!-- image -->\n'
        pics = [_PictureInfo(page_no=1, bbox_l=100, bbox_t=700, bbox_r=500, bbox_b=400,
                             coord_origin='BOTTOMLEFT')]

        with _patch_convert(source, md, pics), \
             patch.object(PdfSource, '_crop_image', return_value='AAAA'):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        # Только первый плейсхолдер заменён
        assert '**[Изображение 1]**' in doc.text
        assert 'Описание' in doc.text
        # Остальные 2 остались как есть
        assert doc.text.count('<!-- image -->') == 2


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
        pics = PdfSource._parse_elements(json_content, 'pictures', 'test.pdf')
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
        pics = PdfSource._parse_elements(json_str, 'pictures', 'test.pdf')
        assert len(pics) == 1

    def test_parse_none_returns_empty(self):
        """None json_content — пустой список."""
        assert PdfSource._parse_elements(None, 'pictures', 'test.pdf') == []

    def test_parse_no_pictures_key(self):
        """JSON без ключа pictures — пустой список."""
        assert PdfSource._parse_elements({'texts': []}, 'pictures', 'test.pdf') == []

    def test_parse_empty_prov(self):
        """Картинка без prov пропускается."""
        json_content = {
            'pictures': [{'label': 'picture', 'prov': []}],
        }
        assert PdfSource._parse_elements(json_content, 'pictures', 'test.pdf') == []

    def test_parse_missing_bbox_fields(self):
        """Картинка с неполным bbox пропускается."""
        json_content = {
            'pictures': [{'prov': [{'page_no': 1, 'bbox': {'l': 10}}]}],
        }
        assert PdfSource._parse_elements(json_content, 'pictures', 'test.pdf') == []

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
        formulas = PdfSource._parse_elements(json_content, 'formulas', 'test.pdf')
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
        assert PdfSource._parse_elements(json_content, 'formulas', 'test.pdf') == []

    def test_parse_unknown_kind(self):
        """Неизвестный kind — пустой список."""
        assert PdfSource._parse_elements({'pictures': []}, 'unknown', 'test.pdf') == []


class TestConvertPdfSync:
    def test_requests_json_when_vision_client(self, pdf_dir):
        """При наличии vision_client запрашивает md + json."""
        vision_client = AsyncMock()
        source = PdfSource(pdf_dir, 'http://localhost:5001', vision_client=vision_client)

        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            'document': {
                'md_content': '# Doc',
                'json_content': json.dumps({'pictures': []}),
            },
        }

        with patch('morag.sources.pdf.httpx.post', return_value=response) as mock_post:
            result = source._convert_pdf_sync(pdf_dir / 'report.pdf')

        assert result is not None
        md, pics, formulas = result
        assert md == '# Doc'
        assert pics == []
        assert formulas == []

        # Проверяем что запрошены оба формата
        call_args = mock_post.call_args
        payload = call_args.kwargs.get('json') or call_args[1].get('json')
        assert 'json' in payload['options']['to_formats']
        assert 'md' in payload['options']['to_formats']

    def test_requests_only_md_without_vision_client(self, pdf_dir):
        """Без vision_client запрашивает только md."""
        source = PdfSource(pdf_dir, 'http://localhost:5001', vision_client=None)

        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = {'document': {'md_content': '# Doc'}}

        with patch('morag.sources.pdf.httpx.post', return_value=response) as mock_post:
            result = source._convert_pdf_sync(pdf_dir / 'report.pdf')

        assert result is not None
        md, pics, formulas = result
        assert md == '# Doc'
        assert pics == []
        assert formulas == []

        call_args = mock_post.call_args
        payload = call_args.kwargs.get('json') or call_args[1].get('json')
        assert payload['options']['to_formats'] == ['md']


class TestPdfSourceFormulaProcessing:
    async def test_formula_placeholders_replaced_by_latex(self, pdf_dir):
        """<!-- formula-not-decoded --> заменяется LaTeX от Vision LLM."""
        vision_client = AsyncMock()
        vision_client.complete_vision.return_value = '$$E = mc^2$$'

        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Math\n\n<!-- formula-not-decoded -->\n\nText after.'
        formulas = [_PictureInfo(page_no=1, bbox_l=50, bbox_t=400, bbox_r=300, bbox_b=350,
                                 coord_origin='BOTTOMLEFT')]

        with _patch_convert(source, md, formulas=formulas), \
             patch.object(PdfSource, '_crop_image', return_value='AAAA'):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert 'formula-not-decoded' not in doc.text
        assert '$$E = mc^2$$' in doc.text
        vision_client.complete_vision.assert_called_once()

    async def test_formula_without_vision_client_removed(self, pdf_dir):
        """Без vision_client плейсхолдеры формул удаляются."""
        source = PdfSource(pdf_dir, 'http://localhost:5001', vision_client=None)

        md = '# Math\n\n<!-- formula-not-decoded -->\n\nText.'

        with _patch_convert(source, md):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert 'formula-not-decoded' not in doc.text

    async def test_formula_without_coordinates_removed(self, pdf_dir):
        """С vision_client, но без координат формул — удаляются."""
        vision_client = AsyncMock()
        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Math\n\n<!-- formula-not-decoded -->\n\nText.'

        with _patch_convert(source, md, formulas=[]):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert 'formula-not-decoded' not in doc.text
        vision_client.complete_vision.assert_not_called()

    async def test_multiple_formulas(self, pdf_dir):
        """Несколько формул обрабатываются корректно."""
        vision_client = AsyncMock()
        vision_client.complete_vision.side_effect = ['$$a^2 + b^2 = c^2$$', '$$\\int_0^1 x dx$$']

        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Math\n\n<!-- formula-not-decoded -->\n\nText.\n\n<!-- formula-not-decoded -->\n'
        formulas = [
            _PictureInfo(page_no=1, bbox_l=50, bbox_t=400, bbox_r=300, bbox_b=350,
                         coord_origin='BOTTOMLEFT'),
            _PictureInfo(page_no=1, bbox_l=50, bbox_t=300, bbox_r=300, bbox_b=250,
                         coord_origin='BOTTOMLEFT'),
        ]

        with _patch_convert(source, md, formulas=formulas), \
             patch.object(PdfSource, '_crop_image', return_value='AAAA'):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert 'formula-not-decoded' not in doc.text
        assert '$$a^2 + b^2 = c^2$$' in doc.text
        assert '$$\\int_0^1 x dx$$' in doc.text
        assert vision_client.complete_vision.call_count == 2

    async def test_formula_vision_error_graceful(self, pdf_dir):
        """При ошибке VLM формула удаляется без описания."""
        vision_client = AsyncMock()
        vision_client.complete_vision.side_effect = Exception('VLM timeout')

        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Math\n\n<!-- formula-not-decoded -->\n\nText.'
        formulas = [_PictureInfo(page_no=1, bbox_l=50, bbox_t=400, bbox_r=300, bbox_b=350,
                                 coord_origin='BOTTOMLEFT')]

        with _patch_convert(source, md, formulas=formulas), \
             patch.object(PdfSource, '_crop_image', return_value='AAAA'):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert 'formula-not-decoded' not in doc.text

    async def test_formula_crop_failure_graceful(self, pdf_dir):
        """При ошибке вырезания формулы — удаляется без описания."""
        vision_client = AsyncMock()
        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Math\n\n<!-- formula-not-decoded -->\n\nText.'
        formulas = [_PictureInfo(page_no=1, bbox_l=50, bbox_t=400, bbox_r=300, bbox_b=350,
                                 coord_origin='BOTTOMLEFT')]

        with _patch_convert(source, md, formulas=formulas), \
             patch.object(PdfSource, '_crop_image', return_value=None):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert 'formula-not-decoded' not in doc.text
        vision_client.complete_vision.assert_not_called()

    async def test_images_and_formulas_together(self, pdf_dir):
        """Изображения и формулы обрабатываются вместе."""
        vision_client = AsyncMock()
        vision_client.complete_vision.side_effect = ['Описание графика', '$$E = mc^2$$']

        source = PdfSource(
            pdf_dir, 'http://localhost:5001',
            vision_client=vision_client,
        )

        md = '# Doc\n\n<!-- image -->\n\nText.\n\n<!-- formula-not-decoded -->\n'
        pics = [_PictureInfo(page_no=1, bbox_l=100, bbox_t=700, bbox_r=500, bbox_b=400,
                             coord_origin='BOTTOMLEFT')]
        formulas = [_PictureInfo(page_no=1, bbox_l=50, bbox_t=300, bbox_r=300, bbox_b=250,
                                 coord_origin='BOTTOMLEFT')]

        with _patch_convert(source, md, pictures=pics, formulas=formulas), \
             patch.object(PdfSource, '_crop_image', return_value='AAAA'):
            doc = await source.load_one('report.pdf')

        assert doc is not None
        assert '<!-- image -->' not in doc.text
        assert 'formula-not-decoded' not in doc.text
        assert '**[Изображение 1]**' in doc.text
        assert 'Описание графика' in doc.text
        assert '$$E = mc^2$$' in doc.text
        assert vision_client.complete_vision.call_count == 2
