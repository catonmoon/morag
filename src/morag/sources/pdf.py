from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from morag.llm.client import LLMClient
from morag.sources.base import Document, Source
from morag.sources.pdf_converter import DoclingPdfConverter, PdfConverter

logger = logging.getLogger(__name__)


class PdfSource(Source):
    """Источник локальных PDF-файлов.

    Рекурсивно сканирует директорию, конвертирует PDF → Markdown через PdfConverter,
    обрабатывает изображения и формулы через Vision LLM.
    parent_doc_ids ссылается на структурные документы директорий (создаются DirectorySource).
    """

    @property
    def source_type(self) -> str:
        return 'pdf'

    def __init__(
        self,
        root: Path | str,
        docling_base_url: str,
        docling_timeout: int = 300,
        vision_client: LLMClient | None = None,
        converter: PdfConverter | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._converter = converter or DoclingPdfConverter(
            docling_base_url=docling_base_url,
            docling_timeout=docling_timeout,
            vision_client=vision_client,
        )

    async def get_metadata(self) -> list[Document]:
        """Вернуть стабы PDF-файлов (без конвертации)."""
        all_pdf_files = sorted(self._root.rglob('*.pdf'))

        stubs: list[Document] = []
        for path in all_pdf_files:
            stub = self._get_file_metadata(path)
            if stub is not None:
                stubs.append(stub)

        stubs.sort(key=lambda s: s.id)
        return stubs

    async def load_one(self, doc_id: str) -> Document | None:
        """Загрузить PDF: конвертировать через PdfConverter."""
        path = self._root / doc_id
        try:
            stat = path.stat()
        except OSError:
            return None

        updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        pdf_bytes = path.read_bytes()
        markdown = await self._converter.convert(pdf_bytes, path.name)
        if markdown is None:
            logger.error('Failed to convert PDF: %s', doc_id)
            return None

        return Document(
            id=doc_id,
            path=[doc_id],
            text=markdown,
            updated_at=updated_at,
            source_type='pdf',
            size=stat.st_size,
            url=path.as_uri(),
            parent_doc_ids=self._parent_doc_ids(path),
        )

    def _parent_doc_ids(self, path: Path) -> list[str]:
        """Вычислить parent_doc_ids для файла."""
        parent_dir = path.parent
        if parent_dir == self._root:
            return []
        return [str(parent_dir.relative_to(self._root)) + '/']

    def _get_file_metadata(self, path: Path) -> Document | None:
        """Получить метаданные PDF-файла без конвертации."""
        try:
            stat = path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            doc_id = str(path.relative_to(self._root))
            return Document(
                id=doc_id,
                path=[doc_id],
                text='',
                updated_at=updated_at,
                source_type='pdf',
                size=stat.st_size,
                url=path.as_uri(),
                parent_doc_ids=self._parent_doc_ids(path),
            )
        except OSError:
            return None
