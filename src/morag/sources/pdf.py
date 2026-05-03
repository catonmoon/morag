from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from morag.sources.base import Document, Source
from morag.sources.pdf_converter import PdfConverter

logger = logging.getLogger(__name__)


class PdfSource(Source):
    """Источник локальных PDF-файлов.

    Рекурсивно сканирует директорию, конвертирует PDF → Markdown через PdfConverter.
    parent_doc_ids ссылается на структурные документы директорий (создаются DirectorySource).
    """

    @property
    def source_type(self) -> str:
        return 'pdf'

    def __init__(
        self,
        root: Path | str,
        converter: PdfConverter,
        name: str = 'default',
    ) -> None:
        self._root = Path(root).resolve()
        self._converter = converter
        self._kind = 'local'
        self._name = name

    async def get_metadata(self) -> list[Document]:
        all_pdf_files = sorted(self._root.rglob('*.pdf'))

        stubs: list[Document] = []
        for path in all_pdf_files:
            stub = self._get_file_metadata(path)
            if stub is not None:
                stubs.append(stub)

        stubs.sort(key=lambda s: s.id)
        return stubs

    async def load_one(self, doc_id: str) -> Document | None:
        external = self._strip_prefix(doc_id)
        path = self._root / external
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
            id=self.make_id(external),
            path=[external],
            text=markdown,
            updated_at=updated_at,
            source_type='pdf',
            title=path.stem,
            size=stat.st_size,
            url=path.as_uri(),
            parent_doc_ids=self._parent_doc_ids(path),
            paged=True,
            payload={'source_name': self._name, 'source_kind': self._kind},
        )

    def _strip_prefix(self, doc_id: str) -> str:
        prefix = f'{self._kind}:{self._name}:'
        return doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id

    def _parent_doc_ids(self, path: Path) -> list[str]:
        parent_dir = path.parent
        if parent_dir == self._root:
            return []
        return [self.make_id(str(parent_dir.relative_to(self._root)) + '/')]

    def _get_file_metadata(self, path: Path) -> Document | None:
        try:
            stat = path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            external = str(path.relative_to(self._root))
            return Document(
                id=self.make_id(external),
                path=[external],
                text='',
                updated_at=updated_at,
                source_type='pdf',
                title=path.stem,
                size=stat.st_size,
                url=path.as_uri(),
                parent_doc_ids=self._parent_doc_ids(path),
                paged=True,
                payload={'source_name': self._name, 'source_kind': self._kind},
            )
        except OSError:
            return None
