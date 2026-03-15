from __future__ import annotations

import logging
from pathlib import Path

from morag.sources.directory import DirectorySource
from morag.sources.markdown import MarkdownSource
from morag.sources.pdf import PdfSource
from morag.sources.pdf_converter import PdfConverter

logger = logging.getLogger(__name__)


class LocalDocumentSource:
    """Композитный источник локальных документов.

    Управляет тремя внутренними источниками и прогоняет их через pipeline
    в правильном порядке: директории → markdown → pdf.
    Не наследует Source — это оркестратор.
    """

    def __init__(
        self,
        root: Path | str,
        pdf_converter: PdfConverter | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._pdf_converter = pdf_converter

    async def run(self, pipeline) -> None:
        """Запустить индексацию всех локальных источников в правильном порядке.

        1. DirectorySource — структурные документы для поддиректорий
        2. MarkdownSource — MD-файлы
        3. PdfSource — PDF-файлы (если pdf_converter задан)
        """
        logger.info('Indexing local documents from %s', self._root)

        # 1. Директории
        dir_source = DirectorySource(self._root)
        logger.info('Phase 1/3: indexing directories...')
        await pipeline.run(dir_source)

        # 2. Markdown
        md_source = MarkdownSource(self._root)
        logger.info('Phase 2/3: indexing markdown files...')
        await pipeline.run(md_source)

        # 3. PDF
        if self._pdf_converter is not None:
            pdf_source = PdfSource(self._root, converter=self._pdf_converter)
            logger.info('Phase 3/3: indexing PDF files...')
            await pipeline.run(pdf_source)
        else:
            logger.info('Phase 3/3: skipping PDF (not configured)')

        logger.info('Local documents indexing complete')
