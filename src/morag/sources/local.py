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

    Управляет тремя внутренними source-классами (Directory/Markdown/Pdf) с
    общим kind='local' и общим name (передаётся из config.LocalSourceConfig.name).
    Не наследует Source — это оркестратор.
    """

    def __init__(
        self,
        root: Path | str,
        pdf_converter: PdfConverter | None = None,
        name: str = 'default',
    ) -> None:
        self._root = Path(root).resolve()
        self._pdf_converter = pdf_converter
        self._name = name

    async def run(self, pipeline) -> None:
        logger.info('Indexing local documents [%s] from %s', self._name, self._root)

        dir_source = DirectorySource(self._root, name=self._name)
        logger.info('Phase 1/3: indexing directories...')
        await pipeline.run(dir_source)

        md_source = MarkdownSource(self._root, name=self._name)
        logger.info('Phase 2/3: indexing markdown files...')
        await pipeline.run(md_source)

        if self._pdf_converter is not None:
            pdf_source = PdfSource(self._root, converter=self._pdf_converter, name=self._name)
            logger.info('Phase 3/3: indexing PDF files...')
            await pipeline.run(pdf_source)
        else:
            logger.info('Phase 3/3: skipping PDF (not configured)')

        logger.info('Local documents indexing complete')
