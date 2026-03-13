from __future__ import annotations

import logging
from pathlib import Path

from morag.llm.client import LLMClient
from morag.sources.directory import DirectorySource
from morag.sources.markdown import MarkdownSource
from morag.sources.pdf import PdfSource

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
        docling_base_url: str | None = None,
        docling_timeout: int = 300,
        vision_client: LLMClient | None = None,
        vision_max_tokens: int | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._docling_base_url = docling_base_url
        self._docling_timeout = docling_timeout
        self._vision_client = vision_client
        self._vision_max_tokens = vision_max_tokens

    async def run(self, pipeline) -> None:
        """Запустить индексацию всех локальных источников в правильном порядке.

        1. DirectorySource — структурные документы для поддиректорий
        2. MarkdownSource — MD-файлы
        3. PdfSource — PDF-файлы (если docling сконфигурирован)
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

        # 3. PDF (только если docling сконфигурирован)
        if self._docling_base_url is not None:
            pdf_source = PdfSource(
                self._root,
                docling_base_url=self._docling_base_url,
                docling_timeout=self._docling_timeout,
                vision_client=self._vision_client,
                vision_max_tokens=self._vision_max_tokens,
            )
            logger.info('Phase 3/3: indexing PDF files...')
            await pipeline.run(pdf_source)
        else:
            logger.info('Phase 3/3: skipping PDF (docling not configured)')

        logger.info('Local documents indexing complete')
