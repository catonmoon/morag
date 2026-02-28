from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from morag.sources.base import Document, Source


class MarkdownSource(Source):
    """Источник локальных Markdown-файлов.

    Рекурсивно сканирует директорию и возвращает Document для каждого *.md файла.
    id документа — путь относительно корневой директории (стабильный, не абсолютный).
    В будущем обработка других форматов (PDF, DOCX) добавляется через DocumentProcessor.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    async def load(self) -> list[Document]:
        """Загрузить все MD-файлы из корневой директории."""
        docs: list[Document] = []

        for path in sorted(self._root.rglob('*.md')):
            doc = self._load_file(path)
            if doc is not None:
                docs.append(doc)

        return docs

    def _load_file(self, path: Path) -> Document | None:
        """Загрузить один MD-файл и создать Document."""
        try:
            stat = path.stat()
            text = path.read_text(encoding='utf-8')
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            doc_id = str(path.relative_to(self._root))

            return Document(
                id=doc_id,
                path=doc_id,
                text=text,
                updated_at=updated_at,
                source_type='markdown',
                size=stat.st_size,
            )
        except OSError:
            return None
