from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from morag.sources.base import Document, Source


class MarkdownSource(Source):
    """Источник локальных Markdown-файлов.

    Рекурсивно сканирует директорию и возвращает Document для каждого *.md файла.
    parent_doc_ids ссылается на структурные документы директорий (создаются DirectorySource).
    """

    @property
    def source_type(self) -> str:
        return 'markdown'

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    async def get_metadata(self) -> list[Document]:
        """Вернуть стабы MD-файлов (без чтения содержимого)."""
        all_md_files = sorted(self._root.rglob('*.md'))

        stubs: list[Document] = []
        for path in all_md_files:
            stub = self._get_file_metadata(path)
            if stub is not None:
                stubs.append(stub)

        stubs.sort(key=lambda s: s.id)
        return stubs

    async def load_one(self, doc_id: str) -> Document | None:
        """Загрузить один MD-файл по doc_id."""
        return self._load_file(self._root / doc_id)

    def _parent_doc_ids(self, path: Path) -> list[str]:
        """Вычислить parent_doc_ids для файла."""
        parent_dir = path.parent
        if parent_dir == self._root:
            return []
        return [str(parent_dir.relative_to(self._root)) + '/']

    def _get_file_metadata(self, path: Path) -> Document | None:
        """Получить метаданные файла без чтения содержимого."""
        try:
            stat = path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            doc_id = str(path.relative_to(self._root))
            return Document(
                id=doc_id,
                path=[doc_id],
                text='',
                updated_at=updated_at,
                source_type='markdown',
                size=stat.st_size,
                url=path.as_uri(),
                parent_doc_ids=self._parent_doc_ids(path),
            )
        except OSError:
            return None

    def _load_file(self, path: Path) -> Document | None:
        """Загрузить один MD-файл и создать Document."""
        try:
            stat = path.stat()
            text = path.read_text(encoding='utf-8')
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            doc_id = str(path.relative_to(self._root))
            return Document(
                id=doc_id,
                path=[doc_id],
                text=text,
                updated_at=updated_at,
                source_type='markdown',
                size=stat.st_size,
                url=path.as_uri(),
                parent_doc_ids=self._parent_doc_ids(path),
            )
        except OSError:
            return None
