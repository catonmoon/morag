from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from morag.sources.base import Document, Source


class MarkdownSource(Source):
    """Источник локальных Markdown-файлов.

    Рекурсивно сканирует директорию и возвращает:
    - фиктивный Document для каждой поддиректории, содержащей *.md файлы (id = 'subdir/')
    - Document для каждого *.md файла (id = относительный путь)

    parent_doc_ids заполняется по дереву: файл/поддиректория → id ближайшей родительской директории.
    Корневые документы (прямо в root) имеют parent_doc_ids = [].
    """

    @property
    def source_type(self) -> str:
        return 'markdown'

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    async def get_metadata(self) -> list[Document]:
        """Вернуть стабы: директории (фиктивные) + MD-файлы, с parent_doc_ids."""
        all_md_files = sorted(self._root.rglob('*.md'))

        # Собрать все поддиректории, содержащие .md файлы (прямо или через вложенные)
        dirs_with_content: set[Path] = set()
        for path in all_md_files:
            for parent in path.relative_to(self._root).parents:
                if parent != Path('.'):
                    dirs_with_content.add(self._root / parent)

        stubs: list[Document] = []
        for dir_path in sorted(dirs_with_content):
            stub = self._get_dir_metadata(dir_path)
            if stub is not None:
                stubs.append(stub)
        for path in all_md_files:
            stub = self._get_file_metadata(path)
            if stub is not None:
                stubs.append(stub)

        stubs.sort(key=lambda s: s.id)
        return stubs

    async def load_one(self, doc_id: str) -> Document | None:
        """Загрузить один документ по doc_id. Директории имеют id оканчивающийся на '/'."""
        if doc_id.endswith('/'):
            return self._load_dir(self._root / doc_id.rstrip('/'))
        return self._load_file(self._root / doc_id)

    def _parent_doc_ids(self, path: Path) -> list[str]:
        """Вычислить parent_doc_ids для файла или директории."""
        parent_dir = path.parent
        if parent_dir == self._root:
            return []
        return [str(parent_dir.relative_to(self._root)) + '/']

    def _get_dir_metadata(self, dir_path: Path) -> Document | None:
        """Получить стаб фиктивного документа для директории."""
        try:
            stat = dir_path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            doc_id = str(dir_path.relative_to(self._root)) + '/'
            return Document(
                id=doc_id,
                path=[doc_id],
                text='',
                updated_at=updated_at,
                source_type='markdown',
                size=0,
                structural=True,
                parent_doc_ids=self._parent_doc_ids(dir_path),
            )
        except OSError:
            return None

    def _load_dir(self, dir_path: Path) -> Document | None:
        """Загрузить фиктивный Document для директории (текст = имя директории)."""
        try:
            stat = dir_path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            doc_id = str(dir_path.relative_to(self._root)) + '/'
            return Document(
                id=doc_id,
                path=[doc_id],
                text=dir_path.name,
                updated_at=updated_at,
                source_type='markdown',
                size=0,
                structural=True,
                parent_doc_ids=self._parent_doc_ids(dir_path),
            )
        except OSError:
            return None

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
