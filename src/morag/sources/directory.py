from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from morag.sources.base import Document, Source

# Расширения файлов, поддерживаемые локальными источниками.
# Директория считается «содержащей контент», если в ней (рекурсивно) есть
# хотя бы один файл с одним из этих расширений.
SUPPORTED_EXTENSIONS = {'*.md', '*.pdf'}


class DirectorySource(Source):
    """Источник структурных документов для поддиректорий.

    Сканирует директорию и создаёт фиктивный Document для каждой поддиректории,
    содержащей файлы поддерживаемых форматов (md, pdf). Документы structural=True,
    не чанкуются — служат для иерархии parent_doc_ids.
    """

    @property
    def source_type(self) -> str:
        return 'directory'

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    async def get_metadata(self) -> list[Document]:
        """Вернуть стабы для всех поддиректорий с контентом."""
        dirs_with_content: set[Path] = set()
        for pattern in SUPPORTED_EXTENSIONS:
            for path in self._root.rglob(pattern):
                for parent in path.relative_to(self._root).parents:
                    if parent != Path('.'):
                        dirs_with_content.add(self._root / parent)

        stubs: list[Document] = []
        for dir_path in sorted(dirs_with_content):
            stub = self._get_dir_metadata(dir_path)
            if stub is not None:
                stubs.append(stub)

        stubs.sort(key=lambda s: s.id)
        return stubs

    async def load_one(self, doc_id: str) -> Document | None:
        """Загрузить структурный документ директории."""
        dir_path = self._root / doc_id.rstrip('/')
        return self._load_dir(dir_path)

    def _parent_doc_ids(self, path: Path) -> list[str]:
        """Вычислить parent_doc_ids для директории."""
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
                source_type='directory',
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
                source_type='directory',
                size=0,
                structural=True,
                parent_doc_ids=self._parent_doc_ids(dir_path),
            )
        except OSError:
            return None
