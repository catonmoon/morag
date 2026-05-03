from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from morag.sources.base import Document, Source

# Расширения файлов, поддерживаемые локальными источниками.
# Директория считается «содержащей контент», если в ней (рекурсивно) есть
# хотя бы один файл с одним из этих расширений.
SUPPORTED_EXTENSIONS = {'*.md', '*.pdf'}


class DirectorySource(Source):
    """Структурные документы для поддиректорий локального источника.

    kind='local' (как часть config.LocalSourceConfig). Document.id префиксится
    через self.make_id() — `local:<name>:<relative-dir>/`.
    """

    @property
    def source_type(self) -> str:
        return 'directory'

    def __init__(self, root: Path | str, name: str = 'default') -> None:
        self._root = Path(root).resolve()
        self._kind = 'local'
        self._name = name

    async def get_metadata(self) -> list[Document]:
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
        external = self._strip_prefix(doc_id)
        dir_path = self._root / external.rstrip('/')
        return self._load_dir(dir_path)

    def _strip_prefix(self, doc_id: str) -> str:
        prefix = f'{self._kind}:{self._name}:'
        return doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id

    def _parent_doc_ids(self, path: Path) -> list[str]:
        parent_dir = path.parent
        if parent_dir == self._root:
            return []
        return [self.make_id(str(parent_dir.relative_to(self._root)) + '/')]

    def _get_dir_metadata(self, dir_path: Path) -> Document | None:
        try:
            stat = dir_path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            external = str(dir_path.relative_to(self._root)) + '/'
            doc_id = self.make_id(external)
            return Document(
                id=doc_id,
                path=[external],
                text='',
                updated_at=updated_at,
                source_type='directory',
                title=dir_path.name,
                size=0,
                structural=True,
                parent_doc_ids=self._parent_doc_ids(dir_path),
                payload={'source_name': self._name, 'source_kind': self._kind},
            )
        except OSError:
            return None

    def _load_dir(self, dir_path: Path) -> Document | None:
        try:
            stat = dir_path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            external = str(dir_path.relative_to(self._root)) + '/'
            doc_id = self.make_id(external)
            return Document(
                id=doc_id,
                path=[external],
                text=dir_path.name,
                title=dir_path.name,
                updated_at=updated_at,
                source_type='directory',
                size=0,
                structural=True,
                parent_doc_ids=self._parent_doc_ids(dir_path),
                payload={'source_name': self._name, 'source_kind': self._kind},
            )
        except OSError:
            return None
