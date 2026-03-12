# Руководство по написанию документации в Markdown

Этот документ описывает продвинутые техники работы с Markdown, включая вложенные блоки кода, шаблоны документации и автоматическую генерацию.

## Базовые блоки кода

Стандартный блок кода оформляется тройными обратными кавычками. Можно указать язык для подсветки синтаксиса:

```python
def hello(name: str) -> str:
    """Приветствие пользователя."""
    return f'Hello, {name}!'
```

Также поддерживается альтернативный синтаксис с тильдами:

~~~bash
echo "Hello from bash"
ls -la /tmp
~~~

## Документирование Markdown внутри Markdown

Одна из частых задач — показать пример Markdown-разметки внутри документации. Для этого используются вложенные code fences с разным количеством кавычек.

### Пример: шаблон README

Вот как должен выглядеть README.md для нового проекта:

````markdown
# Название проекта

Краткое описание проекта в одном-двух предложениях.

## Установка

```bash
pip install my-project
```

## Быстрый старт

```python
from my_project import Client

client = Client(api_key='your-key')
result = client.query('Hello!')
print(result)
```

## Конфигурация

Создайте файл `config.yml` в корне проекта:

```yaml
server:
  host: localhost
  port: 8080
  debug: false

database:
  url: postgresql://user:pass@localhost/db
  pool_size: 10

logging:
  level: INFO
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

## API Reference

### `Client(api_key, base_url=None, timeout=30)`

Основной клиент для работы с API.

**Параметры:**
- `api_key` (str) — ключ API
- `base_url` (str, optional) — базовый URL сервера
- `timeout` (int) — таймаут запросов в секундах

### `Client.query(text, **kwargs)`

Отправляет запрос к API.

```python
result = client.query(
    'Какая погода сегодня?',
    temperature=0.7,
    max_tokens=500,
)
```

## Лицензия

MIT License. See [LICENSE](LICENSE) for details.
````

### Пример: документация API-эндпойнтов

Для документирования REST API удобно использовать следующий шаблон:

````markdown
## POST /api/v1/documents

Создаёт новый документ в системе.

### Запрос

```json
{
  "title": "My Document",
  "content": "Document content in markdown format",
  "tags": ["tutorial", "api"],
  "metadata": {
    "author": "John Doe",
    "version": "1.0"
  }
}
```

### Ответ (201 Created)

```json
{
  "id": "doc_abc123",
  "title": "My Document",
  "content": "Document content in markdown format",
  "tags": ["tutorial", "api"],
  "metadata": {
    "author": "John Doe",
    "version": "1.0"
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "status": "draft"
}
```

### Ошибки

| Код | Описание |
|-----|----------|
| 400 | Невалидный JSON или отсутствуют обязательные поля |
| 401 | Отсутствует или невалидный API-ключ |
| 413 | Размер документа превышает лимит (10MB) |
| 429 | Превышен лимит запросов |
````

## Генерация документации из кода

Многие инструменты позволяют автоматически генерировать документацию из docstrings. Вот пример конфигурации:

### MkDocs с mkdocstrings

```yaml
# mkdocs.yml
site_name: My Project
theme:
  name: material
  palette:
    scheme: slate
    primary: indigo
  features:
    - navigation.tabs
    - navigation.sections
    - search.highlight
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            members_order: source

nav:
  - Home: index.md
  - API Reference:
    - Client: api/client.md
    - Models: api/models.md
    - Exceptions: api/exceptions.md
  - Guides:
    - Quick Start: guides/quickstart.md
    - Configuration: guides/configuration.md
    - Deployment: guides/deployment.md
```

### Sphinx с autodoc

```python
# conf.py — конфигурация Sphinx
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'My Project'
copyright = '2024, Author'
author = 'Author'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'myst_parser',
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'requests': ('https://requests.readthedocs.io/en/latest/', None),
}

html_theme = 'sphinx_rtd_theme'
```

## CI/CD пайплайн для документации

Автоматическая публикация документации при каждом коммите:

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation
on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - 'src/**/*.py'
      - 'mkdocs.yml'

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # для git-revision-date-localized

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install --with docs

      - name: Build docs
        run: poetry run mkdocs build --strict

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site/

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
    steps:
      - uses: actions/deploy-pages@v4
```

## Продвинутые шаблоны

### Шаблон Changelog

````markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- New feature description

### Changed
- Updated dependency versions

## [1.2.0] - 2024-03-15

### Added
- Support for PDF document processing
- New `--format` flag for CLI output
- Batch processing mode for large datasets

### Fixed
- Memory leak in connection pool (#142)
- Incorrect encoding detection for CJK documents (#156)

### Changed
- Upgraded minimum Python version to 3.11
- Improved error messages for configuration errors

### Deprecated
- `Client.sync_query()` — use `Client.query()` with `asyncio.run()` instead

## [1.1.0] - 2024-02-01

### Added
- Retry mechanism with exponential backoff
- Health check endpoint `/api/health`
- Configurable connection pool size

### Fixed
- Race condition in concurrent document updates (#128)
````

### Шаблон ADR (Architecture Decision Record)

````markdown
# ADR-001: Выбор векторной базы данных

## Статус

Принято (2024-01-10)

## Контекст

Нам необходимо выбрать векторную базу данных для хранения эмбеддингов документов. Требования:
- Поддержка гибридного поиска (dense + sparse)
- Горизонтальное масштабирование
- Self-hosted вариант
- Payload-фильтрация

## Рассмотренные варианты

### Qdrant
- Rust, высокая производительность
- Гибридный поиск из коробки (named vectors)
- gRPC и REST API
- Payload индексы

### Milvus
- Go + C++
- Более сложная архитектура (etcd, MinIO, Pulsar)
- Хорошая масштабируемость

### Weaviate
- Go
- GraphQL API
- Встроенные модули для векторизации

## Решение

Выбран **Qdrant** по следующим причинам:
1. Простота деплоя (один бинарник)
2. Named vectors для гибридного поиска
3. Эффективные payload-индексы
4. Активное сообщество и документация

## Последствия

- Все эмбеддинги хранятся в Qdrant
- Payload-индексы на `doc_id` и `parent_doc_ids`
- Зависимость от Qdrant REST API в pipeline

```python
# Пример подключения
from qdrant_client import QdrantClient

client = QdrantClient(host='localhost', port=6333)
```
````

## Заключение

Хорошая документация — это инвестиция, которая окупается многократно. Используйте шаблоны, автоматизируйте генерацию и публикацию, и помните: лучшая документация — та, которую легко поддерживать в актуальном состоянии.
