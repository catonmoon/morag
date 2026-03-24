# ADR-0001: Стратегия парсинга Markdown и чанкинга

**Статус**: принято
**Дата**: 2026-03-13
**Контекст решения**: выбор между кастомной реализацией и open-source библиотеками для разбивки текста

## Контекст

morag использует кастомный пайплайн разбивки Markdown-документов на чанки для RAG-системы.
Пайплайн включает:

- Разбиение по заголовкам Markdown (`_split_by_headers`)
- Разбиение по абзацам с трекингом code fences (`_split_paragraphs`)
- Разбивку таблиц по строкам с дублированием шапки (`TableRowSplitter`)
- Семантический чанкинг через эмбеддинги (`SemanticChunker`)
- Fixed-size fallback (`FixedSizeSplitter`)
- Жадную упаковку блоков (`pack_blocks`)
- LLM-based чанкинг (`LLMChunker`)

При первом прогоне SemanticChunker обнаружен баг: regex-based трекинг code fences
(`in_fence = not in_fence`) некорректно обрабатывает конструкции вида
`[Изображение: ```plantuml...```]` — непарный toggle приводит к склейке всего остатка
документа в один чанк (18272 токена). Баг исправлен ad-hoc через `_fence_toggle()`,
но является симптомом фундаментальной проблемы regex-подхода к парсингу Markdown.

Проведено [исследование open-source библиотек](../../reports/research/2026-03-13-chunking-libraries.md):
Chonkie, semantic-text-splitter, semchunk, LangChain, LlamaIndex, Unstructured, markdown-it-py.

## Решение

### 1. Парсинг Markdown — перевести на `markdown-it-py`

Заменить regex-based парсинг Markdown (`_split_by_headers`, `_split_paragraphs`, `_is_code_fence`,
`_CODE_FENCE_RE`) на `markdown-it-py`.

**Выбор обоснован глубоким сравнением двух финалистов:**

#### `markdown-it-py` vs `semantic-text-splitter` — почему `markdown-it-py`

| Критерий | markdown-it-py | semantic-text-splitter |
|---|---|---|
| **Архитектура** | AST-парсер: текст → токены с типами (`heading_open`, `fence`, `table_open`, `paragraph_open`) | Чёрный ящик: текст → список строк, без доступа к структуре |
| **Source mapping** | `token.map = [start_line, end_line]` — точное соответствие токенов исходному тексту | Нет — возвращает текст чанков без привязки к позициям |
| **Контроль** | Полный: можно итерировать токены и применять разную логику к разным типам элементов | Нет: нельзя обработать таблицу иначе чем параграф |
| **Совместимость с архитектурой** | Drop-in замена regex в `_split_by_headers` и `_split_paragraphs` (~50 строк regex → ~30 строк итерации по токенам) | Ломает архитектуру `RecursiveSplitter` + цепочку `BlockSplitter` — `semantic-text-splitter` сам решает как разбивать, нельзя встроить в нашу цепочку |
| **Таблицы** | `table_open`/`table_close` — точные границы, можно передать в наш `TableRowSplitter` | Не разбивает таблицы по строкам, не дублирует шапку (open issue #422) |
| **Code fences** | `fence` токен — корректный парсинг по CommonMark, никаких regex-багов | Корректный парсинг, но **разрезает большие code fences посередине** (нет контроля) |
| **Зрелость** | v4.0, 340M downloads/month, zero deps, pure Python | v0.29 (pre-1.0), 350K downloads/month, Rust + PyO3 |
| **Стабильность API** | Stable, CommonMark reference implementation | Pre-1.0, API может меняться |

**Ключевой аргумент**: `markdown-it-py` — это **парсер**, а не сплиттер. Он даёт AST, поверх
которого мы строим свои сплиттеры с нужным поведением. `semantic-text-splitter` — это **сплиттер**,
который забирает контроль над разбиением и не позволяет встроить наши стратегии
(TableRowSplitter, SemanticSplitter, цепочку BlockSplitter).

**Конкретный пример замены** (`_split_by_headers`):

Сейчас (regex, источник багов):
```python
_CODE_FENCE_RE = re.compile(r'^\s*```')
# + ручной трекинг in_fence с багами на конструкциях типа [Изображение: ```plantuml...```]
for line in text.splitlines():
    if _CODE_FENCE_RE.match(line):
        in_fence = not in_fence  # ← баг: непарный toggle
    if not in_fence and re.match(r'^#{1,6}\s', line):
        # split here
```

После (markdown-it-py, корректный CommonMark):
```python
from markdown_it import MarkdownIt
md = MarkdownIt()
tokens = md.parse(text)
for token in tokens:
    if token.type == 'heading_open':
        # token.map[0] — номер строки заголовка в исходном тексте
        # split here
```

Fence-баг невозможен в принципе — парсер корректно обрабатывает все edge cases по спецификации
CommonMark, включая вложенные конструкции, незакрытые fences и inline-backticks.

**Rust-ускоренный вариант**: `markdown-it-pyrs` (20x быстрее) доступен как drop-in замена
при необходимости оптимизации.

### 2. SemanticChunker — оставить кастомную реализацию

Кастомный `SemanticChunker` остаётся без замены на библиотечный аналог.

**Причины:**

- **Единый эмбеддер**: SemanticChunker использует тот же FRIDA эмбеддер, что и retrieval pipeline.
  Семантические границы определяются тем же пространством, в котором потом ищутся чанки.
  Библиотечные chunkers (Chonkie и др.) используют свои модели (sentence-transformers),
  что создаёт расхождение между чанкингом и поиском.

- **Контроль размера чанка**: `min_tokens` / `max_tokens` с гарантией через жадный алгоритм.
  Библиотечные semantic chunkers дают рекомендательные лимиты, наш — строгие границы
  (кроме неделимых блоков: code fences, широкие строки таблиц).

- **Статус**: алгоритм только начал тестироваться. Первый прогон (2026-03-13) показал 92% чанков
  в целевом диапазоне. Нужно больше данных прежде чем менять подход.

**Архитектура для замены алгоритма**: реализация должна поддерживать полную замену алгоритма
определения границ. Chonkie `SemanticChunker` использует Savitzky-Golay сглаживание кривой
cosine similarity — это качественнее чем raw cosine distance между соседними предложениями
(наш текущий подход). Если тестирование покажет проблемы с ложными разрывами, мы **перейдём
на Chonkie SemanticChunker целиком** (не заимствование идеи, а полная замена), при условии
что удастся сохранить использование FRIDA эмбеддера.

### 3. TableRowSplitter — оставить кастомную реализацию

Наш `TableRowSplitter` остаётся без замены на Chonkie `TableChunker`.

**Обоснование на основе [сравнительного анализа](../../reports/research/2026-03-13-chunking-libraries.md):**

| Критерий | Наш TableRowSplitter | Chonkie TableChunker |
|---|---|---|
| **Token counting** | Реальный подсчёт через `TokenCounter` (tiktoken) | По умолчанию tokenizer `"row"` — считает строки, не токены. Token-aware режим требует явной настройки |
| **Oversized rows** | Fallback: строка, превышающая лимит, передаётся дальше в цепочку `RecursiveSplitter` для дальнейшей разбивки | **Молча превышает лимит** — строка длиннее `chunk_size` попадает в чанк как есть, без предупреждения |
| **Контекст таблицы** | Обрабатывает `pre_text` / `post_text` вокруг таблицы — текст до и после таблицы не теряется | Принимает только саму таблицу, текст вокруг — ответственность вызывающего кода |
| **Многострочные заголовки** | Корректно обрабатывает заголовки из нескольких строк (GFM-спецификация) | Предполагает однострочный заголовок (`lines[0]` + `lines[1]`) |
| **HTML-таблицы** | Не поддерживает (не нужно — работаем с Markdown после конвертации) | Поддерживает `<table>` через BeautifulSoup (избыточная функциональность для нас) |
| **Интеграция** | Нативно встроен в цепочку `BlockSplitter` + `RecursiveSplitter` | Standalone, не composable |

**Вывод**: Chonkie `TableChunker` не предоставляет преимуществ для нашего use case. Наш
`TableRowSplitter` лучше интегрирован, корректнее обрабатывает edge cases (oversized rows,
многострочные заголовки) и использует реальный подсчёт токенов.

### 4. Что остаётся кастомным

Следующие компоненты не имеют аналогов в open-source и остаются as-is:

- `RecursiveSplitter` + цепочка `BlockSplitter` — композитная архитектура
- `pack_blocks` — жадная упаковка блоков до лимита токенов
- `LLMChunker` — чанкинг через LLM с halving и fallback
- `ContextGenerator` — LLM-суммари для каждого чанка
- `IndexingPipeline` — оркестрация всего пайплайна
- `FixedSizeSplitter` — рекурсивный fallback (абзацы → предложения → слова)

## Последствия

### Положительные

- Устранение класса багов, связанных с regex-парсингом Markdown (конкретный пример: баг
  с `[Изображение: ```plantuml...```]` → 18272-токенный чанк, см. [отчёт эксперимента](../../reports/2026-03-13-semantic-chunker-experiment.md))
- Корректная обработка edge cases по спецификации CommonMark (вложенные fences, GFM extensions,
  inline-backticks, незакрытые блоки)
- Снижение объёма кастомного кода: ~50 строк regex-парсинга → ~30 строк итерации по AST
- SemanticChunker продолжает использовать FRIDA эмбеддер — нет расхождения с retrieval
- TableRowSplitter сохраняет строгий контроль токенов и fallback для oversized строк
- Архитектура `RecursiveSplitter` + `BlockSplitter` не затрагивается — `markdown-it-py`
  заменяет только внутренности отдельных сплиттеров

### Отрицательные

- Новая зависимость: `markdown-it-py` (zero deps, pure Python, 340M downloads/month)
- Рефакторинг `_split_by_headers`, `_split_paragraphs`, удаление `_CODE_FENCE_RE` и `_fence_toggle`
- Потенциальные различия в поведении между regex и CommonMark-парсером (нужны тесты)

### Риски

- CommonMark-парсер может разбивать текст иначе чем regex — возможно изменение качества чанков.
  Митигация: A/B тестирование на существующем корпусе перед полным переключением.
- При переходе на Chonkie SemanticChunker в будущем — необходимо убедиться что Chonkie
  поддерживает кастомный эмбеддер (FRIDA через callback), иначе расхождение с retrieval pipeline.

## Альтернативы, которые были отвергнуты

### 1. `semantic-text-splitter` вместо `markdown-it-py`

Чёрный ящик без доступа к AST. Невозможно:
- встроить в цепочку `BlockSplitter` (сам решает как разбивать)
- обработать таблицы отдельно от параграфов
- предотвратить разрезание code fences посередине

Pre-1.0 (v0.29), 350K downloads/month vs 340M у markdown-it-py.
Не поддерживает table row splitting (issue #422).

### 2. Полная замена на Chonkie

Покрывает ~70% функциональности, но не предоставляет композитную цепочку сплиттеров,
`pack_blocks` и LLM-чанкинг. `SemanticChunker` использует свой эмбеддер, а не FRIDA.
`TableChunker` молча превышает лимит для oversized строк.

### 3. Полная замена на LangChain/LlamaIndex

Тяжёлые фреймворки, markdown-обработка не является их сильной стороной.
LangChain `SemanticChunker` не понимает markdown, regex English-centric.
LlamaIndex `MarkdownNodeParser` не разбивает большие таблицы.

### 4. Оставить всё as-is

Regex-парсинг будет продолжать генерировать edge case баги. Ad-hoc фиксы (`_fence_toggle`)
не масштабируются — каждый новый формат Confluence/Markdown может создать новый класс ошибок.

### 5. Замена TableRowSplitter на Chonkie TableChunker

Отвергнуто по результатам [сравнения](#3-tablerowsplitter--оставить-кастомную-реализацию):
дефолтный tokenizer считает строки а не токены, нет fallback для oversized строк, нет обработки
контекста вокруг таблицы, однострочный заголовок.

## Ссылки

- [Исследование библиотек](../../reports/research/2026-03-13-chunking-libraries.md)
- [Эксперимент SemanticChunker](../../reports/2026-03-13-semantic-chunker-experiment.md)
- [markdown-it-py](https://github.com/executablebooks/markdown-it-py) — выбранный парсер
- [Chonkie](https://github.com/chonkie-inc/chonkie) — потенциальная замена SemanticChunker
- [semantic-text-splitter](https://github.com/benbrandt/text-splitter) — отвергнутая альтернатива
