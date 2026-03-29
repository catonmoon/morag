# ADR-0009: Два токенизатора — FRIDA для чанкинга, TikToken для LLM

## Статус

Принято (2026-03-29)

## Контекст

Система использовала один токенизатор (TikToken cl100k_base) для всех подсчётов:
чанкинг, context generation, doc_summary, doc_title. Но TikToken — токенизатор
OpenAI GPT-4, а FRIDA embedder использует свой HuggingFace токенизатор.

Исследование на 35491 чанке (`experiments/tokenizer_comparison_20260329.md`) показало:

- **Русский текст**: TikToken считает на **43% больше** токенов чем FRIDA (ratio 1.43)
- **Английский текст**: разница минимальна (ratio 0.97)
- `max_tokens: 256` (TikToken) ≈ 179 токенов FRIDA
- FRIDA embedder заполнялся на ~50% — половина capacity не использовалась

## Решение

Два `TokenCounter` в pipeline:

### `embed_counter` — HuggingFaceTokenCounter (ai-forever/FRIDA)

Используется для:
- `HybridChunker` — min_tokens, max_tokens (размер чанков)
- `SemanticChunker` — то же
- `LLMContextGenerator._adaptive_max_tokens` — бюджет embedder
- Точный подсчёт path_tokens (вместо константы `_PATH_OVERHEAD = 20`)

### `llm_counter` — TiktokenCounter (cl100k_base)

Используется для:
- `LLMContextGenerator` — context_window, prompt overhead, window_tokens
- `DocTitleProcessor` — context_window, scan_tokens
- `DocSummaryProcessor` — context_window
- `IndexingPipeline` — логирование размера чанков

### Адаптивный context с точным бюджетом

`LLMContextGenerator.generate()` теперь принимает `path: list[str]` и считает
path_tokens через `embed_counter`:

```python
embed_chunk_tokens = self._embed_counter.count(chunk_text)
path_tokens = self._embed_counter.count('\n'.join(path)) + 1
max_tokens = chunk_max_tokens - embed_chunk_tokens - path_tokens
```

Вместо фиксированного `_PATH_OVERHEAD = 20` — точный подсчёт реального path.

## Реализация

### `HuggingFaceTokenCounter` (token_counter.py)

```python
class HuggingFaceTokenCounter(TokenCounter):
    def __init__(self, model_name: str = 'ai-forever/FRIDA') -> None:
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
```

- Загружает только токенизатор (~1MB), не модель
- `add_special_tokens=False` — считаем только текстовые токены
- 3x медленнее TikToken, но на 35000 чанков это +8s — пренебрежимо

### cli/main.py

```python
llm_counter = TiktokenCounter()
embed_counter = HuggingFaceTokenCounter(config.indexing.dense_embedder.model)
```

`embed_counter` передаётся в HybridChunker, SemanticChunker, LLMContextGenerator.
`llm_counter` — в DocTitleProcessor, DocSummaryProcessor, IndexingPipeline.

## Последствия

- `max_tokens: 256` в конфиге теперь означает 256 **реальных токенов FRIDA**
- Для русского текста это ~370 TikToken — чанки стали больше
- FRIDA embedder заполняется точнее — меньше потерь capacity
- Адаптивный context точнее — path_tokens считается от реального path
- При смене embedder модели достаточно изменить `dense_embedder.model` в конфиге

## Альтернативы

1. **Коэффициент поправки** — `real_tokens ≈ tiktoken * 0.7` для русского.
   Отвергнуто: неточно, зависит от пропорции русского/английского текста.

2. **Один FRIDA токенизатор для всего** — использовать и для LLM context window.
   Отвергнуто: LLM модели (GPT, Grok, Qwen) используют свои токенизаторы,
   TikToken ближе к ним чем FRIDA.
