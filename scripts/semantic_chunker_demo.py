"""Демонстрация SemanticChunker на файлах из examples/.

Использует FRIDA для батчевого эмбеддинга.
Модель загружается один раз, затем применяется SemanticChunker.

Использование:
    python scripts/semantic_chunker_demo.py                           # llm_overview.md
    python scripts/semantic_chunker_demo.py examples/claude_guide.md  # конкретный файл
    python scripts/semantic_chunker_demo.py --min 50 --max 250        # свои лимиты
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from morag.indexing.chunker import SemanticChunker
from morag.indexing.token_counter import TiktokenCounter

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'
DEFAULT_FILE = EXAMPLES_DIR / 'llm_overview.md'


def load_frida():
    """Загружает FRIDA и возвращает батчевый embed_fn."""
    from sentence_transformers import SentenceTransformer

    print('Loading FRIDA model...')
    t0 = time.time()
    model = SentenceTransformer('ai-forever/FRIDA')
    print(f'Model loaded in {time.time() - t0:.1f}s (dim={model.get_sentence_embedding_dimension()})')

    prefix = 'search_document: '

    def batch_embed(texts: list[str]) -> list[list[float]]:
        prefixed = [prefix + t for t in texts]
        embeddings = model.encode(prefixed, normalize_embeddings=False)
        return embeddings.tolist()

    return batch_embed


async def run(file_path: Path, min_tokens: int, max_tokens: int) -> None:
    text = file_path.read_text(encoding='utf-8')
    counter = TiktokenCounter()
    total_tokens = counter.count(text)

    print(f'File: {file_path.name}')
    print(f'Size: {len(text)} chars, {total_tokens} tokens')
    print(f'Chunk limits: [{min_tokens}, {max_tokens}] tokens')
    print()

    embed_fn = load_frida()

    chunker = SemanticChunker(
        embed_fn=embed_fn,
        counter=counter,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )

    # Semantic chunking — весь текст одним блоком
    # (иерархическая нарезка внутри SemanticChunker: заголовки → абзацы → предложения)
    print('Chunking...')
    t0 = time.time()
    chunks = await chunker.chunk(text)
    elapsed = time.time() - t0

    print(f'Done in {elapsed:.2f}s: {len(chunks)} chunks')
    print('=' * 72)

    for i, chunk in enumerate(chunks, 1):
        tokens = counter.count(chunk)
        print(f'\n-- Chunk {i}/{len(chunks)} ({tokens} tokens) ', end='')
        if tokens > max_tokens:
            print('[OVER LIMIT]')
        elif tokens < min_tokens:
            print('[UNDER MIN]')
        else:
            print()
        print('-' * 72)
        print(chunk)

    # Summary
    chunk_sizes = [counter.count(c) for c in chunks]
    over = sum(1 for s in chunk_sizes if s > max_tokens)
    under = sum(1 for s in chunk_sizes[:-1] if s < min_tokens)  # last chunk can be short
    avg = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

    print('\n' + '=' * 72)
    print(f'Chunks: {len(chunks)}, avg: {avg:.0f} tokens')
    print(f'Min: {min(chunk_sizes)} tokens, max: {max(chunk_sizes)} tokens')
    if over:
        print(f'Over limit: {over}')
    if under:
        print(f'Under min (excl. last): {under}')
    if not over and not under:
        print('All chunks within limits')


def main() -> None:
    parser = argparse.ArgumentParser(description='SemanticChunker demo')
    parser.add_argument('file', nargs='?', default=str(DEFAULT_FILE), help='Path to markdown file')
    parser.add_argument('--min', type=int, default=50, help='Min chunk size in tokens')
    parser.add_argument('--max', type=int, default=250, help='Max chunk size in tokens')
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f'File not found: {file_path}', file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(file_path, args.min, args.max))


if __name__ == '__main__':
    main()
