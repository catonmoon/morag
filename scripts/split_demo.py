"""Демонстрация нарезки llm_overview.md на чанки до 512 токенов."""

from pathlib import Path

from morag.indexing.splitter import (
    FixedSizeSplitter,
    MarkdownHeaderSplitter,
    RecursiveSplitter,
    TableRowSplitter,
    pack_blocks,
)
from morag.indexing.token_counter import TiktokenCounter

LIMIT = 512
DATA_DIR = Path(__file__).parent.parent / 'data'


def main() -> None:
    text = (DATA_DIR / 'llm_overview.md').read_text(encoding='utf-8')
    counter = TiktokenCounter()

    splitter = RecursiveSplitter(
        counter=counter,
        limit=LIMIT,
        splitters=[
            MarkdownHeaderSplitter(),
            TableRowSplitter(),
            FixedSizeSplitter(counter=counter, limit=LIMIT),
        ],
    )

    blocks = splitter.split(text)
    # Жадная упаковка: склеиваем мелкие блоки до заполнения лимита
    chunks = ['\n\n'.join(pack) for pack in pack_blocks(blocks, counter, LIMIT)]

    total_tokens = counter.count(text)
    print(f'Исходный документ: {total_tokens} токенов')
    print(f'Чанков: {len(chunks)}, лимит: {LIMIT} токенов')
    print('=' * 72)

    for i, chunk in enumerate(chunks, 1):
        tokens = counter.count(chunk)
        print(f'\n── Чанк {i}/{len(chunks)} ({tokens} токенов) {"⚠ ПРЕВЫШЕНИЕ" if tokens > LIMIT else ""}')
        print('─' * 72)
        print(chunk)

    oversized = [c for c in chunks if counter.count(c) > LIMIT]
    if oversized:
        print(f'\n⚠  Превышений: {len(oversized)} из {len(chunks)}')
    else:
        print(f'\n✓ Все {len(chunks)} чанков в пределах {LIMIT} токенов')


if __name__ == '__main__':
    main()
