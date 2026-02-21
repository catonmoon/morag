"""Quick demo of LLMClient and LLMChunker with local Ollama."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from morag.indexing.chunker import LLMChunker
from morag.indexing.context import LLMContextGenerator
from morag.indexing.splitter import FixedSizeSplitter, MarkdownHeaderSplitter, RecursiveSplitter, TableRowSplitter, pack_blocks
from morag.indexing.token_counter import TiktokenCounter
from morag.llm.client import LLMClient

MODEL = 'qwen2.5-coder:7b'
BASE_URL = 'http://localhost:11434/v1'
BLOCK_LIMIT = 32000

client = LLMClient(base_url=BASE_URL, model=MODEL)
chunker = LLMChunker(client)
counter = TiktokenCounter()
splitter = RecursiveSplitter(
    counter, BLOCK_LIMIT,
    splitters=[
        MarkdownHeaderSplitter(),
        TableRowSplitter(),
        FixedSizeSplitter(counter, BLOCK_LIMIT),
    ],
)


async def demo_complete():
    print('=== complete() ===')
    result = await client.complete([
        {'role': 'system', 'content': 'You are a helpful assistant. Be brief.'},
        {'role': 'user', 'content': 'What is RAG in the context of LLMs? One sentence.'},
    ])
    print(result)


async def demo_chunker():
    print('\n=== LLMChunker on llm_overview.md ===')
    text = (Path(__file__).parent.parent / 'data' / 'llm_overview.md').read_text(encoding='utf-8')
    print(f'File: {len(text)} chars, {counter.count(text)} tokens')

    blocks = splitter.split(text)
    packs = pack_blocks(blocks, counter, BLOCK_LIMIT)
    print(f'Pre-split: {len(blocks)} block(s) → {len(packs)} pack(s)\n')

    all_chunks = []
    for i, pack in enumerate(packs):
        block_text = '\n\n'.join(pack)
        print(f'--- Pack {i} ({counter.count(block_text)} tokens) ---')
        chunks = await chunker.chunk(block_text)
        for j, chunk in enumerate(chunks):
            preview = chunk[:80].replace('\n', ' ')
            print(f'  chunk[{j}]: {preview!r}')
        all_chunks.extend(chunks)

    print(f'\nTotal: {len(all_chunks)} chunk(s)')


async def demo_context():
    print('\n=== LLMContextGenerator on first chunk ===')
    text = (Path(__file__).parent.parent / 'data' / 'llm_overview.md').read_text(encoding='utf-8')
    chunks = await chunker.chunk(text[:3000])  # берём начало документа
    first_chunk = chunks[0] if chunks else text[:500]

    generator = LLMContextGenerator(client)
    context = await generator.generate(doc_text=text, chunk_text=first_chunk)
    print(f'Chunk: {first_chunk[:100]!r}')
    print(f'Context: {context}')


async def demo_full_pipeline():
    print('\n=== Full pipeline: LLMChunker + LLMContextGenerator ===')
    text = (Path(__file__).parent.parent / 'data' / 'llm_overview.md').read_text(encoding='utf-8')
    print(f'Document: {len(text)} chars, {counter.count(text)} tokens\n')

    blocks = splitter.split(text)
    packs = pack_blocks(blocks, counter, BLOCK_LIMIT)

    all_chunks = []
    for pack in packs:
        block_text = '\n\n'.join(pack)
        all_chunks.extend(await chunker.chunk(block_text))

    print(f'Chunks produced: {len(all_chunks)}\n')

    generator = LLMContextGenerator(client)
    for i, chunk_text in enumerate(all_chunks):
        context = await generator.generate(doc_text=text, chunk_text=chunk_text)
        print(f'--- Chunk {i} ---')
        print(f'  text:    {chunk_text[:120].replace(chr(10), " ")!r}')
        print(f'  context: {context!r}')
        print()


async def main():
    await demo_complete()
    await demo_chunker()
    await demo_context()
    await demo_full_pipeline()


asyncio.run(main())
