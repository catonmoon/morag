#!/usr/bin/env python3
"""Send questions from JSONL files to morag pipeline API and save answers."""

import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

LOGO = r"""
    ▄▀▀▀▀▀▀▀▀▄
   █  /\_/\   █      Catonmoon
   █ ( =^.^=) █      ╔╦╗ ╔═╗ ┬─┐ ┌─┐ ┌─┐
   █  /> < /  █      ║║║ ║ ║ ├┬┘ ├─┤ │ ┬
    ▀▄▄▄▄▄▄▄▄▀       ╩ ╩ ╚═╝ ┴└─ ┴ ┴ └─┘
                     Checker      v0.1.0
"""

API_URL = 'http://localhost:9099/v1/chat/completions'
API_KEY = ''
MODEL = 'morag'
TIMEOUT = 180


async def ask_question(
    client: httpx.AsyncClient,
    question_id: str,
    question: str,
    idx: int,
    total: int,
) -> dict:
    q_preview = question[:80] + ('...' if len(question) > 80 else '')
    print(f'  [{idx}/{total}] {question_id}: {q_preview}', flush=True)
    chunks = []
    async with client.stream(
        'POST',
        API_URL,
        json={
            'model': MODEL,
            'messages': [{'role': 'user', 'content': question}],
            'stream': True,
        },
        timeout=TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith('data: '):
                continue
            payload = line[6:]
            if payload == '[DONE]':
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if 'event' in obj:
                continue
            delta = obj.get('choices', [{}])[0].get('delta', {})
            content = delta.get('content', '')
            if content:
                chunks.append(content)

    answer = ''.join(chunks)
    answer = re.sub(r'<think>.*?</think>\s*', '', answer, flags=re.DOTALL)
    a_preview = answer[:80] + ('...' if len(answer) > 80 else '')
    print(f'  [{idx}/{total}] {question_id} done ({len(answer)} chars): {a_preview}', flush=True)
    return {'id': question_id, 'answer': answer}


async def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask_morag.py <input.jsonl> [output.jsonl]')
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_name(
        input_path.stem + '_answers.jsonl'
    )

    all_questions = []
    with open(input_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                all_questions.append(json.loads(line))

    # resume: skip already answered questions
    done_ids = set()
    if output_path.exists():
        with open(output_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)['id'])

    remaining = [q for q in all_questions if q['id'] not in done_ids]
    print(LOGO, flush=True)
    print(
        f'{len(all_questions)} total, {len(done_ids)} already done, '
        f'{len(remaining)} remaining -> {output_path}',
        flush=True,
    )

    errors = 0
    async with httpx.AsyncClient(
        headers={'Authorization': f'Bearer {API_KEY}'},
    ) as client:
        with open(output_path, 'a') as out:
            for i, q in enumerate(remaining, 1):
                try:
                    result = await ask_question(
                        client, q['id'], q['question'], i, len(remaining),
                    )
                    out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    out.flush()
                except Exception as e:
                    errors += 1
                    print(f'  [{q["id"]}] ERROR: {e}', flush=True)

    print(f'\nDone! {len(remaining) - errors} new answers, {errors} errors', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
