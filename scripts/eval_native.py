"""Прогон eval через Pipeline.pipe() напрямую (без OWUI HTTP).

Запускать ВНУТРИ pipelines-контейнера — там есть morag-пакет, /app/conf и доступ
к qdrant/embedder через docker network. Идемпотентный: пропускает уже отвеченные.

Формат:
  input  (JSONL):  {"id": "1", "question": "..."}
                   {"id": "2", "question": "..."}
                   ...
  output (JSON):   [{"id": "1", "question": "...", "answer": "...",
                     "docs": [{"name": "...", "url": "..."}, ...],
                     "elapsed_sec": N, "tool_calls": N, "error": "...",
                     "timestamp": "..."}, ...]

Использование (внутри контейнера):
    python eval_native.py /tmp/questions.jsonl /tmp/answers.json

Output обновляется ПОСЛЕ каждого вопроса (можно мониторить).
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Pipeline-файл лежит в /app/pipelines/ внутри контейнера. Импортируем оттуда.
sys.path.insert(0, '/app/pipelines')
from morag_pipeline import Pipeline  # type: ignore  # noqa: E402


def collect_pipe_output(
    p: Pipeline, question: str,
) -> tuple[str, list[dict], dict, str | None]:
    """Запустить pipe() и собрать (answer_text, docs, stats, error_msg).

    docs: список {name, url, doc_id} в порядке прихода citation-event'ов.
    stats: {tool_calls, docs} — из финального status-event'а.
    """
    text_parts: list[str] = []
    docs: list[dict] = []
    seen_doc_ids: set[str] = set()
    stats = {'docs': 0, 'tool_calls': 0}

    try:
        gen = p.pipe(
            user_message=question,
            model_id='morag_pipeline',
            messages=[{'role': 'user', 'content': question}],
            body={},
        )
        for chunk in gen:
            if isinstance(chunk, str):
                text_parts.append(chunk)
            elif isinstance(chunk, dict):
                ev = chunk.get('event', {})
                ev_type = ev.get('type')
                data = ev.get('data') or {}

                if ev_type == 'citation':
                    src = data.get('source') or {}
                    meta_list = data.get('metadata') or []
                    meta = meta_list[0] if meta_list else {}
                    doc_id = meta.get('source') or src.get('name') or ''
                    if doc_id and doc_id in seen_doc_ids:
                        continue
                    seen_doc_ids.add(doc_id)
                    docs.append({
                        'name': src.get('name') or meta.get('name') or '',
                        'url': src.get('url') or meta.get('url') or '',
                        'doc_id': doc_id,
                    })

                elif ev_type == 'status':
                    desc = data.get('description') or ''
                    if desc.startswith('✅'):
                        m = re.search(r'(\d+)\s+документ', desc)
                        if m:
                            stats['docs'] = int(m.group(1))
                        m2 = re.search(r'за\s+(\d+)', desc)
                        if m2:
                            stats['tool_calls'] = int(m2.group(1))
    except Exception as exc:
        text = ''.join(text_parts)
        return text, docs, stats, f'{type(exc).__name__}: {exc}'

    text = ''.join(text_parts)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text, docs, stats, None


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    if len(sys.argv) < 3:
        print('Usage: python eval_native.py <input.jsonl> <output.json>')
        sys.exit(2)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    print(f'Loading questions from {in_path}...')
    questions = load_jsonl(in_path)
    print(f'  → {len(questions)} questions')

    # Resume: по id, пропускаем уже отвеченные
    answered_by_id: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                if item.get('answer'):
                    answered_by_id[str(item['id'])] = item
    print(f'Resume: {len(answered_by_id)} already done')

    print('Constructing Pipeline (reads /app/conf/config.yml)...')
    t0 = time.time()
    p = Pipeline()
    print(f'  → Pipeline ready in {time.time() - t0:.1f}s')

    todo = [q for q in questions if str(q['id']) not in answered_by_id]
    total = len(todo)
    print(f'To process: {total}')
    print('=' * 70)

    results: list[dict] = list(answered_by_id.values())
    started_at = time.time()

    for i, q in enumerate(todo, 1):
        t_q = time.time()
        question = q['question']
        qid = str(q['id'])
        print(f'[{i}/{total}] Q{qid}: {question[:70]}', flush=True)

        answer, docs, stats, error = collect_pipe_output(p, question)
        elapsed = time.time() - t_q

        result = {
            'id': qid,
            'question': question,
            'answer': answer,
            'docs': docs,
            'elapsed_sec': round(elapsed, 1),
            'tool_calls': stats['tool_calls'],
            'docs_count': stats['docs'] or len(docs),
            'error': error or '',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        results.append(result)
        results.sort(key=lambda r: (len(r['id']), r['id']))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)

        status = '✗' if error else '✓'
        avg = (time.time() - started_at) / i
        eta = avg * (total - i)
        print(
            f'  {status} {elapsed:.1f}s, ans={len(answer)} chars, '
            f'docs={len(docs)}, tool_calls={stats["tool_calls"]}, '
            f'avg={avg:.1f}s/q, ETA={eta / 60:.0f}m',
            flush=True,
        )
        if error:
            print(f'  ERROR: {error}', flush=True)

    print('=' * 70)
    print(f'Done. Total elapsed: {(time.time() - started_at) / 60:.1f}m')


if __name__ == '__main__':
    main()
