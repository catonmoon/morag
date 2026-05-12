"""Прогон eval через Pipeline.pipe() напрямую (без OWUI HTTP).

Запускать ВНУТРИ pipelines-контейнера — там есть morag-пакет, /app/conf и доступ
к qdrant/embedder через docker network. Идемпотентный: пропускает уже отвеченные.

Формат JSON (без зависимостей openpyxl/pandas в контейнере):
  input:  [{'num': 1, 'section': 'Люди', 'question': '...', 'gold': '...'}, ...]
  output: [{...input fields..., 'answer': '...', 'elapsed_sec': N, 'docs': N,
           'tool_calls': N, 'error': '...', 'timestamp': '...'}, ...]

Использование (внутри контейнера):
    python eval_native.py /tmp/eval-in.json /tmp/eval-out.json

Output обновляется ПОСЛЕ каждого вопроса (можно мониторить).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Pipeline-файл лежит в /app/pipelines/ внутри контейнера. Импортируем оттуда.
sys.path.insert(0, '/app/pipelines')
from morag_pipeline import Pipeline  # type: ignore  # noqa: E402


def collect_pipe_output(p: Pipeline, question: str) -> tuple[str, dict, str | None]:
    """Запустить pipe() и собрать (answer_text, stats, error_msg)."""
    text_parts: list[str] = []
    stats = {
        'docs': 0,
        'tool_calls': 0,
        'iters': 0,
    }
    try:
        gen = p.pipe(
            user_message=question,
            model_id='morag_pipeline',
            messages=[{'role': 'user', 'content': question}],
            body={},
        )
        for chunk in gen:
            if isinstance(chunk, str):
                # Strip OWUI <think>...</think> блоки
                text_parts.append(chunk)
            elif isinstance(chunk, dict):
                ev = chunk.get('event', {})
                if ev.get('type') == 'status':
                    desc = (ev.get('data') or {}).get('description') or ''
                    # «✅ Найдено N документов(а) за M шагов» — финальная статистика
                    if desc.startswith('✅'):
                        import re
                        m = re.search(r'(\d+)\s+документ', desc)
                        if m:
                            stats['docs'] = int(m.group(1))
                        m2 = re.search(r'за\s+(\d+)', desc)
                        if m2:
                            stats['tool_calls'] = int(m2.group(1))
    except Exception as exc:
        text = ''.join(text_parts)
        return text, stats, f'{type(exc).__name__}: {exc}'

    text = ''.join(text_parts)
    # Снять <think>...</think> блоки если они есть
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text, stats, None


def main():
    if len(sys.argv) < 3:
        print('Usage: python eval_native.py <input.json> <output.json>')
        sys.exit(2)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    print(f'Loading questions from {in_path}...')
    with open(in_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f'  → {len(questions)} questions')

    # Resume: загружаем уже сделанные, дальше пишем поверх (по номеру)
    answered_by_num: dict = {}
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                if item.get('answer'):
                    answered_by_num[item['num']] = item
    print(f'Resume: {len(answered_by_num)} already done')

    print('Constructing Pipeline (reads /app/conf/config.yml)...')
    t0 = time.time()
    p = Pipeline()
    print(f'  → Pipeline ready in {time.time() - t0:.1f}s')

    todo = [q for q in questions if q['num'] not in answered_by_num]
    total = len(todo)
    print(f'To process: {total}')
    print('=' * 70)

    results: list[dict] = list(answered_by_num.values())
    started_at = time.time()

    for i, q in enumerate(todo, 1):
        t_q = time.time()
        question = q['question']
        print(f'[{i}/{total}] Q{q["num"]}: {question[:70]}', flush=True)

        answer, stats, error = collect_pipe_output(p, question)
        elapsed = time.time() - t_q

        result = {
            **q,
            'answer': answer,
            'elapsed_sec': round(elapsed, 1),
            'docs': stats['docs'],
            'tool_calls': stats['tool_calls'],
            'iters': stats['iters'],
            'error': error or '',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        results.append(result)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)

        status = '✗' if error else '✓'
        avg = (time.time() - started_at) / i
        eta = avg * (total - i)
        print(
            f'  {status} {elapsed:.1f}s, ans={len(answer)} chars, '
            f'docs={stats["docs"]}, tool_calls={stats["tool_calls"]}, '
            f'avg={avg:.1f}s/q, ETA={eta/60:.0f}m',
            flush=True,
        )
        if error:
            print(f'  ERROR: {error}', flush=True)

    print('=' * 70)
    print(f'Done. Total elapsed: {(time.time() - started_at)/60:.1f}m')


if __name__ == '__main__':
    main()