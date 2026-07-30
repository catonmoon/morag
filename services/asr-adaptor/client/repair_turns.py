"""Долечивание реплик, оставшихся без правки сущностей: офлайн, аудио не нужно.

Транзиентный отказ LLM (лаг сети, загрузка провайдера) не должен НАВСЕГДА оставлять реплику
сырой. Конвейер уже отбивается сам (два захода + добивочный проход), а этот скрипт — последний
слой: правка текстовая, поэтому её можно повторить над готовым `epN.json` в любой момент.

    python3 repair_turns.py ep20.json                 # реплики с меткой correction_failed
    python3 repair_turns.py ep20.json --at 2002.5     # адресно (артефакты до введения меток)
    python3 repair_turns.py ep20.json --dry-run       # показать, не трогая файл

Глоссарий и сводку выпуска берёт из артефакта; в старых артефактах их нет — пересчитает (центы).
Файл переписывается на месте, рядом остаётся `.bak`. Пословная разметка (`words`) правленых
реплик может разъехаться на несколько слов — при необходимости выпуск перевыравнивается адаптером.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # каталог asr-adaptor

from config import CFG                                          # noqa: E402
from stages.final_round import correct, doc_summary, recall_entities  # noqa: E402
from stages.glossary import build_glossary, relevant            # noqa: E402


def _context(turns, i: int, n: int) -> str:
    lo, hi = max(0, i - n), min(len(turns), i + n + 1)
    return '\n'.join(f"[{t.get('speaker') or '?'}] {t.get('raw', '')}"
                     for k, t in enumerate(turns[lo:hi], start=lo) if k != i)


def _rebuild_markdown(old_md: str, turns) -> str:
    """Реплики пересобираются из turns, шапка (front-matter) сохраняется как была."""
    head = []
    for line in old_md.splitlines():
        if line.startswith('['):
            break
        head.append(line)
    body = []
    for t in turns:
        body.append(f"[{t['speaker']}] <!-- t:{t['start']:.1f} --> {t['text']}")
        body.append('')
    return '\n'.join(head + body)


async def repair(path: Path, at: list[float], dry: bool) -> None:
    doc = json.loads(path.read_text(encoding='utf-8'))
    x = (doc.get('result') or doc).get('x_enriched') or doc
    turns = x['turns']

    targets = [i for i, t in enumerate(turns)
               if t.get('correction_failed') or any(abs(t['start'] - a) < 1.0 for a in at)]
    if not targets:
        print('лечить нечего: меток correction_failed нет, --at не совпал')
        return
    print(f'реплик к долечиванию: {len(targets)}')

    llm = CFG.build_llm()
    full = ' '.join(t.get('raw', '') for t in turns)
    dsum = x.get('doc_summary') or await doc_summary(full, llm)
    gloss = x.get('glossary') or await build_glossary(full, llm)

    changed = 0
    for i in targets:
        t = turns[i]
        raw = t.get('raw', '')
        recalled = await recall_entities(dsum, raw, llm)
        fixed = await correct(raw, dsum, _context(turns, i, CFG.context_turns),
                              relevant(raw, gloss), llm, CFG.always_terms, recalled)
        mark = '≠' if fixed != t['text'] else '='
        print(f"  {t['start']:>8.1f}с {mark} {'(dry-run)' if dry else ''}")
        if dry:
            continue
        if fixed != t['text']:
            x.setdefault('raw_sidecar', {})[f"{t['start']:.1f}"] = {'raw': raw, 'final': fixed}
            t['text'] = fixed
            changed += 1
        t.pop('correction_failed', None)

    if dry:
        return
    x['markdown'] = _rebuild_markdown(x.get('markdown', ''), turns)
    shutil.copy2(path, path.with_suffix(path.suffix + '.bak'))
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'готово: правок {changed}, метки сняты, бэкап рядом (.bak)')


def main() -> None:
    ap = argparse.ArgumentParser(description='долечивание реплик без правки сущностей')
    ap.add_argument('episode', help='epN.json')
    ap.add_argument('--at', type=float, nargs='*', default=[],
                    help='секунды начала реплик (для артефактов без меток)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    asyncio.run(repair(Path(args.episode), args.at, args.dry_run))


if __name__ == '__main__':
    main()
