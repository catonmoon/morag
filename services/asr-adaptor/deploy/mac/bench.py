#!/usr/bin/env python3
"""Стенд скорости конвейера: записи → адаптер → таблица стадий, очередь ресурсов, память машины.

Чем отличается от `bench.sh`: тот меряет ОДНУ запись и печатает зашитый список из семи стадий;
этот берёт ВСЕ `*_s` из `timing` (новые стадии видны без правки стенда), отдельно — сколько
выпуск ЖДАЛ ресурс и сколько ДЕРЖАЛ его (`wait_*_s` / `busy_*_s`, прибор `pipeline._res`),
счётчики whisper-бэкенда (загрузки весов, секунды инференса — `GET /stats`) и пик занятой
памяти машины по `vm_stat` (у MLX RSS процесса врёт: веса в Metal-буферах).

Пачка: `--parallel` отправляет все записи сразу (очередь решает `ASR_MAX_JOBS` адаптера), без
него — по одной. Итог — таблица в консоль и JSON (`--out`), им сравнивают прогоны и машины.

    python3 bench.py a.mp3 b.mp3 --parallel --out run.json

⚠️ Реестр голосов прогон МЕНЯЕТ (провенанс пишется всегда) — стенд гонять на стеке со своим
черновым `ASR_REGISTRY_PATH`, не на рабочем.
⚠️ Только стандартная библиотека и `curl`: стенд запускается системным питоном рядом со стеком.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ASR = os.environ.get('ASR_BASE', 'http://127.0.0.1:8082')
WHISPER = os.environ.get('WHISPER_BASE', 'http://127.0.0.1:8123')


def get(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:      # noqa: S310 — локальный стек
        return json.loads(r.read())


def whisper_stats() -> dict:
    try:
        return get(f'{WHISPER}/stats')
    except Exception:  # noqa: BLE001 — старый бэкенд без /stats: просто нет счётчиков
        return {}


class Memory(threading.Thread):
    """Занятая память машины раз в `step` секунд: active + wired + сжатая (как «Использовано»
    в Мониторе системы, без кэша файлов). Пик и ряд — в отчёт."""

    def __init__(self, step: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.step, self.stop, self.series = step, threading.Event(), []

    @staticmethod
    def used_gb() -> float:
        out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
        page = int(re.search(r'page size of (\d+)', out).group(1))
        n = {k.strip(): int(v.strip().rstrip('.')) for k, v in re.findall(r'^([^:]+):\s+(\d+)\.', out, re.M)}
        pages = (n.get('Pages active', 0) + n.get('Pages wired down', 0)
                 + n.get('Pages occupied by compressor', 0))
        return round(pages * page / 2**30, 2)

    def run(self) -> None:
        t0 = time.monotonic()
        while not self.stop.is_set():
            self.series.append((round(time.monotonic() - t0, 1), self.used_gb()))
            self.stop.wait(self.step)


def submit(path: Path, episode: str) -> str:
    out = subprocess.run(['curl', '-sS', '--noproxy', '*', '-F', f'file=@{path}', '-F', f'episode={episode}',
                          '-F', f'title={path.stem}', '-F', 'events=1', f'{ASR}/v1/audio/transcriptions'],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)['job_id']


def wait(job: str, poll: float = 5.0) -> dict:
    while True:
        j = get(f'{ASR}/v1/jobs/{job}', timeout=60)
        if j['status'] in ('done', 'error'):
            return j
        time.sleep(poll)


def duration(path: Path) -> float:
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def report(runs: list[dict], wall: float, mem: Memory, w0: dict, w1: dict) -> dict:
    audio = sum(r['audio_s'] for r in runs)
    print(f'\nзаписей {len(runs)}, звука {audio / 60:.1f} мин, стена {wall:.0f} с '
          f'({audio / wall:.2f}× реального времени), пик памяти {max((m for _, m in mem.series), default=0):.1f} ГБ')
    keys: list[str] = []
    for r in runs:
        keys += [k for k in r['timing'] if k.endswith('_s') and k not in keys]
    stage = [k for k in keys if not k.startswith(('wait_', 'busy_'))]
    res = sorted({k.split('_', 1)[1] for k in keys if k.startswith(('wait_', 'busy_'))})
    head = ['стадия'] + [r['name'][:18] for r in runs]
    print('\n' + ' | '.join(f'{h:>18}' for h in head))
    for k in stage:
        print(' | '.join([f'{k:>18}'] + [f"{r['timing'].get(k, ''):>18}" for r in runs]))
    for k in res:
        print(' | '.join([f'{"ресурс " + k[:-2]:>18}']
                         + [f"{r['timing'].get('wait_' + k, 0):>8} / {r['timing'].get('busy_' + k, 0):<7}" for r in runs]))
    whisper = {k: round(w1.get(k, 0) - w0.get(k, 0), 2) for k in ('requests', 'loads', 'load_s', 'infer_s', 'audio_s')
               if isinstance(w1.get(k), (int, float))}
    if whisper:
        print(f'\nwhisper за прогон: {whisper}')
    return {'machine': {'chip': platform.processor() or platform.machine(),
                        'ram_gb': round(int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True,
                                                           text=True).stdout or 0) / 2**30)},
            'wall_s': round(wall, 1), 'audio_s': round(audio, 1), 'runs': runs, 'whisper': whisper,
            'memory': {'peak_gb': max((m for _, m in mem.series), default=0), 'series': mem.series}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('files', nargs='+', type=Path)
    ap.add_argument('--parallel', action='store_true', help='отправить все записи сразу')
    ap.add_argument('--out', type=Path, help='JSON с отчётом')
    ap.add_argument('--tag', default='bench', help='префикс episode (провенанс реестра)')
    a = ap.parse_args()
    get(f'{ASR}/health', timeout=30)
    mem, w0 = Memory(), whisper_stats()
    mem.start()
    t0 = time.monotonic()
    runs: list[dict] = []

    def one(path: Path, job: str | None = None) -> dict:
        job = job or submit(path, f'{a.tag}-{path.stem}')
        j = wait(job)
        if j['status'] == 'error':
            print(f'✗ {path.name}: {j.get("error")}', file=sys.stderr)
            return {'name': path.stem, 'audio_s': duration(path), 'error': j.get('error'), 'timing': {}}
        tm = j['result']['x_enriched']['timing']
        print(f'✓ {path.name}: {tm.get("total_s")} с')
        return {'name': path.stem, 'audio_s': duration(path), 'timing': tm,
                'nodes': (j['result']['x_enriched'].get('conveyor') or {}).get('nodes')}

    if a.parallel:
        jobs = [(p, submit(p, f'{a.tag}-{p.stem}')) for p in a.files]
        runs = [one(p, j) for p, j in jobs]
    else:
        runs = [one(p) for p in a.files]
    wall = time.monotonic() - t0
    mem.stop.set()
    mem.join()
    out = report(runs, wall, mem, w0, whisper_stats())
    if a.out:
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'\nотчёт: {a.out}')
    return 0 if all('error' not in r for r in runs) else 1


if __name__ == '__main__':
    sys.exit(main())
