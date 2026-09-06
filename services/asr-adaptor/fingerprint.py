"""Отпечаток установки: чем и на чём сделана эта расшифровка.

Кладётся в артефакт (`x_enriched.env`), а не печатается командой, и это главное решение модуля.
Отдельная команда — это дисциплина: кто-то должен вспомнить её запустить и сравнить. Отпечаток
внутри артефакта делает КАЖДУЮ расшифровку самодостаточной: через год видно, каким движком и с
какими версиями она получена, без чьей-либо памяти. Тот же приём, что провенанс у метаданных.

Зачем понадобилось: долгие прогоны уезжают на вторую машину, а разработка остаётся на первой. Разъезд
установок замечают не при установке, а через месяц по испорченному корпусу — и тогда уже не отличить
«конвейер стал хуже» от «прогон был на другой машине». Здесь дешёвый ответ на этот вопрос.

⚠️ Что отпечаток ЛОВИТ: другой коммит движка, незакоммиченные правки, другую версию частотника или
трансформеров, другую модель LLM, другой эндпоинт, другой питон, другую машину.
⚠️ Чего НЕ ловит: подменённый файл модели под тем же именем. Модели держат бэкенды, адаптер их не
видит, а хэшировать гигабайты на каждый старт незачем. Честнее назвать границу, чем делать вид.

⚠️ Секретов здесь нет и быть не должно: артефакт уезжает вместе с корпусом. Адрес эндпоинта не
пишем — только его короткий хэш: сравнить машины он позволяет, а прочитать адрес по нему нельзя.
"""
from __future__ import annotations

import hashlib
import logging
import platform
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Пакеты, от которых поведение зависит ПО СУЩЕСТВУ, а не только по API: `wordfreq` — таблица
# частот гейта редкости, `transformers` — токенайзер бюджета подсказки, `tiktoken` — счёт токенов,
# `torch` — выравнивание. Остальное в отпечатке шумело бы.
_PACKAGES = ('wordfreq', 'transformers', 'tiktoken', 'torch')


def _git(*args: str) -> str:
    here = Path(__file__).resolve().parent
    res = subprocess.run(['git', '-C', str(here), *args], capture_output=True, text=True, timeout=5)
    return res.stdout.strip() if res.returncode == 0 else ''


@lru_cache(maxsize=1)
def stack_fingerprint() -> dict:
    """Отпечаток процесса. Считается ОДИН раз: значения за жизнь процесса не меняются."""
    try:
        out: dict = {'host': platform.node().split('.')[0],
                     'python': platform.python_version(),
                     'platform': f'{platform.system()}-{platform.machine()}'}

        commit = _git('rev-parse', '--short', 'HEAD')
        if commit:
            # ⚠️ `+dirty` — не косметика: именно так выглядит «на одной машине прогнали
            # закоммиченный код, на другой — рабочее дерево с правками». Молча это не отличить.
            out['morag'] = commit + ('+dirty' if _git('status', '--porcelain') else '')

        pkgs = {}
        for name in _PACKAGES:
            try:
                pkgs[name] = version(name)
            except PackageNotFoundError:
                continue
        out['packages'] = pkgs

        from config import CFG
        out['llm'] = CFG.llm_model
        if CFG.llm_base_url:
            out['llm_endpoint'] = hashlib.sha256(CFG.llm_base_url.encode()).hexdigest()[:8]
        out['asr_model'] = CFG.asr_model
        return out
    except Exception as e:  # отпечаток не имеет права ронять расшифровку
        logging.getLogger('asr').warning('отпечаток установки не собран: %s: %s',
                                         type(e).__name__, e)
        return {'error': f'{type(e).__name__}'}


def one_line(env: dict) -> str:
    """Отпечаток одной строкой — для логов и приёмки партии."""
    if not env or 'error' in env:
        return '—'
    pkgs = ' '.join(f'{k}{v}' for k, v in sorted((env.get('packages') or {}).items()))
    return ' · '.join(x for x in (env.get('host'), env.get('morag'),
                                  f"py{env.get('python', '?')}", pkgs, env.get('llm')) if x)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    fp = stack_fingerprint()
    print(one_line(fp))
    for k, v in fp.items():
        print(f'  {k}: {v}')
