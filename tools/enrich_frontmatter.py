"""Обогащение front-matter аудио-транскриптов метаданными из источника (RSS + страницы выпусков).

Транскрайб-адаптер кладёт в .md минимальный front-matter (title, url). Этот скрипт добавляет:
дату/время публикации — из RSS `pubDate`; темы для `title` — из `<meta description>` страницы
выпуска (если есть); длительность — ffprobe по локальному аудио-кэшу; speakers — из
`x_enriched.speaker_names` сайдкара (имена, отфильтрованы безымянные Speaker_N).
Пишет синхронно в `epN.md` И в `x_enriched.markdown` внутри `epN.json`.

Env-контракт (генерик; доменные значения — в вашем обвязочном скрипте):
  MP3_URL_TEMPLATE        (обязателен) шаблон URL записи с {n} и {pfx} — идёт в front-matter `url`
  RSS_URL                 (опц.) RSS-фид; без него поле date не заполняется
  RSS_LINK_PATTERN        (опц., default r'-(\\d+)\\.html') regex по <link> item'а, группа 1 = номер
  EPISODE_PAGE_TEMPLATE   (опц.) шаблон URL страницы выпуска с {n} — для тем из meta description
  TITLE_TEMPLATE          (опц., default 'Episode {pfx}{n}') базовый title; темы дописываются через ': '
  TITLE_STRIP_PATTERN     (опц.) regex, вырезаемый из начала description (напр. само название шоу)
  SEASON                  (default 1); SEASON_PREFIX — как в transcribe_one.sh
  TRANSCRIPTS_DIR         (default ./transcripts/season{SEASON})
  MEDIA_CACHE_DIR         (default ./media_cache/season{SEASON})

  python tools/enrich_frontmatter.py 4 20        # выбранные записи
  python tools/enrich_frontmatter.py --dry 4     # превью без записи
"""
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

SEASON = os.environ.get('SEASON', '1')
_pfx_env = os.environ.get('SEASON_PREFIX')
PFX = _pfx_env if _pfx_env is not None else ('' if SEASON == '1' else f'{SEASON}-')

T = Path(os.environ.get('TRANSCRIPTS_DIR', f'./transcripts/season{SEASON}'))
MEDIA = Path(os.environ.get('MEDIA_CACHE_DIR', f'./media_cache/season{SEASON}'))

MP3_TMPL = os.environ.get('MP3_URL_TEMPLATE') or sys.exit('set MP3_URL_TEMPLATE (с {n}/{pfx})')
RSS = os.environ.get('RSS_URL', '')
RSS_LINK_RE = os.environ.get('RSS_LINK_PATTERN', r'-(\d+)\.html')
PAGE_TMPL = os.environ.get('EPISODE_PAGE_TEMPLATE', '')
TITLE_TMPL = os.environ.get('TITLE_TEMPLATE', 'Episode {pfx}{n}')
STRIP_RE = os.environ.get('TITLE_STRIP_PATTERN', '')

# эмодзи, ZWJ (U+200D), variation selectors (U+FE0F), стрелки/символы → разделитель
_JUNK = re.compile(r'[\U0001F000-\U0001FAFF←-⇿⌀-➿⬀-⯿‍️⁦-⁩]+')


def _fill(tmpl: str, n: int) -> str:
    return tmpl.replace('{n}', str(n)).replace('{pfx}', PFX)


def _get(url: str) -> str:
    return subprocess.run(['curl', '-sL', '--max-time', '25', url],
                          capture_output=True, text=True).stdout


def _topics(desc: str) -> str:
    s = _JUNK.sub(' · ', desc)
    if STRIP_RE:
        s = re.sub(STRIP_RE, '', s)
    s = re.sub(r'(\s*·\s*)+', ' · ', s)
    return re.sub(r'\s+', ' ', s).strip(' ·')


def _duration(n: int):
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                              '-of', 'default=noprint_wrappers=1:nokey=1', str(MEDIA / f'ep{n}.mp3')],
                             capture_output=True, text=True).stdout.strip()
        return int(float(out))
    except Exception:
        return None


def _frontmatter(n: int, title: str, date: str, dur, speakers) -> str:
    url = _fill(MP3_TMPL, n)
    # season/episode/speakers — доменные поля; generic-парсер front-matter (markdown.py)
    # кладёт их в payload как есть — retrieval-фильтры и каталог их видят.
    lines = ['---', f'title: "{title}"', f'url: {url}', f'season: {SEASON}', f'episode: {n}']
    if date:
        lines.append(f'date: "{date}"')
    if dur:
        lines.append(f'duration_sec: {dur}')
    if speakers:
        lines.append(f'speakers: {json.dumps(speakers, ensure_ascii=False)}')
    lines.append('---')
    return '\n'.join(lines)


def _splice(markdown: str, fm: str) -> str:
    """Заменить ведущий ---...--- блок (если есть) на новый, сохранив тело."""
    m = re.match(r'^---\n.*?\n---\n', markdown, re.S)
    body = markdown[m.end():] if m else markdown.lstrip('\n')
    return fm + '\n\n' + body.lstrip('\n')


def main(eps, dry):
    dates = {}
    if RSS:
        rss = _get(RSS)
        for it in ET.fromstring(rss).findall('.//item'):
            mm = re.search(RSS_LINK_RE, it.findtext('link') or '')
            if mm:
                dates[int(mm.group(1))] = parsedate_to_datetime(it.findtext('pubDate')).isoformat()

    for n in eps:
        jp = T / f'ep{n}.json'
        if not jp.exists():
            print(f'ep{n}: НЕТ json'); continue
        topics = ''
        if PAGE_TMPL:
            html = _get(_fill(PAGE_TMPL, n))
            d = re.search(r'<meta name="description" content="([^"]*)"', html)
            topics = _topics(d.group(1)) if d and d.group(1).strip() else ''
        title = _fill(TITLE_TMPL, n) + (f': {topics}' if topics else '')
        date = dates.get(n, '')
        dur = _duration(n)
        doc = json.load(open(jp))
        sn = doc['x_enriched'].get('speaker_names', {})
        speakers = [v for v in sn.values() if v and not str(v).startswith('Speaker_')]
        fm = _frontmatter(n, title, date, dur, speakers)
        new_md = _splice(doc['x_enriched']['markdown'], fm)
        print(f'ep{n}: {date[:16]} {dur}s  {len(speakers)}sp  «{title[:60]}»')
        if dry:
            continue
        doc['x_enriched']['markdown'] = new_md
        json.dump(doc, open(jp, 'w'), ensure_ascii=False, indent=1)
        (T / f'ep{n}.md').write_text(new_md)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    dry = '--dry' in sys.argv
    eps = [int(a) for a in args]
    if not eps:
        sys.exit('usage: enrich_frontmatter.py [--dry] <episode-numbers...>')
    main(eps, dry)
    print('=== DONE' + (' (dry)' if dry else ''))
