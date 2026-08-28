"""Караоке-страница из результата asr-adaptor: читать выпуск глазами и ухом.

Формат тот же, что у проверочной страницы выравнивания в продуктовой обвязке
(`media_cache/karaoke.html`): слово подсвечивается ровно когда звучит, клик по слову — перемотка.
Отличия: данные берутся из `x_enriched.words` (morag-words-v1) свежего прогона, аудио — по URL
(качать не надо), и в шапке рядом с караоке — сводка покрытия и правок, чтобы читать прицельно.

    python3 make_karaoke.py ep17.json --out-dir .

Правленые слова подсвечены — по ним и читают: метрики подмену сущности не ловят (класс «H200→H100»
нашло именно чтение), а клик по слову перематывает звук ровно туда, где его можно проверить ухом.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path

# URL-шаблон записи: {slug} — идентификатор выпуска из имени файла (ep2-28 → 2-28).
MP3 = os.environ.get('KARAOKE_MP3_TPL', 'https://example.org/audio/{slug}.mp3')

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1620; --panel:#132132; --ink:#dfe8f2; --dim:#7d93aa;
           --brass:#e4a04b; --sonar:#3fbfad; --fix:#7ec7ff; }}
  * {{ box-sizing:border-box }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
  header {{ position:sticky; top:0; z-index:2;
           background:linear-gradient(180deg,var(--bg) 82%,transparent);
           padding:14px 20px 18px; border-bottom:1px solid #23374d }}
  h1 {{ margin:0 0 10px; font-size:15px; font-weight:600; color:var(--dim); letter-spacing:.02em }}
  audio {{ width:100%; height:36px }}
  .hint {{ margin-top:8px; font-size:13px; color:var(--dim) }}
  .hint b {{ color:var(--brass); font-weight:600 }}
  main {{ max-width:820px; margin:0 auto; padding:22px 20px 60vh }}
  .turn {{ margin:0 0 26px }}
  .who {{ font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.08em;
         text-transform:uppercase; color:var(--sonar); margin-bottom:6px }}
  .who span {{ color:var(--dim); margin-left:8px; font-weight:400 }}
  w {{ cursor:pointer; border-radius:3px; padding:1px 0; transition:color .12s }}
  w:hover {{ background:#1e3348 }}
  w.done {{ color:var(--dim) }}
  w.on {{ color:#fff; background:var(--brass); box-shadow:0 0 0 3px rgba(228,160,75,.25);
         border-radius:3px }}
  w.fix {{ color:var(--fix); border-bottom:1px dotted var(--fix) }}
</style></head><body>
<header>
  <h1>{title}</h1>
  <audio id="a" controls preload="metadata" src="{mp3}"></audio>
  <div class="hint">Слово подсвечивается когда звучит · <b>клик по слову</b> — перемотка ·
    <span style="color:var(--fix)">голубым</span> — слова, тронутые правкой ({n_fixes} мест) ·
    покрытие: не услышано {unheard}с → добрано {recovered}с, потеряно {lost}с</div>
</header>
<main id="doc"></main>
<script>
const TURNS = {turns_json};
const audio = document.getElementById("a");
const doc = document.getElementById("doc");
const flat = [];
for (const t of TURNS) {{
  const box = document.createElement("div"); box.className = "turn";
  const who = document.createElement("div"); who.className = "who";
  who.textContent = t.speaker;
  const mm = String(Math.floor(t.start / 60)).padStart(2, "0");
  const ss = String(Math.floor(t.start % 60)).padStart(2, "0");
  const clock = document.createElement("span"); clock.textContent = mm + ":" + ss;
  who.append(clock); box.append(who);
  const p = document.createElement("p"); p.style.margin = "0";
  for (const word of t.words) {{
    const node = document.createElement("w");
    node.textContent = word.w;
    if (word.f) node.classList.add("fix");
    node.onclick = () => {{ audio.currentTime = word.s; audio.play(); }};
    p.append(node, " ");
    flat.push({{ s: word.s, e: word.e, node }});
  }}
  box.append(p); doc.append(box);
}}
let idx = 0;
function paint() {{
  const t = audio.currentTime;
  let lo = 0, hi = flat.length - 1, cur = -1;
  while (lo <= hi) {{
    const mid = (lo + hi) >> 1;
    if (flat[mid].s <= t) {{ cur = mid; lo = mid + 1; }} else hi = mid - 1;
  }}
  if (cur === idx) return;
  for (let i = 0; i < flat.length; i++) {{
    const n = flat[i].node;
    n.classList.toggle("done", i < cur);
    n.classList.toggle("on", i === cur);
  }}
  if (cur >= 0) {{
    const r = flat[cur].node.getBoundingClientRect();
    if (r.top < 90 || r.bottom > innerHeight - 120)
      flat[cur].node.scrollIntoView({{ block: "center", behavior: "smooth" }});
  }}
  idx = cur;
}}
audio.addEventListener("timeupdate", paint);
setInterval(() => {{ if (!audio.paused) paint(); }}, 120);
</script></body></html>
"""


def fixed_spans(turn) -> list[tuple[int, int]]:
    """Диапазоны слов финального текста, отличающиеся от сырого, — их подсветим как правку."""
    a, b = (turn.get('raw') or '').split(), (turn.get('text') or '').split()
    out = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if op != 'equal':
            out.append((j1, j2))
    return out


def build(path: Path, out: Path) -> None:
    d = json.loads(path.read_text(encoding='utf-8'))
    x = (d.get('result') or d).get('x_enriched') or d
    words = x.get('words')
    if not words:
        raise SystemExit(f'{path}: нет x_enriched.words — выравнивание не отработало')

    # words-реплики и текстовые реплики идут в одном порядке — помечаем правленые слова по диффу
    turns_out, n_fixes = [], 0
    text_turns = x.get('turns') or []
    for i, wt in enumerate(words['turns']):
        marks = set()
        if i < len(text_turns):
            spans = fixed_spans(text_turns[i])
            n_fixes += len(spans)
            for j1, j2 in spans:
                marks.update(range(j1, j2))
        turns_out.append({
            'speaker': wt.get('speaker') or '?',
            'start': wt.get('start') or 0.0,
            'words': [{'w': w[0], 's': w[1], 'e': w[2], **({'f': 1} if k in marks else {})}
                      for k, w in enumerate(wt.get('words') or [])],
        })

    slug = path.stem.replace('ep', '', 1)             # ep17 → 17 | ep2-28 → 2-28
    cov = x.get('coverage') or {}
    html = PAGE.format(
        title=f'Караоке — выпуск {slug} (прогон {path.parent.name})',
        mp3=MP3.format(slug=slug),
        turns_json=json.dumps(turns_out, ensure_ascii=False),
        n_fixes=n_fixes,
        unheard=cov.get('unheard_sec', '?'), recovered=cov.get('recovered_sec', '?'),
        lost=cov.get('lost_sec', '?'),
    )
    out.write_text(html, encoding='utf-8')
    print(f'{out} — {len(turns_out)} реплик, {words["words_total"]} слов, правок {n_fixes}')


def main() -> None:
    ap = argparse.ArgumentParser(description='караоке-страница из результата asr-adaptor')
    ap.add_argument('results', nargs='+', help='epN.json прогона')
    ap.add_argument('--out-dir', default='.', help='куда класть html')
    args = ap.parse_args()
    for p in args.results:
        path = Path(p)
        build(path, Path(args.out_dir) / f'karaoke_{path.stem}.html')


if __name__ == '__main__':
    main()
