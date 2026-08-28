#!/usr/bin/env bash
# Замер машины на сквозном прогоне выпуска: время по стадиям, вызовы LLM, деньги.
# Нужен, чтобы сравнивать РАЗНЫЕ машины одними и теми же числами.
#
#   ./bench.sh /путь/к/выпуску.mp3            # полный замер
#   ./bench.sh /путь/к/выпуску.mp3 --keep     # не откатывать реестр голосов (по умолчанию откат)
#
# Что делает: поднимает стек, если он не поднят; прогоняет файл целиком через адаптер; снимает
# тайминг стадий из ответа, число вызовов OpenRouter из лога и дельту расхода по ключу.
# Результат — таблица в stdout и JSON рядом с выпуском (`<файл>.bench.json`).
#
# ⚠️ Реестр голосов откатывается ПОСЛЕ прогона. Замер — не боевая транскрибация: он не должен
# оставлять следов в общем состоянии. Даже когда все голоса уже известны, прогон дописывает выпуск
# в `provenance`, и запись бывает ложной (CAM++ может сматчить гостя с чужим центроидом — интро
# исправит имя в транскрипте, но в провенанс уйдёт не тот человек).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ADAPTOR_DIR="$(cd "$HERE/../.." && pwd)"
ENV_FILE="${ASR_STACK_ENV:-$HOME/.asr-stack.env}"
[[ -f "$ENV_FILE" ]] || { echo "нет $ENV_FILE — сначала ./install.sh" >&2; exit 1; }
set -a; source "$ENV_FILE"; set +a

MP3="${1:-}"; KEEP="${2:-}"
[[ -f "$MP3" ]] || { sed -n '2,10p' "$0"; exit 2; }

STACK_HOME="${ASR_STACK_HOME:-$HOME/asr-stack}"
REGISTRY="${ASR_REGISTRY_PATH:-$STACK_HOME/state/speaker_registry.json}"
LOG="$STACK_HOME/logs/adaptor.log"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[32m  ✓ %s\033[0m\n' "$*"; }

# --- машина, на которой меряем ------------------------------------------------
MODEL="$(sysctl -n hw.model)"
CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo '?')"
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
CORES="$(sysctl -n hw.ncpu)"
say "машина: $CHIP ($MODEL), $RAM_GB ГБ, $CORES ядер"

DUR="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$MP3" | cut -d. -f1)"
ok "выпуск: $DUR с звука ($(( DUR / 60 )) мин)"

# --- стек ---------------------------------------------------------------------
if ! curl -s -m 5 http://127.0.0.1:8082/health >/dev/null 2>&1; then
  say "стек не поднят — поднимаю"
  "$HERE/stack.sh" up >/dev/null 2>&1 || true
  curl -s -m 5 http://127.0.0.1:8082/health >/dev/null 2>&1 || { echo "стек не поднялся, см. $STACK_HOME/logs" >&2; exit 1; }
fi
ok "стек отвечает"

# --- baseline: расход по ключу и позиция в логе -------------------------------
usage_now() {
  [[ -n "${OR_KEY:-}" ]] || { echo 0; return; }
  curl -s -m 30 ${HTTPS_PROXY:+--proxy "$HTTPS_PROXY"} https://openrouter.ai/api/v1/key \
    -H "Authorization: Bearer $OR_KEY" \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"]["usage"])' 2>/dev/null || echo 0
}
USAGE_BEFORE="$(usage_now)"
LOG_LINES_BEFORE=$(wc -l < "$LOG" 2>/dev/null || echo 0)
cp "$REGISTRY" "$WORK/registry.before.json" 2>/dev/null || true

# --- прогон -------------------------------------------------------------------
say "прогон (сравнимо только с таким же полным выпуском)"
START=$(date +%s)
ASR_BASE=http://127.0.0.1:8082 \
OUT_DIR="$WORK/out" CACHE_DIR="$WORK/cache" \
TITLE_TEMPLATE='bench {n}' SEASON=9 \
bash "$ADAPTOR_DIR/client/transcribe_one.sh" 99 "$MP3" 2>&1 | tail -3
WALL=$(( $(date +%s) - START ))

JOB="$(ls "$WORK/cache"/_job_ep99.json 2>/dev/null || true)"
[[ -f "$JOB" ]] || { echo "прогон не дал артефакта — см. $STACK_HOME/logs/adaptor.log" >&2; exit 1; }

# --- расход -------------------------------------------------------------------
USAGE_AFTER="$(usage_now)"
# ⚠️ Без `|| true` тут падает `set -e`, а с `|| echo 0` получается ДВЕ строки: `grep -c` при нуле
# совпадений печатает 0 и при этом возвращает код 1. Ловилось как `int('0\n0')`.
NEW_LOG="$(tail -n +"$((LOG_LINES_BEFORE + 1))" "$LOG" 2>/dev/null || true)"
CALLS=$(printf '%s' "$NEW_LOG" | grep -c 'openrouter.ai/api/v1/chat/completions' || true)
FAILED=$(printf '%s' "$NEW_LOG" | grep 'openrouter.ai/api/v1/chat/completions' | grep -vc '200 OK' || true)

# --- откат реестра ------------------------------------------------------------
if [[ "$KEEP" == "--keep" ]]; then
  ok "реестр оставлен как есть (--keep)"
elif [[ -f "$WORK/registry.before.json" ]]; then
  cp "$WORK/registry.before.json" "$REGISTRY"
  ok "реестр голосов откачен к состоянию до замера"
fi

# --- отчёт --------------------------------------------------------------------
OUT_JSON="${MP3%.mp3}.bench.json"
python3 - "$JOB" "$DUR" "$WALL" "$CALLS" "$FAILED" "$USAGE_BEFORE" "$USAGE_AFTER" \
         "$CHIP" "$MODEL" "$RAM_GB" "$OUT_JSON" <<'PY'
import json, sys
job, dur, wall, calls, failed, ub, ua, chip, model, ram, out = sys.argv[1:]
dur, wall, calls, failed, ram = int(dur), int(wall), int(calls), int(failed), int(ram)
cost = float(ua) - float(ub)
d = json.load(open(job))
xe = (d.get("result") or d).get("x_enriched") or {}
t = xe.get("timing") or {}
stages = [("diarize_s","диаризация",""),("pass1_s","пасс-1 whisper",""),
          ("glossary_s","глоссарий","LLM"),("pass2_s","пасс-2 whisper",""),
          ("round_s","финал-раунд","LLM"),("naming_s","наминг","LLM"),
          ("align_s","выравнивание слов","")]
total = t.get("total_s") or wall
print(f"\n{'стадия':<22}{'время':>9}{'доля':>8}   LLM")
print("-" * 50)
for k, label, llm in stages:
    v = t.get(k)
    if v is None: continue
    print(f"{label:<22}{v:>8.1f}с{v/total*100:>7.1f}%   {llm}")
print("-" * 50)
print(f"{'итого':<22}{total:>8.1f}с{'':>8}   {calls} вызовов" + (f", ОШИБОК {failed}" if failed else ", 0 ошибок"))
print(f"\nзвука: {dur} с ({dur/60:.0f} мин) → обработано за {total/60:.1f} мин, "
      f"то есть в {dur/total:.1f}× быстрее реального времени")
print(f"чанков: {t.get('n_chunks')} | глоссарий: {t.get('n_glossary')} терминов | правилось реплик: {t.get('n_round_turns')}")
print(f"стоимость: ${cost:.4f}" + (f" (${cost/(dur/3600):.4f} на час звука)" if dur else ""))
cov = xe.get("coverage") or {}
if cov:
    unheard = sum(b - a for a, b in cov.get("unheard", []))
    print(f"потери речи: дыры пасса-1 {cov.get('pass1_hole_sec')} с, осталось неуслышанным {unheard:.1f} с")

json.dump({"machine": {"chip": chip, "model": model, "ram_gb": ram},
           "audio_sec": dur, "wall_sec": wall, "timing": t,
           "llm_calls": calls, "llm_failed": failed, "cost_usd": round(cost, 6),
           "coverage": cov, "turns": len(xe.get("turns") or [])},
          open(out, "w"), ensure_ascii=False, indent=2)
print(f"\nJSON замера: {out}")
PY
