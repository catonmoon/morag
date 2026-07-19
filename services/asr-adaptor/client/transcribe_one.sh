#!/bin/bash
# Клиент на ОДНУ транскрибацию записи: качает аудио (или берёт локальный файл / извлекает
# дорожку из видео через ffmpeg), грузит в asr-adaptor (стандартный OpenAI
# POST /v1/audio/transcriptions, mode=async), поллит джобу, сохраняет epN.{md,json}.
# Оркестрация живёт в клиенте — сервис остаётся тупым.
#
# Использование:
#   ASR_BASE=https://host/asr MP3_URL_TEMPLATE='https://site/ep{pfx}{n}.mp3' ./transcribe_one.sh 19
#   ASR_BASE=https://host/asr ./transcribe_one.sh 19 https://site/any-file.mp3   # прямой URL
#   ASR_BASE=https://host/asr ./transcribe_one.sh 19 /path/to/local.mp4          # локальный файл/видео
#
# Env-контракт:
#   ASR_BASE          (обязателен)  базовый URL asr-adaptor, напр. https://host/asr
#   MP3_URL_TEMPLATE  (если нет 2-го аргумента)  шаблон URL с {n} (номер) и {pfx} (сезонный префикс)
#   SEASON            (default 1)   группировка артефактов по сезонам/плейлистам
#   SEASON_PREFIX     (default: '' для SEASON=1, иначе '{SEASON}-')  значение {pfx} в шаблонах
#   OUT_DIR           (default ./transcripts/season${SEASON})   куда класть epN.{md,json}
#   CACHE_DIR         (default ./media_cache/season${SEASON})   кэш скачанного аудио
#   TITLE_TEMPLATE    (default 'Episode {pfx}{n}')              title для front-matter адаптера
set -euo pipefail

# Большие аплоады (50-100МБ) через корп-CONNECT-прокси таймаутят — ходим напрямую.
# Если вашему адаптеру прокси нужен — уберите строку.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

N="${1:?usage: transcribe_one.sh <episode-number> [url-or-local-file]}"
DIRECT_SRC="${2:-}"
SEASON="${SEASON:-1}"
BASE="${ASR_BASE:?set ASR_BASE, e.g. https://your-host/asr}"

PFX="${SEASON_PREFIX-}"
if [ -z "${SEASON_PREFIX+x}" ]; then    # не задан явно → конвенция «сезон 1 без префикса»
  PFX=""; [ "$SEASON" != "1" ] && PFX="${SEASON}-"
fi

CACHE="${CACHE_DIR:-./media_cache/season${SEASON}}"
OUT="${OUT_DIR:-./transcripts/season${SEASON}}"
mkdir -p "$CACHE" "$OUT"

EPID="ep${PFX}${N}"
AUDIO="$CACHE/ep${N}.mp3"

# источник: прямой аргумент › шаблон
if [ -n "$DIRECT_SRC" ]; then
  SRC="$DIRECT_SRC"
else
  SRC="${MP3_URL_TEMPLATE:?set MP3_URL_TEMPLATE with {n}/{pfx} or pass url as 2nd arg}"
  SRC="${SRC//\{n\}/$N}"; SRC="${SRC//\{pfx\}/$PFX}"
fi
TITLE="${TITLE_TEMPLATE:-Episode {pfx}{n}}"
TITLE="${TITLE//\{n\}/$N}"; TITLE="${TITLE//\{pfx\}/$PFX}"

# идемпотентность: готовая запись не перегоняется (резюм прогона корпуса)
if [ -s "$OUT/ep${N}.json" ]; then
  echo "[$EPID] SKIP (already done: $OUT/ep${N}.json)"
  exit 0
fi

# получить аудио: локальный файл (в т.ч. видео → извлечь дорожку) либо скачать
if [ ! -s "$AUDIO" ]; then
  if [ -f "$SRC" ]; then
    case "$SRC" in
      *.mp4|*.mkv|*.mov|*.webm|*.avi|*.m4v)
        echo "[$EPID] extracting audio track from video (ffmpeg)..."
        ffmpeg -hide_banner -loglevel error -y -i "$SRC" -vn -acodec libmp3lame -q:a 4 "$AUDIO" ;;
      *) cp "$SRC" "$AUDIO" ;;
    esac
  else
    echo "[$EPID] download from $SRC ..."
    curl -fsS --max-time 900 -o "$AUDIO" "$SRC"
    case "$SRC" in
      *.mp4|*.mkv|*.mov|*.webm|*.avi|*.m4v)
        echo "[$EPID] extracting audio track from downloaded video (ffmpeg)..."
        mv "$AUDIO" "$AUDIO.video"
        ffmpeg -hide_banner -loglevel error -y -i "$AUDIO.video" -vn -acodec libmp3lame -q:a 4 "$AUDIO"
        rm -f "$AUDIO.video" ;;
    esac
  fi
fi
echo "[$EPID] $(stat -f%z "$AUDIO" 2>/dev/null || stat -c%s "$AUDIO") bytes; submit (upload ~1 min)..."

RESP=$(curl -fsS --max-time 600 -X POST "$BASE/v1/audio/transcriptions" \
  -F "file=@${AUDIO};type=audio/mpeg" -F "mode=async" -F "episode=${EPID}" \
  -F "title=${TITLE}" -F "url=${SRC}")
JID=$(printf '%s' "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
echo "[$EPID] job $JID; polling..."

i=0
while true; do
  sleep 15
  S=$(curl -fsS --max-time 60 "$BASE/v1/jobs/$JID" || echo '{"status":"poll_fail"}')
  ST=$(printf '%s' "$S" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status","?"))' 2>/dev/null || echo parse_fail)
  case "$ST" in
    done) break ;;
    error) echo "[$EPID] ERROR: $S"; exit 1 ;;
    poll_fail|parse_fail) echo "[$EPID] transient poll issue, retry"; continue ;;
  esac
  i=$((i + 1))
  if [ $((i % 4)) -eq 0 ]; then
    PR=$(printf '%s' "$S" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("progress",""))' 2>/dev/null || true)
    echo "[$EPID] running: $PR (~$((i * 15))s)"
  fi
done

# Сохраняем результат. ВАЖНО: $S пишем в файл, а не в stdin питона — heredoc <<'PY'
# сам занимает stdin (это программа для `python3 -`), пайп бы перебился и json.load пуст.
JOB_JSON="$CACHE/_job_ep${N}.json"
printf '%s' "$S" > "$JOB_JSON"
python3 - "$OUT" "$N" "$JOB_JSON" <<'PY'
import sys, json, pathlib
out, n, src = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
s = json.load(open(src))
r = s["result"]
e = r["x_enriched"]
(out / f"ep{n}.md").write_text(e["markdown"])
json.dump(r, open(out / f"ep{n}.json", "w"), ensure_ascii=False, indent=1)
t = e["timing"]
print(f'[ep{n}] DONE total={t.get("total_s")}s round={t.get("round_s")}s '
      f'n_chunks={t.get("n_chunks")} speakers={e["speaker_map"]}')
PY
