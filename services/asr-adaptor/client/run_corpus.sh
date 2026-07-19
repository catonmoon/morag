#!/bin/bash
# Прогон корпуса записей: по одной вызываем transcribe_one.sh, ПОСЛЕДОВАТЕЛЬНО —
# детерминизм Speaker_N в общем реестре голосов. Идемпотентно (готовые epN.json
# пропускаются → резюмируемо). Падение одной записи не рвёт прогон.
#
#   ASR_BASE=... MP3_URL_TEMPLATE='https://site/ep{pfx}{n}.mp3' ./run_corpus.sh 1 2 3
#   SEASON=2 ASR_BASE=... MP3_URL_TEMPLATE=... ./run_corpus.sh $(seq 1 22)
#
# Все env (SEASON, OUT_DIR, CACHE_DIR, TITLE_TEMPLATE, ...) наследуются transcribe_one.sh.
cd "$(dirname "$0")" || exit 1

SEASON="${SEASON:-1}"
EPS="${*:?usage: run_corpus.sh <episode-numbers...>}"
echo "=== corpus run start @ $(date +%H:%M:%S) [season $SEASON]: $EPS ==="
for n in $EPS; do
  echo "--- season$SEASON ep$n @ $(date +%H:%M:%S) ---"
  bash transcribe_one.sh "$n" || echo "[season$SEASON ep$n] FAILED (continuing)"
done
echo "=== corpus run DONE @ $(date +%H:%M:%S) ==="
