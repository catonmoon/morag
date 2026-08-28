#!/usr/bin/env bash
# Разворачивает Mac-стек транскрибации на ЛЮБОЙ машине Apple Silicon: четыре venv, каталоги
# состояния и (по флагу) launchd-плисты. Код берётся из чекаута morag, НЕ копируется.
#
#   ./install.sh              # поставить/досоздать всё, чего не хватает
#   ./install.sh --check      # ничего не менять, только сказать чего не хватает
#   ./install.sh --launchd    # вдобавок сгенерировать и загрузить плисты (см. README: на 16 ГБ не надо)
#
# Идемпотентен: существующие venv не пересоздаёт, зависимости досыпает. Модели НЕ качает —
# их приносит fetch-assets.sh (pyannote gated, whisper-turbo собирался вручную).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ADAPTOR_DIR="$(cd "$HERE/../.." && pwd)"          # services/asr-adaptor
ENV_FILE="${ASR_STACK_ENV:-$HOME/.asr-stack.env}"

CHECK_ONLY=0; WANT_LAUNCHD=0
for a in "$@"; do
  case "$a" in
    --check) CHECK_ONLY=1 ;;
    --launchd) WANT_LAUNCHD=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "неизвестный флаг: $a" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m  ✓ %s\033[0m\n' "$*"; }

# --- окружение ---------------------------------------------------------------
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
  ok "окружение: $ENV_FILE"
else
  warn "нет $ENV_FILE — создаю из шаблона, ЗАПОЛНИТЕ СЕКРЕТЫ и запустите снова"
  [[ $CHECK_ONLY -eq 1 ]] || { cp "$HERE/env.example" "$ENV_FILE"; chmod 600 "$ENV_FILE"; }
  exit 1
fi

STACK_HOME="${ASR_STACK_HOME:-$HOME/asr-stack}"
MORAG_REPO="${MORAG_REPO:-$(cd "$ADAPTOR_DIR/../.." && pwd)}"
VENVS="$STACK_HOME/venvs"

# Питон для venv. 3.9 (системный) не понимает `X | None` из кода адаптера — ловили при первом
# развёртывании BFF; берём homebrew-питон и проверяем версию явно.
#
# ⚠️ Ищем по абсолютным путям, а НЕ через `command -v`: если установщик запущен из активного venv
# (или venv чужого проекта лежит в PATH), в базу наших venv уедет тот интерпретатор. Ловили:
# базой всех четырёх стал /opt/anaconda3 — стек молча начинал зависеть от анаконды.
if [[ -z "${ASR_PYTHON:-}" ]]; then
  for c in /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 \
           /usr/local/bin/python3.12 /usr/local/bin/python3.13; do
    [[ -x "$c" ]] && { ASR_PYTHON="$c"; break; }
  done
fi
PY="${ASR_PYTHON:-$(command -v python3)}"
PYV="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[[ "${PYV%%.*}" -ge 3 && "${PYV#*.}" -ge 12 ]] || { echo "нужен python ≥3.12, найден $PYV ($PY)" >&2; exit 1; }
ok "python $PYV — $PY"

command -v ffmpeg >/dev/null || warn "ffmpeg не найден: mlx_whisper зовёт его для декода — brew install ffmpeg"
[[ -d "$MORAG_REPO/services/asr-adaptor" ]] || { echo "не вижу чекаут morag: $MORAG_REPO" >&2; exit 1; }
ok "morag: $MORAG_REPO"

# --- локальные Bearer'ы бэкендов ---------------------------------------------
# Диаризатор без DIARIZER_API_KEY не стартует вовсе, CAM++ и transcribe проверяют свой.
# Это секрет между процессами ОДНОЙ машины, и переносить его с донора незачем: генерируем свой.
# С донора нужны только ключ LLM и egress-прокси (import-env.sh).
# Заполняем ПУСТУЮ строку шаблона, а не дописываем в конец: иначе в файле оказываются два
# присваивания одного имени (побеждает последнее), и правка верхней строки молча ни на что не
# влияет. Дописываем только если строки нет вовсе.
set_env_value() {               # set_env_value <ИМЯ> <значение>
  "$PY" - "$ENV_FILE" "$1" "$2" <<'PY'
import os, re, sys
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path, encoding='utf-8').read()
line, pat = f'{key}={val}', re.compile(rf'^{re.escape(key)}=.*$', re.M)
text = pat.sub(lambda _: line, text, count=1) if pat.search(text) else text.rstrip() + f'\n{line}\n'
open(path, 'w', encoding='utf-8').write(text)
os.chmod(path, 0o600)
PY
}

gen_key() {                     # gen_key <ИМЯ> [ИМЯ-СИНОНИМ...]
  local primary="$1"; shift
  local cur="${!primary:-}"
  if [[ -z "$cur" ]]; then
    [[ $CHECK_ONLY -eq 1 ]] && { warn "$primary не задан"; return 0; }
    cur="$("$PY" -c 'import secrets;print(secrets.token_urlsafe(16))')"
    set_env_value "$primary" "$cur"
    ok "$primary — сгенерирован"
  fi
  export "$primary=$cur"
  for alias in "$@"; do         # бэкенды ждут ключ под своим именем, адаптер — под ASR_*
    if [[ -z "${!alias:-}" ]]; then
      [[ $CHECK_ONLY -eq 1 ]] || set_env_value "$alias" "$cur"
      export "$alias=$cur"
    fi
  done
}
say "ключи локальных бэкендов"
gen_key ASR_BACKEND_KEY  TRANSCRIBE_API_KEY
gen_key ASR_CAMPP_KEY    CAMPP_API_KEY
gen_key ASR_DIARIZER_KEY DIARIZER_API_KEY
# ⚠️ Без OR_KEY адаптер не просто теряет LLM-стадии — он НЕ СТАРТУЕТ: app.py строит LLMClient на
# импорте модуля, и AsyncOpenAI падает «Missing credentials» ещё до первого запроса.
[[ -n "${OR_KEY:-}" ]] && ok "ключ LLM задан" \
  || warn "OR_KEY пуст — адаптер не стартует вовсе (LLMClient строится на импорте): ./import-env.sh <донор>"

# --- каталоги ----------------------------------------------------------------
say "каталоги стека — $STACK_HOME"
if [[ $CHECK_ONLY -eq 0 ]]; then mkdir -p "$STACK_HOME"/{models,state,logs,run,venvs}; fi
for d in models state logs run venvs; do
  [[ -d "$STACK_HOME/$d" ]] && ok "$d" || warn "$d — нет"
done

# --- venv-ы ------------------------------------------------------------------
# Четыре, потому что версии несовместимы: pyannote прибит к torch 2.3.1, адаптеру нужен свежий
# torchaudio для MMS_FA, а mlx-whisper тянет свой mlx. Один общий venv тут не собирается.
mkvenv() {                      # mkvenv <имя> <описание> <pip-аргументы...>
  local name="$1" desc="$2"; shift 2
  local v="$VENVS/$name"
  if [[ -x "$v/bin/python" ]]; then ok "venv $name уже есть"; else
    if [[ $CHECK_ONLY -eq 1 ]]; then warn "venv $name — нет ($desc)"; return 0; fi
    say "venv $name — $desc"
    "$PY" -m venv "$v"
    "$v/bin/pip" install --quiet --upgrade pip wheel
  fi
  [[ $CHECK_ONLY -eq 1 ]] && return 0
  say "зависимости $name"
  "$v/bin/pip" install --quiet "$@"
  ok "venv $name готов"
}

mkvenv whisper  "mlx-whisper, пасс-1 и пасс-2" \
        mlx-whisper fastapi uvicorn python-multipart

mkvenv diarizer "pyannote, разметка спикеров" \
        -r "$ADAPTOR_DIR/backends/diarizer/requirements.txt"

mkvenv campp    "CAM++ sherpa-onnx, реестр голосов" \
        sherpa-onnx soundfile numpy fastapi uvicorn python-multipart

# Адаптеру нужно ядро morag из чекаута (LLMClient, TokenCounter) + своё + торч для выравнивания.
# ⚠️ Ядро ставится в site-packages, а не читается из чекаута: после git pull, тронувшего src/morag,
# venv останется со старым ядром (ловили TypeError на свежем RetryPolicy) — прогонять install.sh снова.
mkvenv adaptor  "сам asr-adaptor + morag-core + выравнивание слов" \
        "$MORAG_REPO" \
        -r "$ADAPTOR_DIR/requirements.txt" \
        -r "$ADAPTOR_DIR/requirements-align.txt"

# --- модели и состояние ------------------------------------------------------
say "модели и состояние"
WHISPER_DIR="$STACK_HOME/models/whisper-podlodka-turbo"
CAMPP_ONNX="$STACK_HOME/models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
PYANNOTE_CACHE="$HOME/.cache/torch/pyannote"
REGISTRY="$STACK_HOME/state/speaker_registry.json"

[[ -f "$WHISPER_DIR/weights.safetensors" ]] && ok "whisper-podlodka-turbo" \
  || warn "нет весов whisper: $WHISPER_DIR — забрать fetch-assets.sh"
[[ -f "$CAMPP_ONNX" ]] && ok "CAM++ onnx" || warn "нет CAM++ onnx: $CAMPP_ONNX — забрать fetch-assets.sh"
[[ -d "$PYANNOTE_CACHE/models--pyannote--speaker-diarization-3.1" ]] && ok "pyannote в кэше" \
  || warn "нет pyannote ($PYANNOTE_CACHE) — забрать fetch-assets.sh либо скачать с HF_TOKEN"
if [[ -f "$REGISTRY" ]]; then
  ok "реестр голосов: $("$PY" -c "import json;d=json.load(open('$REGISTRY'));print(f\"next_id={d['next_id']}, голосов {len(d['speakers'])}\")" 2>/dev/null || echo "есть")"
else
  warn "НЕТ РЕЕСТРА ГОЛОСОВ ($REGISTRY). Без него нумерация Speaker_N начнётся заново и наминг"
  warn "всего будущего корпуса разъедется с прежним. Забрать fetch-assets.sh до первого прогона."
fi

# MMS_FA для пословного выравнивания качается сам при первом прогоне (~1.3 ГБ в ~/.cache/torch).
[[ -f "$HOME/.cache/torch/hub/checkpoints/model.pt" ]] && ok "MMS_FA (выравнивание) в кэше" \
  || warn "MMS_FA качнётся при первом выравнивании: +~1.3 ГБ и четверть часа"

# --- launchd (по флагу) ------------------------------------------------------
if [[ $WANT_LAUNCHD -eq 1 && $CHECK_ONLY -eq 0 ]]; then
  say "плисты launchd"
  mkdir -p "$HOME/Library/LaunchAgents"
  for t in "$HERE"/templates/*.plist.in; do
    name="$(basename "${t%.in}")"
    sed -e "s#@STACK_HOME@#$STACK_HOME#g" \
        -e "s#@ADAPTOR_DIR@#$ADAPTOR_DIR#g" \
        -e "s#@ENV_FILE@#$ENV_FILE#g" \
        -e "s#@HOME@#$HOME#g" "$t" > "$HOME/Library/LaunchAgents/$name"
    ok "$name"
  done
  warn "плисты записаны, но НЕ загружены: launchctl load ~/Library/LaunchAgents/<имя>"
  warn "на машине с 16 ГБ держать четыре сервиса резидентно не стоит — см. README и stack.sh"
fi

say "готово"
echo "  поднять стек:  $HERE/stack.sh up"
echo "  проверить:     $HERE/stack.sh health"
