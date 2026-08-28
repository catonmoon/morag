#!/usr/bin/env bash
# Переносит СЕКРЕТЫ работающего инстанса на эту машину: ключ LLM и egress-прокси.
# Значения идут по SSH прямо в ~/.asr-stack.env и НИГДЕ не печатаются — ни в stdout, ни в логах.
#
#   ./import-env.sh user@донор-хост
#
# Донор — машина, где стек уже работает: ключи лежат в плисте launchd (com.podlodka.asr) и в
# ~/.diarizer.env. Скрипт правит только пустые поля; заполненное не трогает (идемпотентно).
#
# Что переносится: OR_KEY (ключ LLM), HTTPS_PROXY (OpenRouter не пускает RU-IP), HF_TOKEN.
# Что НЕ переносится: пути и параллелизм (у этой машины свои) и Bearer'ы локальных бэкендов —
# их install.sh генерирует на месте, это секрет между процессами одной машины.
# Уже заполненные поля не трогаются, так что запуск поверх готового env безопасен.
set -euo pipefail

SRC="${1:-}"
[[ -n "$SRC" ]] || { sed -n '2,12p' "$0"; exit 2; }
ENV_FILE="${ASR_STACK_ENV:-$HOME/.asr-stack.env}"
[[ -f "$ENV_FILE" ]] || { echo "нет $ENV_FILE — сначала ./install.sh" >&2; exit 1; }

TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
chmod 600 "$TMP"

# Снимаем донорское окружение одним заходом, в файл с правами 600. В stdout — ничего.
ssh "$SRC" '
  plutil -extract EnvironmentVariables json -o - ~/Library/LaunchAgents/com.podlodka.asr.plist 2>/dev/null
  echo "@@DIARIZER@@"
  cat ~/.diarizer.env 2>/dev/null
  echo "@@TRANSCRIBE@@"
  plutil -extract EnvironmentVariables.TRANSCRIBE_API_KEY raw -o - \
    ~/Library/LaunchAgents/com.podlodka.transcribe.plist 2>/dev/null
' > "$TMP"

python3 - "$TMP" "$ENV_FILE" <<'PY'
import json, os, re, sys

raw = open(sys.argv[1], encoding='utf-8').read()
env_path = sys.argv[2]

adaptor_json, _, rest = raw.partition('@@DIARIZER@@')
diarizer_txt, _, transcribe_txt = rest.partition('@@TRANSCRIBE@@')

try:
    donor = json.loads(adaptor_json)
except json.JSONDecodeError:
    sys.exit('не разобрал плист донора — стек там точно развёрнут?')

for line in diarizer_txt.splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k, _, v = line.partition('=')
        donor.setdefault(k.strip(), v.strip().strip('"\''))

tk = transcribe_txt.strip()
if tk:
    donor['TRANSCRIBE_API_KEY'] = tk

# Bearer бэкендов на доноре лежат под ASR_*_KEY; сами бэкенды ждут их под своими именами.
donor.setdefault('CAMPP_API_KEY', donor.get('ASR_CAMPP_KEY', ''))
donor.setdefault('DIARIZER_API_KEY', donor.get('ASR_DIARIZER_KEY', ''))
donor.setdefault('TRANSCRIBE_API_KEY', donor.get('ASR_BACKEND_KEY', ''))

# Bearer'ы бэкендов сюда НЕ входят: их генерирует install.sh на этой машине.
WANTED = ['OR_KEY', 'HTTPS_PROXY', 'HF_TOKEN']

text = open(env_path, encoding='utf-8').read()
filled, kept, missing = [], [], []
for key in WANTED:
    val = (donor.get(key) or '').strip()
    if not val:
        missing.append(key)
        continue
    pat = re.compile(rf'^{re.escape(key)}=(.*)$', re.M)
    m = pat.search(text)
    if m and m.group(1).strip():
        kept.append(key)                      # уже заполнено — не перетираем
        continue
    repl = f'{key}={val}'
    text = pat.sub(lambda _: repl, text, count=1) if m else text.rstrip() + f'\n{repl}\n'
    filled.append(key)

with open(env_path, 'w', encoding='utf-8') as fh:
    fh.write(text)
os.chmod(env_path, 0o600)

# Печатаем ТОЛЬКО имена — значения не показываем.
print(f"  ✓ заполнено: {', '.join(filled) or '—'}")
if kept:    print(f"  · уже было (не тронуто): {', '.join(kept)}")
if missing: print(f"  ! на доноре не нашлось: {', '.join(missing)}")
PY

echo "  ✓ $ENV_FILE (0600)"
