# Нативный RAG-стек на macOS без Docker (launchd)

4 сервиса: qdrant (нативный бинарь aarch64-darwin) · GTE sparse-эмбеддер · OWUI Pipelines
с morag_pipeline.py · Open WebUI (публичный, без авторизации). Все bind 127.0.0.1 —
наружу только реверс-прокси (Caddy: `rag.your-domain { reverse_proxy 127.0.0.1:3000 }`).

Перед установкой подставьте своё имя пользователя и прокси (если нужен):

```bash
sed -i '' "s/USERNAME/$(whoami)/g" com.morag.*.plist
# com.morag.pipelines.plist: HTTPS_PROXY — только если LLM-провайдер требует прокси; иначе удалите ключ
cp com.morag.*.plist ~/Library/LaunchAgents/
for s in qdrant gte pipelines owui; do launchctl load -w ~/Library/LaunchAgents/com.morag.$s.plist; done
launchctl list | grep com.morag
```

Грабли (выстрадано, детали в examples/audio-rag/README.md):
- venv'ы — от **Python 3.12** (OWUI требует <3.13; пины GTE без cp313-wheels под darwin-arm64).
- qdrant-бинарь: релиз GitHub `qdrant-aarch64-apple-darwin.tar.gz`, версия = версии данных;
  при карантине `xattr -d com.apple.quarantine qdrant`. Данные — копия каталога storage
  (rsync при остановленных qdrant с обеих сторон).
- Веса GTE — офлайн-кэш (`HF_HOME` + `TRANSFORMERS_OFFLINE=1`); layout two-snapshot:
  models--gte-multilingual-base + new-impl + modules.
- `WEBUI_AUTH=False` работает только на чистой `data/`; `ENABLE_PERSISTENT_CONFIG=False` —
  правки посетителей в Admin UI не переживают рестарт.
- Обновление кода pipeline: scp нового morag_pipeline.py + `pip install <morag>` →
  `pkill -f "uvicorn main:app"` (KeepAlive поднимет). Конфиг — hot-reload без рестарта.
