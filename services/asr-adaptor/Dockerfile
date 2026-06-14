# asr-adaptor — оркестратор аудио-транскрайб-пайплайна на morag-core.
# Контекст сборки = КОРЕНЬ репо (как services/pipeline, services/console): нужен src/morag.
# Docker-вариант — оркестратор без аудио-моделей (Apple-Silicon-bound → на Маке); аудио-URLs
# указывают на Mac по Caddy. Primary-деплой — Mac-native (launchd), см. CLAUDE.md.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY services/asr-adaptor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# morag-core (только core-deps, без [indexing] extras)
COPY pyproject.toml /opt/morag/pyproject.toml
COPY src /opt/morag/src
RUN pip install --no-cache-dir /opt/morag/

# Оркестратор: «оболочка» (app/config/jobs/pipeline/audio_clients) + пакет stages/.
# backends/ (Mac-bound аудио-сервисы) и deploy/ (launchd plist) в образ не идут.
COPY services/asr-adaptor/*.py /app/
COPY services/asr-adaptor/stages /app/stages

EXPOSE 8082
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "1"]
