FROM python:3.12-slim

# Прокси для pip install — передаются через --build-arg или compose build.args.
ARG HTTPS_PROXY
ARG HTTP_PROXY
ARG NO_PROXY

WORKDIR /app

# Console-specific deps — FastAPI, uvicorn, pyyaml, qdrant-client (для /api/stats).
COPY services/console/requirements.txt /tmp/console-requirements.txt
RUN pip install --no-cache-dir -r /tmp/console-requirements.txt

# Установить morag БЕЗ indexing-extras: console больше не запускает cli.main index
# в subprocess'е. Управление индексацией делегируется control-plane'у в morag-indexer
# через HTTP (см. services/console/indexer_client.py). Получаем тонкий image ~250MB
# вместо ~3.5GB.
COPY pyproject.toml /opt/morag/pyproject.toml
COPY src /opt/morag/src
RUN pip install --no-cache-dir /opt/morag/

COPY services/console /app/services/console
RUN touch /app/services/__init__.py

ENV PYTHONPATH=/app:/opt/morag
ENV MORAG_CONFIG_PATH=/data/config.yml
ENV MORAG_INDEXER_URL=http://morag-indexer:9090

EXPOSE 8000

CMD ["uvicorn", "services.console.app:app", "--host", "0.0.0.0", "--port", "8000"]
