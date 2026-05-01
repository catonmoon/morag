FROM python:3.12-slim

WORKDIR /app

# Poetry без создания virtualenv — ставим прямо в системный Python
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir poetry

# Зависимости отдельным слоем для кэша
COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# Скачать tiktoken-энкодинг во время сборки, чтобы не делать это при запуске
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Скачать nltk stopwords (русские + английские) для BM25
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data')"

# Скачать токенизатор Qwen3-Embedding-4B заранее — HuggingFaceTokenCounter
# использует его для точного подсчёта токенов в SectionChunker/HybridChunker.
# Без этого первый запуск индексатора будет тянуть ~6 MB с HuggingFace.
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B')"

# Запретить runtime-вызовы к HuggingFace Hub. Транспортная связность с huggingface.co
# не нужна — токенайзер уже в локальном кэше выше. Без этого `transformers` ≥5
# при каждом from_pretrained() делает скрытые network-проверки (`_patch_mistral_regex`
# → `is_base_mistral` → `model_info`), которые падают при любых DNS-проблемах
# и роняют индексацию (см. cron-job DNS error 2026-05-01).
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Embedding-модели (dense и sparse) сами раздаются через HTTP:
#   - dense: Qwen3-Embedding-4B через Ollama на хосте (host.docker.internal:11434)
#   - sparse: embedder-gte сервис из docker-compose

# Исходный код
COPY src/ ./src/
COPY cli/ ./cli/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "cli/main.py"]
CMD ["index"]
