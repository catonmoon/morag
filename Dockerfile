FROM python:3.12-slim

WORKDIR /app

# Poetry без создания virtualenv — ставим прямо в системный Python
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir poetry

# Зависимости отдельным слоем для кэша
COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev --no-root

# Скачать tiktoken-энкодинг во время сборки, чтобы не делать это при запуске
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Исходный код
COPY src/ ./src/
COPY cli/ ./cli/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "cli/main.py"]
CMD ["index"]
