FROM ghcr.io/open-webui/pipelines:latest

# Прокси для pip install (передаются через --build-arg или compose build.args).
# В runtime не нужны, поэтому не сохраняем как ENV.
ARG HTTPS_PROXY
ARG HTTP_PROXY
ARG NO_PROXY

# Pipeline-specific deps. requests + nltk + numpy уже есть как транзитивы
# (numpy от morag, requests от atlassian — но в core нет, оставляем явно).
# nltk здесь нужен только пока pipeline.py делает локальный fallback стемминг —
# после миграции на HttpGteSparseEmbedder можно будет убрать.
COPY services/pipeline/requirements.txt /tmp/pipeline-requirements.txt
RUN pip install --no-cache-dir -r /tmp/pipeline-requirements.txt

# Скачать nltk stopwords (русские + английские) для BM25-fallback в pipeline
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data')"

# Pre-cache tiktoken cl100k_base BPE — иначе при первом вызове tiktoken.get_encoding()
# в runtime он попытается скачать его с openaipublic.blob.core.windows.net. В air-gapped
# окружении это упадёт; на этапе build'а сеть доступна (через прокси), поэтому
# скачиваем сейчас и кладём в кеш контейнера.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN mkdir -p /opt/tiktoken_cache && \
    python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Установить morag (core deps: openai + httpx + numpy + transitive). НЕ ставим [indexing]
# extras — pipeline им не пользуется (нет чанкеров, sources, tokenizer'ов).
# Это даёт ~120MB site-packages вместо 3GB+.
COPY pyproject.toml /opt/morag/pyproject.toml
COPY src /opt/morag/src
RUN pip install --no-cache-dir /opt/morag/

# Файл переименован из morag.py чтобы не конфликтовать с installed morag-пакетом
# (Python иначе ищет 'morag' в sys.modules и находит pipeline-файл вместо пакета).
# В OWUI это будет отображаться по title-метаданным docstring'а, не по filename.
COPY services/pipeline/morag_pipeline.py /app/pipelines/morag_pipeline.py
