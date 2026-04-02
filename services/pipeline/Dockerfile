FROM ghcr.io/open-webui/pipelines:latest

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачать nltk stopwords (русские + английские) для BM25
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data')"

COPY morag.py /app/pipelines/morag.py
