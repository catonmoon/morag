FROM ghcr.io/open-webui/pipelines:latest

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY morag.py /app/pipelines/morag.py
