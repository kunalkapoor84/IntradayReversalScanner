FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -r requirements.txt -r webapp/requirements.txt

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN mkdir -p logs webapp/logs

WORKDIR /app/webapp
CMD gunicorn --workers 1 --threads 1 --timeout 120 --max-requests 50 --max-requests-jitter 10 --bind 0.0.0.0:$PORT app:app