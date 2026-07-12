FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt webapp/requirements.txt ./
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/webapp/requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN mkdir -p logs webapp/logs

WORKDIR /app/webapp
CMD ["python", "app.py", "10000"]