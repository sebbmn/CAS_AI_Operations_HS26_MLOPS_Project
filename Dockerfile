# Hopsworks requires Python < 3.14
FROM python:3.11-slim

# gcc is needed to build a few of the Hopsworks transitive dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1

CMD ["python", "src/feature_pipeline.py"]
