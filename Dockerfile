# Hopsworks requires Python < 3.14. linux/amd64 is pinned because hops-deltalake
# (part of hopsworks[python]) ships no linux/arm64 wheel; Docker Desktop on Apple
# Silicon runs the image via Rosetta.
FROM --platform=linux/amd64 python:3.11-slim

# gcc is needed to build twofish (a Hopsworks transitive dependency)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1

CMD ["python", "src/feature_pipeline.py"]
