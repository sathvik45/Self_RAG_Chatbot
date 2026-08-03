# syntax=docker/dockerfile:1

# ---------- builder: install Python deps into an isolated prefix ----------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY api ./api
COPY config ./config
COPY src ./src
COPY static ./static
COPY data ./data

# Bake the embedding model download and the FAISS index build into the image
# itself, so the container starts instantly with no cold-start network call
# or embedding computation (and no HF rate-limit risk at container startup).
RUN python -c "from src.self_rag.ingestion.build_index import build_index; build_index()"
ENV HF_HUB_OFFLINE=true

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
