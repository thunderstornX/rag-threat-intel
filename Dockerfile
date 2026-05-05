# syntax=docker/dockerfile:1.7
# Multi-stage build for the API. The pgvector + Ollama services are
# stock images; this Dockerfile is just the FastAPI tier.

FROM python:3.12-slim AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN groupadd -r app -g 10001 && useradd -r -g app -u 10001 -d /app -s /usr/sbin/nologin app
COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=app:app . /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
