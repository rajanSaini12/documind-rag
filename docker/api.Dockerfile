# FastAPI backend: ingestion, retrieval and answer generation.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Install dependencies first so this layer is cached when only the source changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[hf]"

RUN useradd --create-home appuser \
    && mkdir -p /data/chroma /models \
    && chown -R appuser /data /models
USER appuser

ENV CHROMA_DIR=/data/chroma
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "documind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
