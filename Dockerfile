# ── Base image légère ──────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="OualidDR"
LABEL description="ResumeIQ — NLP Resume Screening App"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# ── Dépendances système minimales ─────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Code source ────────────────────────────────────────────────────
COPY app/   ./app/
COPY data/  ./data/
COPY run.py .
RUN echo "LOG_LEVEL=INFO" > .env
RUN mkdir -p artifacts/embeddings_cache

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]