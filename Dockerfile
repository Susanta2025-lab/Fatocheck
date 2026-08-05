# =========================================================
# Base Image
# =========================================================

FROM python:3.10-slim

# =========================================================
# Environment Variables
# =========================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV MODEL_TYPE=xgboost
ENV LOG_LEVEL=INFO
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# =========================================================
# Set Working Directory
# =========================================================

WORKDIR /app

# =========================================================
# Install dependencies (build tools removed after pip install)
# =========================================================

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y gcc build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# Copy Project Files
# =========================================================

COPY . .

# =========================================================
# Runtime user and cache directories
# =========================================================

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p .cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser

# =========================================================
# Expose Port
# =========================================================

EXPOSE 8000

# =========================================================
# Health Check
# =========================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

# =========================================================
# Run FastAPI Application
# =========================================================

CMD ["sh", "-c", "uvicorn api.app:app --host ${API_HOST} --port ${API_PORT} --workers 1"]
