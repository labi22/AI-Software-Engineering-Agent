# Multi-stage production Dockerfile
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir ".[postgres,parsers]"

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed wheels/packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create unprivileged application user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

COPY src/ /app/src/
COPY pyproject.toml README.md /app/

RUN pip install --no-cache-dir -e .

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_MODULE=ai_software_engineering_agent.app:app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "ai_software_engineering_agent.app:app", "--host", "0.0.0.0", "--port", "8000"]
