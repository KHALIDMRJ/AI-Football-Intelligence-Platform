# syntax=docker/dockerfile:1.7
#
# Multi-stage image for the Football AI Intelligence Platform.
#
# Why multi-stage
#   - Build stage pulls the full Python toolchain (gcc, build-essential) so
#     wheels that need compilation (pydantic-core, xgboost, catboost) can
#     be built from source when no prebuilt wheel exists.
#   - Runtime stage is python:slim + curl only — no compiler, no apt cache
#     in the final image. Typical resulting image is ~1.2 GB (mostly
#     numpy/scikit/xgboost/torch footprint), none of which we can trim
#     without losing model-inference capability.
#
# Why we install the project (pip install .) instead of a hand-maintained
# dep list: the previous Dockerfile froze the list at Phase 3 and went stale
# through Phases 6-10 (missing redis, arq, prometheus_client, anthropic,
# passlib, python-jose, ...). Anchoring on pyproject.toml removes the
# single most common production-surprise source: "works on laptop, crashes
# in container because an import isn't there."
#
# Build:
#   docker build -t football-ai:latest .
# Run:
#   docker run --rm -p 8000:8000 football-ai:latest

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy the project manifest + source so `pip install .` resolves the package.
# We do NOT copy data/ or models/ here — those are mounted at runtime.
COPY pyproject.toml README.md ./
COPY football_ai/ ./football_ai/

RUN pip install --upgrade pip wheel setuptools && \
    pip install --prefix=/install .


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Football AI Platform" \
      org.opencontainers.image.description="Tactical analytics + AI prediction API (FastAPI)" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO

# curl is the healthcheck client; tini gives us proper PID-1 signal handling
# so `docker stop` propagates SIGTERM to uvicorn instead of timing out to SIGKILL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* && \
    groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Copy installed site-packages + console scripts from the build stage.
COPY --from=builder /install /usr/local
# Copy application code. Configs/scripts are stable and small; data/ and
# models/ are excluded via .dockerignore and mounted at runtime.
COPY --chown=appuser:appuser football_ai/ ./football_ai/
COPY --chown=appuser:appuser configs/     ./configs/
COPY --chown=appuser:appuser scripts/     ./scripts/
COPY --chown=appuser:appuser alembic/     ./alembic/
COPY --chown=appuser:appuser alembic.ini  ./alembic.ini

# Create the directories the app writes to and hand them to the non-root user.
RUN mkdir -p /app/data/raw /app/data/processed /app/data/features /app/models /app/logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# /health/ready exercises the real dependencies (DB + cache). Liveness alone
# would let a pod with a broken DB keep accepting traffic.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "football_ai.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
