# cortex-agent — multi-stage, non-root production image.
# Lightweight: no model weights, no datasets — models load at runtime via env.
# (TinyBrain's torch stack is intentionally NOT installed here; this image runs
#  the agent with the anthropic/hf/mock backends. See requirements-train.txt.)

# ---- Stage 1: builder — install deps into a venv ---------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for any wheels that need compiling (kept out of the final image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY cortex ./cortex
RUN pip install --no-cache-dir .

# ---- Stage 2: runtime — slim, non-root -------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    CORTEX_HOST=0.0.0.0 \
    CORTEX_PORT=8000 \
    CORTEX_DATABASE_URL=sqlite+aiosqlite:////app/.cortex/cortex.db

# curl is used by the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the prebuilt virtualenv and the app.
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY . .

# Non-root user; writable state dir.
RUN useradd --create-home --uid 1000 cortex \
    && mkdir -p /app/.cortex \
    && chown -R cortex:cortex /app
USER cortex

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Default: serve the API + web UI. Override for the CLI or the worker, e.g.:
#   docker run --rm cortex-agent cortex run "Calculate 21 * 2"
#   docker run --rm cortex-agent arq cortex.worker.worker_settings.WorkerSettings
CMD ["uvicorn", "cortex.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
