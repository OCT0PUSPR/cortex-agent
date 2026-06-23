# cortex-agent container image.
# Lightweight: no model weights, no datasets — models load at runtime via env.
FROM python:3.11-slim

# Avoid interactive prompts; sane Python defaults.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CORTEX_HOST=0.0.0.0 \
    CORTEX_PORT=8000

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project and install it.
COPY . .
RUN pip install --no-cache-dir .

# Non-root user for safety.
RUN useradd --create-home --uid 1000 cortex \
    && mkdir -p /app/.cortex \
    && chown -R cortex:cortex /app
USER cortex

EXPOSE 8000

# Default: serve the API + web UI. Override CMD for the CLI, e.g.:
#   docker run --rm cortex-agent cortex run "Calculate 21 * 2"
CMD ["uvicorn", "cortex.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
