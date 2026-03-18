# STAGE 1: Builder
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# STAGE 2: Final Runtime
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy only the application source
COPY src/ ./src/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

RUN addgroup --system appgroup && adduser --system --group appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "audit_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
