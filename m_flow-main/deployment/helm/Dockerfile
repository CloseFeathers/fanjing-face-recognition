# M-Flow Helm deployment image
# Mirrors the root Dockerfile logic but uses Poetry for compatibility
# with Helm chart's dependency resolution strategy.

FROM python:3.11-slim AS builder

ARG DEBUG
ARG POETRY_EXTRAS="postgres neo4j notebook langchain llama-index huggingface ollama mistral groq deepeval evals posthog codegraph graphiti docs"

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

WORKDIR /opt/m_flow

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
    poetry install --extras "${POETRY_EXTRAS}" --no-root --without dev

# ── Application layer ────────────────────────────────────────────────────────

COPY m_flow/         ./m_flow/
COPY alembic.ini     ./alembic.ini
COPY alembic/        ./alembic/
COPY entrypoint.sh   ./entrypoint.sh

RUN chmod +x entrypoint.sh && sed -i 's/\r$//' entrypoint.sh

ENV PYTHONPATH=/opt/m_flow \
    DEBUG=${DEBUG}

ENTRYPOINT ["./entrypoint.sh"]
