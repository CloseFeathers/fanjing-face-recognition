# DEPRECATED — This Dockerfile references the removed distributed/ directory.
# The distributed worker module was renamed to mflow_workers/ but this
# Dockerfile was not updated. It is not referenced by docker-compose.yml.
# Do NOT use this file. It will be removed in a future release.
#
# M-flow Distributed Worker Image (legacy)

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install M-flow with distributed extras
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[distributed]"

# Copy worker code
COPY distributed/ ./distributed/
COPY m_flow/ ./m_flow/

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run worker via entrypoint
COPY distributed/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
