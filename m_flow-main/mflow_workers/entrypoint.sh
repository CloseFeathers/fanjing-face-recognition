#!/bin/bash
# M-flow Distributed Worker Entrypoint
# Starts the message queue consumer for graph operations

set -euo pipefail

echo "[M-flow] Starting distributed worker..."
echo "[M-flow] Worker ID: ${WORKER_ID:-$(hostname)}"
echo "[M-flow] Queue: ${MFLOW_QUEUE:-default}"

# Apply database migrations if needed
python -m alembic upgrade head 2>/dev/null || true

# Launch the worker process
exec python -m distributed.entrypoint "$@"
