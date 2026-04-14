#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# M-Flow container bootstrap
# ---------------------------------------------------------------------------
set -euo pipefail

log() { printf '[m-flow] %s\n' "$*"; }
die() { printf '[m-flow] FATAL: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Schema migration
# ---------------------------------------------------------------------------
migrate_db() {
    log "Applying schema migrations …"
    local out rc
    set +e; out=$(alembic upgrade head 2>&1); rc=$?; set -e

    if [ $rc -eq 0 ]; then
        log "Schema up to date."
        return
    fi

    # Idempotent: ignore "already exists" on restarts
    case "$out" in
        *UserAlreadyExists*|*"already exists"*)
            log "Seed data present — skipping."
            return ;;
    esac

    log "Alembic failed (rc=$rc). Falling back to direct init …"
    log "$out"
    python -m m_flow.core.domain.operations.setup \
        || die "Database initialisation failed"
    log "Direct init OK."
}

migrate_db

# ---------------------------------------------------------------------------
# 2. Server
# ---------------------------------------------------------------------------
ADDR="0.0.0.0"
PORT="${HTTP_PORT:-8000}"
MODULE="m_flow.api.client:app"

common_flags=(
    --workers 1
    --worker-class uvicorn.workers.UvicornWorker
    --timeout 36000
    --bind "${ADDR}:${PORT}"
    --access-logfile -
    --error-logfile -
)

launch() {
    local level="$1"; shift
    log "Starting Gunicorn (${ENVIRONMENT:-production}, log=$level) on :${PORT}"
    exec gunicorn "${common_flags[@]}" --log-level "$level" "$@" "$MODULE"
}

# Fallback: if gunicorn is not installed, use uvicorn directly
if ! command -v gunicorn &>/dev/null; then
    log "WARNING: gunicorn not found (install 'deploy' extra), falling back to uvicorn"
    exec uvicorn "$MODULE" --host "$ADDR" --port "$PORT"
fi

# Optional: attach remote debugger before starting
if [[ "${ENVIRONMENT:-production}" =~ ^(dev|local)$ ]]; then
    if [[ "${DEBUG:-false}" == "true" ]]; then
        DBG_PORT="${DEBUG_PORT:-9230}"
        log "Debugpy listening on :${DBG_PORT} — waiting for client …"
        exec debugpy --wait-for-client --listen "${ADDR}:${DBG_PORT}" \
             -m gunicorn "${common_flags[@]}" --log-level debug --reload "$MODULE"
    fi
    launch debug --reload
else
    launch warning
fi
