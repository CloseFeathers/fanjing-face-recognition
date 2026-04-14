#!/usr/bin/env bash
# ============================================================================
# M-Flow Quickstart — one command to launch the full stack
# Usage:  ./quickstart.sh
# ============================================================================
set -euo pipefail

# ── Colours & helpers ───────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { printf "  ${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "  ${YELLOW}!${NC} %s\n" "$*"; }
err()  { printf "  ${RED}✗${NC} %s\n" "$*"; }
die()  { err "$*"; exit 1; }

sedi() {
    if [[ "$OSTYPE" == darwin* ]]; then sed -i '' "$@"; else sed -i "$@"; fi
}

banner() {
    echo ""
    printf "${CYAN}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║                                          ║"
    echo "  ║   M - F L O W   Q U I C K S T A R T     ║"
    echo "  ║                                          ║"
    echo "  ╚══════════════════════════════════════════╝"
    printf "${NC}\n"
}

ready_banner() {
    echo ""
    printf "${GREEN}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║  M-Flow is ready!                        ║"
    echo "  ║                                          ║"
    echo "  ║  Frontend : http://localhost:3000         ║"
    echo "  ║  API      : http://localhost:8000         ║"
    echo "  ║  API Docs : http://localhost:8000/docs    ║"
    echo "  ║                                          ║"
    echo "  ║  Press Ctrl+C to stop                    ║"
    echo "  ╚══════════════════════════════════════════╝"
    printf "${NC}\n"
}

# ── Safety checks ──────────────────────────────────────────────────────────

if [[ "${EUID:-$(id -u)}" == "0" ]]; then
    warn "Running as root is not recommended. Proceed? [y/N]"
    read -r ans; [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 0
fi

if [[ ! -f pyproject.toml ]]; then
    die "Not in M-Flow project root. Run this from the cloned repo directory."
fi

# ── Environment checks ────────────────────────────────────────────────────

banner
echo "  Checking environment..."

HAS_DOCKER=false
HAS_PYTHON=false

if command -v docker &>/dev/null && docker compose version &>/dev/null; then
    docker_ver=$(docker --version 2>/dev/null | head -1)
    compose_ver=$(docker compose version 2>/dev/null | head -1)
    ok "Docker: $docker_ver"
    ok "Compose: $compose_ver"
    if docker info &>/dev/null; then
        ok "Docker daemon running"
        HAS_DOCKER=true
    else
        warn "Docker installed but daemon not running"
    fi
else
    warn "Docker not found (install: https://docs.docker.com/get-docker/)"
fi

if command -v python3 &>/dev/null; then
    py_ver=$(python3 --version 2>/dev/null)
    py_minor=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [[ "$py_minor" -ge 10 ]]; then
        ok "$py_ver"
        HAS_PYTHON=true
    else
        warn "$py_ver (need 3.10+)"
    fi
else
    warn "Python 3 not found"
fi

if [[ "$HAS_DOCKER" == false && "$HAS_PYTHON" == false ]]; then
    die "Neither Docker nor Python 3.10+ found. Install one to continue."
fi

# Port checks
for port in 8000 3000; do
    if command -v lsof &>/dev/null; then
        if lsof -iTCP:$port -sTCP:LISTEN &>/dev/null; then
            warn "Port $port already in use"
        else
            ok "Port $port available"
        fi
    elif command -v ss &>/dev/null; then
        if ss -tlnp 2>/dev/null | grep -q ":$port "; then
            warn "Port $port already in use"
        else
            ok "Port $port available"
        fi
    fi
done

# ── Deployment mode selection ──────────────────────────────────────────────

echo ""
echo "  Select deployment mode:"
echo ""
if [[ "$HAS_DOCKER" == true ]]; then
    echo "    [1] Docker (recommended)"
    echo "        Backend + Frontend"
    echo ""
    echo "    [2] Docker + Neo4j"
    echo "        Backend + Frontend + Neo4j graph database"
    echo ""
    echo "    [3] Docker + PostgreSQL"
    echo "        Backend + Frontend + pgvector"
    echo ""
fi
if [[ "$HAS_PYTHON" == true ]]; then
    echo "    [4] Local Python (no Docker)"
    echo "        Requires Python 3.10+, uv, pnpm"
    echo ""
fi
if [[ "$HAS_DOCKER" == true ]]; then
    echo "    [5] Custom Docker profiles"
    echo ""
fi

read -p "  > " mode_choice
mode_choice="${mode_choice:-1}"

# ── API Key configuration ─────────────────────────────────────────────────

echo ""
echo "  Configuring environment..."

if [[ -f .env ]]; then
    read -p "  .env already exists. Overwrite? [y/N]: " overwrite
    if [[ "$overwrite" != "y" && "$overwrite" != "Y" ]]; then
        ok "Keeping existing .env"
    else
        cp .env.template .env
        ok "Created fresh .env from template"
    fi
else
    cp .env.template .env
    ok "Created .env from template"
fi
chmod 600 .env 2>/dev/null || true

# Check if API key is already set
current_key=$(grep "^LLM_API_KEY=" .env 2>/dev/null | head -1 | sed 's/LLM_API_KEY=//' | tr -d '"' || echo "")
if [[ "$current_key" == "your_api_key" || -z "$current_key" ]]; then
    echo ""
    printf "  Enter your LLM API key (OpenAI/Anthropic/etc): "
    read -s api_key
    echo ""
    if [[ -n "$api_key" ]]; then
        sedi "s|^LLM_API_KEY=.*|LLM_API_KEY=\"${api_key}\"|" .env
        ok "API key saved"
    else
        warn "No key entered — you can set LLM_API_KEY in .env later"
    fi
else
    ok "API key already configured"
fi

# Optional: LLM provider
read -p "  LLM provider [openai]: " provider
provider="${provider:-openai}"
sedi "s|^LLM_PROVIDER=.*|LLM_PROVIDER=\"${provider}\"|" .env

read -p "  LLM model [gpt-5-nano]: " model
model="${model:-gpt-5-nano}"
sedi "s|^LLM_MODEL=.*|LLM_MODEL=\"${provider}/${model}\"|" .env

ok "Configuration saved to .env"

# ── Launch functions ──────────────────────────────────────────────────────

wait_for_health() {
    local url="$1" timeout="$2" elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
        printf "."
    done
    echo ""
    warn "Timeout after ${timeout}s waiting for $url"
    return 1
}

open_browser() {
    local url="$1"
    sleep 1
    if command -v open &>/dev/null; then
        open "$url" 2>/dev/null
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$url" 2>/dev/null
    elif command -v wslview &>/dev/null; then
        wslview "$url" 2>/dev/null
    fi
}

launch_docker() {
    local profiles="$1"
    echo ""
    echo "  Starting M-Flow services (first run may take 5-10 minutes)..."
    docker compose $profiles up -d --build

    printf "  Waiting for backend"
    if wait_for_health "http://localhost:8000/health" 300; then
        ok "Backend ready"
    else
        warn "Backend may still be starting — check: docker compose logs m_flow"
    fi

    printf "  Waiting for frontend"
    if wait_for_health "http://localhost:3000" 120; then
        ok "Frontend ready"
    else
        warn "Frontend may still be building — check: docker compose logs frontend"
    fi

    ready_banner
    open_browser "http://localhost:3000"

    echo "  Streaming logs (Ctrl+C to stop all services)..."
    trap 'echo ""; echo "  Stopping containers..."; docker compose '"$profiles"' down; echo "  Done."; exit 0' INT TERM
    docker compose $profiles logs -f 2>/dev/null || true
}

launch_local() {
    echo ""

    # uv
    if ! command -v uv &>/dev/null; then
        echo "  Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        command -v uv &>/dev/null || die "uv installation failed. Install manually: https://docs.astral.sh/uv/"
    fi
    ok "uv $(uv --version 2>/dev/null || echo 'installed')"

    # pnpm
    if ! command -v pnpm &>/dev/null; then
        echo "  Installing pnpm..."
        if command -v npm &>/dev/null; then
            npm install -g pnpm
        elif command -v corepack &>/dev/null; then
            corepack enable && corepack prepare pnpm@latest --activate
        else
            die "pnpm/npm/corepack not found. Install Node.js: https://nodejs.org"
        fi
    fi
    ok "pnpm $(pnpm --version 2>/dev/null)"

    # Install backend
    echo "  Installing backend dependencies..."
    uv sync --extra api --extra dev
    ok "Backend dependencies"

    # Install frontend
    echo "  Installing frontend dependencies..."
    (cd m_flow-frontend && pnpm install --frozen-lockfile) 2>&1 | tail -1
    ok "Frontend dependencies"

    # Start backend
    echo "  Starting backend on :8000..."
    uv run python -m uvicorn m_flow.api.client:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!

    # Start frontend
    echo "  Starting frontend on :3000..."
    (cd m_flow-frontend && pnpm dev) &
    FRONTEND_PID=$!

    # Cleanup on exit
    trap 'echo ""; echo "  Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; wait 2>/dev/null; echo "  Done."; exit 0' INT TERM EXIT

    # Wait for health
    printf "  Waiting for backend"
    wait_for_health "http://localhost:8000/health" 90 && ok "Backend ready"

    printf "  Waiting for frontend"
    wait_for_health "http://localhost:3000" 60 && ok "Frontend ready"

    ready_banner
    open_browser "http://localhost:3000"

    # Block
    wait
}

# ── Execute selected mode ─────────────────────────────────────────────────

case "$mode_choice" in
    1) launch_docker "--profile ui" ;;
    2) launch_docker "--profile ui --profile neo4j" ;;
    3) launch_docker "--profile ui --profile postgres" ;;
    4) launch_local ;;
    5)
        echo ""
        echo "  Available profiles: ui, mcp, neo4j, postgres, chromadb, redis"
        read -p "  Enter profiles (space-separated): " custom
        profiles=""
        for p in $custom; do profiles="$profiles --profile $p"; done
        launch_docker "$profiles"
        ;;
    *) die "Invalid choice: $mode_choice" ;;
esac
