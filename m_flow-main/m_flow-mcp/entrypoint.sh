#!/bin/bash
# M-flow MCP Server Entrypoint
# Initializes and starts the Model Context Protocol server

set -euo pipefail

echo "[M-flow MCP] Initializing server..."
echo "[M-flow MCP] Python: $(python --version)"
echo "[M-flow MCP] Transport: ${TRANSPORT_MODE:-sse}"

# Environment setup
export MFLOW_LOG_LEVEL="${MFLOW_LOG_LEVEL:-INFO}"

# Validate configuration
python -c "import m_flow; print(f'M-flow {m_flow.__version__} loaded')" || {
    echo "[ERROR] Failed to import m_flow — check installation"
    exit 1
}

case "${1:-serve}" in
    serve)
        echo "[M-flow MCP] Starting server..."
        # 传递环境变量到命令行参数
        exec python -m src.server \
            --transport "${TRANSPORT_MODE:-sse}" \
            --host "0.0.0.0" \
            --port "${MCP_PORT:-8000}" \
            --log-level "${MCP_LOG_LEVEL:-info}"
        ;;
    test)
        echo "[M-flow MCP] Running test client..."
        exec python -m src.test_client
        ;;
    *)
        echo "Usage: $0 {serve|test}"
        exit 1
        ;;
esac
