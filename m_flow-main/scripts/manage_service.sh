#!/bin/bash
# M-flow Service Management Script
# Usage: ./scripts/manage_service.sh [start|stop|restart|status]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/mflow_backend.log"
FRONTEND_LOG="/tmp/mflow_frontend.log"

cd "$PROJECT_DIR"

start_backend() {
    echo "Starting M-flow backend..."
    if pgrep -f "uvicorn.*m_flow" > /dev/null; then
        echo "Backend is already running"
        return 1
    fi
    nohup uvicorn m_flow.api.client:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
    sleep 8
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend started successfully"
    else
        echo "Backend failed to start. Check $LOG_FILE"
        return 1
    fi
}

stop_backend() {
    echo "Stopping M-flow backend gracefully..."
    # Try graceful shutdown first (SIGTERM)
    pkill -TERM -f "uvicorn.*m_flow" 2>/dev/null
    sleep 3
    
    # Check if still running
    if pgrep -f "uvicorn.*m_flow" > /dev/null; then
        echo "Process still running, waiting..."
        sleep 5
    fi
    
    # Force kill only if necessary
    if pgrep -f "uvicorn.*m_flow" > /dev/null; then
        echo "Force stopping..."
        pkill -9 -f "uvicorn.*m_flow" 2>/dev/null
        sleep 2
    fi
    
    echo "Backend stopped"
}

start_frontend() {
    echo "Starting M-flow frontend..."
    if pgrep -f "node.*next.*3000" > /dev/null; then
        echo "Frontend is already running"
        return 1
    fi
    cd "$PROJECT_DIR/m_flow-frontend"
    nohup pnpm dev > "$FRONTEND_LOG" 2>&1 &
    sleep 8
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "Frontend started successfully"
    else
        echo "Frontend may need more time. Check $FRONTEND_LOG"
    fi
    cd "$PROJECT_DIR"
}

stop_frontend() {
    echo "Stopping M-flow frontend..."
    pkill -TERM -f "node.*next" 2>/dev/null
    sleep 2
    pkill -9 -f "node.*next" 2>/dev/null 2>&1
    echo "Frontend stopped"
}

status() {
    echo "=== M-flow Service Status ==="
    echo ""
    
    # Backend
    if pgrep -f "uvicorn.*m_flow" > /dev/null; then
        HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo '{"status":"unknown"}')
        echo "Backend:  RUNNING"
        echo "  Health: $HEALTH"
        echo "  PID:    $(pgrep -f 'uvicorn.*m_flow' | head -1)"
    else
        echo "Backend:  STOPPED"
    fi
    
    echo ""
    
    # Frontend
    if pgrep -f "node.*next" > /dev/null; then
        echo "Frontend: RUNNING"
        echo "  URL:    http://localhost:3000"
        echo "  PID:    $(pgrep -f 'node.*next' | head -1)"
    else
        echo "Frontend: STOPPED"
    fi
}

case "$1" in
    start)
        start_backend
        start_frontend
        ;;
    stop)
        stop_frontend
        stop_backend
        ;;
    restart)
        stop_frontend
        stop_backend
        sleep 2
        start_backend
        start_frontend
        ;;
    backend)
        case "$2" in
            start) start_backend ;;
            stop) stop_backend ;;
            restart) stop_backend; sleep 2; start_backend ;;
            *) echo "Usage: $0 backend [start|stop|restart]" ;;
        esac
        ;;
    frontend)
        case "$2" in
            start) start_frontend ;;
            stop) stop_frontend ;;
            restart) stop_frontend; sleep 2; start_frontend ;;
            *) echo "Usage: $0 frontend [start|stop|restart]" ;;
        esac
        ;;
    status)
        status
        ;;
    *)
        echo "M-flow Service Manager"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start     Start both backend and frontend"
        echo "  stop      Stop both services gracefully"
        echo "  restart   Restart both services"
        echo "  status    Show service status"
        echo ""
        echo "  backend [start|stop|restart]   Manage backend only"
        echo "  frontend [start|stop|restart]  Manage frontend only"
        ;;
esac
