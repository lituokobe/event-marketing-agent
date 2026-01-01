#!/bin/sh
# entrypoint.sh - Minimal version

set -e

# Add project root to Python path
export PYTHONPATH=/app:$PYTHONPATH

echo "Starting AI services..."

# Start services
python models/ai_gateway_service.py &
GATEWAY_PID=$!

python models/ai_service.py &
SERVICE_PID=$!

echo "Services started:"
echo "  Gateway PID: $GATEWAY_PID"
echo "  AI Service PID: $SERVICE_PID"

# Shutdown handler
shutdown() {
    echo "Shutting down services..."
    kill $GATEWAY_PID $SERVICE_PID 2>/dev/null
    wait $GATEWAY_PID $SERVICE_PID 2>/dev/null
    echo "Services stopped"
    exit 0
}

trap shutdown 15 2

# Wait for processes
wait $GATEWAY_PID $SERVICE_PID