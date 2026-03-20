#!/bin/bash
set -e

# Start the FastAPI API server in background
echo "Starting FastAPI API server on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Graceful shutdown: kill API server when container stops
trap "kill $API_PID 2>/dev/null; wait $API_PID 2>/dev/null; exit 0" SIGTERM SIGINT

# Wait for API to be ready (model loading takes 2-3 minutes)
echo "Waiting for API server to load models..."
until curl -sf http://localhost:8000/health | python3 -c "import sys,json; sys.exit(0 if json.load(sys.stdin).get('model_loaded') else 1)" 2>/dev/null; do
    # Check API process is still alive
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "ERROR: API server process died during startup."
        exit 1
    fi
    sleep 5
    echo "  Still loading models..."
done
echo "API server ready."

# Ensure LLM model is pulled in Ollama (non-blocking — analysis is optional)
LLM_MODEL="${LLM_MODEL:-granite3.3:8b}"
LLM_BASE_URL="${LLM_BASE_URL:-http://ollama:11434}"
echo "Ensuring LLM model '$LLM_MODEL' is available in Ollama..."
curl -sf -X POST "${LLM_BASE_URL}/api/pull" \
  -d "{\"name\":\"${LLM_MODEL}\",\"stream\":false}" \
  --max-time 600 \
  && echo "LLM model ready." \
  || echo "WARNING: LLM model pull failed — meeting analysis will be unavailable."

# Start Gradio frontend in foreground
echo "Starting Gradio frontend on port 8001..."
exec python app.py
