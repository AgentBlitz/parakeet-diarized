#!/bin/bash
# Kill all processes listening on port 8000
pids=$(lsof -ti tcp:8000 2>/dev/null)
if [ -n "$pids" ]; then
    echo "$pids" | xargs kill -9 2>/dev/null
    echo "Parakeet stopped (killed PIDs: $pids)"
else
    pkill -9 -f 'uvicorn main:app' 2>/dev/null \
        && echo "Parakeet stopped." \
        || echo "Nothing was running on port 8000."
fi
