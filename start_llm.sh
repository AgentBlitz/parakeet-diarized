#!/usr/bin/env bash
# Start llama.cpp server with CUDA for entity extraction
# Serves Qwen 3.5 9B (Q4_K_M) on port 8003

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-Qwen3.5-9B-Q4_K_M.gguf}"
PORT="${LLM_PORT:-8003}"

# Stop existing container if running
docker stop llama-server 2>/dev/null && docker rm llama-server 2>/dev/null || true

docker run -d \
  --name llama-server \
  --gpus all \
  -p "${PORT}:8003" \
  -v "${SCRIPT_DIR}/models:/models:ro" \
  --restart unless-stopped \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model "/models/${MODEL}" \
  --host 0.0.0.0 \
  --port 8003 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --flash-attn

echo "llama.cpp server starting on port ${PORT}"
echo "Health check: curl http://localhost:${PORT}/health"
