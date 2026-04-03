#!/usr/bin/env bash
# Stop llama.cpp server container
docker stop llama-server && docker rm llama-server
