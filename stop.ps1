# Parakeet one-click stopper
# Kills the API server on port 8000 and stops the llama.cpp LLM container

wsl bash -c "fuser -k 8000/tcp 2>/dev/null && echo 'Parakeet API stopped.' || echo 'Nothing was running on port 8000.'"

# Stop llama.cpp server
$llamaRunning = docker ps --filter "name=llama-server" --format "{{.Names}}" 2>$null
if ($llamaRunning -eq "llama-server") {
    Write-Host "Stopping llama.cpp LLM server..."
    docker stop llama-server | Out-Null
    docker rm llama-server | Out-Null
    Write-Host "llama.cpp LLM server stopped."
} else {
    Write-Host "llama.cpp LLM server was not running."
}
