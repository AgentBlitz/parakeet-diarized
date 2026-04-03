# Parakeet one-click launcher
# Kills any existing instance on port 8000, starts the server in WSL,
# and ensures llama.cpp LLM server is running on port 8003

$wslPath = "/mnt/c/_Dev/parakeet-diarized"

# Start llama.cpp server (entity extraction LLM) if not already running
$llamaRunning = docker ps --filter "name=llama-server" --format "{{.Names}}" 2>$null
if ($llamaRunning -ne "llama-server") {
    Write-Host "Starting llama.cpp LLM server on port 8003..."
    wsl bash -c "cd '$wslPath' && bash start_llm.sh"
} else {
    Write-Host "llama.cpp LLM server already running on port 8003."
}

# Start parakeet API server
wsl bash -c "fuser -k 8000/tcp 2>/dev/null; cd '$wslPath' && bash run.sh"
