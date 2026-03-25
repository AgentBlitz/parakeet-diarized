# Parakeet one-click launcher
# Kills any existing instance on port 8000, then starts the server in WSL

$wslPath = "/mnt/c/_Dev/parakeet-diarized"

wsl bash -c "pkill -f '[u]vicorn main:app' 2>/dev/null; cd '$wslPath' && bash run.sh"
