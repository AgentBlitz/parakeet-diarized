# Parakeet one-click launcher
# Kills any existing instance on port 8000, then starts the server in WSL

$wslPath = "/mnt/c/_Dev/parakeet-diarized"

wsl bash -c "fuser -k 8000/tcp 2>/dev/null; cd '$wslPath' && bash run.sh"
