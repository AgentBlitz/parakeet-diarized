# Parakeet one-click stopper
# Kills the server process listening on port 8000

wsl bash -c "fuser -k 8000/tcp 2>/dev/null && echo 'Parakeet stopped.' || echo 'Nothing was running on port 8000.'"
