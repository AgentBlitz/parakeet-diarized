# Parakeet one-click launcher
# Kills any existing instance on port 8000, then starts the server in WSL
# Sets up port proxy so remote machines can reach WSL2

$wslPath = "/mnt/c/_Dev/parakeet-diarized"
$port = 8000

# --- Port proxy: forward host:8000 → WSL:8000 (WSL2 IP changes on reboot) ---
# Requires admin — spawns an elevated sub-process (one UAC prompt)
$wslIp = (wsl -e hostname -I).Trim().Split()[0]
if ($wslIp) {
    $proxyCmd = "netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 2>`$null; netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIp"
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -Command $proxyCmd" -Wait -WindowStyle Hidden
    Write-Host "Port proxy: 0.0.0.0:$port -> ${wslIp}:$port" -ForegroundColor Green
} else {
    Write-Host "WARNING: Could not get WSL IP - remote access won't work" -ForegroundColor Yellow
}

wsl bash -c "pkill -f '[u]vicorn main:app' 2>/dev/null; cd '$wslPath' && bash run.sh"
