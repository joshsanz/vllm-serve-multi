#!/bin/bash
# Install the GPU serving telemetry collector for telegraf (needs sudo).
# Re-run after editing gpu_serve_telemetry.py to redeploy.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dest=/opt/serve-telemetry
uv="$(command -v uv)"  # sudo's PATH usually lacks ~/.local/bin

sudo install -d -m 755 "$dest"
if [ ! -x "$dest/venv/bin/python" ]; then
    sudo "$uv" venv --quiet --python /usr/bin/python3 "$dest/venv"
fi
sudo "$uv" pip install --quiet --python "$dest/venv/bin/python" 'nvidia-ml-py>=13'
sudo install -m 644 "$here/gpu_serve_telemetry.py" "$dest/gpu_serve_telemetry.py"
sudo install -m 644 "$here/gpu_serve.conf" /etc/telegraf/telegraf.d/gpu_serve.conf

# Smoke test as the telegraf user (docker group, no access to $HOME).
sudo -u telegraf ONCE_WINDOW_S=1 "$dest/venv/bin/python" "$dest/gpu_serve_telemetry.py" --once >/dev/null
sudo systemctl restart telegraf
echo "installed; telegraf restarted"
