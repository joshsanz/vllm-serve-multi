#!/usr/bin/env python3
"""Docker HEALTHCHECK for vLLM sleep-mode services.
Healthy once the model has finished loading. On the FIRST successful check
only, puts the model to sleep (level 2) so the sequential boot chain's
`depends_on: condition: service_healthy` can safely let the next service
start loading without two models' memory coexisting on the shared GPU.

After that one-time transition, this script never calls /sleep again --
real wake/sleep while serving traffic is owned exclusively by
fastapi-router's swap_manager. A healthcheck that kept re-sleeping an
awake, actively-serving model on every 10s tick would race the router's
wakes and corrupt in-flight requests (observed empirically: it does).
"""
import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://localhost:8000"
SENTINEL = Path("/tmp/.initial-sleep-done")


def _get(path: str):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as resp:
        return json.load(resp)


def _post(path: str):
    req = urllib.request.Request(f"{BASE}{path}", method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=280) as resp:
        return resp.status


def main() -> int:
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=10)
    except Exception:
        return 1  # still loading, or not up yet

    if SENTINEL.exists():
        return 0  # steady state: reachable is healthy; sleep state is the router's business

    try:
        state = _get("/is_sleeping")
    except Exception:
        return 1  # dev-mode endpoints not up yet

    if not state.get("is_sleeping"):
        try:
            _post("/sleep?level=2")
        except Exception:
            return 1

    SENTINEL.touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
