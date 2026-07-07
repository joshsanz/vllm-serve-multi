MODELS: dict[str, str] = {
    "gemma4": "http://gemma4:8000",
    "Ornith-1.0": "http://ornith:8000",
}

# vLLM has no API to report its configured output cap, so this must be
# kept in sync by hand with each service's --override-generation-config
# '{"max_new_tokens": ...}' in docker-compose.yml -- it's surfaced here
# purely for the proxy's /v1/models response, not enforced by this code
# (enforcement happens engine-side).
MAX_OUTPUT_TOKENS: dict[str, int] = {
    "gemma4": 32768,
    "Ornith-1.0": 32768,
}
