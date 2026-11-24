#!/bin/bash
# Qwen3.8-Flash-Next — 177B params / 6B active + 51B n-gram embed / 4B MTP.
# UD-IQ2_XS (~75.8 GB, Unsloth Dynamic V3.1 re-upload of 2026-11-04): leaves room in the 121 GB
# unified pool for a 256k KV cache, the MTP head and d3-flash (systemone, :8001) running alongside.
# Previously UD-IQ4_XS (~93.7 GB); UD-Q4_K_XL (upstream's pick) does not fit.
#
# Text-only: --no-mmproj drops the ~1.1 GiB vision projector; --parallel 3 (was 4) trims KV/recurrent
# state in the shared pool. --cache-ram 2048 and --ctx-checkpoints 10 cap the growable host-side
# caches (prompt cache + checkpoints) at ~5 GiB worst case so they cannot OOM the 121 GB pool.
#
# MTP spec decoding, per unsloth.ai/docs/models/qwen3.8-next:
#   llama-server -hf ...:<quant> -md <repo>/MTP/mtp-...-shared-Q8_0.gguf \
#     --spec-type draft-mtp --spec-draft-n-max N
# The head is a separate sidecar under MTP/ (not in the main quant), so
# studio's --speculative-type mtp / --hf-repo-draft can't find it. We resolve
# the cached sidecar and pass it via --model-draft (unsloth run forwards it
# verbatim). n_max via env var: studio swallows --spec-draft-n-max.
# n_max 2 measured best here (1.65x / 72.5% accept, c=1); see
# ../vllm/MTP_BENCHMARKS.md (2026-09-01).

set -euo pipefail

DRAFT_REPO=unsloth/Qwen3.8-Flash-Next-GGUF
DRAFT_FILE=MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

# Prefer an already-cached copy; only hit the network if it isn't there.
DRAFT_HEAD="$(ls -t "$HOME"/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/*/"$DRAFT_FILE" 2>/dev/null | head -1 || true)"
if [ -z "${DRAFT_HEAD:-}" ] || [ ! -f "$DRAFT_HEAD" ]; then
    DRAFT_HEAD="$(hf download --quiet "$DRAFT_REPO" "$DRAFT_FILE")"
fi
if [ ! -f "$DRAFT_HEAD" ]; then
    echo "qwen38-flash-next: could not resolve MTP draft head '$DRAFT_FILE' (got '$DRAFT_HEAD')" >&2
    exit 1
fi
echo "qwen38-flash-next: MTP draft head -> $DRAFT_HEAD" >&2

LLAMA_ARG_SPEC_DRAFT_N_MAX=3 unsloth run \
    --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ2_XS \
    --parallel 3 \
    --no-mmproj \
    --cache-ram 2048 \
    --ctx-checkpoints 10 \
    -c 262144 \
    --cache-type-k f16 --cache-type-v f16 \
    --threads 20 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 \
    --chat-template-kwargs '{"reasoning_effort":"medium"}' \
    --model-draft "$DRAFT_HEAD" \
    --spec-type draft-mtp \
    -H 0.0.0.0 -p 8000
