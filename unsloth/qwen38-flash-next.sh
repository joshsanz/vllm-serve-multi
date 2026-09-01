#!/bin/bash
# Qwen3.8-Flash-Next — 177B params / 6B active + 51B n-gram embed / 4B MTP
# Recommended from https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
# Downloaded quant in cache is UD-IQ4_XS (~93.7 GB per HF card). Matches
# unified memory — ~93.7 GB IQ4_XS, comfortable headroom. Thinking
# mode is on by default; sampling uses HF-recommended thinking-mode params.
#
# MTP speculative decoding: as of unsloth studio 2026.8.22, studio's own
# `--speculative-type mtp` doesn't work for this model — it looks for an MTP
# head baked into the main quant, doesn't find one (the head ships as a
# separate sidecar GGUF under MTP/ in the repo), and silently falls back to
# n-gram. `--hf-repo-draft ...:quant` also fails — llama.cpp's quant matcher
# doesn't look in the repo's MTP/ subdir. What works: hand llama-server an
# explicit filesystem path to the sidecar head via the long-form
# --model-draft (unsloth run forwards unrecognized args verbatim), plus
# --spec-type draft-mtp to select the MTP graph. The head borrows token-embd
# + output proj from the main model. n_max is set via env var because
# unsloth swallows --spec-draft-n-max as its own flag. Measured 1.43x decode
# speedup at concurrency 1, ~58% draft acceptance — see
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

LLAMA_ARG_SPEC_DRAFT_N_MAX=2 unsloth run \
    --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS \
    --parallel 4 \
    -c 262144 \
    --cache-type-k f16 --cache-type-v f16 \
    --threads 20 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 \
    --chat-template-kwargs '{"reasoning_effort":"medium"}' \
    --model-draft "$DRAFT_HEAD" \
    --spec-type draft-mtp \
    -H 0.0.0.0 -p 8000
