#!/bin/bash
# Muse-Glimmer-30B — dense 29.6B causal LM + perception encoder, native
# 131,072-token context, [Local,Local,Local,Global] sliding-window attention
# (window 2048 on local layers) per the model card.
#
# UD-Q4_K_XL (~15.9 GB, Meta's own "K-Quant-Dynamic" recommendation) +
# DFlash2 drafter Q4_K_M (~1.6 GB): both fit comfortably in the box's 121 GB
# unified memory, leaving room for KV cache. -c matches the model's native
# 131072 context (kv-unified, shared across --parallel slots) — going
# higher just gets silently capped back down by studio with a warning, per
# serve memory dflash2-dspark-unsupported.
#
# DFlash2 speculative decoding, per incoai/Muse-Glimmer-30B-DFlash2-GGUF's
# README quick-start: --spec-type draft-dflash --spec-draft-n-max 15.
# Studio's own `--speculative-type dflash` auto-resolution doesn't let us
# pick a specific draft quant/repo, so we resolve the cached draft file
# ourselves and pass it via --model-draft, same pattern as
# qwen38-flash-next.sh's MTP head. n_max via env var: studio swallows
# --spec-draft-n-max on its own flag path.
#
# Two rounds of benchmarking on 2026-09-08 (bench/bench_llamacpp_timings.py,
# 6-file agentic-coding suite, see vllm/MTP_BENCHMARKS.md for full tables):
#   1. Base quant dominates: UD-Q4_K_XL beat UD-Q8_K_XL by ~62% mean decode
#      (24.81 vs 15.35 tok/s) at the same DFlash2 accept rate — this box's
#      single-stream decode is memory-bandwidth-bound, and Q4_K_XL costs
#      only ~0.2% degradation per Meta's model card. Prompted by a
#      third-party DGX Spark recipe reporting similar numbers
#      (sxuff/muse-glimmer-dgx-spark on GitHub). Q8_K_XL stays cached if the
#      extra quality is ever wanted.
#   2. Drafter choice is a smaller, second-order effect on UD-Q4_K_XL: swept
#      all 4 available drafters (bundled dflash-kquant.gguf + incoai's
#      Q4_K_M/Q8_0/BF16), all within ~11% of each other by mean. Q4_K_M
#      picked: essentially tied with the bundled drafter by median, no
#      outlier runs (the bundled drafter had one noisy 35.5 tok/s file).
#
# Confirmed working 2026-09-08 on unsloth build 10715 (llama.cpp commit
# 92cedc867) — the "wrong number of tensors; expected 81, got 58" load
# failure seen on 2026-08-20 is gone; draft loads and serves real
# draft_n/draft_n_accepted stats. See dflash2-dspark-unsupported memory.

set -euo pipefail

DRAFT_REPO=incoai/Muse-Glimmer-30B-DFlash2-GGUF
DRAFT_FILE=Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf

# Prefer an already-cached copy; only hit the network if it isn't there.
DRAFT_HEAD="$(ls -t "$HOME"/.cache/huggingface/hub/models--incoai--Muse-Glimmer-30B-DFlash2-GGUF/snapshots/*/"$DRAFT_FILE" 2>/dev/null | head -1 || true)"
if [ -z "${DRAFT_HEAD:-}" ] || [ ! -f "$DRAFT_HEAD" ]; then
    DRAFT_HEAD="$(hf download --quiet "$DRAFT_REPO" "$DRAFT_FILE")"
fi
if [ ! -f "$DRAFT_HEAD" ]; then
    echo "muse-glimmer-30b: could not resolve DFlash2 draft '$DRAFT_FILE' (got '$DRAFT_HEAD')" >&2
    exit 1
fi
echo "muse-glimmer-30b: DFlash2 draft -> $DRAFT_HEAD" >&2

LLAMA_ARG_SPEC_DRAFT_N_MAX=15 unsloth run \
    --model unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL \
    --parallel 4 \
    -c 131072 \
    --cache-type-k f16 --cache-type-v f16 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --min-p 0.0 \
    --model-draft "$DRAFT_HEAD" \
    --spec-type draft-dflash \
    -H 0.0.0.0 -p 8000
