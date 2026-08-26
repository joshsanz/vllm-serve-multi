#!/bin/bash
# Qwen3.8-Flash-Next — 177B params / 6B active + 51B n-gram embed / 4B MTP
# Recommended from https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
# Downloaded quant in cache is UD-IQ4_XS (~93.7 GB per HF card). Matches
# unified memory — ~93.7 GB IQ4_XS, comfortable headroom. Thinking
# mode is on by default; sampling uses HF-recommended thinking-mode params.
# Uses MTP speculative draft (model has 1 MTP layer) matching qwen38-27b.sh.

unsloth run \
    --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS \
    --speculative-type mtp \
    --spec-draft-n-max 3 \
    --parallel 4 \
    -c 262144 \
    --cache-type-k f16 --cache-type-v f16 \
    --threads 20 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 \
    --chat-template-kwargs '{"reasoning_effort":"medium"}' \
    -H 0.0.0.0 -p 8000
