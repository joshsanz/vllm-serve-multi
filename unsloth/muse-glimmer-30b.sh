#!/bin/bash

unsloth run \
    --model unsloth/Muse-Glimmer-30B-GGUF:UD-Q8_K_XL \
    --speculative-type dflash \
    --spec-draft-n-max 3 --spec-draft-p-min 0.2 \
    --parallel 4 \
    -c $((131072 * 4)) \
    --cache-type-k f16 --cache-type-v f16 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --min-p 0.0 \
    -H 0.0.0.0 -p 8000
