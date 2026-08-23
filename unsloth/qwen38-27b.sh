#!/bin/bash

unsloth run \
    --model unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL \
    --speculative-type mtp \
    --spec-draft-n-max 3 \
    --parallel 4 \
    -c 262144 \
    --cache-type-k f16 --cache-type-v f16 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 \
    --chat-template-kwargs '{"reasoning_effort":"medium"}' \
    -H 0.0.0.0 -p 8000
