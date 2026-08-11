#!/bin/bash

unsloth run \
  --model unsloth/DeepSeek-V4-Flash-0731-GGUF:UD-Q2_K_XL \
  --temp 1.0 --top-p 0.95 --min-p 0.01 \
  -ngl 999 --no-mmap \
  --threads 20 \
  --batch-size 2048 --ubatch-size 2048 \
  -c $((262144 + 32768 + 32768)) --parallel 3 --kv-unified \
  -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
  -H 0.0.0.0 -p 8000 \
  --disable-tools

