#!/bin/bash
# Standalone launcher for nemotron-3-super — NOT part of docker-compose.yml.
#
# Removed from the compose stack: on this box (DGX Spark / GB10, 121.63GiB
# unified CPU+GPU memory), NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4's NVFP4 MoE
# marlin weight-repack step peaks at ~108GiB of GPU memory. That leaves too
# little margin against the 121.63GiB total once ANY sibling vLLM process
# (gemma4, ornith) is also resident — even just their sleep-mode residual
# floor (observed ~13-17GiB per asleep engine) is enough to push the repack
# into `CUDA error: out of memory`. Confirmed by testing: identical failure
# at --gpu-memory-utilization 0.85 AND 0.75 (that flag does not bound this
# transient spike). Even with gemma4/ornith fully stopped (not just asleep),
# the 74.8GiB checkpoint size against ~121GiB system RAM caused the box to
# swap to disk during load — this model is marginal-to-unviable on this
# hardware, full stop, not a config bug.
#
# Keeping this script (not the compose service) preserves the exact tested
# settings in case this ever runs on bigger hardware, or you want to try it
# standalone with the entire box dedicated to it. To run:
#   1. docker compose down   (frees host port 8000 AND the whole GPU/unified
#      memory pool — this container needs ALL of it, not just "most")
#   2. HF_TOKEN=... ./nemotron-3-super.sh
#   3. Watch `docker logs -f nemotron-3-super` — cold load is ~10-15 min
#      (weight load + NVFP4 marlin repack), assuming it doesn't OOM/swap.

docker run \
  --name nemotron-3-super \
  -d \
  --gpus all \
  --restart unless-stopped \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)/super_v3_reasoning_parser.py:/app/super_v3_reasoning_parser.py" \
  -v "$(pwd)/healthcheck.py:/healthcheck.py:ro" \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_SERVER_DEV_MODE=1 \
  --health-cmd="python3 /healthcheck.py" \
  --health-interval=10s \
  --health-timeout=30s \
  --health-start-period=20m \
  --health-retries=6 \
  vllm/vllm-openai:v0.24.0-ubuntu2404 \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.75 \
    --enable-chunked-prefill \
    --max-num-seqs 4 \
    --max-model-len 1000000 \
    --moe-backend marlin \
    --mamba_ssm_cache_dtype float32 \
    --quantization fp4 \
    --enable-sleep-mode \
    --speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
    --reasoning-parser-plugin /app/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
