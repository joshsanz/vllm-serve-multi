# Inkling-Small is a 276B-total/12B-active MoE — Unsloth's own docs example
# offloads all but 2 layers to CPU (`--n-gpu-layers 2`), but that's written
# for a discrete-GPU + separate-RAM box. This box's GB10 unified memory pool
# doesn't have that split, so it's left off and Unsloth's own default
# --gpu-memory-mode auto handles placement. IQ3-XXS weights alone are
# ~105GB against this box's ~121.63GiB unified pool — very tight, watch
# for an OOM on first run.

unsloth run \
    --model unsloth/Inkling-Small-GGUF:UD-IQ3_XXS \
    -c 32768 \
    --threads 20 \
    --temperature 1.0 --top-p 1.0 --min-p 0.0 \
    -H 0.0.0.0 -p 8000
