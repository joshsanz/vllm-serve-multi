# vllm-srun (vLLM Semantic Router model runtime) for aarch64 + CUDA (DGX Spark / GB10).
#
# Upstream only publishes a CPU image for arm64 (vllm-sr) and an amd64-only CUDA image
# (vllm-sr-cuda). The runtime itself is a pure-Python package, so we install it from a
# pinned upstream commit on top of the vLLM image, which already carries aarch64 CUDA
# PyTorch. Its CUDA accelerator is "implemented, unvalidated" upstream.
FROM vllm/vllm-openai:v0.33.0

ARG SR_REF=6dd029835fee5c80f864d442868a7be4102bf128

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN pip install --no-deps \
       "vllm-srun @ https://github.com/vllm-project/semantic-router/archive/${SR_REF}.tar.gz#subdirectory=src/model-runtime" \
    && pip install "numpy>=1.26" "safetensors>=0.4" "tokenizers>=0.20" "huggingface_hub>=0.25" \
       "starlette>=0.37" "uvicorn>=0.30" uvloop httptools prometheus_client pyyaml \
       pillow "opencv-python-headless>=5.0.0.93,<5.1" flash-linear-attention

EXPOSE 8100
ENTRYPOINT ["vllm-srun"]
