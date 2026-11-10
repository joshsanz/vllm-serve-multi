"""HTTP server for Ripple-65B (NVFP4 MLP / BF16 attention) on a DGX Spark, with optional DFlash2.

Mirrors the launch config of Ricursive's spark.py (offline Engine) but serves the OpenAI API.
Env: RIPPLE_SPEC=1|0 (DFlash2 on/off), RIPPLE_DRAFT_TOKENS (default 16), RIPPLE_TARGET / RIPPLE_DRAFT (HF repos or local dirs),
extra CLI args are appended to the server args.
"""
import functools
import os
import shlex
import sys
from pathlib import Path

os.environ.setdefault("SGLANG_FLASHINFER_USE_PAGED", "1")
os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")

import torch
from huggingface_hub import snapshot_download
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import prepare_server_args

from model import RicursiveForCausalLM

import ripple_format

# Module-level so SGLang's spawned scheduler processes (which re-import this module to find
# run_worker) register the model and parsers too.
ModelRegistry.models["RicursiveForCausalLM"] = RicursiveForCausalLM
ripple_format.register()


def run_worker(*args, **kwargs):
    """The two numerical fixes spark.py applies on SM121."""
    from sglang.kernels.ops.speculative.dspark import dspark_accept
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.model_executor.model_runner import ModelRunner

    softmax = dspark_accept._flashinfer_softmax

    @functools.wraps(softmax)
    def safe_softmax(*a, **kw):
        kw["enable_pdl"] = False  # SM121 PDL softmax can produce NaNs.
        return softmax(*a, **kw)

    dspark_accept._flashinfer_softmax = safe_softmax
    load_model = ModelRunner.load_model

    @functools.wraps(load_model)
    def load(self):
        result = load_model(self)
        for _, module in self.model.named_modules():
            if isinstance(module, RadixAttention):
                # Quantized exports can carry FP8 KV scales although our KV is BF16.
                for kind in ("k", "v"):
                    scale = getattr(module, kind + "_scale", None)
                    if scale is not None:
                        scale.data.fill_(1.0)
                        setattr(module, kind + "_scale_float", 1.0)
        return result

    ModelRunner.load_model = load
    return run_scheduler_process(*args, **kwargs)


def main():
    target = os.environ.get("RIPPLE_TARGET", "ricursiveAI/Ripple-65B-Preview-NVFP4")
    draft = os.environ.get("RIPPLE_DRAFT", "ricursiveAI/Ripple-65B-Preview-DFlash2")
    spec = os.environ.get("RIPPLE_SPEC", "1") == "1"
    target_path = Path(target) if Path(target).is_dir() else Path(snapshot_download(target))
    spec_args = []
    if spec:
        draft_path = Path(draft) if Path(draft).is_dir() else Path(snapshot_download(draft))
        spec_args = [
            "--speculative-draft-model-path", str(draft_path),
            "--speculative-algorithm", "DFLASH", "--speculative-draft-model-quantization", "unquant",
            "--speculative-num-draft-tokens", os.environ.get("RIPPLE_DRAFT_TOKENS", "16"), "--speculative-draft-window-size", "16384",
            "--speculative-draft-attention-backend", "triton",
        ]
    args = prepare_server_args(
        ["--model-path", str(target_path), "--served-model-name", "ripple-65b",
         "--host", "0.0.0.0", "--port", "30000",
         "--chat-template", str(target_path / "chat_template.jinja")]
        + spec_args
        + shlex.split("""
            --trust-remote-code --dtype bfloat16 --kv-cache-dtype bfloat16 --tp-size 1
            --attention-backend flashinfer --fp4-gemm-backend flashinfer_cutlass
            --flashinfer-autotune-skip-ops fp8_gemm
            --context-length 131072 --max-total-tokens 139264 --mem-fraction-static 0.78
            --max-running-requests 1 --cuda-graph-bs-decode 1
            --cuda-graph-backend-prefill disabled --disable-radix-cache
            --weight-loader-disable-mmap --weight-loader-drop-cache-after-load
            --model-loader-extra-config '{"enable_multithread_load": false}'
            --random-seed 42 --watchdog-timeout 1800
            --tool-call-parser ripple --reasoning-parser ripple
        """)
        + sys.argv[1:]
    )
    from sglang.srt.entrypoints.http_server import launch_server
    launch_server(args, run_scheduler_process_func=run_worker)


if __name__ == "__main__":
    main()
