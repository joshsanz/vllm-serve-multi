"""Offline Ripple inference on DGX Spark: NVFP4 MLPs, BF16 attention + BF16 DFlash2."""

import asyncio
import fnmatch
import functools
import json
import os
import shlex
from pathlib import Path

# Keep the Spark kernel choices used by the original notebook.
os.environ["SGLANG_FLASHINFER_USE_PAGED"] = "1"
os.environ["FLASHINFER_USE_CUDA_NORM"] = "1"
os.environ.setdefault("MAX_JOBS", "2")
os.environ.setdefault("NVCC_THREADS", "1")

import torch
from huggingface_hub import snapshot_download
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import prepare_server_args

from model import RicursiveForCausalLM

import ripple_format
from ripple_transport import patch_request_cleanup

ModelRegistry.models["RicursiveForCausalLM"] = RicursiveForCausalLM
ripple_format.register()
patch_request_cleanup()


def checkpoint(value):
    """Accept a complete local checkpoint or download a Hugging Face repo."""
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    if str(value).startswith(("/", "~", ".", "gs://")):
        raise FileNotFoundError(value)
    return Path(snapshot_download(str(value), ignore_patterns=["*.mp4", "*.ipynb"]))


def check_checkpoints(target, draft=None):
    """Require NVFP4 MLPs, BF16 attention, and an unquantized DFlash2 draft."""
    config = json.loads((target / "config.json").read_text())
    if config.get("architectures") != ["RicursiveForCausalLM"]:
        raise ValueError("Target must be a RicursiveForCausalLM checkpoint")
    quant = config.get("quantization_config") or {}
    if (target / "hf_quant_config.json").exists():
        quant = json.loads((target / "hf_quant_config.json").read_text())["quantization"]
    # ModelOpt stores the same recipe as config_groups/ignore in config.json
    # and as group_size/exclude_modules in the optional legacy sidecar.
    group_size = quant.get("group_size")
    if "config_groups" in quant:
        groups = list(quant["config_groups"].values())
        if len(groups) != 1 or groups[0].get("targets") != ["Linear"]:
            raise ValueError("Expected one NVFP4 group targeting Linear layers")
        for kind in ("weights", "input_activations"):
            scheme = groups[0].get(kind, {})
            if any(scheme.get(k) != v for k, v in
                   {"type": "float", "num_bits": 4, "group_size": 16, "dynamic": False}.items()):
                raise ValueError(f"Expected static NVFP4 {kind}, group size 16")
        group_size = 16
    if quant.get("quant_algo") != "NVFP4" or group_size != 16:
        raise ValueError("Spark requires an NVFP4/BF16 ModelOpt checkpoint (MLP group size 16)")
    if quant.get("kv_cache_quant_algo") is not None or quant.get("kv_cache_scheme") is not None:
        raise ValueError("Spark requires an unquantized BF16 KV cache")
    if quant.get("quantized_layers"):
        raise ValueError("Per-layer mixed-precision overrides are not supported")
    excluded = quant.get("exclude_modules", quant.get("ignore", []))

    def require(name, bf16):
        is_excluded = any(fnmatch.fnmatchcase(name, pattern) for pattern in excluded)
        if is_excluded != bf16:
            raise ValueError(f"Expected {'BF16' if bf16 else 'NVFP4'} for {name}")

    for name in ("lm_head", "model.embed_tokens"):
        require(name, True)
    for index in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{index}"
        for name in ("gate", "up", "gate_up", "down"):
            require(f"{prefix}.mlp.{name}_proj", False)
        heads = ("q", "k", "v", "o", "g") if config.get("use_gated_attention", True) else ("q", "k", "v", "o")
        for name in heads:
            require(f"{prefix}.self_attn.{name}_proj", True)
    if draft is None:
        return
    draft_config = json.loads((draft / "config.json").read_text())
    if draft_config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("Draft must be a DFlash2DraftModel checkpoint")
    if draft_config.get("quantization_config") or (draft / "hf_quant_config.json").exists():
        raise ValueError("The DFlash2 draft must be unquantized BF16")


def run_worker(server_args, *args, **kwargs):
    """Install an explicitly selected adapter, then retain Spark compatibility fixes."""
    overrides = server_args.json_model_override_args or {}
    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    tree = overrides.get("ripple_spark_tree")
    if tree is not None:
        from ripple_tree import install
        install(tree, server_args)
    from sglang.kernels.ops.speculative.dspark import dspark_accept
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.model_executor.model_runner import ModelRunner

    softmax = dspark_accept._flashinfer_softmax

    @functools.wraps(softmax)
    def safe_softmax(*args, **kwargs):
        kwargs["enable_pdl"] = False  # SM121 PDL softmax can produce NaNs.
        return softmax(*args, **kwargs)

    dspark_accept._flashinfer_softmax = safe_softmax
    load_model = ModelRunner.load_model

    @functools.wraps(load_model)
    def load(self):
        result = load_model(self)
        for name, module in self.model.named_modules():
            if isinstance(module, RadixAttention):
                # Quantized exports can carry FP8 scales although our KV is BF16.
                for kind in ("k", "v"):
                    scale = getattr(module, kind + "_scale", None)
                    if scale is not None:
                        scale.data.fill_(1.0)
                        setattr(module, kind + "_scale_float", 1.0)
            weight = getattr(module, "weight", None)
            if self.is_draft_worker and weight is not None and weight.dtype != torch.bfloat16:
                raise ValueError(f"Draft weight {name} must be BF16")
        return result

    ModelRunner.load_model = load
    return run_scheduler_process(server_args, *args, **kwargs)


class Spark(Engine):
    """One local inference engine; no HTTP server or port configuration."""

    run_scheduler_process_func = staticmethod(run_worker)

    def __init__(self, target, draft=None, **options):
        self.max_batch_size = options.get("max_batch_size", 4)
        super().__init__(server_args=self.build_args(target, draft, **options))
        self._chat = OpenAIServingChat(self.tokenizer_manager, self.template_manager)

    @staticmethod
    def build_args(target, draft=None, *, speculative=True, decoding=None, max_batch_size=4,
                 context_length=131072, max_total_tokens=139264, draft_tokens=32,
                 prefix_cache=True, kernels="combined", deterministic=True,
                 attention_backend="triton", prefill_chunk_size=4096, graph_batches=None,
                 fp4_gemm_backend="flashinfer_cutlass", experimental_concurrency_8=False):
        # Target-only callers keep working without selecting a speculative decoder.
        if decoding is None:
            decoding = "tree" if speculative else "chain"
        if decoding not in ("chain", "tree"):
            raise ValueError("decoding must be chain or tree")
        if decoding == "tree" and not speculative:
            raise ValueError("Experimental tree decoding requires a DFlash2 draft")
        if decoding == "tree" and (not deterministic or attention_backend != "triton"):
            raise ValueError("Experimental tree decoding requires deterministic Triton attention")
        if speculative and draft is None:
            raise ValueError("Provide a DFlash2 draft or set speculative=False")
        if type(experimental_concurrency_8) is not bool:
            raise ValueError("experimental_concurrency_8 must be a boolean")
        batch_limit = 8 if experimental_concurrency_8 else 4
        if type(max_batch_size) is not int or not 1 <= max_batch_size <= batch_limit:
            raise ValueError(f"max_batch_size must be an integer from 1 through {batch_limit}; C5–8 requires experimental_concurrency_8=True")
        if experimental_concurrency_8 and decoding == "tree" and draft_tokens != 32:
            raise ValueError("Experimental C1–8 tree benchmarking requires draft_tokens=32")
        if type(context_length) is not int or not 1 <= context_length <= 131072:
            raise ValueError("context_length must be an integer from 1 through 131072")
        if type(max_total_tokens) is not int or max_total_tokens < context_length:
            raise ValueError("max_total_tokens must cover context_length; it is shared across requests")
        budgets = (32, 64, 128) if decoding == "tree" else (8, 16, 32)
        if type(draft_tokens) is not int or draft_tokens not in budgets:
            raise ValueError(f"draft_tokens must be one of {budgets} for {decoding} decoding")
        if kernels not in ("stock", "fused", "stream", "combined", "ported"):
            raise ValueError("kernels must be stock, fused, stream, combined or ported")
        if attention_backend not in ("flashinfer", "triton"):
            raise ValueError("attention_backend must be flashinfer or triton")
        if fp4_gemm_backend not in ("flashinfer_cutlass", "flashinfer_cudnn"):
            raise ValueError("fp4_gemm_backend must be flashinfer_cutlass or flashinfer_cudnn")
        if type(prefill_chunk_size) is not int or prefill_chunk_size < 4096 or prefill_chunk_size % 4096:
            raise ValueError("prefill_chunk_size must be a positive multiple of the native 4096-token alignment")
        graph_batches = list(range(1, max_batch_size + 1)) if graph_batches is None else list(graph_batches)
        if (not graph_batches or any(type(b) is not int or not 1 <= b <= batch_limit for b in graph_batches)
                or graph_batches != sorted(set(graph_batches)) or graph_batches[0] != 1
                or graph_batches[-1] < max_batch_size):
            raise ValueError(f"graph_batches must be increasing sizes 1–{batch_limit}, starting at 1 and covering max_batch_size")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError("Run this notebook on a DGX Spark / GB10 GPU")
        target = checkpoint(target)
        draft = checkpoint(draft) if speculative else None
        check_checkpoints(target, draft)
        chat_template = Path(__file__).with_name("chat_template.jinja")
        if not chat_template.is_file():
            chat_template = Path(__file__).resolve().parents[2] / "chat_template.jinja"
        if not chat_template.is_file():
            raise FileNotFoundError("Download chat_template.jinja alongside spark.py")
        overrides = {"ripple_spark_kernels": kernels}
        if decoding == "tree":
            # Experimental: native target-only regret gate remains FAIL (0.25 > 0.20).
            # This is the verification node budget; the draft backbone stays 16 tokens.
            overrides["ripple_spark_tree"] = {
                "nodes": draft_tokens, "draft_width": 16, "children": 16,
                "proposal_temperature": 1.5, "sampler": "native_eagle_target_only",
                "exact_thresholds": [1.0, 1.0],
            }
        if decoding == "tree" and experimental_concurrency_8:
            overrides["ripple_spark_tree"]["max_concurrency"] = 8
        spec_args = [
            "--speculative-draft-model-path", str(draft),
            "--speculative-algorithm", "DFLASH", "--speculative-draft-model-quantization", "unquant",
            "--speculative-num-draft-tokens", str(draft_tokens), "--speculative-draft-window-size", "16384",
            "--speculative-draft-attention-backend", "triton",
        ] if speculative else []
        args = prepare_server_args(
            [
                "--model-path",
                str(target),
                "--chat-template",
                str(chat_template),
            ]
            + spec_args + ["--context-length", str(context_length),
                           "--max-total-tokens", str(max_total_tokens),
                           "--max-running-requests", str(max_batch_size),
                           "--cuda-graph-bs-decode", *map(str, graph_batches),
                           "--attention-backend", attention_backend,
                           "--fp4-gemm-backend", fp4_gemm_backend,
                           "--chunked-prefill-size", str(prefill_chunk_size)]
            + ([] if prefix_cache else ["--disable-radix-cache"])
            + (["--enable-deterministic-inference"] if deterministic else [])
            + ["--json-model-override-args", json.dumps(overrides)]
            + shlex.split("""
                --trust-remote-code --dtype bfloat16 --kv-cache-dtype bfloat16 --tp-size 1
                --flashinfer-autotune-skip-ops fp8_gemm
                --mem-fraction-static 0.78
                --cuda-graph-backend-prefill disabled
                --weight-loader-disable-mmap --weight-loader-drop-cache-after-load
                --model-loader-extra-config '{"enable_multithread_load": false}'
                --random-seed 42 --watchdog-timeout 1800 --log-level warning
                --tool-call-parser ripple --reasoning-parser ripple
            """)
        )
        return args

    def complete(self, messages, **kwargs):
        """Return an OpenAI-style response with reasoning and tool calls, locally."""
        return self.complete_batch([messages], **kwargs)[0]

    def complete_batch(self, conversations, **kwargs):
        """Run up to max_batch_size conversations concurrently, preserving input order.

        SGLang owns request scheduling, independent KV state and EOS retirement. This is
        a concurrency limit; prompts and outputs may have different lengths.
        """
        conversations = list(conversations)
        if not 1 <= len(conversations) <= self.max_batch_size:
            raise ValueError(f"Provide 1 through {self.max_batch_size} conversations")
        requests = [ChatCompletionRequest(**{
            "model": "ripple", "messages": messages, "max_tokens": 256, "temperature": 0,
            "chat_template_kwargs": {"clear_thinking": False}, **kwargs,
        }) for messages in conversations]
        if any(request.stream for request in requests):
            raise ValueError("complete_batch() returns full responses; use generate() for raw streaming")

        async def run():
            # Wait for all submitted requests even if one fails; do not leave work in flight.
            return await asyncio.gather(
                *(self._chat.handle_request(request, None) for request in requests),
                return_exceptions=True,
            )

        responses = self.loop.run_until_complete(run())
        result = []
        for response in responses:
            if isinstance(response, BaseException):
                raise response
            if not isinstance(response, ChatCompletionResponse):
                raise ValueError(json.loads(response.body))
            result.append(response.model_dump(exclude_none=True))
        return result

    def chat(self, prompt, *, max_tokens=256, temperature=0, top_p=1):
        """Return final-answer text; use complete() to retain reasoning and tool calls."""
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        response = self.complete(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return response["choices"][0]["message"].get("content") or ""

    def chat_batch(self, prompts, *, max_tokens=256, temperature=1, top_p=.95):
        """Return one final-answer string per prompt; use complete_batch() for full outputs."""
        conversations = [[{"role": "user", "content": p}] if isinstance(p, str) else p
                         for p in prompts]
        responses = self.complete_batch(conversations, max_tokens=max_tokens,
                                        temperature=temperature, top_p=top_p)
        return [r["choices"][0]["message"].get("content") or "" for r in responses]


def main():
    """Serve the same model through SGLang's standard HTTP entrypoint."""
    import argparse
    from sglang.srt.entrypoints.http_server import launch_server

    parser = argparse.ArgumentParser(description="Serve Ripple on DGX Spark with SGLang")
    parser.add_argument("--target", default="ricursiveAI/Ripple-65B-Preview-NVFP4")
    parser.add_argument("--draft", default="ricursiveAI/Ripple-65B-Preview-DFlash2")
    parser.add_argument("--target-only", dest="speculative", action="store_false",
                        help="Run without DFlash2; do not download or load a draft")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch-size", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--experimental-concurrency-8", action="store_true",
                        help="Allow unqualified C5–8 benchmarks; tree decoding is limited to 32 nodes")
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--max-total-tokens", type=int, default=139264)
    parser.add_argument("--decoding", choices=("chain", "tree"), default=None,
                        help="Default: tree with a draft, chain for target-only; tree remains experimental")
    parser.add_argument("--draft-tokens", type=int, choices=(8, 16, 32, 64, 128), default=32,
                        help="Chain block size (8/16/32), or experimental tree node budget (32/64/128)")
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse matching prefixes across turns (default); disable for cold-cache benchmarks")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="Deterministic inference (default); tree decoding requires it")
    parser.add_argument("--attention-backend", choices=("flashinfer", "triton"), default="triton")
    parser.add_argument("--fp4-gemm-backend", choices=("flashinfer_cutlass", "flashinfer_cudnn"),
                        default="flashinfer_cutlass", help="Native NVFP4 GEMM backend; cuDNN is an unqualified candidate")
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--graph-batches", type=int, nargs="+")
    parser.add_argument("--kernels", choices=("stock", "fused", "stream", "combined", "ported"), default="combined")
    options = vars(parser.parse_args())
    host, port = options.pop("host"), options.pop("port")
    args = Spark.build_args(**options)
    args.host, args.port, args.served_model_name = host, port, "ripple"
    launch_server(args, run_scheduler_process_func=run_worker)


if __name__ == "__main__":
    main()
