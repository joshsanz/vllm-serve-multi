"""HTTP server for Ripple-65B (NVFP4 MLP / BF16 attention) on a DGX Spark, with optional DFlash2.

Thin wrapper over Ricursive's spark.py (vendored): it reuses Spark.build_args for the launch config
and spark.run_worker for the SM121 fixes, tree adapter and request-cleanup patch, and serves the
OpenAI API on :30000 under the name ripple-65b.
Env: RIPPLE_SPEC=1|0 (DFlash2 on/off), RIPPLE_DRAFT_TOKENS (default 16; chain 8/16/32, tree 32/64/128),
RIPPLE_DECODING=chain|tree (default chain; tree is experimental upstream), RIPPLE_KERNELS (default combined),
RIPPLE_MAX_BATCH (default 4), RIPPLE_PREFIX_CACHE=1|0 (default 1), RIPPLE_DETERMINISTIC=1|0 (default 0 here; upstream default 1),
RIPPLE_TARGET / RIPPLE_DRAFT (HF repos or local dirs).
"""
import os

from sglang.srt.entrypoints.http_server import launch_server

# Importing spark registers the model class and parsers at module level, so SGLang's spawned
# scheduler processes (which re-import it to find run_worker) get them too.
import spark


def flag(name, default):
    return os.environ.get(name, default) == "1"


def main():
    spec = flag("RIPPLE_SPEC", "1")
    args = spark.Spark.build_args(
        os.environ.get("RIPPLE_TARGET", "ricursiveAI/Ripple-65B-Preview-NVFP4"),
        os.environ.get("RIPPLE_DRAFT", "ricursiveAI/Ripple-65B-Preview-DFlash2") if spec else None,
        speculative=spec,
        decoding=os.environ.get("RIPPLE_DECODING", "chain") if spec else "chain",
        draft_tokens=int(os.environ.get("RIPPLE_DRAFT_TOKENS", "16")),
        max_batch_size=int(os.environ.get("RIPPLE_MAX_BATCH", "4")),
        kernels=os.environ.get("RIPPLE_KERNELS", "combined"),
        prefix_cache=flag("RIPPLE_PREFIX_CACHE", "1"),
        deterministic=flag("RIPPLE_DETERMINISTIC", "0"),
    )
    args.host, args.port, args.served_model_name = "0.0.0.0", 30000, "ripple-65b"
    launch_server(args, run_scheduler_process_func=spark.run_worker)


if __name__ == "__main__":
    main()
