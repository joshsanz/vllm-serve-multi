# serve

Model serving configs for a single-GPU box (DGX Spark / GB10, ~121.63GiB
unified CPU+GPU memory) that can't hold more than one large model resident
at once. Every stack/script across every engine below publishes its
OpenAI-compatible API on host port `:8000` and is meant to run alone — bring
the current one down before bringing up a different one, regardless of
which engine directory it lives in.

## Engines

| Directory | Engine | What's there |
|---|---|---|
| `vllm/` | vLLM | Standalone `docker compose` stack per model. See `vllm/README.md` for the full model table, running instructions, and box-specific notes. |
| `sglang/` | SGLang | Standalone `docker compose` stack per model. See below. |
| `unsloth/` | unsloth (llama.cpp-based `unsloth run`) | Standalone launcher script per model, run directly (not compose). See `unsloth/README.md` for the model table, running instructions, and a systemd user service for durable (no-tmux-needed) launches. |
| `ollama/` | Ollama | **unused/deprecated** — kept in the tree but not run; see below. |

## sglang/

| Directory | Model | Notes |
|---|---|---|
| `laguna/` | `poolside/Laguna-S-2.1-NVFP4` | **parked** — sglang lacks upstream fixes vLLM already has for this model on SM12x; see `laguna/README.md` for the full root-cause writeup. Serving moved to `../vllm/laguna-s-21/`. |

Run any stack the same way as vLLM's:

```bash
cd sglang/<model-dir>
HF_TOKEN=... docker compose up -d
```

`sglang/archive/` holds compose recipes whose weights were removed from the
HF cache (2026-09-03) to reclaim disk — currently the `ling-30-flash-int4/`
(`inclusionAI/Ling-3.0-flash-int4`, hybrid Mamba+MoE, NEXTN/MTP speculative
decoding) stack. Move a dir up out of `archive/` and let sglang re-download
on first `up -d` to revive it.

## ollama/ (unused/deprecated)

Runs the `ollama` daemon plus a one-shot `puller` sidecar that pre-fetches
its configured model list (`north-mini-code-1.0`, `laguna-xs-2.1`, `gemma4`,
`qwen3.6:35b-a3b-mtp`) so the blobs are on disk before first request, with
model residency in VRAM left to Ollama's own request-driven load/evict
behavior (see the comments in `ollama/docker-compose.yml` for the
tradeoffs — `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_KEEP_ALIVE`, etc). Not
currently run — Ollama's serving interface didn't work out — in favor of
the vLLM/SGLang/unsloth stacks above. Kept in the tree for reference only.
