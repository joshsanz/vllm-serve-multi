# serve

vLLM model serving stack for a single-GPU box (DGX Spark / GB10, ~121.63GiB
unified CPU+GPU memory) that can't hold more than one large model resident at
once. Each model gets its own standalone `docker compose` stack in its own
directory; only one stack is meant to be up at a time. Bring one down before
bringing another up.

This box previously ran `gemma4` and `ornith` behind a FastAPI router
(`router/`) that used vLLM sleep-mode to swap models on/off the GPU on
demand, sharing one OpenAI-compatible endpoint. That approach has been
retired in favor of standalone stacks, matching every other model here — the
sleep/wake cycle turned out to not reliably restore speculative-decoding
draft-model state after waking (0% draft acceptance observed on gemma4 after
a wake, despite the main model still generating correct output). `router/`
is kept in the tree but unused; nothing currently depends on it.

All compose services share the `x-logging` anchor (`json-file`, 10MB × 3
files) so container logs can't grow unboundedly; `nemotron-3-super.sh`
applies the same cap via `--log-opt` since it runs outside compose.

## Running any stack

```bash
cd <model-dir>
docker compose up -d
```

Requires the NVIDIA Container Toolkit and a Hugging Face cache at
`~/.cache/huggingface` (run `huggingface-cli login` first for gated models
like Gemma). Each stack publishes its OpenAI-compatible API on `:8000`:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Bring the current stack down (`docker compose down`) before bringing up a
different one — they all bind the same host port and the box doesn't have
GPU memory to spare for two at once.

## Model directories

| Directory | Model | Notes |
|---|---|---|
| `gemma4/` | `google/gemma-4-31B-it-qat-w4a16-ct` | |
| `ornith/` | `deepreinforce-ai/Ornith-1.0-35B-FP8` | model overridable via `MODEL` env var |
| `laguna-s-21/` | `poolside/Laguna-S-2.1-NVFP4` | dflash speculative decoding |
| `qwen38-27b/` | `Qwen/Qwen3.8-27B-FP8` | dense hybrid attention (Gated DeltaNet + full attention); vLLM auto-manages the hybrid KV cache, no extra flags needed |
| `kat-coder-v25-dev-awq/` | `cyankiwi/KAT-Coder-V2.5-Dev-AWQ-INT4` | fine-tune of Qwen3.6-35B-A3B, INT4 compressed-tensors quant, no speculative decoding |
| `nemotron-35-lightning-dspark/` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | hybrid Mamba-2+MoE+Attention, DSpark speculative decoding, NVIDIA's DGX Spark recipe |
| `systemone/` | `vllm-sr/d3-flash` + `LiquidAI/LFM2.5-24B-A2B` (BF16) | two containers: d3-flash decision model (9B multimodal, text/image/video in; choice/yes-no/score answers, not chat; runtime places it at ~19.2 GiB) in a `vllm-srun` process on :8001 via `POST /v1/decisions` (locally built image, upstream has no arm64+CUDA one); LFM2.5 hybrid conv/attention MoE as a regular chat LLM on :8000 (`--max-num-seqs 4`, 131072 ctx (native max), util 0.44 for ~320k KV tokens = two full sessions, `qwen3` reasoning + `lfm2` tool parsers; FP8 variant is the fallback if memory gets tight); vLLM v0.33.0; see compose header for the page-cache gotcha. The LFM chat pairing is superseded by `../sglang/clover/` (Clover + d3-flash on the same ports). Not meant to run both containers next to another chat stack: `docker compose up -d systemone` alone (d3-flash only, :8001) pairs with `../unsloth/qwen38-flash-next.sh` (UD-IQ2_XS GGUF, chat on :8000), as a tool-use reviewer; the placement check needs the page cache dropped repeatedly while it loads |

To iterate on settings, edit the `command:` list and re-run `up -d` —
compose recreates the container in place instead of requiring a manual
`docker rm -f` first.

See `MTP_BENCHMARKS.md` for throughput benchmarks across these stacks
(note: it also covers the archived `qwen36-27b`/`qwen36-35b-a3b` stacks,
which were pinned to fp8-only, dropping the nvfp4 profile it favored).

`archive/` holds compose recipes whose weights were removed from the HF
cache (2026-09-03) to reclaim disk. To bring one back: `docker compose pull`
/ let vLLM re-download on first `up -d`, then move the dir up out of
`archive/`.

## nemotron-3-super.sh

Standalone launcher for `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`,
deliberately kept **out** of compose: its NVFP4 MoE marlin weight-repack step
peaks at ~108GiB GPU memory, which doesn't leave enough margin against this
box's 121.63GiB total once any sibling vLLM engine is resident. It needs the
entire GPU/unified memory pool to itself. See the comments in the script for
the full rationale and run steps (bring down any other running stack first,
then `HF_TOKEN=... ./nemotron-3-super.sh`).

It uses `super_v3_reasoning_parser.py`, a custom vLLM reasoning-parser plugin
that fixes how the stock `deepseek_r1` parser (which Nemotron 3 reuses)
handles thinking-disabled requests and truncated-reasoning edge cases, so
final content isn't dropped into `reasoning_content` when it shouldn't be.

## Layout

```
gemma4/docker-compose.yml              standalone gemma4 stack
ornith/docker-compose.yml              standalone Ornith-1.0 stack
laguna-s-21/docker-compose.yml         standalone Laguna-S-2.1 stack
qwen38-27b/docker-compose.yml          standalone Qwen3.8-27B stack (fp8, hybrid attention)
kat-coder-v25-dev-awq/docker-compose.yml  standalone KAT-Coder-V2.5-Dev-AWQ-INT4 stack
nemotron-35-lightning-dspark/docker-compose.yml  standalone Nemotron-3.5-Lightning stack (DSpark speculative decoding)
systemone/{Dockerfile,models.yaml,docker-compose.yml}  d3 decision model (vllm-srun) + LFM2.5-24B-A2B BF16 chat LLM
MTP_BENCHMARKS.md                      throughput benchmark results across stacks
nemotron-3-super.sh                    standalone Nemotron 3 Super launcher
super_v3_reasoning_parser.py           reasoning parser plugin for Nemotron 3
archive/                               compose recipes for models no longer cached on the box
router/                                retired FastAPI swap-aware proxy (unused)
  app/main.py                            routes + request dispatch
  app/config.py                          served-model registry
  app/swap_manager.py                    sleep/wake orchestration
```
