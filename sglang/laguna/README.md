# Laguna-S-2.1 on sglang — parked

This stack is **not in active use**. We moved serving of `poolside/Laguna-S-2.1-NVFP4`
to `../../vllm/laguna-s-21/` after finding correctness bugs in sglang on this hardware
(NVIDIA GB10 / DGX Spark, SM121) that are already fixed upstream in vLLM. Kept here for
reference and in case we want to revisit once the relevant sglang PR lands.

## What went wrong

Running Laguna-S-2.1 (+ its DFlash draft model) via sglang on SM121 produced garbled,
repetitive output ("The harsh. The harsh. The harsh...") on long input contexts
(confirmed reproducible at ~10.7k prompt tokens, well under the model's 262144-token
window), independent of whether DFlash speculative decoding was enabled. Short prompts
looked fine, which made this easy to miss initially.

## Root cause

SGLang's `main` branch is missing two pieces of hardware/model-specific support that
Laguna needs on SM12x (GB10 / RTX Pro 6000 Blackwell):

1. **Attention planning for mixed per-layer query-head counts.** Laguna's full-attention
   and sliding-window-attention layer groups have *different* query-head counts, but
   stock sglang's FlashInfer attention planner uses model-global metadata that doesn't
   account for this. This is what caused the long-context collapse — the bug compounds
   as context grows, so it's invisible on short prompts.
2. **NVFP4 MoE kernel dispatch for SM12x.** Stock sglang doesn't route compressed-tensors
   NVFP4 MoE to FlashInfer's `b12x_fused_moe` kernel on SM120/SM121, so MoE expert
   computation isn't using the checkpoint's intended weight layout/scale contract on
   this hardware.

## The fix (draft, not yet merged)

[sgl-project/sglang#32119](https://github.com/sgl-project/sglang/pull/32119) — adds the
SM12x NVFP4 MoE kernel adaptation. It's stacked on two other PRs and **requires all
three together**:

- [#32062](https://github.com/sgl-project/sglang/pull/32062) — the attention-planning fix
  for Laguna's mixed query-head-count layer groups.
- [#31927](https://github.com/sgl-project/sglang/pull/31927) — FlashInfer upgrade to
  `0.6.15.post1` (main is pinned to `0.6.14`, which lacks the required kernel surface).
- #32119 itself — the SGLang-side weight reorder / scale-layout adaptation for the
  `b12x_fused_moe` kernel.

The PR author's own ablation table confirms stock `main` "does not run correctly" for
this model on SM121, and that all three pieces are needed together — any subset fails.
They also flag a known residual numerical issue even with all three applied (the
current scale-folding approach pushes ~95% of block scales into FP8 subnormal range,
~5% median relative error), with a better decoupled-scale approach still blocked on
upstream FlashInfer/vLLM work.

None of this is merged as of writing. Adopting it today would mean building sglang
from source against these branches plus the FlashInfer upgrade — not available as a
released image.

## Why vLLM instead

The equivalent fixes are already merged in vLLM (running on `v0.25.1`, the latest
release as of this writing):

- [#40082](https://github.com/vllm-project/vllm/pull/40082) (merged 2026-05-20) — adds
  the same `b12x_fused_moe`/FP4 GEMM kernel support for SM120/SM121.
- [#42650](https://github.com/vllm-project/vllm/pull/42650) (merged 2026-05-22) — fixes
  `FlashInferMetadataBuilder`/`TritonAttentionMetadataBuilder` sourcing `num_qo_heads`
  from model-wide config instead of per-layer — the same architectural bug class as
  sglang's attention-planning gap, root-caused on a sibling Poolside model
  (`Laguna-XS.2-FP8`) via
  [vllm-project/vllm#41651](https://github.com/vllm-project/vllm/issues/41651).
- [#42080](https://github.com/vllm-project/vllm/pull/42080) (merged 2026-05-19) — FP8
  Q-scale support needed alongside #42650.

We validated `../../vllm/laguna-s-21/` against the exact same long-context test that
broke sglang (needle-in-haystack retrieval at ~10.7k prompt tokens), both with and
without DFlash — clean, correct output in both cases.

One open, unresolved gap on the vLLM side worth knowing about:
[vllm-project/vllm#49379](https://github.com/vllm-project/vllm/issues/49379) — the
`poolside_v1` reasoning parser fails to auto-initialize reasoning token IDs on v0.25.1
(logs a warning). Looked like a feature-completeness gap, not a correctness bug — it
didn't affect generation quality in our testing.

## Other findings while debugging (unrelated to the above, but worth remembering)

- `--load-format fastsafetensors` triggers a real kernel hang on this GB10 box (not an
  sglang bug specifically) — its bulk `torch.cuda.caching_allocator_alloc()` call hits
  an open, unfixed PyTorch bug on unified-memory hardware:
  [pytorch/pytorch#174358](https://github.com/pytorch/pytorch/issues/174358). We
  switched this stack to `--load-format runai_streamer` with a bounded `memory_limit`
  instead, which loads nearly as fast without the crash risk.
- Disabling host swap turns memory-pressure incidents into clean OOM kills instead of
  the kernel-level hang/reclaim-lock deadlock we hit multiple times — relevant for any
  service on this box, not just sglang.
