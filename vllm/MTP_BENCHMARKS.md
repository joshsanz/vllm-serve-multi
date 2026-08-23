# MTP speculative decoding benchmark results

## Concurrency-1 agentic-shaped benchmark: unsloth vs vLLM on qwen38-27b (recorded 2026-08-21)

All benchmarks below (`v0.26.0 baseline` and the original N-sweeps) use
`vllm bench serve`'s standard methodology: 40 requests, 512 input / 256
output tokens, `--ignore-eos`, max concurrency 8. That measures aggregate
*serving* throughput under batched load — useful for sizing a
multi-user deployment, but it doesn't represent this box's actual use case:
a single coding agent driving one model with mostly sequential requests, no
batching. Real-world observation on the vLLM `qwen38-27b` stack under that
usage pattern was **6-8 tok/s, with bursts to 15 tok/s** — far below the
~90-260 tok/s figures elsewhere in this doc, prompting a methodology that
actually reflects single-stream decode speed.

**Methodology**: a custom client (`bench_unsloth.py` / `bench_vllm.py`,
written ad hoc for this comparison, not checked into this repo) rotates
through 6 real source files from this box's local `llama.cpp` checkout
(8k-18k tokens each: `server-http.cpp`, `llama-chat.cpp`,
`llama-memory-recurrent.cpp`, `server-common.cpp`, `server-task.cpp`,
`llama-quant.cpp`) as long input context, pairing each with a realistic
single coding-question prompt ("explain this file", "find unhandled edge
cases", etc.). Requests are sent **one at a time** (concurrency 1, no
batching), and generation **stops naturally** (chat template default,
`reasoning_effort: medium`) with a 1500-token cap as a safety net rather
than `--ignore-eos`. Each file is only used once per run, so every request
hits a cold/differently-shaped prefix — no artificial prefix-cache speedup
between requests, matching how a real agent session's context keeps
growing and changing turn to turn.

Both engines ran the model at the same nominal config: Qwen3.8-27B, MTP
speculative decoding with `num_speculative_tokens: 3`. unsloth serves
`unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL` via `unsloth run` (llama.cpp
`llama-server` underneath); vLLM serves `Qwen/Qwen3.8-27B-FP8` per its
usual `qwen38-27b/docker-compose.yml` stack — different quant format and
precision, so this compares the two *deployments* as actually run on this
box, not an apples-to-apples quant comparison. unsloth's decode tok/s comes
from llama.cpp's own per-request `timings.predicted_per_second` (exact,
server-measured); vLLM doesn't expose an equivalent field over its OpenAI
API, so its decode tok/s is computed client-side from SSE chunk arrival
timestamps (`(completion_tokens - 1) / (time_of_last_token -
time_of_first_token)`) — comparable in practice, but sourced differently.

| File | Prompt tok | unsloth decode tok/s | unsloth accept% | vLLM decode tok/s | vLLM TTFT |
|---|---|---|---|---|---|
| server-http.cpp | 8,279 | 15.97 | 66.2% | 16.39 | 3.02s |
| llama-chat.cpp | 11,948 | 16.39 | 70.0% | 16.42 | 12.67s |
| llama-memory-recurrent.cpp | 12,159 | 14.14 | 55.9% | 15.33 | 14.90s |
| server-common.cpp | 17,332 | 15.39 | 64.8% | 17.32 | 18.54s |
| server-task.cpp | 16,032 | 14.99 | 62.0% | 16.30 | 17.42s |
| llama-quant.cpp | 18,262 | 15.27 | 64.1% | 16.77 | 22.39s |
| **Mean** | | **15.36** | | **16.42** | |
| **Range** | | 14.14-16.39 | | 15.33-17.32 | |

**vLLM edges out unsloth by ~7% on single-stream decode speed** (16.42 vs
15.36 tok/s mean) in this test — the opposite of what the 6-8 tok/s
real-world observation would suggest, and both landed well above that
figure. Two things stand out:

- **Neither engine reproduces the reported real-world 6-8 tok/s (bursts to
  15) on this synthetic test** — both sit in the 14-17 tok/s band across
  every file size tested (8k-18k input tokens), with no meaningful
  slowdown as context grew. This points to something in the *real*
  multi-turn agent workload — growing conversation history re-processed
  each turn, tool-call round trips, prompt structure/caching behavior
  specific to the coding harness in use — as the actual bottleneck, not
  raw single-shot decode throughput. Worth a follow-up benchmark that
  simulates an actual multi-turn session (accumulating context across
  several sequential requests) rather than independent single-shot ones.
- **vLLM's TTFT scales much more with prompt size than unsloth's implied
  prefill cost did** (3s at 8k tokens up to 22s at 18k tokens) — this
  single-request test doesn't isolate why (prefix-caching effectiveness,
  scheduler warm-up, FP8 vs GGUF prefill kernel differences are all
  plausible) and wasn't investigated further here.

This is a **synthetic single-turn proxy**, not a captured real agent
session — one Q&A turn against a pasted file, no multi-turn history, no
tool calls. Useful as a same-day, same-box, matched comparison between the
two engines, but not a substitute for profiling the actual workload
directly.

## Concurrency-1 benchmark: full cached-model sweep (recorded 2026-08-22)

Extended the concurrency-1 methodology above to every other model on this
box that was actually cached and loadable, not just qwen38-27b. The two ad
hoc scripts from the 2026-08-21 run are now checked into the repo as
`bench/bench_openai_client_timed.py` (client-timed via SSE, for vLLM and
sglang — neither exposes a server-side timings block over the OpenAI API)
and `bench/bench_llamacpp_timings.py` (uses llama.cpp's own per-request
`timings` block, for the unsloth stacks). Same 6 files, same questions, same
1500-token cap, same natural-stop/no-batching/concurrency-1 methodology as
before. Raw per-file output for every stack is in `bench/results/`.

Cache state was checked directly against `~/.cache/huggingface/hub` before
running anything (not assumed from prior benchmark docs). Three vLLM stacks
(`laguna-s-21`, `nemotron-35-lightning-dspark`, `ornith`) and sglang
`laguna` were skipped — their weights simply aren't downloaded on this box,
and none were fetched to avoid multi-hour/many-GB downloads mid-benchmark.
`unsloth/inkling-small.sh`'s model was only 1-of-3 GGUF shards downloaded
(stalled) at the start of this run but finished mid-session, so it was
added back in.

**One unrelated bug found and fixed along the way**:
`unsloth/inkling-small.sh` referenced the GGUF variant as `UD-IQ3-XXS`
(hyphen); the actual variant name unsloth serves is `UD-IQ3_XXS`
(underscore) — confirmed against both the cached filename and the "closest
available variants" list vLLM's — sorry, unsloth studio's — own load-error
returned. One-character fix, not a benchmark-methodology issue.

**Also found**: `unsloth run`'s `-p 8000` only binds the *unsloth-studio*
management API (auth-gated, returns `401` on `/v1/models` without a
bearer token) on that port — the actual `llama-server` OpenAI endpoint
listens on a separate, dynamically-assigned `127.0.0.1:<port>` printed in
the studio startup log (`Starting llama-server: ... --port NNNNN`). All
three unsloth benchmarks below hit that internal port directly, not `:8000`.
This also explains why `bench_unsloth.py`'s original hardcoded
`localhost:45949` (2026-08-21 run, above) worked — that was the port
assigned on that particular run, not a fixed convention.

| Stack (engine) | Model | Spec. method | Mean decode tok/s | Range | Accept% | Notes |
|---|---|---|---|---|---|---|
| kat-coder-v25-dev-awq (vLLM) | KAT-Coder-V2.5-Dev-AWQ-INT4 | none | **37.06** | 36.42–37.72 | n/a | Fastest no-speculation stack; AWQ-INT4 + hybrid linear attention is just cheap per step. |
| qwen36-27b (vLLM) | Qwen3.6-27B-FP8 | MTP N=3 | 16.40 | 15.02–17.29 | n/a* | Matches qwen38-27b/FP8's 16.42 tok/s from 2026-08-21 almost exactly. |
| qwen36-35b-a3b (vLLM) | Qwen3.6-35B-A3B-FP8 | MTP N=3 | **58.88** | 55.88–62.88 | n/a* | Fastest stack overall — MoE, ~3B active params/token. |
| ling-30-flash-int4 (sglang) | Ling-3.0-flash-int4 | NEXTN | 15.08 | 12.96–17.50 | n/a* | First TTFT (51s, cold file) far higher than the rest — likely first-request JIT/kernel warmup, not representative of steady state. |
| muse-glimmer-30b (unsloth/llama.cpp) | Muse-Glimmer-30B-GGUF Q8_K_XL | dflash (v1) | 14.35 | 13.27–15.43 | 43.8–57.1% | Own bundled `dflash-kquant.gguf` draft, not the broken DFlash2 checkpoint (see `[[dflash2-dspark-unsupported]]`). |
| deepseek-v4-flash-0731 (unsloth/llama.cpp) | DeepSeek-V4-Flash-0731-GGUF UD-IQ2_M | dspark | 28.94 | 26.78–30.70 | 48.4–62.4% | Largest checkpoint run (~95GB); loaded and benchmarked cleanly despite the tight memory budget. |
| inkling-small (unsloth/llama.cpp) | Inkling-Small-GGUF UD-IQ3_XXS | auto (`--spec-default`) | 18.76 | 18.12–19.50 | **~0%** | 276B-total/12B-active MoE, ~91GB weights against ~121GB unified memory — loaded fine, no OOM despite the script's own "very tight" warning. Speculative accept rate was ~0% on 5 of 6 files (one hit 4.7%) — whatever draft method Studio auto-selected isn't landing hits on this checkpoint; not chased further. |
| gemma4 (vLLM) | gemma-4-31B-it-qat-w4a16-ct | MTP N=4 | — | — | — | **Failed to start.** Crash-loops on this box's current `vllm/vllm-openai:v0.27.1` image: `AmbiguousGlobalPerLayerAttributeError` reading `head_dim` from `transformers`' config layer — gemma4 has heterogeneous per-layer attention (local/global mix with different head dims) and vLLM's config convertor assumes a single global value. Not a resource or timeout issue; every restart hit the same traceback within ~2s. Last known-good run was the v0.26.0 baseline (2026-08-05); something in the v0.26.0→v0.27.1 vLLM/transformers version jump broke it. Needs a real fix (vLLM or transformers version pin/patch) before it can be benchmarked again. |

*vLLM and sglang don't expose per-request draft-accept counts over the
OpenAI streaming API the way llama.cpp does, so accept% is only available
for the three unsloth/llama.cpp rows — consistent with the qwen38-27b
comparison above.

**What stood out:**

- **Raw decode speed here doesn't track speculative-decoding accept rate at
  all** — `qwen36-35b-a3b` (MoE, cheap per-step) is 4x faster than
  `muse-glimmer-30b` despite both running MTP-family speculation, simply
  because the underlying dense-compute cost per token differs that much
  between model sizes/architectures. Model size and architecture dominate;
  speculative decoding is a second-order effect on top.
- **`inkling-small`'s near-zero speculative accept rate is a real finding,
  not noise** — it ran the same file-rotation methodology as
  `muse-glimmer-30b` and `deepseek-v4-flash-0731`, which both landed
  44-62% accept, so this isn't a fluke of the test; the auto-selected
  draft method just isn't compatible with (or well-tuned for) this
  particular checkpoint. Worth an explicit `--speculative-type off`
  comparison run if `inkling-small` becomes a regular stack, to see
  whether disabling speculation entirely is actually faster (avoided
  wasted draft passes) than the current ~0%-hit-rate default.
- **`gemma4` being broken is the most actionable result of this run** — it
  isn't a benchmark artifact, it's a real regression somewhere in the
  v0.26.0→v0.27.1 vLLM/transformers upgrade path that will block *any* use
  of that stack, not just benchmarking, until fixed.
- Absolute decode-speed ranking across everything benchmarked on this box so
  far (concurrency-1, natural-stop): `qwen36-35b-a3b` (58.88) >
  `kat-coder-v25-dev-awq` (37.06) > `deepseek-v4-flash-0731` (28.94) >
  `inkling-small` (18.76) > `qwen38-27b`/vLLM (16.42, 2026-08-21) >
  `qwen36-27b` (16.40) > `qwen38-27b`/unsloth (15.36, 2026-08-21) >
  `ling-30-flash-int4` (15.08) > `muse-glimmer-30b` (14.35).

## v0.26.0 baseline (recorded 2026-08-05)

Bumped every stack from `vllm/vllm-openai:v0.25.1-ubuntu2404` to
`v0.26.0-ubuntu2404` and re-ran `vllm bench serve` (same methodology as
below: 40 requests, 512 input / 256 output tokens, `--ignore-eos`, max
concurrency 8, seed 42, 3 warmup requests) against every stack at its
**current default speculative config** — single-point runs, not the N-sweeps
below. `laguna-s-21` was skipped: poolside changed the last 8 layers of
Laguna-S-2.1 from NVFP4 to bf16 upstream, which no longer fits this box's
121.63GiB unified memory, pending a fix on their end.

Two things came up worth recording as findings, not just numbers:

- **gemma4 + ornith moved out of the shared-GPU sleep/swap architecture.**
  They used to run behind `fastapi-router`, which put whichever engine was
  idle to sleep (`--enable-sleep-mode`) so the two could timeshare the one
  GPU. On v0.26.0, waking gemma4 from sleep left its MTP draft/assistant
  model producing garbage (0% speculative acceptance, `Accepted: 0` across
  thousands of drafts) even though the *main* model still verified and
  produced correct output — just at a fraction of normal speed, since every
  draft was wasted compute. Root cause wasn't chased further; the fix was
  architectural: `gemma4` and `ornith` are now standalone stacks
  (`gemma4/`, `ornith/`) like every other model here, never sleep, and
  `router/` is retired (kept in the tree, unused). See `README.md`.
- **gemma4's `--gpu-memory-utilization` dropped from 0.78 to 0.75.** Even
  from a clean idle boot with nothing else running, this box only had
  92.59GiB of its 121.63GiB unified pool free for CUDA (desktop environment
  + OS overhead), while 0.78 wanted 94.87GiB — a hard failure at engine
  init, not a KV-cache-sizing warning. 0.75 leaves enough margin to boot
  reliably. Separately (and more dramatically): a concurrent `hf download`
  of `laguna-s-21`'s weights running in the background pushed the free pool
  down to 67.83GiB at one point, which is worth remembering next time a
  benchmark looks inexplicably bad on this box — check for other processes
  competing for the shared memory pool first.

| Stack | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Accept rate | Failures | vs v0.25.1 N=3 |
|---|---|---|---|---|---|---|---|
| kat-coder-v25-dev-awq | 196.2 | 596.5 | 667ms | 38.3ms | n/a (no MTP) | 0/40 | -4.8% (206.0) |
| gemma4 | 105.5 | 323.2 | 2198ms | 65.1ms | 44.5% | 0/40 | new |
| ornith | 146.5 | 445.4 | 772ms | 51.8ms | n/a (no MTP) | 0/40 | new |
| qwen36-27b / fp8 | 90.7 | 275.8 | 2119ms | 75.1ms | 62.6% | 0/40 | +14.2% (79.4) |
| qwen36-27b / nvfp4 | 106.7 | 324.3 | 2432ms | 59.9ms | 56.0% | 0/40 | +9.3% (97.6) |
| qwen36-35b-a3b / fp8 | 183.0 | 556.4 | 532ms | 38.9ms | 56.6% | 0/40 | +15.8% (158.1) |
| qwen36-35b-a3b / nvfp4 | 262.1 | 796.8 | 554ms | 27.0ms | 58.2% | 0/40 | +5.1% (249.4) |

All qwen36 profiles picked up a real, consistent throughput gain on v0.26.0
(+5-16%) at their existing default `num_speculative_tokens`, with no
retuning — reads as genuine inference-engine improvements in the new vLLM
release rather than noise, though it's one run per stack, not an average.
kat-coder regressed slightly (-4.8%), within run-to-run noise for a
single-point measurement. gemma4 and ornith have no prior baseline in this
doc (first time they've been benchmarked this way) so there's nothing to
compare against yet — worth a repeat next time to establish their own
trend.

## Original v0.24.0/v0.25.1 N-sweeps

Recorded 2026-07-10. Benchmarked with `vllm bench serve` (built into the
`vllm/vllm-openai:v0.24.0-ubuntu2404` image) run inside each container against
its own server: 40 requests, 512 input / 256 output tokens (`--ignore-eos`),
max concurrency 8, fixed seed 42, 3 warmup requests. Method is `mtp` in all
cases; only `num_speculative_tokens` was varied per sweep.

All four stacks were restored to their original `num_speculative_tokens: 3`
config after benchmarking. The live default (`qwen36-27b`, nvfp4 profile) was
left running afterward; the other three were torn down.

## qwen36-27b / nvfp4 (Qwen3.6-27B, nvidia/Qwen3.6-27B-NVFP4) — current default

| N | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Accept rate | Accept length | Failures |
|---|---|---|---|---|---|---|---|
| 2 | 96.5 | 289.4 | 1944ms | 67.4ms | 61.3% | 2.23 | 0/40 |
| **3 (default)** | 97.6 | 292.9 | 1628ms | 65.2ms | 51.0% | 2.53 | 0/40 |
| **4** | **102.7** | **308.2** | 2039ms | 64.9ms | 47.9% | 2.92 | 0/40 |
| 6 | 100.8 | 331.1 | 2503ms | 88.8ms | 36.1% | 3.17 | 10/40 ⚠️ low confidence |
| 8 | 79.1 | 237.2 | 1698ms | 90.4ms | 27.2% | 3.18 | 0/40 |

**Best: N=4** (~5% more output throughput than the current default N=3, latency roughly flat). N=6 had reliability issues (10 failed requests) and shouldn't be trusted without a clean re-run; N=8 clearly regresses.

## qwen36-27b / fp8 (Qwen/Qwen3.6-27B-FP8)

| N | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Accept rate | Accept length | Failures |
|---|---|---|---|---|---|---|---|
| 2 | 77.4 | 232.1 | 2151ms | 86.7ms | 64.7% | 2.29 | 0/40 |
| **3 (default)** | 79.4 | 238.2 | 1243ms | 83.4ms | 51.1% | 2.53 | 0/40 |
| 4 | 85.4 | 311.4 | 3286ms | 110.2ms | 53.2% | 3.13 | 19/40 ⚠️ unreliable |
| 6 | 67.0 | 250.0 | 4449ms | 128.4ms | — | — | 25/40 ⚠️ unreliable |
| 8 | 74.6 | 223.8 | 1719ms | 92.5ms | 32.5% | 3.60 | 0/40 |

**fp8 profile is substantially slower than nvfp4 for this model** (~75-85 tok/s vs ~80-103 tok/s) and showed real instability (`Internal Server Error`s) at N=4 and N=6 that did not reproduce at N=8 — looks like intermittent flakiness in the fp8 path under load rather than a clean function of N (docker logs from the failing runs were lost to container recreation before they could be inspected — worth re-running N=4/N=6 in isolation with logs captured if this profile is going to be used). **Recommendation: prefer the nvfp4 profile over fp8 for this model** — it's faster and cleaner at every N tested.

## qwen36-35b-a3b / fp8 (Qwen/Qwen3.6-35B-A3B-FP8)

| N | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Accept rate | Accept length | Failures |
|---|---|---|---|---|---|---|---|
| 2 | 146.7 | 440.2 | 550ms | 50.4ms | 48.3% | 1.97 | 0/40 |
| **3 (default)** | **158.1** | **474.4** | 581ms | 45.2ms | 46.0% | 2.38 | 0/40 |
| 4 | 142.7 | 434.9 | 615ms | 53.8ms | 32.8% | 2.31 | 0/40 |
| 6 | 136.1 | 422.6 | 687ms | 55.8ms | 29.9% | 2.79 | 0/40 |
| 8 | 107.3 | 321.9 | 746ms | 64.6ms | 21.6% | 2.73 | 0/40 |

All runs clean. **N=3 (current default) is already the peak** — throughput declines monotonically on either side of it.

## qwen36-35b-a3b / nvfp4 (nvidia/Qwen3.6-35B-A3B-NVFP4)

| N | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Accept rate | Accept length | Failures |
|---|---|---|---|---|---|---|---|
| 2 | 229.2 | 687.5 | 1095ms | 29.2ms | 56.9% | 2.14 | 0/40 |
| **3 (default)** | **249.4** | **748.2** | 416ms | 28.9ms | 51.7% | 2.55 | 0/40 |
| 4 | 208.8 | 656.0 | 1164ms | 33.8ms | 39.9% | 2.60 | 0/40 |
| 6 | 189.0 | 600.9 | 1156ms | 39.1ms | 30.7% | 2.84 | 0/40 |
| 8 | 172.3 | 546.0 | 1054ms | 43.0ms | 24.5% | 2.96 | 0/40 |

All runs clean. **N=3 (current default) is already the peak** here too, and by a wider margin than fp8 (249 tok/s vs 209 at N=4, a 16% drop). This is also by far the fastest of the four stack/profile combos — expected, since Qwen3.6-35B-A3B is a MoE model with only ~3B active params per token, so it's cheaper per step than the 27B dense model despite being nominally larger.

## kat-coder-v25-dev-awq (cyankiwi/KAT-Coder-V2.5-Dev-AWQ-INT4)

Recorded 2026-08-04. Same `vllm bench serve` methodology (40 requests, 512
input / 256 output tokens, `--ignore-eos`, max concurrency 8, seed 42, 3
warmup requests) run inside the container against its own server, but this
is a **single-configuration run, not an N-sweep** — this checkpoint ships no
MTP draft weights (`mtp_num_hidden_layers: 0` in `config.json`), so there's
no `--speculative-config` to vary.

| Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT | Mean TPOT | Failures |
|---|---|---|---|---|---|
| 206.0 | 626.4 | 232.0 | 655ms | 36.4ms | 0/40 |

No speculative decoding, so not directly comparable to the MTP-assisted
numbers above — included here as a throughput data point for this model
alongside the others. For context: it lands between `qwen36-35b-a3b` fp8+MTP
N=3 (158 tok/s) and nvfp4+MTP N=3 (249 tok/s) despite having no draft
assist, likely thanks to the INT4 quant and the hybrid linear-attention
layers being cheaper per step than plain attention.

## Summary / recommendations

- **qwen36-27b**: bump `num_speculative_tokens` from 3 → **4** on the nvfp4 profile for a modest (~5%) throughput gain. Avoid the fp8 profile — it's slower and showed unexplained request failures under load at N=4/N=6.
- **qwen36-35b-a3b**: leave `num_speculative_tokens` at **3** on both profiles — it's already the peak for this model, and prefer the **nvfp4** profile over fp8 (roughly 1.5-1.6x the throughput at matching N).
- General pattern: acceptance rate decays with each added speculative position (compounding error from reusing the same MTP layer autoregressively, as vLLM's own startup warning notes), and past N=3-4 the extra drafting compute costs more than the added accepted tokens return. The 35B-A3B (MoE) model peaks earlier and drops off harder than the 27B dense model, likely because its per-step compute is cheaper, so the relative overhead of wasted draft passes is larger.
