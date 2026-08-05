# MTP speculative decoding benchmark results

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
