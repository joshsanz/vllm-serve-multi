# MTP speculative decoding benchmark results

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

## Summary / recommendations

- **qwen36-27b**: bump `num_speculative_tokens` from 3 → **4** on the nvfp4 profile for a modest (~5%) throughput gain. Avoid the fp8 profile — it's slower and showed unexplained request failures under load at N=4/N=6.
- **qwen36-35b-a3b**: leave `num_speculative_tokens` at **3** on both profiles — it's already the peak for this model, and prefer the **nvfp4** profile over fp8 (roughly 1.5-1.6x the throughput at matching N).
- General pattern: acceptance rate decays with each added speculative position (compounding error from reusing the same MTP layer autoregressively, as vLLM's own startup warning notes), and past N=3-4 the extra drafting compute costs more than the added accepted tokens return. The 35B-A3B (MoE) model peaks earlier and drops off harder than the 27B dense model, likely because its per-step compute is cheaper, so the relative overhead of wasted draft passes is larger.
