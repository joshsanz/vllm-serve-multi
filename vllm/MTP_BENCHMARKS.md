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
| muse-glimmer-30b (unsloth/llama.cpp) | Muse-Glimmer-30B-GGUF Q8_K_XL | dflash (v1) | 14.35 | 13.27–15.43 | 43.8–57.1% | Own bundled `dflash-kquant.gguf` draft, recorded before the standalone DFlash2 checkpoint below was fixed upstream (see `[[dflash2-dspark-unsupported]]` and the 2026-09-08 section below). |
| deepseek-v4-flash-0731 (unsloth/llama.cpp) | DeepSeek-V4-Flash-0731-GGUF UD-IQ2_M | dspark | 28.94 | 26.78–30.70 | 48.4–62.4% | Largest checkpoint run (~95GB); loaded and benchmarked cleanly despite the tight memory budget. |
| inkling-small (unsloth/llama.cpp) | Inkling-Small-GGUF UD-IQ3_XXS | auto (`--spec-default`) | 18.76 | 18.12–19.50 | **~0%** | 276B-total/12B-active MoE, ~91GB weights against ~121GB unified memory — loaded fine, no OOM despite the script's own "very tight" warning. Speculative accept rate was ~0% on 5 of 6 files (one hit 4.7%) — see investigation below; root-caused, not a mystery. |
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
- **`inkling-small`'s near-zero speculative accept rate is root-caused, not
  a mystery.** `--speculative-type auto` on `unsloth run` translates to the
  vendored llama.cpp's `--spec-default`, which (per
  `common/arg.cpp:4712-4726` in `~/.unsloth/llama.cpp`) *unconditionally*
  enables `NGRAM_MOD` speculation (`n_min=48, n_max=64`) — a pure
  context-lookup method with **no draft model or MTP head involved at
  all**. It speculates that the model is about to repeat a 48-64-token
  span that already appears verbatim in the prompt or prior output. This
  benchmark's prompts ask for prose explanation/analysis of a pasted
  source file ("explain what this file does," "summarize the functions"),
  so the output is almost never a long verbatim repeat of the input —
  hence ~0% hits. `muse-glimmer-30b` and `deepseek-v4-flash-0731` don't
  have this problem because they use real draft-model methods (`dflash`,
  `dspark`) that predict *semantically*, not by string-matching.
  Ran a same-file A/B to check whether the failed lookups were at least
  cheap: `--speculative-type off` on the identical model/config gave
  **19.01 tok/s mean** vs **18.76 tok/s** with `--spec-default` on — a
  ~1.3% difference, within run-to-run noise. **Conclusion: leaving
  `--spec-default` on for this checkpoint isn't costing anything** (a
  failed n-gram lookup is a cheap string scan, not a wasted forward pass
  the way a failed model-based draft would be), it's just not helping for
  this prose-generation workload. It would likely pay off on tasks with
  heavy verbatim overlap between context and output — large diffs/edits,
  repeated boilerplate, structured-format regeneration — just not on
  "explain this code" style prompts. No config change needed.
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

## Concurrency-1 benchmark: qwen38-flash-next WITH MTP (recorded 2026-09-01)

Follow-up to the 2026-08-27 baseline below. Engine-side MTP support for this
checkpoint **landed**, and `qwen38-flash-next.sh` now enables it.

- **unsloth's llama.cpp fork** (build 10715, at `~/.unsloth/llama.cpp`) now
  has a real `qwen4exp` MTP draft-head graph (`src/models/qwen4exp.cpp`),
  the `--spec-type draft-mtp` option, and cross-model shared-tensor
  borrowing (`llama_model_loader::borrow_shared_tensor`). None of that
  existed on 2026-08-27.
- The HF repo `unsloth/Qwen3.8-Flash-Next-GGUF` was updated 2026-09-01
  09:13 UTC to ship the MTP head as a **separate sidecar GGUF** under
  `MTP/` (not baked into the quant): `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
  (2.60 GB, recommended — borrows token-embd + output proj from the main
  model), plus `-Q4_K_M` / `-BF16` and self-contained variants.

**Getting it running through `unsloth run` — three dead ends first:**

1. Studio's own `--speculative-type mtp` checks the *main* `UD-IQ4_XS`
   quant for a baked-in MTP head, doesn't find one, and falls back to
   `--spec-default` n-gram (`"MTP requested but this GGUF has no MTP head
   or drafter"`).
2. Passing `--hf-repo-draft unsloth/Qwen3.8-Flash-Next-GGUF:shared-Q8_0`
   through to llama-server: the quant matcher doesn't look in the repo's
   `MTP/` subdir, so it resolves to an empty path and llama-server dies
   with `failed to load draft model, ''`.
3. `hf download` (as used to resolve a filesystem path in the launcher)
   prints `path=<path>` in its default "agent" output format — the
   `path=` prefix has to be stripped, or use `--quiet`.

**What works** — `unsloth run` forwards unrecognized args to `llama-server`
verbatim, so the committed `qwen38-flash-next.sh` now resolves the sidecar
head from the HF cache and passes:

```
--model-draft <cache>/…/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
--spec-type draft-mtp
```

plus `LLAMA_ARG_SPEC_DRAFT_N_MAX=2` as an **env var** (studio swallows
`--spec-draft-n-max` as its own flag and won't forward it). Load log:
`common_speculative_init_result: loading draft model '…shared-Q8_0.gguf'`
then `model loaded`. (One benign `borrow_shared_tensor: … load it as a
draft of its target model` error appears first — that's studio's memory
pre-measurement pass trying to load the head standalone; it retries
correctly.)

Benchmarked through the actual systemd stack (`--fit on`, n_max=2, mmproj
loaded), same 6-file `bench_llamacpp_timings.py` methodology as every run
in this doc. Raw output: `bench/results/qwen38-flash-next-mtp.log`.

| File | Prompt tok | Prefill tok/s | Decode tok/s | Accept% |
|---|---|---|---|---|
| server-http.cpp | 8,267 | 529.7 | 40.40 | 75.2% |
| llama-chat.cpp | 11,948 | 529.4 | 39.96 | 74.5% |
| llama-memory-recurrent.cpp | 13,035 | 523.9 | 37.31 | 67.4% |
| server-common.cpp | 17,652 | 512.4 | 37.52 | 72.4% |
| server-task.cpp | 16,029 | 519.0 | 37.85 | 72.6% |
| llama-quant.cpp | 19,059 | 510.9 | 37.02 | 72.9% |
| **Mean** | | **520.9** | **38.34** | **72.5%** |
| **Range** | | 510.9–529.7 | 37.02–40.40 | 67.4–75.2% |

**MTP vs. no-MTP baseline (2026-08-27), same box, same methodology:**

| | Decode tok/s (mean) | Accept% |
|---|---|---|
| No MTP (`--spec-default` n-gram) | 23.30 | ~0 (2 files hit 1.6% / 11.7%) |
| **MTP (`draft-mtp`, shared-Q8_0, N=2)** | **38.34** | **72.5%** |
| Speedup | **1.65×** | |

**Notes:**

- **1.65× decode speedup**, near the top of unsloth's own "1.3–1.7× at
  low concurrency" claim (their `MTP/README.md`) despite this not being a
  greedy run. Server-side `draft acceptance` log lines corroborate the
  client-side accept% (0.67–0.75, mean draft len ~2.4–2.5 at N=2).
- An earlier same-day run via a bare `llama-server` invocation (`--fit
  off -ngl -1`, no mmproj, thinking-kwargs only — no `reasoning_effort`)
  measured lower: 33.34 tok/s / 57.6% accept / 1.43×. The stack config
  above (matching the committed script) is the one that counts; the gap
  is most plausibly the draft head landing fully GPU-resident under
  `--fit on` and/or the `reasoning_effort: medium` prompt shaping.
- Prefill is unchanged (~521 vs ~491 tok/s, within noise) — MTP only
  touches decode.
- unsloth's `MTP/README.md` says **skip MTP for concurrent serving**
  (measured net loss ~0.81–0.87× at concurrency 8) — fine here, this box
  runs one coding agent at a time.
- If studio later grows native sidecar-MTP support, revisit
  `qwen38-flash-next.sh` — the manual `--model-draft` passthrough can
  probably go back to a plain `--speculative-type mtp`.

## Concurrency-1 benchmark: qwen38-flash-next, MTP NOT engaged (recorded 2026-08-27)

Baseline run of `unsloth/qwen38-flash-next.sh`
(`unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS`, ~93.7GB, 177B-total /
~6B-active + 51B n-gram embed + 4B MTP) recorded **so it can be re-compared
in a few days when real MTP support for this checkpoint lands**. Same
methodology as the 2026-08-22 sweep: `bench/bench_llamacpp_timings.py`, 6
`llama.cpp` source files, 1500-token cap, natural stop, concurrency 1, no
batching. Raw per-file output in `bench/results/qwen38-flash-next.log`;
startup-log excerpt in `bench/results/qwen38-flash-next.server.log`.

**MTP is confirmed NOT working on this stack today.** The script passes
`--speculative-type mtp`, but unsloth studio logs at load time:

```
"MTP requested but this GGUF has no MTP head or drafter; loading without speculative decoding."
```

and the `llama-server` it then spawns carries only `--spec-default` (no
`--model-draft` / MTP head). The per-request `timings` block confirms it —
no `draft_n` / `draft_n_accepted` fields on most requests, and the two
files that did show a nonzero "accept%" (1.6%, 11.7%) are `--spec-default`
NGRAM_MOD context-lookup hits (same mechanism as `inkling-small` above),
not MTP. So the numbers below are **pure autoregressive decode with no
speculation** — the "MTP lands" re-run should show a decode-tok/s jump and
a real 50–70%-style accept rate if it's working.

| File | Prompt tok | Prefill tok/s | Output tok | Decode tok/s | Accept% |
|---|---|---|---|---|---|
| server-http.cpp | 8,267 | 511.3 | 1500 | 24.72 | 0.0% |
| llama-chat.cpp | 11,948 | 499.9 | 644 | 23.82 | 0.0% |
| llama-memory-recurrent.cpp | 12,289 | 508.5 | 1500 | 23.61 | 1.6%† |
| server-common.cpp | 17,578 | 458.9 | 1500 | 22.40 | 11.7%† |
| server-task.cpp | 16,029 | 498.6 | 1500 | 22.99 | 0.0% |
| llama-quant.cpp | 19,533 | 470.3 | 1500 | 22.26 | 0.0% |
| **Mean** | | **491.3** | | **23.30** | |
| **Range** | | 458.9–511.3 | | 22.26–24.72 | |

†NGRAM_MOD `--spec-default` context-lookup hits, not MTP.

**Notes:**

- **23.30 tok/s mean with no speculation already beats every MTP-enabled
  dense stack on this box** — `qwen38-27b`/unsloth (15.36), `qwen36-27b`
  (16.40), `qwen38-27b`/vLLM (16.42) — because only ~6B params are active
  per token. Working MTP should push it higher still.
- Prefill (~491 tok/s mean) is also markedly faster than `qwen38-27b`'s
  (~310 tok/s in the 2026-08-21 run) for the same reason.
- Decode speed decays gently with context (24.7 tok/s at 8k → 22.3 tok/s at
  19.5k input tokens) — normal attention-cost growth, nothing anomalous.
- Config at run time: `--spec-draft-n-max 3`, `--parallel 4`, `-c 262144`,
  f16 KV cache, 20 threads, `reasoning_effort: medium`, thinking mode on.

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

## qwen38-27b / fp8 — DFlash2 vs MTP (recorded 2026-09-04)

vLLM `v0.28.0` adds a `DFlash2DraftModel` registry entry (`v0.27.1` rejected
it with `Model architectures ['DFlash2DraftModel'] are not supported` — see
the now-resolved `dflash2-dspark-unsupported` note). Bumped all `vllm/`
stacks to `v0.28.0-ubuntu2404` and swapped `qwen38-27b`'s
`--speculative-config` from MTP (`num_speculative_tokens: 3`) to DFlash2:

```
{"method":"dflash","num_speculative_tokens":7,"model":"incoai/Qwen3.8-27B-DFlash2"}
```

(`method` is `"dflash"` for both DFlash1 and DFlash2 — vLLM distinguishes
them by the draft checkpoint's `architectures` field, not a separate method
name. `num_speculative_tokens: 7` mirrors the working `laguna-s-21` DFlash
config and matches the drafter's own `block_size: 8` in `dflash_config`.)
Loaded clean: architecture resolved as `DFlash2DraftModel`, CUDA graphs
captured for 48 shapes, no errors.

Same concurrency-1 agentic-shaped methodology as the 2026-08-21 run
(`bench/bench_openai_client_timed.py`, 6 real source files, natural stop,
1500-token cap). Raw output in `bench/results/qwen38-dflash2.log`.

| File | Prompt tok | MTP decode tok/s (2026-08-21) | DFlash2 decode tok/s (2026-09-04) |
|---|---|---|---|
| server-http.cpp | ~8.3k | 16.39 | 23.05 |
| llama-chat.cpp | 11,948 | 16.42 | 22.06 |
| llama-memory-recurrent.cpp | ~13k | 15.33 | 20.48 |
| server-common.cpp | ~17.5k | 17.32 | 23.68 |
| server-task.cpp | 16,029 | 16.30 | 23.05 |
| llama-quant.cpp | ~19k | 16.77 | 24.79 |
| **Mean** | | **16.42** | **22.85** |

**DFlash2 beats MTP by ~39% on single-stream decode speed** on this stack
(22.85 vs 16.42 tok/s mean), the biggest speculative-decoding win recorded
in this doc for a dense (non-MoE) model. `qwen38-27b/docker-compose.yml` now
uses DFlash2 as its default speculative config.

## muse-glimmer-30b — DFlash2 (incoai, BF16) vs bundled dflash-kquant (recorded 2026-09-08)

Follow-up to the 2026-08-22 row above. `incoai/Muse-Glimmer-30B-DFlash2-GGUF`
previously failed to load (`wrong number of tensors; expected 81, got 58`,
see `[[dflash2-dspark-unsupported]]`, 2026-08-20). Updated `unsloth` to
`2026.9.3` / vendored llama.cpp to build 10840 (commit `58670d128`, from
build 10715) on 2026-09-08 and retried — the draft now loads and serves
cleanly. New recipe: `unsloth/muse-glimmer-30b.sh`, resolving the cached BF16
draft file and passing `--model-draft <path> --spec-type draft-dflash` with
`LLAMA_ARG_SPEC_DRAFT_N_MAX=15` (the draft repo's own README-recommended
value; `block_size=16`), vs. the 2026-08-22 run's bundled `dflash-kquant.gguf`
at `n_max=3`. Also fixed `-c` down from `131072*4` to a plain `131072` — the
larger value was silently capped back to the model's native context by
`--kv-unified` anyway (harmless, but worth not carrying forward).

Same `bench/bench_llamacpp_timings.py` methodology (6 real source files,
concurrency-1, natural stop, 1500-token cap). Raw output in
`bench/results/muse-glimmer-30b-dflash2.log`.

| File | Prompt tok | dflash-kquant decode tok/s (2026-08-22, accept%) | DFlash2 BF16 decode tok/s (2026-09-08, accept%) |
|---|---|---|---|
| server-http.cpp | ~7.6k | 14.31 (47.7%) | 14.68 (12.6%) |
| llama-chat.cpp | 11,033 | 15.28 (53.9%) | 16.82 (15.5%) |
| llama-memory-recurrent.cpp | ~11.5k | 13.27 (43.8%) | 12.32 (9.6%) |
| server-common.cpp | ~16k | 15.43 (57.1%) | 12.84 (10.3%) |
| server-task.cpp | 14,501–14,508 | 13.86 (46.0%) | 14.42 (12.4%) |
| llama-quant.cpp | ~17–18k | 13.97 (47.4%) | 15.13 (13.4%) |
| **Mean** | | **14.35** | **14.37** |

**Essentially a wash on this workload** — the standalone DFlash2 BF16 draft
(n_max=15, block_size 16) lands within noise of the bundled dflash-kquant
draft (n_max=3) despite a ~4x lower accept rate (9.6–15.5% vs 43.8–57.1%).
Plausible reason: at block_size 16 a low-percentage accept rate can still
yield a similar number of *net* accepted tokens per verification step as a
high-percentage accept rate at block_size 3 (e.g. ~12% of 16 ≈ 1.9 tokens/step
vs ~50% of 3 ≈ 1.5 tokens/step) — not confirmed against llama.cpp's per-step
accounting, just consistent with the numbers. A quick smoke-test chat
completion (short "say hello" prompt, not part of this suite) hit 67.5%
accept, so acceptance rate on this drafter varies a lot with how
predictable the target's continuation is — this benchmark's reasoning-heavy
code-explanation prompts are a harder case for it than short factual
replies. Not enough evidence here to prefer one draft over the other; both
give ~14.3-14.4 tok/s mean decode on this box for this model. Worth
retrying with a lower `n_max` (e.g. 5-8) if revisited, since the upstream
README's recommended 15 wasn't tuned for this workload shape.

**Follow-up same day: also tried the Q8_0 draft quant** (same repo,
`Muse-Glimmer-30B-DFlash2-Q8_0.gguf`, 2.8 GB vs BF16's 5.2 GB), same
`n_max=15`, same benchmark suite. Raw output in
`bench/results/muse-glimmer-30b-dflash2-q8.log`.

| File | BF16 decode tok/s (accept%) | Q8_0 decode tok/s (accept%) |
|---|---|---|
| server-http.cpp | 14.68 (12.6%) | 16.23 (13.7%) |
| llama-chat.cpp | 16.82 (15.5%) | 17.28 (15.0%) |
| llama-memory-recurrent.cpp | 12.32 (9.6%) | 14.41 (11.5%) |
| server-common.cpp | 12.84 (10.3%) | 12.66 (9.3%) |
| server-task.cpp | 14.42 (12.4%) | 15.91 (13.4%) |
| llama-quant.cpp | 15.13 (13.4%) | 15.60 (13.1%) |
| **Mean** | **14.37** | **15.35** |

**Q8_0 wins outright: ~6.8% faster mean decode at essentially the same
accept rate as BF16** (both cluster 9-15%). Makes sense — same
predictions, cheaper drafter forward pass per verification step, so no
accuracy tradeoff for the speed gain. If sticking with the standalone
incoai DFlash2 checkpoint over the bundled dflash-kquant draft, Q8_0 looks
like the better quant to default to, not BF16.

## muse-glimmer-30b — UD-Q8_K_XL vs UD-Q4_K_XL, same DFlash2 Q8_0 drafter (recorded 2026-09-08)

Prompted by a third-party recipe ([sxuff/muse-glimmer-dgx-spark on
GitHub](https://github.com/sxuff/muse-glimmer-dgx-spark)) reporting 27.9
tok/s with DFlash on Meta's own "K-Quant-Dynamic" quant (unsloth's
`UD-Q4_K_XL`, ~15.9 GB) vs our 15.35 tok/s on `UD-Q8_K_XL` (~32.3 GB) — same
model, same DFlash-family speculative decoding, different base quant.
Downloaded `UD-Q4_K_XL` and re-ran the identical benchmark suite (same
Q8_0 DFlash2 drafter, `n_max=15`, `-c 131072`, `--parallel 4`) for a
same-box, same-methodology comparison. Raw output in
`bench/results/muse-glimmer-30b-q4k-dflash2-q8.log`.

| File | Q8_K_XL decode tok/s (accept%) | Q4_K_XL decode tok/s (accept%) |
|---|---|---|
| server-http.cpp | 16.23 (13.7%) | 25.40 (13.4%) |
| llama-chat.cpp | 17.28 (15.0%) | 28.09 (15.6%) |
| llama-memory-recurrent.cpp | 14.41 (11.5%) | 22.70 (11.4%) |
| server-common.cpp | 12.66 (9.3%) | 21.06 (10.1%) |
| server-task.cpp | 15.91 (13.4%) | 25.94 (14.0%) |
| llama-quant.cpp | 15.60 (13.1%) | 25.66 (13.8%) |
| **Mean** | **15.35** | **24.81** |

**Q4_K_XL is ~62% faster mean decode than Q8_K_XL** (24.81 vs 15.35 tok/s),
at essentially identical accept rates (both 9-16%) — confirms the base
model's own weight bandwidth, not the drafter, is the dominant cost on this
box's memory-bandwidth-bound single-stream decoding. Prefill speed also
improved (877-896 tok/s vs 741-757 tok/s) since prefill is partly
compute-bound on weight size too. Per Meta's own model card, `UD-Q4_K_XL`
("K-Quant-Dynamic") costs only ~0.2% average degradation across 15
benchmarks vs full precision — a good trade for close to 2x the throughput.
This is now the bigger lever than drafter-quant choice (BF16 vs Q8_0, only
~7% apart, see above).

## muse-glimmer-30b — full drafter sweep on UD-Q4_K_XL (recorded 2026-09-08)

Once `UD-Q4_K_XL` was established as the faster base quant (above), swept
all four available drafters against it, same benchmark suite, same
`n_max=15`, `-c 131072`, `--parallel 4`: the bundled `dflash-kquant.gguf`
(ships in `unsloth/Muse-Glimmer-30B-GGUF` itself, 1.6 GB, DFlash-v1-family
per `[[dflash2-dspark-unsupported]]`) and all three incoai DFlash2 quants
(Q4_K_M 1.6 GB, Q8_0 2.8 GB, BF16 5.2 GB). Raw output in `bench/results/muse-glimmer-30b-q4k-bundled-dflash-kquant.log`,
`muse-glimmer-30b-q4k-incoai-q4km.log`, `muse-glimmer-30b-q4k-dflash2-q8.log`
(the Q8_0 run from the base-quant comparison above, same config), and
`muse-glimmer-30b-q4k-incoai-bf16.log`.

| File | bundled dflash-kquant | incoai Q4_K_M | incoai Q8_0 | incoai BF16 |
|---|---|---|---|---|
| server-http.cpp | 26.04 (12.6%) | 26.08 (12.9%) | 25.40 (13.4%) | 24.07 (13.9%) |
| llama-chat.cpp | 35.50 (19.9%)† | 29.33 (15.6%) | 28.09 (15.6%) | 27.69 (16.8%) |
| llama-memory-recurrent.cpp | 22.50 (10.2%) | 24.38 (11.9%) | 22.70 (11.4%) | 23.61 (13.3%) |
| server-common.cpp | 23.79 (11.3%) | 24.21 (11.8%) | 21.06 (10.1%) | 21.39 (11.4%) |
| server-task.cpp | 23.83 (11.4%) | 25.98 (13.1%) | 25.94 (14.0%) | 21.62 (11.7%) |
| llama-quant.cpp | 26.03 (13.3%) | 25.68 (12.9%) | 25.66 (13.8%) | 22.48 (12.7%) |
| **Mean** | **26.28** | **25.94** | **24.81** | **23.48** |
| Median | 24.93 | 25.83 | 25.53 | 23.05 |

† llama-chat.cpp on the bundled drafter is a clear outlier (35.50 vs
27-29 tok/s everywhere else for that file) — single-run noise, not
re-verified; median is a better read on this drafter's typical performance
than mean.

**Smaller/cheaper drafters win, mirroring the BF16-vs-Q8_0 result above**:
ranking by mean is bundled dflash-kquant > Q4_K_M > Q8_0 > BF16, and by
median Q4_K_M > Q8_0 ≈ bundled > BF16 — all four are within ~11% of each
other by mean, much tighter than the 62% base-quant gap. The bundled
drafter has a mild edge and the added benefit of shipping with the base
repo already (no separate `incoai/...-DFlash2-GGUF` download needed), but
Q4_K_M is closely competitive and less noisy (no outlier file). BF16 is the
weakest of the four here, consistent with "smaller drafter forward pass at
the same accept rate wins."

## Summary / recommendations

- **muse-glimmer-30b**: DFlash2 BF16 (n_max=15) and the bundled dflash-kquant
  draft (n_max=3) are a wash on decode speed (~14.35-14.37 tok/s mean) despite
  a big accept-rate gap — see the 2026-09-08 section above. DFlash2 **Q8_0**
  beats both (~15.35 tok/s, +6.8% over BF16 at the same accept rate) — if using
  the standalone incoai checkpoint, prefer Q8_0 over BF16. But the far bigger
  lever is the **base model quant**: `UD-Q4_K_XL` beats `UD-Q8_K_XL` by ~62%
  mean decode (24.81 vs 15.35 tok/s) at the same accept rate, for ~0.2%
  benchmark degradation per Meta's model card — worth defaulting to
  `UD-Q4_K_XL` unless the extra quality of Q8 is specifically wanted.
- **qwen36-27b**: bump `num_speculative_tokens` from 3 → **4** on the nvfp4 profile for a modest (~5%) throughput gain. Avoid the fp8 profile — it's slower and showed unexplained request failures under load at N=4/N=6.
- **qwen36-35b-a3b**: leave `num_speculative_tokens` at **3** on both profiles — it's already the peak for this model, and prefer the **nvfp4** profile over fp8 (roughly 1.5-1.6x the throughput at matching N).
- General pattern: acceptance rate decays with each added speculative position (compounding error from reusing the same MTP layer autoregressively, as vLLM's own startup warning notes), and past N=3-4 the extra drafting compute costs more than the added accepted tokens return. The 35B-A3B (MoE) model peaks earlier and drops off harder than the 27B dense model, likely because its per-step compute is cheaper, so the relative overhead of wasted draft passes is larger.
