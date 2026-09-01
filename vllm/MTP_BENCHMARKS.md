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
checkpoint **landed**, but not through the path the stack currently uses:

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
- **`unsloth run` (studio 2026.8.22) still can't drive it.** It reads
  `--speculative-type mtp`, checks the *main* `UD-IQ4_XS` quant for a
  baked-in MTP head, doesn't find one (correct — it's a separate file now),
  logs `"MTP requested but this GGUF has no MTP head or drafter; loading
  without speculative decoding."`, and falls back to `--spec-default`
  n-gram. The `qwen38-flash-next.sh` stack as committed is therefore still
  running with **no MTP** — the 2026-08-27 numbers remain current for it.

This run **bypassed `unsloth run`** and invoked `llama-server` directly:

```
~/.unsloth/llama.cpp/llama-server \
  -m .../UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  -md .../MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --parallel 4 -c 262144 --flash-attn on --cache-type-k f16 --cache-type-v f16 \
  --threads 20 --jinja
```

The draft model loaded cleanly (`common_speculative_init_result: loading
draft model '...shared-Q8_0.gguf'`, no rejection — the shared-tensor
borrow worked). Same 6-file `bench_llamacpp_timings.py` methodology as
every run in this doc. Raw output: `bench/results/qwen38-flash-next-mtp.log`.

| File | Prompt tok | Prefill tok/s | Decode tok/s | Accept% |
|---|---|---|---|---|
| server-http.cpp | 8,309 | 512.4 | 35.44 | 59.0% |
| llama-chat.cpp | 11,990 | 526.5 | 35.41 | 61.9% |
| llama-memory-recurrent.cpp | 13,077 | 520.4 | 30.60 | 48.1% |
| server-common.cpp | 17,694 | 512.6 | 32.43 | 57.0% |
| server-task.cpp | 16,071 | 518.6 | 33.12 | 59.2% |
| llama-quant.cpp | 19,101 | 508.6 | 33.03 | 60.2% |
| **Mean** | | **516.5** | **33.34** | **57.6%** |
| **Range** | | 508.6–526.5 | 30.60–35.44 | 48.1–61.9% |

**MTP vs. no-MTP baseline (2026-08-27), same box, same methodology:**

| | Decode tok/s (mean) | Accept% |
|---|---|---|
| No MTP (`--spec-default` n-gram) | 23.30 | ~0 (2 files hit 1.6% / 11.7%) |
| **MTP (`draft-mtp`, shared-Q8_0, N=2)** | **33.34** | **57.6%** |
| Speedup | **1.43×** | |

**Notes:**

- **1.43× decode speedup**, in the middle of unsloth's own "1.3–1.7× at
  low concurrency" claim (their `MTP/README.md`). Server-side
  `draft acceptance` log lines corroborate the client-side accept%
  (0.48–0.62, mean draft len ~2.1–2.2 at N=2).
- Prefill is unchanged (~516 vs ~491 tok/s, within noise) — MTP only
  touches decode.
- These prompts run at the HF-recommended thinking-mode sampler
  (temp 1.0 / top-p 0.95 / top-k 20), **not greedy** — unsloth's headline
  1.67× is a greedy number, so 1.43× at temp 1.0 is consistent with their
  "higher temperature → less predictable → smaller speedup" caveat.
- unsloth's `MTP/README.md` says **skip MTP for concurrent serving**
  (measured net loss ~0.81–0.87× at concurrency 8) — fine here, this box
  runs one coding agent at a time.
- **To make `qwen38-flash-next.sh` actually use this**: either wait for a
  studio release that auto-fetches the `MTP/` sidecar, or convert the
  script to a direct `llama-server` invocation with the `-md` +
  `--spec-type draft-mtp` flags above. Not changed yet — flagged for a
  follow-up once studio support (or a decision to bypass it) lands.

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

## Summary / recommendations

- **qwen36-27b**: bump `num_speculative_tokens` from 3 → **4** on the nvfp4 profile for a modest (~5%) throughput gain. Avoid the fp8 profile — it's slower and showed unexplained request failures under load at N=4/N=6.
- **qwen36-35b-a3b**: leave `num_speculative_tokens` at **3** on both profiles — it's already the peak for this model, and prefer the **nvfp4** profile over fp8 (roughly 1.5-1.6x the throughput at matching N).
- General pattern: acceptance rate decays with each added speculative position (compounding error from reusing the same MTP layer autoregressively, as vLLM's own startup warning notes), and past N=3-4 the extra drafting compute costs more than the added accepted tokens return. The 35B-A3B (MoE) model peaks earlier and drops off harder than the 27B dense model, likely because its per-step compute is cheaper, so the relative overhead of wasted draft passes is larger.
