#!/usr/bin/env python3
"""Concurrency-1, agentic-shaped benchmark against a llama.cpp llama-server
OpenAI-compatible endpoint (as used by the unsloth/* stacks). Uses
llama.cpp's own per-request `timings` block (exact server-side
prefill/decode speed, draft accept rate) instead of client-side wall-clock
math, and rotates through the same 6 real source files as
bench_openai_client_timed.py for direct cross-engine comparison.

Usage: bench_llamacpp_timings.py --base-url http://localhost:8000 --model <model-name>
"""
import argparse
import json
import statistics
import sys
import time

import requests

LLAMA_CPP = "/home/jsanz/.unsloth/llama.cpp"

CASES = [
    (f"{LLAMA_CPP}/tools/server/server-http.cpp",
     "Explain what this file does and trace the main HTTP request-handling flow."),
    (f"{LLAMA_CPP}/src/llama-chat.cpp",
     "Summarize the main functions in this file and how chat template dispatch works."),
    (f"{LLAMA_CPP}/src/llama-memory-recurrent.cpp",
     "What does this class do? Point out any parts that look error-prone or worth refactoring."),
    (f"{LLAMA_CPP}/tools/server/server-common.cpp",
     "Find any TODOs or obvious unhandled edge cases in this file."),
    (f"{LLAMA_CPP}/tools/server/server-task.cpp",
     "Explain how this file models a generation task's lifecycle, start to finish."),
    (f"{LLAMA_CPP}/src/llama-quant.cpp",
     "Walk through the quantization pipeline implemented in this file."),
]

MAX_TOKENS = 1500


def run_case(base_url, model, path, question):
    with open(path) as f:
        code = f.read()
    prompt = f"Here is a source file (`{path.split('/')[-1]}`):\n\n```cpp\n{code}\n```\n\n{question}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.monotonic()
    ttft = None
    final = None
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=600)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if ttft is None and chunk.get("choices") and chunk["choices"][0].get("delta", {}).get(
            "content"
        ) is not None:
            ttft = time.monotonic() - t0
        if "timings" in chunk:
            final = chunk
    wall = time.monotonic() - t0
    return wall, ttft, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    results = []
    for i, (path, question) in enumerate(CASES, 1):
        name = path.split("/")[-1]
        print(f"[{i}/{len(CASES)}] {name} ...", file=sys.stderr, flush=True)
        wall, ttft, final = run_case(args.base_url, args.model, path, question)
        if final is None:
            print(f"  no timings returned, skipping", file=sys.stderr)
            continue
        timings = final["timings"]
        usage = final["usage"]
        row = {
            "file": name,
            "prompt_tokens": usage["prompt_tokens"],
            "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            "completion_tokens": usage["completion_tokens"],
            "prompt_tok_s": timings["prompt_per_second"],
            "decode_tok_s": timings["predicted_per_second"],
            "draft_n": timings.get("draft_n", 0),
            "draft_accepted": timings.get("draft_n_accepted", 0),
            "wall_s": wall,
        }
        results.append(row)
        acc_rate = (row["draft_accepted"] / row["draft_n"] * 100) if row["draft_n"] else 0.0
        print(
            f"  prompt={row['prompt_tokens']}tok (cached {row['cached_tokens']}) "
            f"prefill={row['prompt_tok_s']:.1f}tok/s  "
            f"output={row['completion_tokens']}tok decode={row['decode_tok_s']:.2f}tok/s  "
            f"accept={acc_rate:.1f}%  wall={wall:.1f}s",
            file=sys.stderr,
        )

    print("\n=== Summary (concurrency=1, natural stop) ===")
    print(f"{'file':<28}{'prompt':>8}{'cached':>8}{'output':>8}{'prefill t/s':>13}{'decode t/s':>12}{'accept%':>9}")
    for r in results:
        acc_rate = (r["draft_accepted"] / r["draft_n"] * 100) if r["draft_n"] else 0.0
        print(
            f"{r['file']:<28}{r['prompt_tokens']:>8}{r['cached_tokens']:>8}{r['completion_tokens']:>8}"
            f"{r['prompt_tok_s']:>13.1f}{r['decode_tok_s']:>12.2f}{acc_rate:>8.1f}%"
        )

    decode_speeds = [r["decode_tok_s"] for r in results]
    if decode_speeds:
        print(
            f"\nmean decode tok/s: {statistics.mean(decode_speeds):.2f}  "
            f"median: {statistics.median(decode_speeds):.2f}  "
            f"min: {min(decode_speeds):.2f}  max: {max(decode_speeds):.2f}"
        )


if __name__ == "__main__":
    main()
