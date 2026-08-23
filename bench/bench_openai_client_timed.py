#!/usr/bin/env python3
"""Concurrency-1, agentic-shaped benchmark against any OpenAI-compatible
chat-completions endpoint (vLLM, sglang, ...). Rotates through 6 real source
files from this box's local llama.cpp checkout as long input context, each
paired with a realistic single coding question. Requests are sent one at a
time (no batching), generation stops naturally (1500-token cap as a safety
net, not --ignore-eos). Each file is used once per run, so every request
hits a cold/differently-shaped prefix.

These engines don't return a per-request server-side timings block like
llama.cpp does, so TTFT and decode tok/s are computed client-side from SSE
chunk arrival times instead.

Usage: bench_openai_client_timed.py --base-url http://localhost:8000 --model <served-model-name>
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


def has_token(delta):
    return bool(delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"))


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
    t_first_token = None
    t_last_token = None
    usage = None
    finish_reason = None
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
        now = time.monotonic()
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta", {})
            if has_token(delta):
                if t_first_token is None:
                    t_first_token = now
                t_last_token = now
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
        if chunk.get("usage"):
            usage = chunk["usage"]
    wall = time.monotonic() - t0
    return wall, t0, t_first_token, t_last_token, usage, finish_reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    results = []
    for i, (path, question) in enumerate(CASES, 1):
        name = path.split("/")[-1]
        print(f"[{i}/{len(CASES)}] {name} ...", file=sys.stderr, flush=True)
        wall, t0, t_first, t_last, usage, finish_reason = run_case(args.base_url, args.model, path, question)
        if usage is None:
            print(f"  no usage returned, skipping", file=sys.stderr)
            continue
        ttft = (t_first - t0) if t_first else None
        completion_tokens = usage["completion_tokens"]
        decode_span = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
        decode_tok_s = ((completion_tokens - 1) / decode_span) if decode_span else None
        row = {
            "file": name,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": completion_tokens,
            "ttft_s": ttft,
            "decode_tok_s": decode_tok_s,
            "finish_reason": finish_reason,
            "wall_s": wall,
        }
        results.append(row)
        print(
            f"  prompt={row['prompt_tokens']}tok  ttft={ttft:.2f}s  "
            f"output={completion_tokens}tok ({finish_reason})  "
            f"decode={decode_tok_s:.2f}tok/s  wall={wall:.1f}s",
            file=sys.stderr,
        )

    print("\n=== Summary (concurrency=1, natural stop, client-timed) ===")
    print(f"{'file':<28}{'prompt':>8}{'output':>8}{'ttft s':>9}{'decode t/s':>12}{'finish':>8}")
    for r in results:
        print(
            f"{r['file']:<28}{r['prompt_tokens']:>8}{r['completion_tokens']:>8}"
            f"{r['ttft_s']:>9.2f}{r['decode_tok_s']:>12.2f}{r['finish_reason']:>8}"
        )

    decode_speeds = [r["decode_tok_s"] for r in results if r["decode_tok_s"]]
    if decode_speeds:
        print(
            f"\nmean decode tok/s: {statistics.mean(decode_speeds):.2f}  "
            f"median: {statistics.median(decode_speeds):.2f}  "
            f"min: {min(decode_speeds):.2f}  max: {max(decode_speeds):.2f}"
        )


if __name__ == "__main__":
    main()
