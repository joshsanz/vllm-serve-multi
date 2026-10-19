# unsloth

GGUF models served via `unsloth run` (llama.cpp under the hood), one
standalone launcher script per model — same "single stack at a time" rule as
`vllm/` and `sglang/` (see top-level `README.md`), just invoked as a shell
script instead of `docker compose` since these aren't containerized.

## Running a script directly

```bash
bash unsloth/<model>.sh
```

This is fine for a quick foreground run, but it dies with your terminal/tmux
session. For anything meant to stay up, use the systemd user service below
instead.

**Gotcha**: `unsloth run`'s `-H 0.0.0.0 -p 8000` only binds the
*unsloth-studio* management API on `:8000` (auth-gated — `GET /v1/models`
there returns `401` without a bearer token, that's expected, not a health
check failure). The actual `llama-server` OpenAI-compatible endpoint listens
separately on `127.0.0.1:<dynamically-assigned-port>`, printed in the studio
startup log as `Starting llama-server: ... --port NNNNN`. Point clients (and
benchmark scripts — see `../bench/`) at that port, not `:8000`, for direct
inference. `:8000` is still the right port for the box's general "every
stack answers on :8000" convention if you're going through the studio proxy
with a token.

## Running as a systemd user service (durable, no tmux needed)

`systemd/unsloth@.service` is a templated unit — `%i` is the script's
basename (without `.sh`). Install once:

```bash
mkdir -p ~/.config/systemd/user
cp unsloth/systemd/unsloth@.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger "$USER"   # let user services run without an active login session
```

`enable-linger` is the part that actually removes the tmux dependency — without
it, systemd kills your user service when your last session logs out.

Then, per model:

```bash
# bring down whatever's currently running first (only one stack at a time)
systemctl --user stop unsloth@<currently-running-model>

# bring the new one up
systemctl --user start unsloth@<model>          # e.g. unsloth@inkling-small

systemctl --user status unsloth@<model>          # is it up?
journalctl --user -u unsloth@<model> -f          # follow logs (replaces manual log redirection)
systemctl --user stop unsloth@<model>            # tear down
```

Restarts automatically on crash (`Restart=on-failure`), capped at 5 restarts
per 2 minutes so a persistent failure (bad config, OOM, etc.) doesn't loop
forever burning GPU init attempts. Not enabled at boot
(`systemctl --user enable unsloth@<model>` would do that) — matches this
box's convention of bringing up a stack deliberately rather than
auto-starting a model on every reboot.

The unit's `Environment=PATH=...` is copied from this box's actual shell
`$PATH` at the time it was written (systemd user services don't inherit your
login shell's environment). If `unsloth run` or CUDA tooling ever moves, or
this is adapted for another box, that line needs updating to match.

## Layout

```
inkling-small.sh            Inkling-Small-GGUF (UD-IQ3_XXS), auto speculative (resolves to ngram-mod, see MTP_BENCHMARKS.md)
qwen38-flash-next.sh        Qwen3.8-Flash-Next-GGUF (UD-IQ4_XS), mtp speculative decoding via sidecar head passed through to llama-server as `--model-draft ... --spec-type draft-mtp` (studio's own `--speculative-type mtp` can't find the split-out head); 1.65x decode, ~72% accept (see MTP_BENCHMARKS.md, 2026-09-01)
muse-glimmer-30b.sh         Muse-Glimmer-30B-GGUF (UD-Q4_K_XL — ~62% faster decode than UD-Q8_K_XL at ~0.2% quality cost per Meta's card, see MTP_BENCHMARKS.md), DFlash2 speculative decoding via incoai/Muse-Glimmer-30B-DFlash2-GGUF (Q4_K_M draft, picked from a 4-way drafter sweep — all within ~11% of each other), passed through as `--model-draft ... --spec-type draft-dflash` for the same reason as the Flash-Next MTP head; confirmed working 2026-09-08 (previously hit a tensor-count mismatch, now fixed upstream — see the script's header comment)
archive/                    launcher recipes for models no longer cached on the box
systemd/unsloth@.service    templated systemd user unit for durable, tmux-free launches
```

`archive/` holds launcher recipes whose weights were removed from the HF
cache (2026-09-03) to keep the studio `/v1/models` cold scan fast — it
parses GGUF headers for every cached model on each rebuild. To bring one
back: `hf download` the repo, then move the script up out of `archive/`.

See `../vllm/MTP_BENCHMARKS.md` for throughput benchmarks across these
stacks (and every other cached model on the box).
