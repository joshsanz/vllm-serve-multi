# telemetry/

GPU-utilization telemetry for the serving stacks, attributed to the model
being served, shipped by telegraf to the same InfluxDB bucket as the rest of
the host stats (`dgx-monitor`).

`inputs.nvidia_smi` is weak on the GB10: unified memory means it reports no
memory at all, and `utilization.gpu` is one instantaneous sample with no idea
which model caused it. `gpu_serve_telemetry.py` is a long-running telegraf
`inputs.execd` collector (NVML via `nvidia-ml-py`) that adds:

| Measurement | Tags | Fields |
|---|---|---|
| `gpu_serve` (per GPU) | `gpu`, `gpu_name`, `models` (all resident models, `+`-joined) | `sm_util_{avg,min,max,p50,p90}` over **every** NVML sample (~6/s) in the interval, `sm_busy_frac`, `sm_clock_mhz`/`sm_clock_ratio`, `sm_util_clock_weighted`, `power_w`, `energy_j` + `power_avg_w` (from the cumulative energy counter, exact), `gpu_mem_used_bytes` (sum of per-process allocations), `host_mem_{total,available}_bytes`, `gpu_mem_frac_of_pool`, `temp_c`, `pstate`, `throttle_*` flags |
| `gpu_serve_model` (per serving stack) | `model`, `engine`, `stack` (e.g. `sglang/ripple`, `unsloth/qwen38-flash-next`), `service` (container or unit name), `gpu` | `gpu_mem_bytes`, `gpu_mem_frac_of_pool`, `sm_util_avg`/`sm_util_max` (NVML per-process samples), `host_rss_bytes`, `n_procs`, `energy_j_attributed` / `power_w_attributed` (interval energy and its average power, split by per-process SM share; 0 when idle) |

Stacks are found from each GPU process's cgroup: docker containers (compose
project / working dir) and systemd units (`unsloth@<name>`). The model comes
from the process command line (`--served-model-name`, `--model`,
`--model-path`, `vllm serve X`, `unsloth run --model X`, llama-server's
HF-cache `-m` path), falling back to the container's `/v1/models`, then the
stack name. To pin a name for a docker stack, add a `serve.model` label to
the service in its compose file.

## Install / redeploy

```bash
./install.sh
```

Copies the script to `/opt/serve-telemetry` (own venv; telegraf can't read
`$HOME`), drops `gpu_serve.conf` into `/etc/telegraf/telegraf.d/`, smoke-tests
as the `telegraf` user and restarts telegraf. Re-run after editing the script.

Debug locally (prints one interval, logs identification to stderr):

```bash
GPU_SERVE_DEBUG=1 /opt/serve-telemetry/venv/bin/python gpu_serve_telemetry.py --once
```

## Limits on GB10

- No NVML GPM and no framebuffer counters on this driver (580.178), so there
  are no SM-occupancy, tensor-pipe or DRAM-bandwidth fields. DCGM 4.8's
  Profiling module doesn't support GB10 either (`NVPW_DCGM_LoadDriver returned
  1`).
- `sm_util` is NVML's "a kernel was running" percentage, not SM occupancy. A
  memory-bound decode can read 95% while the SMs are mostly stalled. Read it
  next to `power_avg_w` (low watts at high util means memory-bound).
- Per-process `sm_util_avg` is sampled more coarsely than the device value
  and doesn't add up to it exactly.

## Example Flux

Energy per stack per hour:

```flux
from(bucket: "dgx-monitor")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "gpu_serve_model" and r._field == "energy_j_attributed")
  |> aggregateWindow(every: 1h, fn: sum)
  |> group(columns: ["model"])
```

Memory held vs SM work done, per model:

```flux
from(bucket: "dgx-monitor")
  |> range(start: -6h)
  |> filter(fn: (r) => r._measurement == "gpu_serve_model")
  |> filter(fn: (r) => r._field == "gpu_mem_bytes" or r._field == "sm_util_avg")
  |> aggregateWindow(every: 5m, fn: mean)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
```
