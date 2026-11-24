#!/usr/bin/env python3
"""GPU-utilization telemetry for LLM serving, attributed to the model being served.

Runs as a telegraf `inputs.execd` plugin (signal = "STDIN"): NVML stays
initialized between collections, and every newline telegraf writes on stdin
triggers one collection, emitted as influx line protocol on stdout.

Why not just `inputs.nvidia_smi`: on the GB10 (unified memory) nvidia-smi
reports no memory at all, and its utilization.gpu is a single instantaneous
sample. This collector instead reports, per interval:

  gpu_serve        one row per GPU, tagged with the model(s) holding it
                   - sm_util_{avg,min,max,p50,p90}: every NVML utilization
                     sample (~6/s) since the previous collection, not one
                     snapshot; sm_busy_frac = fraction of samples > 0
                   - sm_clock_ratio / sm_util_clock_weighted: utilization
                     scaled by SM clock vs max, so a throttled or down-clocked
                     "100%" doesn't read as full use
                   - energy_j / power_avg_w from the cumulative energy counter
                     (exact, not sampled), power_w instantaneous
                   - gpu_mem_used_bytes: sum of per-process GPU allocations
                     (the only GPU-side memory NVML exposes on GB10), plus
                     host_mem_{total,available}_bytes since the pool is shared
                   - throttle reason flags, temperature, pstate
  gpu_serve_model  one row per serving stack (systemd unit or docker
                   container), tags model/engine/stack/service
                   - gpu_mem_bytes, sm_util_{avg,max} (NVML per-process
                     samples), host_rss_bytes, n_procs
                   - energy_j_attributed / power_w_attributed: the
                     interval's GPU energy (and its average power) split by
                     each stack's share of per-process SM utilization; 0 when
                     the GPU is idle

Model identification, first hit wins:
  1. docker label `serve.model` on the container — explicit override
  2. the GPU process's (or an ancestor's, same cgroup) command line:
     --served-model-name / --model / --model-path / `vllm serve X` /
     `unsloth run --model X` / llama-server -m <hf-cache path>
  3. GET /v1/models on the container's published port
  4. the stack name (compose project or unit instance)

Only stdlib + nvidia-ml-py (pynvml).
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import sys
import time
import urllib.request

import pynvml as nv

MEASUREMENT_GPU = "gpu_serve"
MEASUREMENT_MODEL = "gpu_serve_model"
DOCKER_SOCK = os.environ.get("DOCKER_SOCK", "/var/run/docker.sock")
HTTP_TIMEOUT = 2.0
# telegraf logs every stderr line at E! level, so stay quiet unless asked.
DEBUG = bool(os.environ.get("GPU_SERVE_DEBUG"))

THROTTLE_REASONS = {
    "sw_power_cap": "nvmlClocksEventReasonSwPowerCap",
    "hw_slowdown": "nvmlClocksEventReasonHwSlowdown",
    "hw_thermal": "nvmlClocksEventReasonHwThermalSlowdown",
    "hw_power_brake": "nvmlClocksEventReasonHwPowerBrakeSlowdown",
    "sw_thermal": "nvmlClocksEventReasonSwThermalSlowdown",
    "sync_boost": "nvmlClocksEventReasonSyncBoost",
    "idle": "nvmlClocksEventReasonGpuIdle",
}


def log(msg: str) -> None:
    print(f"gpu_serve_telemetry: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- line protocol

def _esc_tag(v: str) -> str:
    return re.sub(r"([,= ])", r"\\\1", str(v).replace("\\", "\\\\")) or "unknown"


def _fmt_field(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v}i"
    if isinstance(v, float):
        return repr(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def line(measurement: str, tags: dict, fields: dict, ts_ns: int) -> str | None:
    fields = {k: v for k, v in fields.items() if v is not None}
    if not fields:
        return None
    t = "".join(f",{k}={_esc_tag(v)}" for k, v in sorted(tags.items()) if v not in (None, ""))
    f = ",".join(f"{k}={_fmt_field(v)}" for k, v in sorted(fields.items()))
    return f"{measurement}{t} {f} {ts_ns}"


# ---------------------------------------------------------------- /proc helpers

def read(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            return fh.read().decode(errors="replace")
    except OSError:
        return None


def cmdline(pid: int) -> list[str]:
    raw = read(f"/proc/{pid}/cmdline")
    return [a for a in raw.split("\0") if a] if raw else []


def ppid(pid: int) -> int | None:
    stat = read(f"/proc/{pid}/stat")
    if not stat:
        return None
    # comm may contain spaces/parens; fields after the last ')'
    return int(stat.rsplit(")", 1)[1].split()[1])


def starttime(pid: int) -> str | None:
    stat = read(f"/proc/{pid}/stat")
    return stat.rsplit(")", 1)[1].split()[19] if stat else None


def cgroup(pid: int) -> str:
    raw = read(f"/proc/{pid}/cgroup") or ""
    for ln in raw.splitlines():
        if ln.startswith("0::"):
            return ln[3:]
    return ""


def rss_bytes(pid: int) -> int | None:
    raw = read(f"/proc/{pid}/status") or ""
    m = re.search(r"^VmRSS:\s+(\d+) kB", raw, re.M)
    return int(m.group(1)) * 1024 if m else None


def meminfo() -> dict[str, int]:
    out = {}
    for ln in (read("/proc/meminfo") or "").splitlines():
        k, _, rest = ln.partition(":")
        parts = rest.split()
        if parts:
            out[k] = int(parts[0]) * 1024
    return out


# ---------------------------------------------------------------- docker API

class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str):
        super().__init__("localhost", timeout=HTTP_TIMEOUT)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(HTTP_TIMEOUT)
        self.sock.connect(self._path)


def docker_inspect(cid: str) -> dict | None:
    try:
        conn = _UnixHTTPConnection(DOCKER_SOCK)
        conn.request("GET", f"/containers/{cid}/json")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return json.loads(body) if resp.status == 200 else None
    except (OSError, ValueError, http.client.HTTPException) as e:
        log(f"docker inspect {cid[:12]} failed: {e}")
        return None


# ---------------------------------------------------------------- model identification

HF_CACHE_RE = re.compile(r"models--([^/]+)--([^/]+)/snapshots/[^/]+/(.+)$")


def model_from_gguf_path(path: str) -> str:
    m = HF_CACHE_RE.search(path)
    if not m:
        return os.path.basename(path)
    org, repo, rest = m.groups()
    # quant = subdir (UD-IQ2_XS/...) or the quant suffix in the file name
    if "/" in rest:
        quant = rest.split("/")[0]
    else:
        q = re.search(r"[-.]((?:UD-)?I?Q\d[\w]*|BF16|F16|F32)(?:-\d{5}-of-\d{5})?\.gguf$", rest, re.I)
        quant = q.group(1) if q else rest
    return f"{org}/{repo}:{quant}"


def _flag(args: list[str], *names: str) -> str | None:
    for i, a in enumerate(args):
        for n in names:
            if a == n and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def parse_cmdline(args: list[str]) -> tuple[str | None, str | None]:
    """-> (model, engine) from one process's argv."""
    if not args:
        return None, None
    joined = " ".join(args[:4])
    base = os.path.basename(args[0])

    if base == "llama-server" or "llama-server" in joined:
        alias = _flag(args, "--alias", "-a")
        m = _flag(args, "-m", "--model")
        hf = _flag(args, "-hf", "--hf-repo")
        return alias or (model_from_gguf_path(m) if m else hf), "llama.cpp"
    if "unsloth" in joined and " run" in f" {joined}":
        return _flag(args, "--model"), "unsloth"
    if "sglang" in joined:
        return _flag(args, "--served-model-name") or _flag(args, "--model-path", "--model"), "sglang"
    if "vllm-srun" in joined:
        return None, "vllm-srun"  # model list lives in a yaml; fall through to /v1/models
    if "vllm" in joined:
        model = _flag(args, "--served-model-name") or _flag(args, "--model")
        if not model:
            for i, a in enumerate(args):
                if a == "serve" and i + 1 < len(args) and not args[i + 1].startswith("-"):
                    model = args[i + 1]
                    break
        return model, "vllm"
    return None, None


def query_v1_models(port: int) -> str | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=HTTP_TIMEOUT) as r:
            data = json.load(r).get("data") or []
        ids = sorted({d.get("id") for d in data if d.get("id")})
        return "+".join(ids) or None
    except Exception:
        return None


SYSTEMD_UNIT_RE = re.compile(r"/([^/]+)\.service$")
DOCKER_SCOPE_RE = re.compile(r"docker-([0-9a-f]{64})\.scope")


class Identity(dict):
    """tags: service, stack, engine, model."""


def identify(pid: int) -> Identity:
    cg = cgroup(pid)
    ident = Identity(service=None, stack=None, engine=None, model=None)
    labels: dict = {}
    host_ports: list[int] = []

    if m := DOCKER_SCOPE_RE.search(cg):
        info = docker_inspect(m.group(1)) or {}
        cfg = info.get("Config") or {}
        labels = cfg.get("Labels") or {}
        ident["service"] = (info.get("Name") or m.group(1)[:12]).lstrip("/")
        ident["stack"] = labels.get("com.docker.compose.project") or ident["service"]
        wd = labels.get("com.docker.compose.project.working_dir", "")
        if "/serve/" in wd:
            ident["stack"] = wd.split("/serve/", 1)[1]  # e.g. sglang/ripple
        ident["model"] = labels.get("serve.model")
        for bindings in ((info.get("NetworkSettings") or {}).get("Ports") or {}).values():
            for b in bindings or []:
                if b.get("HostPort", "").isdigit():
                    host_ports.append(int(b["HostPort"]))
    elif m := SYSTEMD_UNIT_RE.search(cg):
        unit = m.group(1) + ".service"
        ident["service"] = unit.removesuffix(".service")
        ident["stack"] = ident["service"].replace("@", "/")  # unsloth@x -> unsloth/x
    else:
        ident["service"] = os.path.basename(cmdline(pid)[0]) if cmdline(pid) else str(pid)
        ident["stack"] = ident["service"]

    # Walk the GPU process and its ancestors inside the same cgroup: vLLM's
    # GPU process is a bare "VLLM::EngineCore" whose argv names no model.
    # Keep the most specific name seen (the unsloth launcher's repo:quant beats
    # llama-server's --alias), unless a serve.model label pinned it.
    pinned = bool(ident["model"])
    cur, seen = pid, 0
    while cur and cur > 1 and seen < 8 and cgroup(cur) == cg:
        model, engine = parse_cmdline(cmdline(cur))
        ident["engine"] = ident["engine"] or engine
        if model and not pinned and _specificity(model) > _specificity(ident["model"]):
            ident["model"] = model
        cur, seen = ppid(cur), seen + 1

    if not ident["model"]:
        for port in host_ports:
            if model := query_v1_models(port):
                ident["model"] = model
                break
    ident["model"] = ident["model"] or ident["stack"]
    ident["engine"] = ident["engine"] or "unknown"
    return ident


# ---------------------------------------------------------------- collector

def _pct(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return float(sorted_vals[i])


def _specificity(model: str | None) -> tuple[int, int, int]:
    """Rank model strings: org/repo:quant > org/repo > bare name."""
    m = model or ""
    return (":" in m, "/" in m, len(m))


def _nv(fn, *a):
    try:
        return fn(*a)
    except nv.NVMLError:
        return None


class GpuState:
    def __init__(self, idx: int):
        self.idx = idx
        self.h = nv.nvmlDeviceGetHandleByIndex(idx)
        self.name = nv.nvmlDeviceGetName(self.h)
        self.uuid = nv.nvmlDeviceGetUUID(self.h)
        self.last_sample_ts = 0       # NVML us timestamp of the newest util sample seen
        self.last_proc_ts = int(time.time() * 1e6)
        self.last_energy_mj = _nv(nv.nvmlDeviceGetTotalEnergyConsumption, self.h)
        self.last_wall = time.monotonic()
        self.max_sm_clock = _nv(nv.nvmlDeviceGetMaxClockInfo, self.h, nv.NVML_CLOCK_SM)
        self.has_reasons = {k: getattr(nv, v) for k, v in THROTTLE_REASONS.items() if hasattr(nv, v)}


class Collector:
    def __init__(self):
        nv.nvmlInit()
        self.gpus = [GpuState(i) for i in range(nv.nvmlDeviceGetCount())]
        self.ident_cache: dict[tuple[int, str], Identity] = {}
        for g in self.gpus:  # prime the sample cursor so the first interval is bounded
            self._util_samples(g)

    def _identity(self, pid: int) -> Identity:
        key = (pid, starttime(pid) or "")
        if key not in self.ident_cache:
            self.ident_cache[key] = identify(pid)
            if DEBUG:
                log(f"pid {pid}: {dict(self.ident_cache[key])}")
        return self.ident_cache[key]

    def _util_samples(self, g: GpuState) -> list[int]:
        try:
            # Always ask for the whole ~14 s ring buffer: passing a non-zero lastSeen
            # makes this driver return NOT_FOUND most of the time. Dedupe client-side.
            _, samples = nv.nvmlDeviceGetSamples(g.h, nv.NVML_GPU_UTILIZATION_SAMPLES, 0)
        except nv.NVMLError:
            return []
        fresh = [s for s in samples if s.timeStamp > g.last_sample_ts]
        if fresh:
            g.last_sample_ts = max(s.timeStamp for s in fresh)
        return [int(s.sampleValue.uiVal) for s in fresh]

    def collect(self) -> list[str]:
        ts = time.time_ns()
        out: list[str] = []
        mi = meminfo()
        live_keys = set()

        for g in self.gpus:
            now = time.monotonic()
            dt = max(now - g.last_wall, 1e-3)
            g.last_wall = now
            tags = {"gpu": str(g.idx), "gpu_name": g.name}
            f: dict = {}

            # --- SM utilization over the interval
            vals = sorted(self._util_samples(g))
            if vals:
                f.update(
                    sm_util_avg=sum(vals) / len(vals),
                    sm_util_min=float(vals[0]),
                    sm_util_max=float(vals[-1]),
                    sm_util_p50=_pct(vals, 0.5),
                    sm_util_p90=_pct(vals, 0.9),
                    sm_busy_frac=sum(1 for v in vals if v > 0) / len(vals),
                    sm_util_samples=len(vals),
                )
            else:
                u = _nv(nv.nvmlDeviceGetUtilizationRates, g.h)
                if u is not None:
                    f["sm_util_avg"] = float(u.gpu)

            clk = _nv(nv.nvmlDeviceGetClockInfo, g.h, nv.NVML_CLOCK_SM)
            if clk is not None:
                f["sm_clock_mhz"] = clk
                if g.max_sm_clock:
                    f["sm_clock_max_mhz"] = g.max_sm_clock
                    f["sm_clock_ratio"] = clk / g.max_sm_clock
                    if "sm_util_avg" in f:
                        f["sm_util_clock_weighted"] = f["sm_util_avg"] * clk / g.max_sm_clock

            # --- power / energy
            p = _nv(nv.nvmlDeviceGetPowerUsage, g.h)
            f["power_w"] = p / 1000.0 if p is not None else None
            e = _nv(nv.nvmlDeviceGetTotalEnergyConsumption, g.h)
            energy_j = None
            if e is not None and g.last_energy_mj is not None and e >= g.last_energy_mj:
                energy_j = (e - g.last_energy_mj) / 1000.0
                f["energy_j"] = energy_j
                f["power_avg_w"] = energy_j / dt
            g.last_energy_mj = e
            f["interval_s"] = dt

            # --- thermals / throttling
            f["temp_c"] = _nv(nv.nvmlDeviceGetTemperature, g.h, nv.NVML_TEMPERATURE_GPU)
            f["pstate"] = _nv(nv.nvmlDeviceGetPerformanceState, g.h)
            reasons = _nv(nv.nvmlDeviceGetCurrentClocksEventReasons, g.h)
            if reasons is not None:
                f["throttle_reasons"] = int(reasons)
                for k, bit in g.has_reasons.items():
                    f[f"throttle_{k}"] = bool(reasons & bit)

            # --- memory: GB10 has no framebuffer counters; sum per-process allocations
            fb = _nv(nv.nvmlDeviceGetMemoryInfo, g.h)
            if fb is not None:
                f.update(fb_total_bytes=int(fb.total), fb_used_bytes=int(fb.used))
            procs = _nv(nv.nvmlDeviceGetComputeRunningProcesses, g.h) or []
            gpu_mem = {p.pid: int(p.usedGpuMemory or 0) for p in procs}
            f["gpu_mem_used_bytes"] = sum(gpu_mem.values())
            if mi.get("MemTotal"):
                f["host_mem_total_bytes"] = mi["MemTotal"]
                f["host_mem_available_bytes"] = mi.get("MemAvailable")
                f["gpu_mem_frac_of_pool"] = f["gpu_mem_used_bytes"] / mi["MemTotal"]
            f["n_procs"] = len(gpu_mem)

            # --- per-process SM utilization since last collection
            putil: dict[int, list[int]] = {}
            try:
                for s in nv.nvmlDeviceGetProcessUtilization(g.h, g.last_proc_ts):
                    putil.setdefault(s.pid, []).append(int(s.smUtil))
                    g.last_proc_ts = max(g.last_proc_ts, int(s.timeStamp))
            except nv.NVMLError:
                pass  # NOT_FOUND when no samples in window

            # --- aggregate per serving stack
            by_svc: dict[str, dict] = {}
            for pid in set(gpu_mem) | set(putil):
                ident = self._identity(pid)
                live_keys.add((pid, starttime(pid) or ""))
                a = by_svc.setdefault(ident["service"], {"ident": Identity(ident), "mem": 0, "rss": 0,
                                                         "util": [], "pids": 0, "top_mem": -1})
                # One stack, one tag set: engine from the process holding the most
                # GPU memory (the real worker), model from whichever process names
                # it most specifically (e.g. the unsloth launcher's repo:quant
                # beats llama-server's --alias).
                if gpu_mem.get(pid, 0) > a["top_mem"]:
                    a["top_mem"] = gpu_mem.get(pid, 0)
                    a["ident"]["engine"] = ident["engine"]
                if _specificity(ident["model"]) > _specificity(a["ident"]["model"]):
                    a["ident"]["model"] = ident["model"]
                a["mem"] += gpu_mem.get(pid, 0)
                a["rss"] += rss_bytes(pid) or 0
                a["pids"] += 1
                if putil.get(pid):
                    a["util"].append((sum(putil[pid]) / len(putil[pid]), max(putil[pid])))

            total_util = sum(sum(u for u, _ in a["util"]) for a in by_svc.values())
            models = sorted({a["ident"]["model"] for a in by_svc.values() if a["mem"] > 0})
            tags["models"] = "+".join(models) or "none"
            out.append(line(MEASUREMENT_GPU, tags, f, ts))

            for svc, a in by_svc.items():
                util_avg = sum(u for u, _ in a["util"])  # sum over the stack's processes
                mf = {
                    "gpu_mem_bytes": a["mem"],
                    "host_rss_bytes": a["rss"],
                    "n_procs": a["pids"],
                    "sm_util_avg": float(util_avg),
                    "sm_util_max": float(max((m for _, m in a["util"]), default=0)),
                    "gpu_mem_frac_of_pool": a["mem"] / mi["MemTotal"] if mi.get("MemTotal") else None,
                }
                if energy_j is not None:
                    share = util_avg / total_util if total_util > 0 else 0.0
                    mf["energy_j_attributed"] = energy_j * share
                    mf["power_w_attributed"] = energy_j * share / dt
                mt = {"gpu": str(g.idx), **{k: a["ident"][k] for k in ("model", "engine", "stack", "service")}}
                out.append(line(MEASUREMENT_MODEL, mt, mf, ts))

        for k in list(self.ident_cache):
            if k not in live_keys:
                del self.ident_cache[k]
        return [ln for ln in out if ln]


def main() -> int:
    once = "--once" in sys.argv
    c = Collector()
    if once:
        time.sleep(float(os.environ.get("ONCE_WINDOW_S", "2")))
        print("\n".join(c.collect()), flush=True)
        return 0
    for _ in sys.stdin:  # telegraf execd signal = "STDIN"
        try:
            lines = c.collect()
            if lines:
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
        except Exception as e:  # keep the daemon alive across transient errors
            log(f"collect failed: {e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
