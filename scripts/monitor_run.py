#!/usr/bin/env python3
"""
monitor_run.py — sidecar diagnostics for train_rl.py runs.

Samples GPU utilisation, CPU, and memory every --interval seconds while a
training run is live, then prints a phase-breakdown report and saves a JSON
sidecar next to the monitored log file.

Usage
-----
  # In a second terminal, before or after launching train_rl.py:
  python scripts/monitor_run.py --log runs/aug_v3_rewardA/run.txt

  # Stop with Ctrl-C or let it exit when the run finishes.

Output
------
  <log_basename>.monitor.json   — raw sample data
  Summary printed to stdout on exit.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _gpu_stats():
    """Return list of {index, util_pct, mem_used_mib, mem_total_mib} dicts."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode()
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            rows.append({
                "index":          int(parts[0]),
                "util_pct":       int(parts[1]),
                "mem_used_mib":   int(parts[2]),
                "mem_total_mib":  int(parts[3]),
            })
        return rows
    except Exception:
        return []


def _system_stats():
    """Return {cpu_pct, ram_used_gb, ram_total_gb, swap_used_gb}."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                k, v = line.split(":", 1)
                meminfo[k.strip()] = int(v.strip().split()[0])  # kB
        ram_total = meminfo["MemTotal"] / 1048576
        ram_free  = (meminfo["MemFree"] + meminfo["Buffers"] + meminfo.get("Cached", 0)) / 1048576
        swap_used = (meminfo["SwapTotal"] - meminfo["SwapFree"]) / 1048576
    except Exception:
        ram_total = ram_free = swap_used = 0.0

    try:
        with open("/proc/stat") as f:
            cpu_line = f.readline().split()
        user, nice, system, idle, iowait = [int(x) for x in cpu_line[1:6]]
        total = user + nice + system + idle + iowait
        busy  = total - idle - iowait
        cpu_pct = busy * 100.0 / total if total else 0.0
    except Exception:
        cpu_pct = 0.0

    return {
        "cpu_pct":      round(cpu_pct, 1),
        "ram_used_gb":  round(ram_total - ram_free, 2),
        "ram_total_gb": round(ram_total, 2),
        "swap_used_gb": round(swap_used, 2),
    }


def _tail_new(path: Path, offset: int) -> tuple[str, int]:
    """Read new bytes from path since offset, return (new_text, new_offset)."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        return chunk.decode("utf-8", errors="replace"), offset + len(chunk)
    except Exception:
        return "", offset


# ── phase detection ───────────────────────────────────────────────────────────

_RE_POOLING    = re.compile(r"\[PoolingSavings @ (\d+)\]")
_RE_TIMESTEPS  = re.compile(r"total_timesteps\s*\|\s*(\d+)")
_RE_FPS        = re.compile(r"\|\s*fps\s*\|\s*(\d+)")


@dataclass
class Phase:
    name: str           # "training" or "callback"
    start_ts: float
    end_ts: float = 0.0
    samples: list = field(default_factory=list)

    def duration(self) -> float:
        return (self.end_ts or time.time()) - self.start_ts

    def avg_gpu_util(self) -> float:
        vals = [s["gpu"][0]["util_pct"] for s in self.samples if s.get("gpu")]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_ram_gb(self) -> float:
        vals = [s["sys"]["ram_used_gb"] for s in self.samples]
        return sum(vals) / len(vals) if vals else 0.0


# ── main ──────────────────────────────────────────────────────────────────────

def monitor(log_path: Path, interval: float):
    out_path = log_path.with_suffix(".monitor.json")
    print(f"[monitor] watching {log_path}  →  {out_path}")
    print(f"[monitor] sampling every {interval}s  |  Ctrl-C to stop\n")

    all_samples = []
    phases: list[Phase] = []
    current_phase = Phase("startup", start_ts=time.time())
    phases.append(current_phase)

    log_offset = 0
    last_timesteps = 0
    last_fps_vals = []
    in_callback = False

    try:
        while True:
            t = time.time()
            gpu  = _gpu_stats()
            sys_ = _system_stats()
            sample = {"t": round(t, 2), "gpu": gpu, "sys": sys_}
            all_samples.append(sample)
            current_phase.samples.append(sample)

            # Parse new log lines
            new_text, log_offset = _tail_new(log_path, log_offset)
            for line in new_text.splitlines():
                m_cb = _RE_POOLING.search(line)
                m_ts = _RE_TIMESTEPS.search(line)
                m_fps = _RE_FPS.search(line)

                if m_fps:
                    last_fps_vals.append(int(m_fps.group(1)))

                if m_ts:
                    last_timesteps = int(m_ts.group(1))

                if m_cb and not in_callback:
                    # Callback just logged — end the training phase, start callback phase
                    current_phase.end_ts = t
                    cb_step = int(m_cb.group(1))
                    current_phase = Phase(f"callback@{cb_step}", start_ts=t)
                    phases.append(current_phase)
                    in_callback = True

                # Detect end of callback: next fps/timesteps log after a callback
                if in_callback and (m_fps or m_ts):
                    current_phase.end_ts = t
                    current_phase = Phase("training", start_ts=t)
                    phases.append(current_phase)
                    in_callback = False

            # Live status line
            g0 = gpu[0] if gpu else {}
            fps_disp = last_fps_vals[-1] if last_fps_vals else "?"
            phase_tag = "CB  " if in_callback else "train"
            print(
                f"\r[{phase_tag}] step={last_timesteps:>9,}  fps={fps_disp:>5}  "
                f"GPU={g0.get('util_pct','?'):>3}%  {g0.get('mem_used_mib','?')}MiB  "
                f"RAM={sys_['ram_used_gb']:.1f}/{sys_['ram_total_gb']:.0f}GB  "
                f"swap={sys_['swap_used_gb']:.1f}GB     ",
                end="", flush=True
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[monitor] interrupted")
    finally:
        current_phase.end_ts = time.time()
        _report(phases, last_fps_vals, out_path, all_samples)


def _report(phases, fps_vals, out_path, all_samples):
    print("\n\n" + "=" * 70)
    print("  EXECUTION DIAGNOSIS REPORT")
    print("=" * 70)

    training_phases = [p for p in phases if p.name == "training"]
    callback_phases = [p for p in phases if p.name.startswith("callback")]
    startup_phases  = [p for p in phases if p.name == "startup"]

    def _fmt(secs):
        m, s = divmod(int(secs), 60)
        return f"{m}m {s:02d}s"

    total_train   = sum(p.duration() for p in training_phases)
    total_cb      = sum(p.duration() for p in callback_phases)
    total_startup = sum(p.duration() for p in startup_phases)
    total         = total_startup + total_train + total_cb

    print(f"\n  Startup:        {_fmt(total_startup):>12}  ({100*total_startup/total:.1f}%)")
    print(f"  Training:       {_fmt(total_train):>12}  ({100*total_train/total:.1f}%)")
    print(f"  Callbacks:      {_fmt(total_cb):>12}  ({100*total_cb/total:.1f}%)")
    print(f"  Total:          {_fmt(total):>12}")

    if callback_phases:
        cb_durs = [p.duration() for p in callback_phases]
        print(f"\n  Callback stats ({len(callback_phases)} evals):")
        print(f"    avg duration:    {_fmt(sum(cb_durs)/len(cb_durs))}")
        print(f"    min/max:         {_fmt(min(cb_durs))} / {_fmt(max(cb_durs))}")

    if fps_vals:
        print(f"\n  Reported fps (SB3):  min={min(fps_vals)}  max={max(fps_vals)}  "
              f"median={sorted(fps_vals)[len(fps_vals)//2]}")

    # GPU utilisation by phase type
    if all_samples and all_samples[0].get("gpu"):
        train_utils = [s["gpu"][0]["util_pct"] for p in training_phases for s in p.samples]
        cb_utils    = [s["gpu"][0]["util_pct"] for p in callback_phases for s in p.samples]
        all_ram     = [s["sys"]["ram_used_gb"] for s in all_samples]
        print(f"\n  GPU utilisation:")
        if train_utils:
            print(f"    during training:   avg {sum(train_utils)/len(train_utils):.1f}%  "
                  f"max {max(train_utils)}%")
        if cb_utils:
            print(f"    during callbacks:  avg {sum(cb_utils)/len(cb_utils):.1f}%  "
                  f"max {max(cb_utils)}%")
        if all_ram:
            print(f"\n  RAM: avg {sum(all_ram)/len(all_ram):.1f} GB  peak {max(all_ram):.1f} GB")

    print(f"\n  Full sample data → {out_path}")
    print("=" * 70)

    # Save JSON
    data = {
        "phases": [
            {**asdict(p), "duration_s": p.duration(),
             "avg_gpu_util": p.avg_gpu_util(), "avg_ram_gb": p.avg_ram_gb()}
            for p in phases
        ],
        "fps_samples": fps_vals,
        "n_samples": len(all_samples),
    }
    # Remove heavy per-sample lists from phases to keep JSON lean
    for d in data["phases"]:
        d.pop("samples", None)

    out_path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sidecar diagnostics for train_rl.py")
    ap.add_argument("--log", required=True, help="Path to run.txt being written")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="Sampling interval in seconds (default: 10)")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        # Create empty file so tail works immediately
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()

    monitor(log_path, args.interval)
