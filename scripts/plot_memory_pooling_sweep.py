#!/usr/bin/env python3
"""
Generate publication-style plots for memory pooling experiments.

This script is meant to reproduce the kinds of figures shown in slides:
  - Required Memory Capacity (%) vs Pod Size
  - Fully-connected vs Regular Octopus (N=4) under varying CXL%
  - (Optional) Greedy vs Optimal vs Fully-connected (for a chosen CXL%)
  - (Optional) MPD/MHD usage time series for a representative run

It uses the same trace + event logic as `evaluate.py` / `memory-pooling-v0.ipynb`,
but adds (a) CXL% scaling and (b) parameter sweeps + plotting.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from octopus.data import load_trace
from octopus.baselines import greedy_alloc
from octopus.topology import generate_pod_to_nodes, expand_M_to_all_nodes
from octopus.optimal import find_optimal


MEM_IDX = 1  # index into rss/machine_sz: memory (GB)


def _fully_connected_matrix(pod_size: int, pools_factor: int = 2) -> List[List[int]]:
    num_hosts = int(pod_size)
    num_pools = int(pools_factor) * num_hosts
    return [[1] * num_pools for _ in range(num_hosts)]


def _regular_octopus_matrix(
    pod_size: int,
    host_degree: int = 8,
    pools_factor: int = 2,
    seed: int = 0,
    max_tries: int = 200,
) -> List[List[int]]:
    """
    Generate a bipartite regular-ish Octopus topology:
      - L = pod_size hosts
      - Y = pools_factor * L pools
      - Each host has exactly host_degree edges
      - Each pool has exactly N edges, where N is implied by degree balance
    """
    L = int(pod_size)
    Y = int(pools_factor) * L
    X = int(host_degree)
    if L <= 0 or Y <= 0 or X <= 0:
        raise ValueError("Invalid pod_size / pools_factor / host_degree")
    total_edges = L * X
    if total_edges % Y != 0:
        raise ValueError(
            f"Degree mismatch: L*X={total_edges} not divisible by Y={Y} "
            f"(choose pools_factor so that Y divides total edges)"
        )
    N = total_edges // Y
    if N <= 0:
        raise ValueError("Implied pool degree must be positive")

    rng = random.Random(seed)

    for _attempt in range(max_tries):
        # remaining pool capacities
        rem = [N] * Y
        M = [[0] * Y for _ in range(L)]
        ok = True

        # Randomize host order to reduce bias.
        hosts = list(range(L))
        rng.shuffle(hosts)

        for h in hosts:
            # Candidate pools with remaining capacity
            cand = [p for p in range(Y) if rem[p] > 0]
            if len(cand) < X:
                ok = False
                break

            # Weighted sampling without replacement (weight = remaining cap)
            chosen = []
            local_rem = rem[:]  # small; fine at these sizes
            for _k in range(X):
                cand = [p for p in cand if local_rem[p] > 0 and M[h][p] == 0]
                if not cand:
                    ok = False
                    break
                weights = [local_rem[p] for p in cand]
                p = rng.choices(cand, weights=weights, k=1)[0]
                chosen.append(p)
                local_rem[p] -= 1
            if not ok:
                break

            for p in chosen:
                if rem[p] <= 0:
                    ok = False
                    break
                M[h][p] = 1
                rem[p] -= 1
            if not ok:
                break

        if ok and all(r == 0 for r in rem):
            return M

    raise RuntimeError("Failed to construct a regular Octopus matrix (try more max_tries)")


@dataclass
class _Trace:
    all_vms: Dict
    node_to_vms: Dict
    node_to_machine: Dict
    machine_sz: Dict


def _load_trace(trace: str) -> _Trace:
    all_vms, node_to_vms, node_to_machine, _vtsz, machine_sz = load_trace(trace)
    return _Trace(
        all_vms=all_vms,
        node_to_vms=node_to_vms,
        node_to_machine=node_to_machine,
        machine_sz=machine_sz,
    )


def _to_tick(base: _dt.datetime, t: _dt.datetime) -> int:
    return int((t - base).total_seconds() // 300)


def _build_events_and_skip(
    tr: _Trace, node_ids: List[int], base: _dt.datetime, dur: int, cxl_pct: float
) -> Tuple[List[List[Tuple[int, int, float]]], np.ndarray]:
    """
    Build alloc events: per tick list of (node_idx, dealloc_tick, mem_gb).
    Also returns pod_rss (vector) for denominator.
    """
    pod_rss = np.zeros(4, dtype=float)
    vmkey_to_skip: set = set()

    # First pass: per-node skip VMs that exceed local DRAM (not CXL scaled)
    for node in node_ids:
        pod_rss += np.asarray(tr.machine_sz[tr.node_to_machine[node]], dtype=float)

        alloc_events_node: List[List[Tuple[int, int, float]]] = [[] for _ in range(dur)]
        dealloc_events_node = np.zeros(dur, dtype=float)
        node_rss = np.asarray(tr.machine_sz[tr.node_to_machine[node]], dtype=float)

        for vmkey in tr.node_to_vms.get(node, []):
            vm = tr.all_vms[vmkey]
            vm_start = _to_tick(base, vm.start_time)
            vm_end = _to_tick(base, vm.end_time)
            if vm_end < 0:
                vmkey_to_skip.add(vmkey)
                continue
            vm_start = max(vm_start, 0)
            vm_end = min(vm_end, dur - 1)

            mem = float(np.asarray(vm.rss, dtype=float)[MEM_IDX])
            alloc_events_node[vm_start].append((vmkey, vm_end + 1, mem))
            if vm_end + 1 < dur:
                dealloc_events_node[vm_end + 1] += mem

        cur_mem = 0.0
        for ts in range(dur):
            cur_mem -= dealloc_events_node[ts]
            if cur_mem < 0:
                cur_mem = 0.0
            for vmkey, end_ts, mem in alloc_events_node[ts]:
                cur_mem += mem
                if cur_mem > node_rss[MEM_IDX]:
                    cur_mem -= mem
                    vmkey_to_skip.add(vmkey)
                    if end_ts < dur:
                        dealloc_events_node[end_ts] -= mem

    # Second pass: build events for simulation (CXL scaled)
    alloc_events_sim: List[List[Tuple[int, int, float]]] = [[] for _ in range(dur)]
    for node_idx, node in enumerate(node_ids):
        for vmkey in tr.node_to_vms.get(node, []):
            if vmkey in vmkey_to_skip:
                continue
            vm = tr.all_vms[vmkey]
            vm_start = _to_tick(base, vm.start_time)
            vm_end = _to_tick(base, vm.end_time)
            if vm_end < 0:
                continue
            vm_start = max(vm_start, 0)
            vm_end = min(vm_end, dur - 1)

            mem = float(np.asarray(vm.rss, dtype=float)[MEM_IDX]) * float(cxl_pct)
            if mem <= 0:
                continue
            alloc_events_sim[vm_start].append((node_idx, vm_end + 1, mem))

    return alloc_events_sim, pod_rss


def _time_window(tr: _Trace, node_ids: List[int]) -> Tuple[_dt.datetime, _dt.datetime]:
    n_start_time = _dt.datetime.max
    n_end_time = _dt.datetime.min
    any_vm = False
    for node in node_ids:
        for vmkey in tr.node_to_vms.get(node, []):
            vm = tr.all_vms[vmkey]
            any_vm = True
            if vm.start_time < n_start_time:
                n_start_time = vm.start_time
            if vm.end_time > n_end_time:
                n_end_time = vm.end_time
    if not any_vm:
        return _dt.datetime(2000, 1, 1), _dt.datetime(2000, 1, 1)
    return n_start_time, n_end_time


def greedy_required_capacity(
    tr: _Trace,
    node_to_M: Dict[int, int],
    M: List[List[int]],
    cxl_pct: float,
) -> float:
    """Return required capacity in [0, 1] (same as notebook: 1 - ratio)."""
    nodes = list(node_to_M.keys())
    start, end = _time_window(tr, nodes)
    base = _dt.datetime(start.year, start.month, start.day, start.hour, start.minute)
    pod_start = _to_tick(base, start)
    pod_end = _to_tick(base, end)
    dur = pod_end - pod_start + 1
    if dur <= 0:
        return 0.0

    # normalize ticks relative to pod_start
    # We'll shift by pod_start by just re-basing all times on the same base and
    # clipping during event build (base tick 0 == base time).
    alloc_events, pod_rss = _build_events_and_skip(tr, nodes, base, dur, cxl_pct)

    M_np = np.asarray(M, dtype=np.int8)
    num_nodes, num_mhd = M_np.shape
    acc = [np.nonzero(M_np[i])[0].tolist() for i in range(num_nodes)]

    cur = np.zeros(num_mhd, dtype=float)
    maxv = np.zeros(num_mhd, dtype=float)
    dealloc = np.zeros((dur, num_mhd), dtype=float)

    for ts, evs in enumerate(alloc_events):
        cur -= dealloc[ts, :]
        cur = np.maximum(cur, 0.0)

        for node_idx, end_ts, mem in evs:
            mhd_list = acc[node_idx]
            if not mhd_list:
                continue
            alloc_vec = greedy_alloc(mem, mhd_list, cur)
            cur += alloc_vec
            if end_ts < dur:
                dealloc[end_ts, :] += alloc_vec

        maxv = np.maximum(maxv, cur)

    denom = float(pod_rss[MEM_IDX])
    if denom <= 0:
        return 0.0
    ratio = float(np.max(maxv)) * float(num_mhd) / denom
    return float(1.0 - ratio)


def rl_required_capacity(
    tr: _Trace,
    node_to_M: Dict[int, int],
    M: List[List[int]],
    cxl_pct: float,
    model,
    max_degree: int,
) -> float:
    """RL per-event allocation. Returns required capacity in [0, 1]."""
    nodes = list(node_to_M.keys())
    start, end = _time_window(tr, nodes)
    base = _dt.datetime(start.year, start.month, start.day, start.hour, start.minute)
    dur = _to_tick(base, end) - _to_tick(base, start) + 1
    if dur <= 0:
        return 0.0

    alloc_events, pod_rss = _build_events_and_skip(tr, nodes, base, dur, cxl_pct)

    M_np = np.asarray(M, dtype=np.int8)
    num_nodes, num_mhd = M_np.shape
    acc = [np.nonzero(M_np[i])[0].tolist() for i in range(num_nodes)]

    cur = np.zeros(num_mhd, dtype=float)
    maxv = np.zeros(num_mhd, dtype=float)
    dealloc = np.zeros((dur, num_mhd), dtype=float)

    denom = float(pod_rss[MEM_IDX]) if float(pod_rss[MEM_IDX]) > 0 else 1.0

    for ts, evs in enumerate(alloc_events):
        cur -= dealloc[ts, :]
        cur = np.maximum(cur, 0.0)

        for node_idx, end_ts, mem in evs:
            mhd_list = acc[node_idx]
            n_acc = len(mhd_list)
            if n_acc == 0:
                continue
            if n_acc > int(max_degree):
                raise ValueError(
                    f"RL policy max_degree={max_degree} < accessible degree={n_acc}. "
                    f"Use a topology with degree <= max_degree or retrain."
                )

            loads = np.zeros(int(max_degree), dtype=np.float32)
            for idx, mhd in enumerate(mhd_list):
                loads[idx] = float(cur[mhd]) / denom

            mask = np.zeros(int(max_degree), dtype=np.float32)
            mask[:n_acc] = 1.0

            vm_norm = np.float32(float(mem) / denom)
            peak_norm = np.float32(float(np.max(cur)) / denom)

            abs_time = base + _dt.timedelta(minutes=ts * 5)
            hour = abs_time.hour + abs_time.minute / 60.0
            hour_sin = np.float32(np.sin(2.0 * np.pi * hour / 24.0))
            hour_cos = np.float32(np.cos(2.0 * np.pi * hour / 24.0))

            obs = np.concatenate(
                [
                    loads,
                    mask,
                    np.array([vm_norm, peak_norm, hour_sin, hour_cos], dtype=np.float32),
                ]
            ).astype(np.float32)

            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            raw = action[:n_acc]
            raw = raw - raw.max()
            exp_raw = np.exp(raw)
            proportions = exp_raw / (exp_raw.sum() + 1e-12)

            alloc_arr = np.zeros(num_mhd, dtype=np.float64)
            for idx, mhd in enumerate(mhd_list):
                alloc_arr[mhd] = proportions[idx] * float(mem)

            cur += alloc_arr
            if end_ts < dur:
                dealloc[end_ts, :] += alloc_arr

        maxv = np.maximum(maxv, cur)

    denom2 = float(pod_rss[MEM_IDX])
    if denom2 <= 0:
        return 0.0
    ratio = float(np.max(maxv)) * float(num_mhd) / denom2
    return float(1.0 - ratio)


def optimal_required_capacity(
    tr: _Trace,
    node_to_M: Dict[int, int],
    M: List[List[int]],
    cxl_pct: float,
    window: int = 1,
) -> float:
    """Optimal per-tick placement (windowed). Returns required capacity in [0,1]."""
    nodes = list(node_to_M.keys())
    start, end = _time_window(tr, nodes)
    base = _dt.datetime(start.year, start.month, start.day, start.hour, start.minute)
    pod_start = _to_tick(base, start)
    pod_end = _to_tick(base, end)
    dur = pod_end - pod_start + 1
    if dur <= 0:
        return 0.0

    # Events build also computes denominator and skip list; for optimal we need
    # per-node demand time series (difference array).
    alloc_events, pod_rss = _build_events_and_skip(tr, nodes, base, dur, cxl_pct)
    num_nodes = len(nodes)
    num_mhd = len(M[0])

    diff = np.zeros((dur, num_nodes), dtype=float)
    for ts, evs in enumerate(alloc_events):
        for node_idx, end_ts, mem in evs:
            diff[ts, node_idx] += mem
            if end_ts < dur:
                diff[end_ts, node_idx] -= mem

    node_cxl = np.zeros(num_nodes, dtype=float)
    max_cxl_mem = 0.0
    ts_count = 0
    M_bool = np.asarray(M, dtype=bool)

    for ts in range(dur):
        node_cxl += diff[ts, :]
        # Gate on current active load, not instantaneous delta at this tick.
        if float(np.max(node_cxl)) <= 0:
            continue
        ts_count += 1
        if ts_count % int(window) == 0:
            cur_cxl_mem = float(find_optimal(node_cxl, M_bool)) * float(num_mhd)
            if cur_cxl_mem > max_cxl_mem:
                max_cxl_mem = cur_cxl_mem

    denom = float(pod_rss[MEM_IDX])
    if denom <= 0:
        return 0.0
    ratio = float(max_cxl_mem) / denom
    return float(1.0 - ratio)


def _generic_mhd_timeseries(
    tr: _Trace,
    node_to_M: Dict[int, int],
    M: List[List[int]],
    cxl_pct: float,
    alloc_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times_minutes, usage_per_mhd[time, mhd]) for one mapping.

    *alloc_fn(mem, mhd_list, cur_vec, ts, base, denom) -> alloc_arr*
    """
    nodes = list(node_to_M.keys())
    start, end = _time_window(tr, nodes)
    base = _dt.datetime(start.year, start.month, start.day, start.hour, start.minute)
    pod_start = _to_tick(base, start)
    pod_end = _to_tick(base, end)
    dur = pod_end - pod_start + 1
    if dur <= 0:
        return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)

    alloc_events, pod_rss = _build_events_and_skip(tr, nodes, base, dur, cxl_pct)
    denom = float(pod_rss[MEM_IDX]) if float(pod_rss[MEM_IDX]) > 0 else 1.0

    M_np = np.asarray(M, dtype=np.int8)
    num_nodes, num_mhd = M_np.shape
    acc = [np.nonzero(M_np[i])[0].tolist() for i in range(num_nodes)]

    cur = np.zeros(num_mhd, dtype=float)
    dealloc_arr = np.zeros((dur, num_mhd), dtype=float)
    usage = np.zeros((dur, num_mhd), dtype=float)

    for ts, evs in enumerate(alloc_events):
        cur -= dealloc_arr[ts, :]
        cur = np.maximum(cur, 0.0)

        for node_idx, end_ts, mem in evs:
            mhd_list = acc[node_idx]
            if not mhd_list:
                continue
            alloc_vec = alloc_fn(mem, mhd_list, cur, ts, base, denom)
            cur += alloc_vec
            if end_ts < dur:
                dealloc_arr[end_ts, :] += alloc_vec

        usage[ts, :] = cur

    times = np.arange(dur, dtype=float) * 5.0
    return times, usage


def greedy_mhd_timeseries(
    tr: _Trace,
    node_to_M: Dict[int, int],
    M: List[List[int]],
    cxl_pct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times, usage_per_mhd[time, mhd]) for one mapping (greedy)."""
    def _alloc(mem, mhd_list, cur, ts, base, denom):
        return greedy_alloc(mem, mhd_list, cur)
    return _generic_mhd_timeseries(tr, node_to_M, M, cxl_pct, _alloc)


def rl_mhd_timeseries(
    tr: _Trace,
    node_to_M: Dict[int, int],
    M: List[List[int]],
    cxl_pct: float,
    model,
    max_degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times, usage_per_mhd[time, mhd]) for one mapping (RL)."""
    num_mhd = len(M[0])

    def _alloc(mem, mhd_list, cur, ts, base, denom):
        n_acc = len(mhd_list)
        loads = np.zeros(int(max_degree), dtype=np.float32)
        for idx, mhd in enumerate(mhd_list):
            loads[idx] = float(cur[mhd]) / denom
        mask = np.zeros(int(max_degree), dtype=np.float32)
        mask[:n_acc] = 1.0
        vm_norm = np.float32(float(mem) / denom)
        peak_norm = np.float32(float(np.max(cur)) / denom)
        abs_time = base + _dt.timedelta(minutes=ts * 5)
        hour = abs_time.hour + abs_time.minute / 60.0
        hour_sin = np.float32(np.sin(2.0 * np.pi * hour / 24.0))
        hour_cos = np.float32(np.cos(2.0 * np.pi * hour / 24.0))
        obs = np.concatenate([
            loads, mask,
            np.array([vm_norm, peak_norm, hour_sin, hour_cos], dtype=np.float32),
        ]).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        raw = action[:n_acc]
        raw = raw - raw.max()
        exp_raw = np.exp(raw)
        proportions = exp_raw / (exp_raw.sum() + 1e-12)
        alloc_arr = np.zeros(num_mhd, dtype=np.float64)
        for idx, mhd in enumerate(mhd_list):
            alloc_arr[mhd] = proportions[idx] * float(mem)
        return alloc_arr

    return _generic_mhd_timeseries(tr, node_to_M, M, cxl_pct, _alloc)


def _style():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "figure.titlesize": 22,
            "lines.linewidth": 2.5,
            "lines.markersize": 6,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="LON23PrdApp01-troundgrt5m.sqlite")
    ap.add_argument("--outdir", default="output/plots/memory_pooling")
    ap.add_argument(
        "--pod-sizes",
        nargs="+",
        type=int,
        default=[4, 8, 13, 16, 20],
    )
    ap.add_argument(
        "--cxl-pcts",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5],
    )
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=10086)
    ap.add_argument("--host-degree", type=int, default=8)
    ap.add_argument("--pools-factor", type=int, default=2)
    ap.add_argument("--do-rl", action="store_true", help="Include RL curve if a model checkpoint is available")
    ap.add_argument(
        "--rl-model",
        nargs="+",
        default=["output/checkpoints/best_model.zip", "output/checkpoints/octopus_sac_final.zip", "output/checkpoints/octopus_sac_final"],
        help="SB3 SAC model path(s). First existing one is used.",
    )
    ap.add_argument("--rl-max-degree", type=int, default=8, help="Max degree the RL policy was trained with (e.g., 8)")
    ap.add_argument("--rl-n-iter", type=int, default=5, help="Number of random pod mappings to average for RL curves")
    ap.add_argument("--wait-rl-secs", type=int, default=0, help="If >0, wait up to this many seconds for an RL model file to appear")
    ap.add_argument("--wait-rl-interval", type=int, default=30, help="Seconds between RL model existence checks")
    ap.add_argument("--do-optimal", action="store_true")
    ap.add_argument("--optimal-window", type=int, default=24)
    ap.add_argument("--do-timeseries", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    _style()

    tr = _load_trace(args.trace)
    title = args.trace

    pod_sizes = list(args.pod_sizes)
    cxl_pcts = list(args.cxl_pcts)

    # Sweep: fully-connected vs regular octopus (greedy)
    colors = {0.1: "tab:blue", 0.3: "tab:orange", 0.5: "tab:green"}

    fc_means = {pct: [] for pct in cxl_pcts}
    oct_means = {pct: [] for pct in cxl_pcts}
    rl_means = {pct: [] for pct in cxl_pcts}

    rl_model = None
    if args.do_rl:
        # Try to load a model (optionally waiting for training to create one)
        from stable_baselines3 import SAC

        deadline = time.time() + int(args.wait_rl_secs) if int(args.wait_rl_secs) > 0 else None
        while True:
            chosen = None
            for p in list(args.rl_model):
                if os.path.exists(p):
                    chosen = p
                    break
                # SB3 also supports providing stems without ".zip"
                if p.endswith(".zip") and os.path.exists(p[:-4]):
                    chosen = p[:-4]
                    break
                if (not p.endswith(".zip")) and os.path.exists(p + ".zip"):
                    chosen = p
                    break
            if chosen is not None:
                print(f"Loading RL model from: {chosen}")
                rl_model = SAC.load(chosen)
                break
            if deadline is None or time.time() >= deadline:
                print("RL model not found yet; continuing without RL curves.")
                break
            time.sleep(int(args.wait_rl_interval))

    # Also collect per-pct results for the 3-panel figure
    opt_means = {pct: [] for pct in cxl_pcts}

    for pod_size in pod_sizes:
        print(f"[Fig 1] pod_size={pod_size} …")
        fc = _fully_connected_matrix(pod_size, args.pools_factor)
        try:
            octo = _regular_octopus_matrix(
                pod_size,
                host_degree=args.host_degree,
                pools_factor=args.pools_factor,
                seed=args.seed + 1234 + pod_size,
            )
        except Exception as e:
            print(f"  Skipping octopus for pod_size={pod_size}: {e}")
            octo = None

        for pct in cxl_pcts:
            fc_vals = []
            oct_vals = []
            rl_vals = []
            opt_vals = []
            n_opt = max(3, int(args.n_iter // 4)) if args.do_optimal else 0
            for i in range(int(args.n_iter)):
                pod_to_nodes = generate_pod_to_nodes(tr.node_to_vms, pod_size, args.seed + i)
                node_to_M, M_fc = expand_M_to_all_nodes(fc, pod_to_nodes)
                fc_vals.append(greedy_required_capacity(tr, node_to_M, M_fc, pct))
                if octo is not None:
                    _node_to_M2, M_oct = expand_M_to_all_nodes(octo, pod_to_nodes)
                    oct_vals.append(greedy_required_capacity(tr, node_to_M, M_oct, pct))
                    if rl_model is not None and i < int(args.rl_n_iter):
                        rl_vals.append(
                            rl_required_capacity(
                                tr, node_to_M, M_oct, pct,
                                model=rl_model,
                                max_degree=int(args.rl_max_degree),
                            )
                        )
                    if args.do_optimal and i < n_opt:
                        opt_vals.append(
                            optimal_required_capacity(
                                tr, node_to_M, M_oct, pct,
                                window=int(args.optimal_window),
                            )
                        )
            fc_means[pct].append(float(np.mean(fc_vals)) * 100.0)
            oct_means[pct].append(float(np.mean(oct_vals)) * 100.0 if oct_vals else float("nan"))
            rl_means[pct].append(float(np.mean(rl_vals)) * 100.0 if rl_vals else float("nan"))
            opt_means[pct].append(float(np.mean(opt_vals)) * 100.0 if opt_vals else float("nan"))

    short_title = title.replace("-troundgrt5m.sqlite", "").replace("-tround.sqlite", "")

    # ── Fig 1: FC vs Octopus overview ─────────────────────────────────
    print("[Fig 1] FC vs Octopus overview …")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(short_title, fontsize=20, fontweight="bold")
    ax.set_xlabel("Pod Size")
    ax.set_ylabel("Required Memory Capacity (%)")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    pct_labels = {0.1: "10%", 0.3: "30%", 0.5: "50%"}
    for pct in cxl_pcts:
        c = colors.get(pct, None)
        lbl = pct_labels.get(pct, f"{pct:.0%}")
        ax.plot(pod_sizes, fc_means[pct], marker="s", markersize=8,
                color=c, linestyle="-", linewidth=2.5,
                label=f"{lbl} CXL, fully-connected")
        ax.plot(pod_sizes, oct_means[pct], marker="o", markersize=8,
                color=c, linestyle="--", linewidth=2.5,
                label=f"{lbl} CXL, octopus (N=4)")
        if rl_model is not None:
            ax.plot(pod_sizes, rl_means[pct], marker="D", markersize=7,
                    color=c, linestyle="-.", linewidth=2.0,
                    label=f"{lbl} CXL, octopus (N=4) RL")

    all_vals = [v for d in [fc_means, oct_means, rl_means] for vs in d.values() for v in vs if not np.isnan(v)]
    if all_vals:
        ylo = max(0, min(all_vals) - 2)
        yhi = min(100.5, max(all_vals) + 2)
        ax.set_ylim(ylo, yhi)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95,
              borderaxespad=0.8, handlelength=3)
    fig.tight_layout()
    p = os.path.join(args.outdir, f"{args.trace}_fc_vs_octopus")
    fig.savefig(p + ".png", dpi=200)
    fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  saved {p}.png")

    # ── Fig 2: Greedy vs Optimal vs RL (cxl_pct = max) ──────────────
    if args.do_optimal:
        pct = float(max(cxl_pcts))
        print(f"[Fig 2] Greedy vs Optimal vs RL (cxl_pct={pct}) …")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(f"{title}")
        ax.set_xlabel("Pod Size")
        ax.set_ylabel("Required Memory Capacity (%)")
        ax.grid(True, alpha=0.35)
        ax.plot(pod_sizes, fc_means[pct], marker="s", color="tab:green",
                label=f"cxl_pct={pct}, fully-connected")
        ax.plot(pod_sizes, oct_means[pct], marker="o", color="tab:green",
                linestyle=":", label=f"cxl_pct={pct}, regular octopus (N=4), Greedy")
        ax.plot(pod_sizes, opt_means[pct], marker="^", color="tab:green",
                linestyle="--", label=f"cxl_pct={pct}, regular octopus (N=4), Optimal")
        if rl_model is not None:
            ax.plot(pod_sizes, rl_means[pct], marker="D", color="tab:green",
                    linestyle="-.", label=f"cxl_pct={pct}, regular octopus (N=4), RL")
        ax.set_ylim(80, 100.5)
        ax.legend(loc="lower left", framealpha=0.9)
        fig.tight_layout()
        p = os.path.join(args.outdir, f"{args.trace}_greedy_vs_optimal_pct{pct}")
        fig.savefig(p + ".png", dpi=200)
        fig.savefig(p + ".pdf")
        plt.close(fig)
        print(f"  saved {p}.png")

    # ── Fig 3: 3-panel by CXL% ───────────────────────────────────────
    print("[Fig 3] 3-panel by CXL% …")
    n_panels = len(cxl_pcts)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 7), sharey=True)
    if n_panels == 1:
        axes = [axes]
    pct_titles = {0.1: "10% VM Memory in CXL",
                  0.3: "30% VM Memory in CXL",
                  0.5: "50% VM Memory in CXL"}

    all_panel_vals = []
    for ax, pct in zip(axes, cxl_pcts):
        c = colors.get(pct, None)
        ax.set_title(pct_titles.get(pct, f"cxl_pct={pct}"),
                     fontsize=16, fontweight="bold", pad=10)
        ax.plot(pod_sizes, fc_means[pct], marker="s", markersize=9, color=c,
                linewidth=2.5, linestyle="-", label="Fully-connected")
        ax.plot(pod_sizes, oct_means[pct], marker="o", markersize=9, color=c,
                linewidth=2.5, linestyle="--", label="Octopus (N=4), Greedy")
        if args.do_optimal:
            ax.plot(pod_sizes, opt_means[pct], marker="^", markersize=9, color=c,
                    linewidth=2.5, linestyle=":", label="Octopus (N=4), Optimal")
        if rl_model is not None:
            ax.plot(pod_sizes, rl_means[pct], marker="D", markersize=8, color=c,
                    linewidth=2.0, linestyle="-.", label="Octopus (N=4), RL")
        ax.set_xlabel("Pod Size", fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=12)
        ax.legend(loc="lower left", fontsize=11, framealpha=0.95, handlelength=3)
        for d in [fc_means, oct_means, rl_means, opt_means]:
            all_panel_vals.extend([v for v in d.get(pct, []) if not np.isnan(v)])

    if all_panel_vals:
        ylo = max(0, min(all_panel_vals) - 2)
        yhi = min(100.5, max(all_panel_vals) + 2)
        axes[0].set_ylim(ylo, yhi)
    axes[0].set_ylabel("Required Memory Capacity (%)", fontsize=14)
    fig.suptitle(
        f"Fully-Connected vs. Octopus (Greedy) vs. Octopus (RL)\n{short_title}",
        fontsize=18, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(args.outdir, f"{args.trace}_3panel_by_cxl")
    fig.savefig(p + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(p + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}.png")

    # ── Fig 4 & 5: MHD time series (Greedy + RL) ─────────────────────
    if args.do_timeseries:
        pod_size = int(max(pod_sizes))
        pct = float(max(cxl_pcts))
        pct_str = pct_labels.get(pct, f"{pct:.0%}")
        print(f"[Fig 4] MHD time series (Greedy), pod={pod_size}, pct={pct} …")
        octo = _regular_octopus_matrix(
            pod_size,
            host_degree=args.host_degree,
            pools_factor=args.pools_factor,
            seed=args.seed + 1234 + pod_size,
        )
        pod_to_nodes = generate_pod_to_nodes(tr.node_to_vms, pod_size, args.seed)
        node_to_M, M_oct = expand_M_to_all_nodes(octo, pod_to_nodes)

        def _plot_ts(t_arr, usage_arr, policy_name, out_stem):
            peaks = np.max(usage_arr, axis=0)
            top = np.argsort(-peaks)[:12]
            fig, ax = plt.subplots(figsize=(12, 6))
            t_hours = t_arr / 60.0
            for j in top:
                ax.plot(t_hours, usage_arr[:, j], linewidth=1.2, label=f"MHD {j}")
            max_peak = float(np.max(peaks))
            min_peak = float(np.min(peaks[peaks > 0])) if np.any(peaks > 0) else 0.0
            ax.set_title(
                f"MHD Usage — {policy_name}\n{short_title}  "
                f"({pct_str} CXL, pod size={pod_size})",
                fontsize=16, fontweight="bold",
            )
            ax.set_xlabel("Time (hours)", fontsize=14)
            ax.set_ylabel("MHD Usage (GB)", fontsize=14)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.legend(ncol=3, fontsize=10, framealpha=0.9,
                      loc="upper center", bbox_to_anchor=(0.5, -0.12))
            ax.annotate(
                f"Max peak: {max_peak:,.0f} GB\nMin peak: {min_peak:,.0f} GB",
                xy=(0.98, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
            )
            fig.tight_layout(rect=[0, 0.08, 1, 1])
            fig.savefig(out_stem + ".png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out_stem}.png")

        t, usage_g = greedy_mhd_timeseries(tr, node_to_M, M_oct, pct)
        if usage_g.size > 0:
            _plot_ts(t, usage_g, "Octopus Greedy",
                     os.path.join(args.outdir, f"{args.trace}_mhd_timeseries_greedy_pct{pct}_pod{pod_size}"))

        if rl_model is not None:
            print(f"[Fig 5] MHD time series (RL), pod={pod_size}, pct={pct} …")
            t, usage_r = rl_mhd_timeseries(
                tr, node_to_M, M_oct, pct,
                model=rl_model, max_degree=int(args.rl_max_degree),
            )
            if usage_r.size > 0:
                _plot_ts(t, usage_r, "Octopus RL",
                         os.path.join(args.outdir, f"{args.trace}_mhd_timeseries_rl_pct{pct}_pod{pod_size}"))

    print("\nDone.")


if __name__ == "__main__":
    main()

