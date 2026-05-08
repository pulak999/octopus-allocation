#!/usr/bin/env python3
"""
Evaluate greedy / RL / optimal policies on Octopus memory pooling.

Code structure mirrors ``memory-pooling-v0.ipynb`` as closely as possible
so that results are directly comparable.

Usage
-----
  python evaluate.py                          # greedy + RL on AG16x6 / AMS20
  python evaluate.py --policy greedy          # greedy only
  python evaluate.py --policy rl --model output/checkpoints/octopus_sac_final
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import datetime as _dt
import json
import os

import numpy as np
import torch

from octopus.data import load_trace, load_topology
from octopus.baselines import greedy_alloc, pid_alloc
from octopus.topology import (
    generate_pod_to_nodes,
    expand_M_to_all_nodes,
)

MEM_IDX = 1  # index into VM.rss / machine_sz for memory (GB)
SUMMARY_FIELDNAMES = [
    "policy",
    "trace",
    "topology",
    "n_iter",
    "savings_mean",
    "savings_min",
    "savings_max",
    "savings_std",
    "pooling_ratio_mean",
    "pooling_ratio_std",
    "timestamp",
]
DETAIL_FIELDNAMES = [
    "policy",
    "trace",
    "topology",
    "mapping_seed",
    "pooling_ratio",
    "savings",
]


# ═══════════════════════════════════════════════════════════════════════
#  Generic simulation (accepts an alloc_fn callback)
# ═══════════════════════════════════════════════════════════════════════
def pooling_simulation(
    node_to_M,
    M,
    alloc_fn,
    all_vms,
    node_to_vms,
    node_to_machine,
    machine_sz,
    track_vm_allocs: bool = False,
):
    """Run one pooling simulation.  Structure matches the notebook's
    ``greedy_pooling_simulation`` line-for-line, except the allocation
    call is delegated to *alloc_fn*.

    Parameters
    ----------
    alloc_fn : callable
        ``alloc_fn(cxl_mem, mhd_list, cur_cxl_mem_vec, ctx) -> alloc_arr``
        where *ctx* is a dict with optional extra info (tick, etc.).
    """
    pod_size = len(M)
    assert len(node_to_M) == pod_size
    assert pod_size > 0
    num_mhd = len(M[0])
    assert num_mhd > 0

    # 1) Pod resource totals & time range
    pod_rss = np.zeros(4, dtype=float)
    n_start_time = _dt.datetime.max
    n_end_time = _dt.datetime.min
    any_vm = False

    for cur_node in node_to_M.keys():
        pod_rss += np.asarray(
            machine_sz[node_to_machine[cur_node]], dtype=float
        )
        for cur_vmkey in node_to_vms.get(cur_node, []):
            cur_vm = all_vms[cur_vmkey]
            any_vm = True
            if cur_vm.start_time < n_start_time:
                n_start_time = cur_vm.start_time
            if cur_vm.end_time > n_end_time:
                n_end_time = cur_vm.end_time

    if not any_vm:
        return 0.0

    base = _dt.datetime(
        n_start_time.year, n_start_time.month, n_start_time.day,
        n_start_time.hour, n_start_time.minute,
    )

    def to_tick(t):
        return int((t - base).total_seconds() // 300)

    pod_start_ts = to_tick(n_start_time)
    pod_end_ts = to_tick(n_end_time)
    pod_dur = pod_end_ts - pod_start_ts + 1
    if pod_dur <= 0:
        return 0.0

    # C4: Extract ctx constants as locals to avoid redundant attribute lookups
    _pod_start_ts = pod_start_ts
    _base = base
    _num_mhd = num_mhd
    _pod_rss_mem = float(pod_rss[MEM_IDX])

    # C1: Precompute mhd_list per node once (not per VM per tick)
    node_mhd_lists = {
        node_id: [mhd for mhd in range(num_mhd) if M[node_id][mhd] != 0]
        for node_id in range(len(M))
    }

    # Precompute mhd_to_hosts and Q_j from expanded topology for obs variants A/B.
    # Must use expanded_M (not base M) — expand_M_to_all_nodes creates a
    # block-diagonal layout where pod k's MPDs start at k*192, so indices >= 192
    # only exist here.
    _mhd_to_hosts_exp = {j: [] for j in range(num_mhd)}
    for h, mhds in node_mhd_lists.items():
        for j in mhds:
            _mhd_to_hosts_exp[j].append(h)
    _Q_j_exp = np.zeros(num_mhd, dtype=np.float32)
    for j in range(num_mhd):
        hosts = _mhd_to_hosts_exp[j]
        if hosts:
            inv_deg = [1.0 / len(node_mhd_lists[h]) for h in hosts if node_mhd_lists[h]]
            if inv_deg:
                _Q_j_exp[j] = float(np.mean(inv_deg))

    # C2: Cache VM tick bounds and memory upfront
    base_sec = base.timestamp()
    vm_cache = {}  # vmkey -> (vm_tick_start, vm_tick_end, mem_gb)
    for cur_node in node_to_M.keys():
        for cur_vmkey in node_to_vms.get(cur_node, []):
            if cur_vmkey not in vm_cache:
                cur_vm = all_vms[cur_vmkey]
                vm_tick_start = int((cur_vm.start_time.timestamp() - base_sec) // 300) - pod_start_ts
                vm_tick_end = int((cur_vm.end_time.timestamp() - base_sec) // 300) - pod_start_ts
                mem_gb = float(np.asarray(cur_vm.rss, dtype=float)[MEM_IDX])
                vm_cache[cur_vmkey] = (vm_tick_start, vm_tick_end, mem_gb)

    # 2) HOTFIX: filter VMs that would exceed per-node memory
    vmkey_to_skip: set = set()
    for cur_node in node_to_M.keys():
        alloc_events = [[] for _ in range(pod_dur)]
        dealloc_events_node = np.zeros(pod_dur, dtype=float)
        node_rss = np.asarray(
            machine_sz[node_to_machine[cur_node]], dtype=float
        )

        for cur_vmkey in node_to_vms.get(cur_node, []):
            vm_start, vm_end, mem = vm_cache[cur_vmkey]
            if vm_end < 0:
                vmkey_to_skip.add(cur_vmkey)
                continue
            alloc_events[vm_start].append((cur_vmkey, vm_end + 1, mem))
            if vm_end + 1 < pod_dur:
                dealloc_events_node[vm_end + 1] += mem

        cur_mem = 0.0
        for ts in range(pod_dur):
            cur_mem -= dealloc_events_node[ts]
            if cur_mem < 0:
                cur_mem = 0.0
            for vmkey, end_ts, mem in alloc_events[ts]:
                cur_mem += mem
                if cur_mem > node_rss[MEM_IDX]:
                    cur_mem -= mem
                    vmkey_to_skip.add(vmkey)
                    if end_ts < pod_dur:
                        dealloc_events_node[end_ts] -= mem

    # 3) Build flat sorted event list instead of list-of-lists (C3)
    flat_events = []
    for cur_node, node_in_pod_id in node_to_M.items():
        for cur_vmkey in node_to_vms.get(cur_node, []):
            if cur_vmkey in vmkey_to_skip:
                continue
            vm_start, vm_end, mem = vm_cache[cur_vmkey]
            flat_events.append((vm_start, node_in_pod_id, vm_end + 1, mem))
    flat_events.sort(key=lambda e: e[0])

    # 4) Sweep ticks — allocate using alloc_fn (C3: only visit ticks with events)
    cur_cxl_mem_vec = np.zeros(num_mhd, dtype=float)
    max_cxl_mem_vec = np.zeros(num_mhd, dtype=float)
    dealloc_events = np.zeros((pod_dur, num_mhd), dtype=float)

    # Per-VM tracking state (only allocated when track_vm_allocs=True)
    if track_vm_allocs:
        mpd_vm_allocs: list = [[] for _ in range(num_mhd)]
        active_mpds: set = set()  # MPDs with at least one pending alloc entry
        host_cxl_load = np.zeros(pod_size, dtype=float)
        host_dealloc_events = np.zeros((pod_dur, pod_size), dtype=float)
    else:
        mpd_vm_allocs = host_cxl_load = host_dealloc_events = None  # type: ignore[assignment]
        active_mpds = set()  # type: ignore[assignment]

    prev_tick = -1
    evt_idx = 0
    n_events = len(flat_events)

    while evt_idx < n_events:
        tick = flat_events[evt_idx][0]

        # Apply deallocs for all skipped ticks up to and including current tick
        if tick > prev_tick + 1:
            cur_cxl_mem_vec -= dealloc_events[prev_tick + 1:tick + 1, :].sum(axis=0)
        else:
            cur_cxl_mem_vec -= dealloc_events[tick, :]
        cur_cxl_mem_vec = np.maximum(cur_cxl_mem_vec, 0.0)

        # Drain per-host loads and purge expired MPD VM lists
        if track_vm_allocs:
            start_t = max(prev_tick + 1, 0)
            end_t = min(tick + 1, pod_dur)
            if end_t > start_t:
                host_cxl_load -= host_dealloc_events[start_t:end_t, :].sum(axis=0)
            np.maximum(host_cxl_load, 0.0, out=host_cxl_load)
            _to_remove = set()
            for mhd in active_mpds:
                mpd_vm_allocs[mhd] = [
                    (dt, m) for dt, m in mpd_vm_allocs[mhd] if dt > tick
                ]
                if not mpd_vm_allocs[mhd]:
                    _to_remove.add(mhd)
            active_mpds -= _to_remove

        # Process all events at this tick
        while evt_idx < n_events and flat_events[evt_idx][0] == tick:
            _, node_in_pod_id, dealloc_time, mem = flat_events[evt_idx]
            evt_idx += 1

            if mem <= 0:
                continue

            mhd_list = node_mhd_lists[node_in_pod_id]
            assert len(mhd_list) != 0

            ctx = {
                "tick": tick,
                "dealloc_time": dealloc_time,
                "pod_start_ts": _pod_start_ts,
                "base_time": _base,
                "num_mhd": _num_mhd,
                "pod_rss_mem": _pod_rss_mem,
                "mhd_to_hosts": _mhd_to_hosts_exp,
                "Q_j": _Q_j_exp,
            }
            if track_vm_allocs:
                ctx["mpd_vm_allocs"] = mpd_vm_allocs
                ctx["host_cxl_load"] = host_cxl_load

            alloc_vec = np.asarray(
                alloc_fn(mem, mhd_list, cur_cxl_mem_vec, ctx), dtype=float
            )
            assert alloc_vec.shape == (num_mhd,)

            cur_cxl_mem_vec += alloc_vec

            assert dealloc_time > tick
            if dealloc_time < pod_dur:
                dealloc_events[dealloc_time, :] += alloc_vec

            # Update per-VM tracking after allocation.
            # Only mhd_list entries of alloc_vec are non-zero; iterating over
            # range(num_mhd) ~11,904 times per event was O(num_mhd) like the D_j bug.
            if track_vm_allocs:
                for mhd in mhd_list:
                    if alloc_vec[mhd] > 0 and dealloc_time < pod_dur:
                        mpd_vm_allocs[mhd].append((dealloc_time, float(alloc_vec[mhd])))
                        active_mpds.add(mhd)
                host_cxl_load[node_in_pod_id] += mem
                if dealloc_time < pod_dur:
                    host_dealloc_events[dealloc_time, node_in_pod_id] += mem

        max_cxl_mem_vec = np.maximum(max_cxl_mem_vec, cur_cxl_mem_vec)
        prev_tick = tick

    denom = float(pod_rss[MEM_IDX])
    if denom <= 0:
        return 0.0
    return float(np.max(max_cxl_mem_vec)) * num_mhd / denom


# ═══════════════════════════════════════════════════════════════════════
#  Allocation callbacks
# ═══════════════════════════════════════════════════════════════════════
def _greedy_alloc_cb(cxl_mem, mhd_list, cur_cxl_mem_vec, ctx):
    """Greedy callback (wraps baselines.greedy_alloc)."""
    return greedy_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec)


def make_pid_alloc_cb(kp=1.0, ki=0.01, kd=0.1):
    """Return a fresh PID callback (new pid_state per call → reset per pod mapping)."""
    pid_state = {}

    def _pid_alloc_cb(cxl_mem, mhd_list, cur_cxl_mem_vec, ctx):
        return pid_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec, pid_state,
                         kp=kp, ki=ki, kd=kd)

    return _pid_alloc_cb


def make_rl_alloc_cb(model, max_degree, obs_variant="current",
                     mhd_to_hosts=None, Q_j=None, lookahead_window=200):
    """Return a callback that uses the trained RL policy.

    Parameters
    ----------
    obs_variant : str
        "current" (20-dim legacy) or "A"/"B" (50-dim new state space).
    mhd_to_hosts : dict[int, list[int]] | None
        Required when obs_variant != "current". Maps MPD index → host indices.
    Q_j : np.ndarray | None
        Required when obs_variant != "current". Neighbour-scarcity per MPD.
    lookahead_window : int
        W (timesteps) for D_j / S_j computation when obs_variant != "current".
    """
    # Bypass SB3's model.predict wrapper (which calls set_training_mode and
    # allocates tensors on every invocation).  Extract the policy once, set
    # eval mode once, and call _predict directly with a pre-allocated buffer.
    policy = model.policy
    policy.set_training_mode(False)
    _device = next(policy.parameters()).device
    _SIMPLE_OBS = {"current", "R1"}
    _obs_dim = max_degree * 2 + 4 if obs_variant in _SIMPLE_OBS else max_degree * 6 + 2
    _obs_buf = torch.zeros(1, _obs_dim, dtype=torch.float32, device=_device)

    def _rl_alloc_cb(cxl_mem, mhd_list, cur_cxl_mem_vec, ctx):
        n_acc = len(mhd_list)
        num_mhd = ctx["num_mhd"]
        norm = ctx["pod_rss_mem"] if ctx["pod_rss_mem"] > 0 else 1.0
        tick = ctx["tick"]

        if obs_variant in _SIMPLE_OBS:
            # --- Original 20-dim obs ------------------------------------
            loads = np.zeros(max_degree, dtype=np.float32)
            for idx, mhd in enumerate(mhd_list):
                loads[idx] = float(cur_cxl_mem_vec[mhd]) / norm

            mask = np.zeros(max_degree, dtype=np.float32)
            mask[:n_acc] = 1.0

            vm_norm = np.float32(cxl_mem / norm)
            peak_norm = np.float32(float(np.max(cur_cxl_mem_vec)) / norm)

            abs_time = ctx["base_time"] + _dt.timedelta(
                minutes=(tick + ctx["pod_start_ts"]) * 5
            )
            hour = abs_time.hour + abs_time.minute / 60.0
            hour_sin = np.float32(np.sin(2.0 * np.pi * hour / 24.0))
            hour_cos = np.float32(np.cos(2.0 * np.pi * hour / 24.0))

            obs = np.concatenate(
                [loads, mask, np.array([vm_norm, peak_norm, hour_sin, hour_cos])]
            ).astype(np.float32)

        else:
            # --- New 50-dim obs (variants A and B) ----------------------
            mpd_vm_allocs = ctx["mpd_vm_allocs"]
            host_cxl_load = ctx["host_cxl_load"]
            W = lookahead_window
            t_plus_W = tick + W
            # Hoist ctx lookups out of the per-MPD loop
            _mhd_to_hosts = ctx.get("mhd_to_hosts", mhd_to_hosts)
            _Q_j = ctx.get("Q_j", Q_j)

            # Only compute D_j/S_j for the MPDs we actually observe (mhd_list,
            # max 8).  The expanded topology has O(num_pods * 192) MPDs total;
            # iterating over all of them per-event costs ~1.2B iterations/callback.
            obs = np.zeros(max_degree * 6 + 2, dtype=np.float32)
            for k, mhd in enumerate(mhd_list):
                alloc_data = mpd_vm_allocs[mhd]
                if alloc_data:
                    # Vectorised D_j/S_j — avoids Python loop over active VMs
                    _arr = np.array(alloc_data)      # (n, 2): col0=dt, col1=mem
                    _dts = _arr[:, 0]
                    _mems = _arr[:, 1]
                    _mask = _dts <= t_plus_W
                    dj = float(np.dot(_mems[_mask], 1.0 - (_dts[_mask] - tick) / W))
                    sj = float(_mems[~_mask].sum())
                else:
                    dj = sj = 0.0
                base_idx = k * 6
                obs[base_idx]     = float(cur_cxl_mem_vec[mhd]) / norm
                obs[base_idx + 1] = dj / norm
                obs[base_idx + 2] = sj / norm
                obs[base_idx + 3] = 1.0
                P_j = float(sum(host_cxl_load[h] for h in _mhd_to_hosts[mhd]))
                obs[base_idx + 4] = P_j / norm
                obs[base_idx + 5] = float(_Q_j[mhd])
            obs[-2] = float(np.max(cur_cxl_mem_vec)) / norm
            obs[-1] = float(cxl_mem) / norm

        # Predict — write obs directly into pre-allocated GPU buffer,
        # call policy._predict once (no set_training_mode, no tensor alloc).
        _obs_buf[0].copy_(torch.from_numpy(obs))
        with torch.no_grad():
            action = policy._predict(_obs_buf, deterministic=True).cpu().numpy()[0]

        # Softmax → proportions → allocation
        raw = action[:n_acc].astype(np.float64)
        raw = raw - raw.max()
        exp_raw = np.exp(raw)
        proportions = exp_raw / (exp_raw.sum() + 1e-12)

        alloc_arr = np.zeros(num_mhd, dtype=np.float64)
        for idx, mhd in enumerate(mhd_list):
            alloc_arr[mhd] = proportions[idx] * cxl_mem

        return alloc_arr

    return _rl_alloc_cb


# ═══════════════════════════════════════════════════════════════════════
#  Main evaluation loop (mirrors notebook cells 14-16)
# ═══════════════════════════════════════════════════════════════════════
def run_eval(policy_name, alloc_fn, matrix, trace_data, n_iter=50, alloc_fn_factory=None,
             track_vm_allocs=False):
    """Run *n_iter* random pod mappings, print summary statistics.

    If *alloc_fn_factory* is provided it is called once per iteration (with no
    arguments) to produce a fresh alloc_fn — needed for stateful policies like PID.
    """
    all_vms, node_to_vms, node_to_machine, _vtsz, machine_sz = trace_data

    savings_list = []
    ratio_list = []
    detail_rows = []
    for i in range(n_iter):
        print(f"\r  {policy_name}: iteration {i}/{n_iter}", end="", flush=True)
        fn = alloc_fn_factory() if alloc_fn_factory is not None else alloc_fn
        seed = 10086 + i
        pod_to_nodes = generate_pod_to_nodes(node_to_vms, len(matrix), seed)
        node_to_M, expanded_M = expand_M_to_all_nodes(matrix, pod_to_nodes)
        ratio = pooling_simulation(
            node_to_M, expanded_M, fn,
            all_vms, node_to_vms, node_to_machine, machine_sz,
            track_vm_allocs=track_vm_allocs,
        )
        savings = 1.0 - ratio
        ratio_list.append(ratio)
        savings_list.append(savings)
        detail_rows.append(
            {
                "mapping_seed": seed,
                "pooling_ratio": float(ratio),
                "savings": float(savings),
            }
        )

    print()
    avg = np.mean(savings_list)
    mn = np.min(savings_list)
    mx = np.max(savings_list)
    std = np.std(savings_list)
    print(f"  {policy_name}:")
    print(f"    avg:  {avg:.4f}")
    print(f"    min:  {mn:.4f}")
    print(f"    max:  {mx:.4f}")
    print(f"    std:  {std:.4f}")
    return {
        "savings": savings_list,
        "ratios": ratio_list,
        "detail_rows": detail_rows,
    }


# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate policies on Octopus memory pooling"
    )
    ap.add_argument(
        "--policy",
        nargs="+",
        default=["greedy", "rl"],
        choices=["greedy", "rl", "pid"],
        help="Policies to evaluate",
    )
    ap.add_argument("--kp", type=float, default=1.0, help="PID proportional gain")
    ap.add_argument("--ki", type=float, default=0.01, help="PID integral gain")
    ap.add_argument("--kd", type=float, default=0.1, help="PID derivative gain")
    ap.add_argument(
        "--trace",
        default="AMS20PrdApp19-tround.sqlite",
    )
    ap.add_argument(
        "--topology",
        default="data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Path to trained SB3 model (required when --policy includes rl)",
    )
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for CSVs (summary/detail). "
        "If unset, no files are written.",
    )
    ap.add_argument(
        "--out-prefix",
        default="evaluate",
        help="Filename prefix for output CSVs when --out-dir is set.",
    )
    args = ap.parse_args()

    print(f"Loading trace: {args.trace} …")
    trace_data = load_trace(args.trace)

    M, num_hosts, num_pools = load_topology(args.topology)
    max_deg = max(sum(row) for row in M)
    print(
        f"Topology: {num_hosts} hosts, {num_pools} MPDs, "
        f"max_degree={max_deg}\n"
    )

    results = {}

    # ── Greedy ───────────────────────────────────────────────────────
    if "greedy" in args.policy:
        results["greedy"] = run_eval(
            "Greedy", _greedy_alloc_cb, M, trace_data, args.n_iter
        )

    # ── RL ───────────────────────────────────────────────────────────
    if "rl" in args.policy:
        from stable_baselines3 import SAC
        import hashlib

        if args.model is None:
            ap.error("--model is required when --policy includes rl")

        model_path = os.path.abspath(args.model)
        # Append .zip if needed (SB3 convention)
        check_path = model_path if os.path.exists(model_path) else model_path + ".zip"
        if not os.path.exists(check_path):
            ap.error(f"Model file not found: {check_path}")
        md5 = hashlib.md5(open(check_path, "rb").read()).hexdigest()
        print(f"Loading RL model: {check_path}")
        print(f"  md5: {md5}")

        # Read reward_variant and lookahead_window from checkpoint config if present
        reward_variant = "current"
        lookahead_window = 200
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        if os.path.exists(config_path):
            with open(config_path) as _f:
                _cfg = json.load(_f)
            reward_variant = _cfg.get("reward_variant", "current")
            lookahead_window = int(_cfg.get("lookahead_window", 200))
            print(f"  reward_variant: {reward_variant}  lookahead_window: {lookahead_window}")

        # Compute mhd_to_hosts and Q_j from base topology (same structure for all pods)
        num_mhd = len(M[0])
        pod_size = len(M)
        _host_to_mhds = {h: [j for j in range(num_mhd) if M[h][j] != 0] for h in range(pod_size)}
        mhd_to_hosts = {j: [] for j in range(num_mhd)}
        for h, mhds in _host_to_mhds.items():
            for j in mhds:
                mhd_to_hosts[j].append(h)
        Q_j = np.zeros(num_mhd, dtype=np.float32)
        for j in range(num_mhd):
            hosts = mhd_to_hosts[j]
            if hosts:
                inv_deg = [1.0 / len(_host_to_mhds[h]) for h in hosts if _host_to_mhds[h]]
                if inv_deg:
                    Q_j[j] = float(np.mean(inv_deg))

        model = SAC.load(args.model)
        rl_cb = make_rl_alloc_cb(
            model, max_deg,
            obs_variant=reward_variant,
            mhd_to_hosts=mhd_to_hosts,
            Q_j=Q_j,
            lookahead_window=lookahead_window,
        )
        results["rl"] = run_eval(
            "RL (SAC)", rl_cb, M, trace_data, args.n_iter,
            track_vm_allocs=(reward_variant not in ("current", "R1")),
        )

    # ── PID ──────────────────────────────────────────────────────────
    if "pid" in args.policy:
        results["pid"] = run_eval(
            "PID",
            alloc_fn=None,
            matrix=M,
            trace_data=trace_data,
            n_iter=args.n_iter,
            alloc_fn_factory=lambda: make_pid_alloc_cb(
                kp=args.kp, ki=args.ki, kd=args.kd
            ),
        )

    # ── Summary ──────────────────────────────────────────────────────
    if len(results) > 1:
        print("\n── Comparison ──")
        for name, payload in results.items():
            vals = payload["savings"]
            print(f"  {name:>10s}:  avg={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        ts = _dt.datetime.utcnow().isoformat()
        summary_path = os.path.join(args.out_dir, f"{args.out_prefix}_summary.csv")
        detail_path = os.path.join(args.out_dir, f"{args.out_prefix}_detail.csv")

        summary_rows = []
        detail_rows = []
        for name, payload in results.items():
            savings_arr = np.asarray(payload["savings"], dtype=float)
            ratio_arr = np.asarray(payload["ratios"], dtype=float)
            summary_rows.append(
                {
                    "policy": name,
                    "trace": args.trace,
                    "topology": args.topology,
                    "n_iter": args.n_iter,
                    "savings_mean": float(np.mean(savings_arr)),
                    "savings_min": float(np.min(savings_arr)),
                    "savings_max": float(np.max(savings_arr)),
                    "savings_std": float(np.std(savings_arr)),
                    "pooling_ratio_mean": float(np.mean(ratio_arr)),
                    "pooling_ratio_std": float(np.std(ratio_arr)),
                    "timestamp": ts,
                }
            )
            for row in payload["detail_rows"]:
                detail_rows.append(
                    {
                        "policy": name,
                        "trace": args.trace,
                        "topology": args.topology,
                        "mapping_seed": row["mapping_seed"],
                        "pooling_ratio": row["pooling_ratio"],
                        "savings": row["savings"],
                    }
                )

        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
            w.writeheader()
            w.writerows(summary_rows)

        with open(detail_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DETAIL_FIELDNAMES)
            w.writeheader()
            w.writerows(detail_rows)

        print(f"\nSaved summary CSV: {summary_path}")
        print(f"Saved detail CSV:  {detail_path}")


if __name__ == "__main__":
    main()
