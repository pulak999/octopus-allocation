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

import argparse
import datetime as _dt
import os
import sys

import numpy as np

from octopus.data import load_trace, load_topology
from octopus.baselines import greedy_alloc
from octopus.topology import (
    generate_pod_to_nodes,
    expand_M_to_all_nodes,
)

MEM_IDX = 1  # index into VM.rss / machine_sz for memory (GB)


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

    # 2) HOTFIX: filter VMs that would exceed per-node memory
    vmkey_to_skip: set = set()
    for cur_node in node_to_M.keys():
        alloc_events = [[] for _ in range(pod_dur)]
        dealloc_events_node = np.zeros(pod_dur, dtype=float)
        node_rss = np.asarray(
            machine_sz[node_to_machine[cur_node]], dtype=float
        )

        for cur_vmkey in node_to_vms.get(cur_node, []):
            cur_vm = all_vms[cur_vmkey]
            vm_start = to_tick(cur_vm.start_time) - pod_start_ts
            vm_end = to_tick(cur_vm.end_time) - pod_start_ts
            if vm_end < 0:
                vmkey_to_skip.add(cur_vmkey)
                continue
            vm_rss_vec = np.asarray(cur_vm.rss, dtype=float)
            mem = float(vm_rss_vec[MEM_IDX])
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

    # 3) Build per-tick arrival events
    alloc_events_sim = [[] for _ in range(pod_dur)]
    for cur_node, node_in_pod_id in node_to_M.items():
        for cur_vmkey in node_to_vms.get(cur_node, []):
            if cur_vmkey in vmkey_to_skip:
                continue
            cur_vm = all_vms[cur_vmkey]
            vm_start = to_tick(cur_vm.start_time) - pod_start_ts
            vm_end = to_tick(cur_vm.end_time) - pod_start_ts
            vm_rss_vec = np.asarray(cur_vm.rss, dtype=float)
            alloc_events_sim[vm_start].append(
                (node_in_pod_id, vm_end + 1, vm_rss_vec)
            )

    # 4) Sweep ticks — allocate using alloc_fn
    cur_cxl_mem_vec = np.zeros(num_mhd, dtype=float)
    max_cxl_mem_vec = np.zeros(num_mhd, dtype=float)
    dealloc_events = np.zeros((pod_dur, num_mhd), dtype=float)

    for i, cur_event_list in enumerate(alloc_events_sim):
        cur_cxl_mem_vec -= dealloc_events[i, :]
        cur_cxl_mem_vec = np.maximum(cur_cxl_mem_vec, 0.0)

        for node_in_pod_id, dealloc_time, cur_event in cur_event_list:
            mem = float(cur_event[MEM_IDX])
            if mem <= 0:
                continue

            mhd_list = [
                mhd for mhd in range(num_mhd) if M[node_in_pod_id][mhd] != 0
            ]
            assert len(mhd_list) != 0

            ctx = {
                "tick": i,
                "pod_start_ts": pod_start_ts,
                "base_time": base,
                "num_mhd": num_mhd,
                "pod_rss_mem": float(pod_rss[MEM_IDX]),
            }

            alloc_vec = np.asarray(
                alloc_fn(mem, mhd_list, cur_cxl_mem_vec, ctx), dtype=float
            )
            assert alloc_vec.shape == (num_mhd,)

            cur_cxl_mem_vec += alloc_vec

            assert dealloc_time > i
            if dealloc_time < pod_dur:
                dealloc_events[dealloc_time, :] += alloc_vec

        max_cxl_mem_vec = np.maximum(max_cxl_mem_vec, cur_cxl_mem_vec)

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


def make_rl_alloc_cb(model, max_degree):
    """Return a callback that uses the trained RL policy."""

    def _rl_alloc_cb(cxl_mem, mhd_list, cur_cxl_mem_vec, ctx):
        n_acc = len(mhd_list)
        num_mhd = ctx["num_mhd"]
        norm = ctx["pod_rss_mem"] if ctx["pod_rss_mem"] > 0 else 1.0

        # Build observation (same layout as OctopusMemPoolEnv._get_obs)
        loads = np.zeros(max_degree, dtype=np.float32)
        for idx, mhd in enumerate(mhd_list):
            loads[idx] = float(cur_cxl_mem_vec[mhd]) / norm

        mask = np.zeros(max_degree, dtype=np.float32)
        mask[:n_acc] = 1.0

        vm_norm = np.float32(cxl_mem / norm)
        peak_norm = np.float32(float(np.max(cur_cxl_mem_vec)) / norm)

        # Time features
        tick = ctx["tick"]
        abs_time = ctx["base_time"] + _dt.timedelta(
            minutes=(tick + ctx["pod_start_ts"]) * 5
        )
        hour = abs_time.hour + abs_time.minute / 60.0
        hour_sin = np.float32(np.sin(2.0 * np.pi * hour / 24.0))
        hour_cos = np.float32(np.cos(2.0 * np.pi * hour / 24.0))

        obs = np.concatenate(
            [loads, mask, np.array([vm_norm, peak_norm, hour_sin, hour_cos])]
        ).astype(np.float32)

        # Predict
        action, _ = model.predict(obs, deterministic=True)

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
def run_eval(policy_name, alloc_fn, matrix, trace_data, n_iter=50):
    """Run *n_iter* random pod mappings, print summary statistics."""
    all_vms, node_to_vms, node_to_machine, _vtsz, machine_sz = trace_data

    result_list = []
    for i in range(n_iter):
        print(f"\r  {policy_name}: iteration {i}/{n_iter}", end="", flush=True)
        pod_to_nodes = generate_pod_to_nodes(
            node_to_vms, len(matrix), 10086 + i
        )
        node_to_M, expanded_M = expand_M_to_all_nodes(matrix, pod_to_nodes)
        ratio = pooling_simulation(
            node_to_M, expanded_M, alloc_fn,
            all_vms, node_to_vms, node_to_machine, machine_sz,
        )
        result_list.append(1.0 - ratio)

    print()
    avg = np.mean(result_list)
    mn = np.min(result_list)
    mx = np.max(result_list)
    std = np.std(result_list)
    print(f"  {policy_name}:")
    print(f"    avg:  {avg:.4f}")
    print(f"    min:  {mn:.4f}")
    print(f"    max:  {mx:.4f}")
    print(f"    std:  {std:.4f}")
    return result_list


# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate policies on Octopus memory pooling"
    )
    ap.add_argument(
        "--policy",
        nargs="+",
        default=["greedy", "rl"],
        choices=["greedy", "rl"],
        help="Policies to evaluate",
    )
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
        default="output/checkpoints/octopus_sac_final",
        help="Path to trained SB3 model (for --policy rl)",
    )
    ap.add_argument("--n-iter", type=int, default=50)
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

        print(f"Loading RL model: {args.model} …")
        model = SAC.load(args.model)
        rl_cb = make_rl_alloc_cb(model, max_deg)
        results["rl"] = run_eval(
            "RL (SAC)", rl_cb, M, trace_data, args.n_iter
        )

    # ── Summary ──────────────────────────────────────────────────────
    if len(results) > 1:
        print("\n── Comparison ──")
        for name, vals in results.items():
            print(f"  {name:>10s}:  avg={np.mean(vals):.4f}  std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
