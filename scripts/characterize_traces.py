#!/usr/bin/env python3
"""
Characterize all 10 Azure VM traces for train/val/test split assignment.

For each trace computes:
  - Total VM count, unique VM type count
  - rss[1] (memory demand) mean/std
  - VM lifetime mean/std (in ticks, 1 tick = 5 min)
  - Arrival rate (VMs per tick)
  - Peak aggregate memory demand (GB)
  - Episode event counts with and without HOTFIX

Usage
-----
  python scripts/characterize_traces.py
  python scripts/characterize_traces.py --out-csv output/trace_characterization.csv
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import datetime
import os

import numpy as np

from octopus.data import load_trace, load_topology

# Default topology for event counting (HOTFIX depends on topology)
DEFAULT_TOPOLOGY = "data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv"

ALL_TRACES = [
    "AMS20PrdApp19-tround.sqlite",
    "BLAPrdApp19-troundgrt5m.sqlite",
    "BN9PrdApp18-troundgrt5m.sqlite",
    "DSM08PrdApp05-troundgrt5m.sqlite",
    "DUB24PrdApp09-troundgrt5m.sqlite",
    "LON23PrdApp01-troundgrt5m.sqlite",
    "LVL01PrdApp05-troundgrt5m.sqlite",
    "SG2PrdApp35-troundgrt5m.sqlite",
    "SYD21PrdApp07-troundgrt5m.sqlite",
    "YTO21PrdApp05-troundgrt5m.sqlite",
]

MEM_IDX = 1


def _count_events_for_trace(trace_data, M, seed, skip_hotfix):
    """Count events for one pod assignment using OctopusMemPoolEnv logic."""
    from octopus.env import OctopusMemPoolEnv

    all_vms, node_to_vms, node_to_machine, _, machine_sz = trace_data
    env = OctopusMemPoolEnv(
        all_vms=all_vms,
        node_to_vms=node_to_vms,
        node_to_machine=node_to_machine,
        machine_sz=machine_sz,
        M=M,
        seed=seed,
        skip_hotfix=skip_hotfix,
    )
    _, info = env.reset(seed=seed)
    return info["num_events"]


def characterize_trace(trace_name, M, n_seeds=5):
    """Compute characterization metrics for one trace."""
    trace_data = load_trace(trace_name)
    all_vms, node_to_vms, node_to_machine, _, machine_sz = trace_data

    # --- Basic VM stats ---
    vm_count = len(all_vms)
    vm_types = set()
    rss_values = []
    lifetimes = []  # in ticks (5-min intervals)

    for vm in all_vms.values():
        rss_val = float(np.asarray(vm.rss, dtype=float)[MEM_IDX])
        rss_values.append(rss_val)

        # VM type = (cores, memory tier) — approximate by rss[0] (cores) + rss[1] (mem)
        rss_tuple = tuple(float(r) for r in vm.rss)
        vm_types.add(rss_tuple)

        # Lifetime in ticks
        if vm.end_time > vm.start_time:
            lifetime_sec = (vm.end_time - vm.start_time).total_seconds()
            lifetimes.append(lifetime_sec / 300.0)  # convert to ticks

    rss_arr = np.array(rss_values)
    life_arr = np.array(lifetimes) if lifetimes else np.array([0.0])

    # --- Arrival rate ---
    # Compute time span in ticks
    min_time = min(vm.start_time for vm in all_vms.values())
    max_time = max(vm.end_time for vm in all_vms.values())
    span_ticks = max(1, (max_time - min_time).total_seconds() / 300.0)
    arrival_rate = vm_count / span_ticks  # VMs per tick

    # --- Peak aggregate memory demand ---
    # Sample at tick resolution: build a demand curve
    tick_start = {}
    tick_end = {}
    for vm in all_vms.values():
        rss_val = float(np.asarray(vm.rss, dtype=float)[MEM_IDX])
        if rss_val <= 0:
            continue
        s = int((vm.start_time - min_time).total_seconds() // 300)
        e = int((vm.end_time - min_time).total_seconds() // 300)
        tick_start[s] = tick_start.get(s, 0.0) + rss_val
        tick_end[e] = tick_end.get(e, 0.0) + rss_val

    # Sweep through change points to find peak
    all_change_ticks = sorted(set(list(tick_start.keys()) + list(tick_end.keys())))
    current_demand = 0.0
    peak_demand = 0.0
    for t in all_change_ticks:
        current_demand += tick_start.get(t, 0.0)
        current_demand -= tick_end.get(t, 0.0)
        peak_demand = max(peak_demand, current_demand)

    # --- Event counts with/without HOTFIX (average over n_seeds pod assignments) ---
    events_with_hotfix = []
    events_without_hotfix = []
    for seed in range(n_seeds):
        events_with_hotfix.append(
            _count_events_for_trace(trace_data, M, seed=seed, skip_hotfix=False)
        )
        events_without_hotfix.append(
            _count_events_for_trace(trace_data, M, seed=seed, skip_hotfix=True)
        )

    return {
        "trace": trace_name.replace(".sqlite", ""),
        "vm_count": vm_count,
        "vm_type_count": len(vm_types),
        "rss1_mean": float(np.mean(rss_arr)),
        "rss1_std": float(np.std(rss_arr)),
        "lifetime_mean": float(np.mean(life_arr)),
        "lifetime_std": float(np.std(life_arr)),
        "arrival_rate": arrival_rate,
        "peak_demand_gb": peak_demand,
        "events_hotfix_mean": float(np.mean(events_with_hotfix)),
        "events_no_hotfix_mean": float(np.mean(events_without_hotfix)),
        "hotfix_survival_pct": (
            100.0 * np.mean(events_with_hotfix) / max(1, np.mean(events_without_hotfix))
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="Characterize Azure VM traces")
    ap.add_argument(
        "--out-csv",
        default="output/trace_characterization.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--topology",
        default=DEFAULT_TOPOLOGY,
        help="Topology CSV for event counting",
    )
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of pod seeds to average event counts over",
    )
    args = ap.parse_args()

    M, _, _ = load_topology(args.topology)

    results = []
    for trace_name in ALL_TRACES:
        print(f"Characterizing {trace_name} ...", flush=True)
        row = characterize_trace(trace_name, M, n_seeds=args.n_seeds)
        results.append(row)
        print(f"  VMs={row['vm_count']:,}  types={row['vm_type_count']}  "
              f"rss1={row['rss1_mean']:.2f}±{row['rss1_std']:.2f}  "
              f"lifetime={row['lifetime_mean']:.1f}±{row['lifetime_std']:.1f}  "
              f"arrival={row['arrival_rate']:.2f}/tick  "
              f"peak={row['peak_demand_gb']:.0f}GB  "
              f"events(hf/no)={row['events_hotfix_mean']:.0f}/{row['events_no_hotfix_mean']:.0f}  "
              f"survival={row['hotfix_survival_pct']:.1f}%")

    # Print summary table
    print("\n" + "=" * 120)
    print(f"{'Trace':<35} {'VMs':>7} {'Types':>6} {'rss1μ':>7} {'rss1σ':>7} "
          f"{'lifeμ':>7} {'lifeσ':>7} {'arr/t':>6} {'peakGB':>8} "
          f"{'ev_hf':>7} {'ev_no':>7} {'surv%':>6}")
    print("-" * 120)
    for r in results:
        print(f"{r['trace']:<35} {r['vm_count']:>7,} {r['vm_type_count']:>6} "
              f"{r['rss1_mean']:>7.2f} {r['rss1_std']:>7.2f} "
              f"{r['lifetime_mean']:>7.1f} {r['lifetime_std']:>7.1f} "
              f"{r['arrival_rate']:>6.2f} {r['peak_demand_gb']:>8.0f} "
              f"{r['events_hotfix_mean']:>7.0f} {r['events_no_hotfix_mean']:>7.0f} "
              f"{r['hotfix_survival_pct']:>6.1f}")
    print("=" * 120)

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV written → {args.out_csv}")


if __name__ == "__main__":
    main()
