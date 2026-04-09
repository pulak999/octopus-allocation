#!/usr/bin/env python3
"""
Evaluate greedy, PID, and optional optimal baselines across all
traces × topologies.  Saves results to a single CSV; idempotent
(skips combos already in the CSV).

Usage
-----
  python scripts/eval_baselines.py --policies greedy pid --n-iter 5 \\
      --traces AMS20PrdApp19-tround --topologies data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv

  # Full sweep (slow):
  python scripts/eval_baselines.py --policies greedy pid --n-iter 50
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import datetime
import glob
import itertools
import os
import shutil
import tempfile
import time
from multiprocessing import Pool

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

from octopus.data import load_trace, load_topology
from octopus.topology import generate_pod_to_nodes, expand_M_to_all_nodes
from scripts.evaluate import (
    pooling_simulation,
    _greedy_alloc_cb,
    make_pid_alloc_cb,
)

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

ALL_TOPOLOGIES = [
    "data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv",
    "data/topologies/AG16x4_latin3_r5_sym.csv",
    "data/topologies/bibd_25.csv",
    # "data/topologies/random_L64_X8_N4_Y128.csv",  # empty file - skip until populated
    "data/topologies/random_L96_X8_N4_Y192.csv",
]

CSV_FIELDNAMES = [
    "policy", "topology", "trace",
    "savings_mean", "savings_min", "savings_max", "savings_std",
    "mpd_load_variance_mean", "n_iter", "timestamp", "wall_sec",
]


def _load_existing(csv_path):
    """Return set of (policy, topology, trace) already in the CSV."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["policy"], row["topology"], row["trace"]))
    return done


def _run_combo(args_tuple):
    """Run one (policy, topology, trace) combo and return a result dict.
    Accepts a tuple so it can be pickled for multiprocessing.
    """
    policy, topology_path, trace_name, n_iter, kp, ki, kd = args_tuple
    trace_data = load_trace(trace_name)
    all_vms, node_to_vms, node_to_machine, _, machine_sz = trace_data
    M, num_hosts, num_pools = load_topology(topology_path)

    savings_list = []
    total_wall_sec = 0.0

    for i in range(n_iter):
        if policy == "pid":
            alloc_fn = make_pid_alloc_cb(kp=kp, ki=ki, kd=kd)
        else:
            alloc_fn = _greedy_alloc_cb

        pod_to_nodes = generate_pod_to_nodes(node_to_vms, len(M), 10086 + i)
        node_to_M, expanded_M = expand_M_to_all_nodes(M, pod_to_nodes)

        t0 = time.perf_counter()
        ratio = pooling_simulation(
            node_to_M, expanded_M, alloc_fn,
            all_vms, node_to_vms, node_to_machine, machine_sz,
        )
        wall = time.perf_counter() - t0
        savings_list.append(1.0 - ratio)
        total_wall_sec += wall

    arr = np.array(savings_list)
    return {
        "policy": policy,
        "topology": topology_path,
        "trace": trace_name,
        "savings_mean": float(np.mean(arr)),
        "savings_min": float(np.min(arr)),
        "savings_max": float(np.max(arr)),
        "savings_std": float(np.std(arr)),
        "mpd_load_variance_mean": float("nan"),  # not tracked at sim level
        "n_iter": n_iter,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "wall_sec": round(total_wall_sec, 2),
    }


def _print_latex_table(csv_path):
    """Print tab:greedy_vs_opt LaTeX tabular to stdout."""
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    greedy_rows = [r for r in rows if r["policy"] == "greedy"]
    topologies = sorted(set(r["topology"] for r in greedy_rows))
    traces = sorted(set(r["trace"] for r in greedy_rows))

    # Build lookup
    lookup = {}
    for r in greedy_rows:
        lookup[(r["topology"], r["trace"])] = r

    print("\n% LaTeX table: greedy savings (mean ± std)")
    print("\\begin{tabular}{l" + "c" * len(topologies) + "}")
    print("\\hline")
    short_tops = [os.path.basename(t).replace(".csv", "") for t in topologies]
    print("Trace & " + " & ".join(short_tops) + " \\\\")
    print("\\hline")
    for trace in traces:
        short_trace = trace.replace("-troundgrt5m.sqlite", "").replace("-tround.sqlite", "")
        cells = []
        for topo in topologies:
            r = lookup.get((topo, trace))
            if r:
                cells.append(f"{float(r['savings_mean']):.3f}±{float(r['savings_std']):.3f}")
            else:
                cells.append("--")
        print(short_trace + " & " + " & ".join(cells) + " \\\\")
    print("\\hline")
    print("\\end{tabular}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate baseline policies")
    ap.add_argument(
        "--policies", nargs="+", default=["greedy", "pid"],
        choices=["greedy", "pid", "optimal"],
        help="Policies to evaluate",
    )
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--out", default="output/baselines/results.csv")
    ap.add_argument(
        "--topologies", nargs="+", default=ALL_TOPOLOGIES,
        help="Topology CSV paths",
    )
    ap.add_argument(
        "--traces", nargs="+", default=None,
        help="Cluster name stems (without .pkl). Default: all 10.",
    )
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--ki", type=float, default=0.01)
    ap.add_argument("--kd", type=float, default=0.1)
    ap.add_argument("--n-workers", type=int, default=os.cpu_count(),
                    help="Parallel workers for combo evaluation (1 = serial)")
    args = ap.parse_args()

    # Normalise trace names
    traces = []
    for t in (args.traces or ALL_TRACES):
        if not t.endswith(".sqlite"):
            t = t + ".sqlite"
        traces.append(t)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    done = _load_existing(args.out)
    combos = list(itertools.product(args.policies, args.topologies, traces))
    combos_todo = [(p, top, tr) for p, top, tr in combos if (p, top, tr) not in done]

    print(f"Combos total: {len(combos)}, already done: {len(done)}, to run: {len(combos_todo)}")

    # Filter out optimal (not implemented) before building work list
    combos_todo = [(p, top, tr) for p, top, tr in combos_todo if p != "optimal"]

    # Ensure output CSV has a header
    write_header = not os.path.exists(args.out)
    if write_header:
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()

    todo_args = [
        (policy, topology_path, trace_name, args.n_iter, args.kp, args.ki, args.kd)
        for policy, topology_path, trace_name in combos_todo
    ]

    all_results = []
    wall_start = time.perf_counter()

    n_workers = min(args.n_workers, len(todo_args)) if todo_args else 1
    print(f"Using {n_workers} worker(s) for {len(todo_args)} combos …\n")

    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_run_combo, todo_args),
            total=len(todo_args),
            desc="Combos",
        ):
            all_results.append(result)
            # Atomic append: write to temp then copy-append to main CSV
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
            try:
                with os.fdopen(tmp_fd, "w", newline="") as tf:
                    w = csv.DictWriter(tf, fieldnames=CSV_FIELDNAMES)
                    w.writeheader()
                    w.writerow(result)
                with open(args.out, "a") as main_f, open(tmp_path) as tmp_f:
                    next(tmp_f)  # skip header
                    shutil.copyfileobj(tmp_f, main_f)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            print(
                f"  [done] {result['policy']} / {result['trace'][:30]}"
                f"  savings={result['savings_mean']:.4f}"
                f"  wall={result['wall_sec']:.1f}s"
            )

    elapsed = time.perf_counter() - wall_start
    if all_results:
        total_compute = sum(r["wall_sec"] for r in all_results)
        print(
            f"\n[timing] TOTAL: {total_compute:.0f}s compute across "
            f"{len(all_results)} combos, {n_workers} workers, "
            f"wall={elapsed:.0f}s"
        )

    print(f"\nResults saved to {args.out}")
    _print_latex_table(args.out)


if __name__ == "__main__":
    main()
