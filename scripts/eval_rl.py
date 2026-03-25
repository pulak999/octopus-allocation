#!/usr/bin/env python3
"""
Evaluate one or more RL checkpoints using pooling_simulation.

Usage
-----
  python scripts/eval_rl.py --run-id smoke_test --n-iter 5 --traces AMS20PrdApp19-tround
  python scripts/eval_rl.py --run-id v0_fast v1_lam05 --n-iter 50
  python scripts/eval_rl.py --model-path output/checkpoints/best_model.zip --n-iter 10
"""

import argparse
import csv
import datetime
import os

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

from octopus.data import load_trace, load_topology
from octopus.topology import generate_pod_to_nodes, expand_M_to_all_nodes
from scripts.evaluate import pooling_simulation, make_rl_alloc_cb

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

CSV_FIELDNAMES = [
    "policy", "topology", "trace",
    "pooling_ratio_mean", "pooling_ratio_std",
    "savings_mean", "savings_min", "savings_max", "savings_std",
    "mpd_load_variance_mean", "n_iter", "timestamp",
]

DETAIL_FIELDNAMES = [
    "policy", "topology", "trace", "mapping_seed", "pooling_ratio", "savings",
]


def _eval_model_on_trace(model, max_deg, trace_data, M, n_iter):
    """Run n_iter pod mappings; return (savings_list, ratio_list, detail_rows)."""
    all_vms, node_to_vms, node_to_machine, _, machine_sz = trace_data
    savings_list = []
    ratio_list = []
    detail_rows = []

    rl_cb = make_rl_alloc_cb(model, max_deg)

    for i in range(n_iter):
        seed = 10086 + i
        pod_to_nodes = generate_pod_to_nodes(node_to_vms, len(M), seed)
        node_to_M, expanded_M = expand_M_to_all_nodes(M, pod_to_nodes)
        ratio = pooling_simulation(
            node_to_M, expanded_M, rl_cb,
            all_vms, node_to_vms, node_to_machine, machine_sz,
        )
        savings = 1.0 - ratio
        savings_list.append(savings)
        ratio_list.append(ratio)
        detail_rows.append({"mapping_seed": seed, "pooling_ratio": ratio, "savings": savings})

    return savings_list, ratio_list, detail_rows


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate RL checkpoints on pooling simulation"
    )
    ap.add_argument("--run-id", nargs="+", default=None,
                    help="One or more run IDs (looks up output/checkpoints/<run-id>/best_model.zip)")
    ap.add_argument("--model-path", default=None,
                    help="Explicit model path (alternative to --run-id)")
    ap.add_argument(
        "--traces", nargs="+", default=None,
        help="Cluster name stems. Default: all 10.",
    )
    ap.add_argument(
        "--topology",
        default="data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv",
    )
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--out-dir", default="output/rl_evals/")
    ap.add_argument(
        "--skip-hotfix",
        action="store_true",
        help="Skip the HOTFIX VM filter (match training config).",
    )
    args = ap.parse_args()

    from stable_baselines3 import SAC

    # Determine models to evaluate
    models_to_eval = []  # list of (label, model_path)
    if args.run_id:
        for rid in args.run_id:
            mp = os.path.join("output", "checkpoints", rid, "best_model")
            models_to_eval.append((rid, mp))
    if args.model_path:
        label = os.path.splitext(os.path.basename(args.model_path))[0]
        models_to_eval.append((label, args.model_path))

    if not models_to_eval:
        ap.error("Provide --run-id or --model-path")

    # Normalise trace names
    traces = []
    for t in (args.traces or ALL_TRACES):
        if not t.endswith(".sqlite"):
            t = t + ".sqlite"
        traces.append(t)

    M, num_hosts, num_pools = load_topology(args.topology)
    max_deg = max(sum(row) for row in M)
    print(f"Topology: {num_hosts} hosts, {num_pools} MPDs, max_degree={max_deg}")

    for label, model_path in models_to_eval:
        print(f"\n=== Evaluating run: {label} ===")

        check_path = model_path if os.path.exists(model_path) else model_path + ".zip"
        if not os.path.exists(check_path):
            print(f"  Model not found: {check_path} — skipping.")
            continue

        model = SAC.load(model_path)

        out_dir = os.path.join(args.out_dir, label)
        os.makedirs(out_dir, exist_ok=True)

        results_path = os.path.join(out_dir, "results.csv")
        detail_path = os.path.join(out_dir, "episode_detail.csv")

        results_rows = []
        detail_rows_all = []

        for trace_name in tqdm(traces, desc=f"{label}"):
            print(f"  trace: {trace_name}")
            try:
                trace_data = load_trace(trace_name)
                savings_list, ratio_list, detail_rows = _eval_model_on_trace(
                    model, max_deg, trace_data, M, args.n_iter
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            sav_arr = np.array(savings_list)
            rat_arr = np.array(ratio_list)
            ts = datetime.datetime.utcnow().isoformat()
            results_rows.append({
                "policy": f"rl:{label}",
                "topology": args.topology,
                "trace": trace_name,
                "pooling_ratio_mean": float(np.mean(rat_arr)),
                "pooling_ratio_std": float(np.std(rat_arr)),
                "savings_mean": float(np.mean(sav_arr)),
                "savings_min": float(np.min(sav_arr)),
                "savings_max": float(np.max(sav_arr)),
                "savings_std": float(np.std(sav_arr)),
                "mpd_load_variance_mean": float("nan"),
                "n_iter": args.n_iter,
                "timestamp": ts,
            })
            for dr in detail_rows:
                detail_rows_all.append({
                    "policy": f"rl:{label}",
                    "topology": args.topology,
                    "trace": trace_name,
                    "mapping_seed": dr["mapping_seed"],
                    "pooling_ratio": dr["pooling_ratio"],
                    "savings": dr["savings"],
                })

            print(f"    pooling_ratio: {np.mean(rat_arr):.4f} ± {np.std(rat_arr):.4f}  "
                  f"savings: {np.mean(sav_arr):.4f} ± {np.std(sav_arr):.4f}")

        # Write results CSV
        with open(results_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            w.writeheader()
            w.writerows(results_rows)
        print(f"  Results → {results_path}")

        # Write detail CSV
        with open(detail_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DETAIL_FIELDNAMES)
            w.writeheader()
            w.writerows(detail_rows_all)
        print(f"  Episode detail → {detail_path}")

        # Summary table
        print(f"\n  Summary for {label}:")
        print(f"  {'Trace':<40} {'pool_ratio':>10} {'savings':>10}")
        for r in results_rows:
            print(f"  {r['trace']:<40} {r['pooling_ratio_mean']:>10.4f} {r['savings_mean']:>10.4f}")


if __name__ == "__main__":
    main()
