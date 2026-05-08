#!/usr/bin/env python3
"""
Evaluate one or more RL checkpoints using pooling_simulation.

Usage
-----
  python scripts/eval_rl.py --run-id smoke_test --n-iter 5 --traces AMS20PrdApp19-tround
  python scripts/eval_rl.py --run-id v0_fast v1_lam05 --n-iter 50
  python scripts/eval_rl.py --model-path output/checkpoints/best_model.zip --n-iter 10
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


def _eval_model_on_trace(model, max_deg, trace_data, M, n_iter,
                         obs_variant="current", mhd_to_hosts=None,
                         Q_j=None, lookahead_window=200):
    """Run n_iter pod mappings; return (savings_list, ratio_list, detail_rows)."""
    all_vms, node_to_vms, node_to_machine, _, machine_sz = trace_data
    savings_list = []
    ratio_list = []
    detail_rows = []
    track = obs_variant not in {"current", "R1"}

    rl_cb = make_rl_alloc_cb(
        model, max_deg,
        obs_variant=obs_variant,
        mhd_to_hosts=mhd_to_hosts,
        Q_j=Q_j,
        lookahead_window=lookahead_window,
    )

    for i in range(n_iter):
        seed = 10086 + i
        pod_to_nodes = generate_pod_to_nodes(node_to_vms, len(M), seed)
        node_to_M, expanded_M = expand_M_to_all_nodes(M, pod_to_nodes)
        ratio = pooling_simulation(
            node_to_M, expanded_M, rl_cb,
            all_vms, node_to_vms, node_to_machine, machine_sz,
            track_vm_allocs=track,
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
    ap.add_argument(
        "--reward-variant",
        choices=["current", "R1", "R2", "R3", "R4", "R5", "A", "B"],
        default=None,
        help="Reward variant used during training (determines obs layout). "
             "If omitted, read from config.json in the checkpoint directory. "
             "A/B are deprecated aliases for R2/R3.",
    )
    ap.add_argument(
        "--lookahead-window",
        type=int,
        default=None,
        help="Lookahead window W for D_j (default: read from config.json or 200).",
    )
    args = ap.parse_args()

    import json as _json
    import numpy as _np
    from stable_baselines3 import SAC

    # Determine models to evaluate
    models_to_eval = []  # list of (label, model_path, ckpt_dir)
    if args.run_id:
        for rid in args.run_id:
            ckpt_dir = os.path.join("output", "checkpoints", rid)
            mp = os.path.join(ckpt_dir, "best_model")
            models_to_eval.append((rid, mp, ckpt_dir))
    if args.model_path:
        label = os.path.splitext(os.path.basename(args.model_path))[0]
        ckpt_dir = os.path.dirname(args.model_path)
        models_to_eval.append((label, args.model_path, ckpt_dir))

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

    for label, model_path, ckpt_dir in models_to_eval:
        print(f"\n=== Evaluating run: {label} ===")

        check_path = model_path if os.path.exists(model_path) else model_path + ".zip"
        if not os.path.exists(check_path):
            print(f"  Model not found: {check_path} — skipping.")
            continue

        # Resolve reward variant and lookahead window (CLI > config.json > defaults)
        obs_variant = args.reward_variant
        lookahead_window = args.lookahead_window
        config_path = os.path.join(ckpt_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as _f:
                _cfg = _json.load(_f)
            if obs_variant is None:
                obs_variant = _cfg.get("reward_variant", "current")
            if lookahead_window is None:
                lookahead_window = _cfg.get("lookahead_window", 200)
        obs_variant = obs_variant or "current"
        lookahead_window = lookahead_window or 200
        print(f"  obs_variant={obs_variant}  lookahead_window={lookahead_window}")

        # Precompute topology-derived structures for rich obs variants
        # R1 and "current" use simple obs — no mhd_to_hosts/Q_j needed
        _SIMPLE_OBS = {"current", "R1"}
        mhd_to_hosts = None
        Q_j = None
        if obs_variant not in _SIMPLE_OBS:
            num_mhd = len(M[0])
            mhd_to_hosts = {j: [] for j in range(num_mhd)}
            host_to_mhds_tmp = {}
            for h, row in enumerate(M):
                host_to_mhds_tmp[h] = [j for j, v in enumerate(row) if v]
                for j in host_to_mhds_tmp[h]:
                    mhd_to_hosts[j].append(h)
            Q_j = _np.zeros(num_mhd, dtype=_np.float32)
            for j in range(num_mhd):
                hosts = mhd_to_hosts[j]
                if hosts:
                    inv_deg = [1.0 / len(host_to_mhds_tmp[h]) for h in hosts if host_to_mhds_tmp[h]]
                    if inv_deg:
                        Q_j[j] = float(_np.mean(inv_deg))

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
                    model, max_deg, trace_data, M, args.n_iter,
                    obs_variant=obs_variant,
                    mhd_to_hosts=mhd_to_hosts,
                    Q_j=Q_j,
                    lookahead_window=lookahead_window,
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
