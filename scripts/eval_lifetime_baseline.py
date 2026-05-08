"""
Lifetime-conditioned allocation baseline.

For each VM arrival the callback checks whether the VM is "long-lived"
(dealloc_time - tick >= LONG_LIVED_TICKS).  Long-lived VMs are consolidated
(anti-greedy: assign to the *most* loaded accessible MPD, bin-pack style).
Short-lived VMs use the standard greedy water-fill.

The threshold LONG_LIVED_TICKS = 288 corresponds to 24 h at 5 min/tick.

Run:
    python scripts/eval_lifetime_baseline.py --traces AMS20PrdApp19-tround LVL01PrdApp05-troundgrt5m --n-iter 20
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from octopus.baselines import greedy_alloc
from octopus.topology import generate_pod_to_nodes, expand_M_to_all_nodes
from scripts.evaluate import pooling_simulation

# ── constants ──────────────────────────────────────────────────────────────
LONG_LIVED_TICKS = 288          # 24 h × 12 ticks/h
DEFAULT_TOPOLOGY = "data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv"
ALL_TRACES = [
    "AMS20PrdApp19-tround",
    "LVL01PrdApp05-troundgrt5m",
]


def anti_greedy_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec):
    """Assign entire VM to the single most-loaded accessible MPD (bin-pack)."""
    mhd_arr = np.asarray(mhd_list)
    loads = cur_cxl_mem_vec[mhd_arr]
    best = int(mhd_arr[np.argmax(loads)])
    alloc = np.zeros(len(cur_cxl_mem_vec))
    alloc[best] = float(cxl_mem)
    return alloc


def make_lifetime_cb(threshold=LONG_LIVED_TICKS):
    def _cb(cxl_mem, mhd_list, cur_cxl_mem_vec, ctx):
        tick = ctx["tick"]
        dealloc_time = ctx.get("dealloc_time", None)
        if dealloc_time is not None and (dealloc_time - tick) >= threshold:
            return anti_greedy_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec)
        return greedy_alloc(cxl_mem, mhd_list, cur_cxl_mem_vec)
    return _cb


def run_on_trace(trace_name, M, n_iter, threshold):
    from octopus.data import load_topology
    import pickle
    from octopus.data import _VMUnpickler

    path = f"data/traces/{trace_name}.sqlite.pkl"
    with open(path, "rb") as f:
        data = _VMUnpickler(f).load()
    all_vms, node_to_vms, node_to_machine, vm_to_sz, machine_sz = data

    cb_greedy   = lambda cxl, mhds, vec, ctx: greedy_alloc(cxl, mhds, vec)
    cb_lifetime = make_lifetime_cb(threshold)

    ratios_g, ratios_l = [], []
    for i in range(n_iter):
        seed = 10086 + i
        pod_to_nodes = generate_pod_to_nodes(node_to_vms, len(M), seed)
        node_to_M, expanded_M = expand_M_to_all_nodes(M, pod_to_nodes)

        r_g = pooling_simulation(node_to_M, expanded_M, cb_greedy,
                                 all_vms, node_to_vms, node_to_machine, machine_sz)
        r_l = pooling_simulation(node_to_M, expanded_M, cb_lifetime,
                                 all_vms, node_to_vms, node_to_machine, machine_sz)
        ratios_g.append(r_g)
        ratios_l.append(r_l)

    rg = np.array(ratios_g); rl = np.array(ratios_l)
    return {
        "greedy_ratio":   rg.mean(), "greedy_savings":   1 - rg.mean(),
        "lifetime_ratio": rl.mean(), "lifetime_savings": 1 - rl.mean(),
        "delta_pp": (1 - rl.mean() - (1 - rg.mean())) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", default=ALL_TRACES)
    ap.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--threshold-h", type=float, default=24.0,
                    help="Long-lived threshold in hours (default 24h = 288 ticks)")
    args = ap.parse_args()

    threshold_ticks = int(args.threshold_h * 12)  # 12 ticks per hour
    print(f"Long-lived threshold: {args.threshold_h}h = {threshold_ticks} ticks\n")

    import csv as _csv
    from octopus.data import load_topology
    M, _, _ = load_topology(args.topology)

    print(f"{'Trace':<30} {'Greedy':>8} {'Lifetime':>9} {'Delta(pp)':>10}")
    print("-" * 62)
    rows = []
    for trace in args.traces:
        stem = trace.replace(".sqlite", "")
        r = run_on_trace(stem, M, args.n_iter, threshold_ticks)
        print(f"{stem:<30} {r['greedy_savings']:>+8.3f} {r['lifetime_savings']:>+9.3f} {r['delta_pp']:>+10.2f}")
        rows.append({"trace": stem, **r})

    out = "output/lifetime_baseline_results.csv"
    os.makedirs("output", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
