#!/usr/bin/env python3
"""
Read saved CSVs from eval_baselines.py and eval_rl.py; produce all figures.
Never runs any simulation itself.

Usage
-----
  python scripts/plot_results.py
  python scripts/plot_results.py --baselines-csv output/baselines/results.csv \\
      --rl-dirs output/rl_evals/smoke_test --out-dir output/figures/
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Data loading ──────────────────────────────────────────────────────────
def _load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_all_rl(rl_dirs):
    rows = []
    for d in rl_dirs:
        p = os.path.join(d, "results.csv")
        rows.extend(_load_csv(p))
    return rows


def _load_rl_detail(rl_dirs):
    rows = []
    for d in rl_dirs:
        p = os.path.join(d, "episode_detail.csv")
        rows.extend(_load_csv(p))
    return rows


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ── Figure 1: Savings comparison bar chart ────────────────────────────────
def plot_savings_by_trace(baseline_rows, rl_rows, topology_filter, trace_filter, out_dir):
    all_rows = baseline_rows + rl_rows
    if topology_filter:
        all_rows = [r for r in all_rows if topology_filter in r.get("topology", "")]

    policies = sorted(set(r["policy"] for r in all_rows))
    traces = sorted(set(r["trace"] for r in all_rows))
    if trace_filter:
        traces = [t for t in traces if t in trace_filter]

    # Sort traces by greedy savings descending
    greedy_mean = {}
    for r in all_rows:
        if r["policy"] == "greedy":
            greedy_mean[r["trace"]] = _float(r["savings_mean"])
    traces.sort(key=lambda t: greedy_mean.get(t, 0.0), reverse=True)

    lookup = {}
    for r in all_rows:
        lookup[(r["policy"], r["trace"])] = r

    x = np.arange(len(traces))
    width = 0.8 / max(len(policies), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(traces) * 1.2), 6))

    for pi, policy in enumerate(policies):
        means = [_float(lookup.get((policy, t), {}).get("savings_mean", "nan")) for t in traces]
        stds = [_float(lookup.get((policy, t), {}).get("savings_std", "nan")) for t in traces]
        ax.bar(x + pi * width - (len(policies) - 1) * width / 2,
               means, width * 0.9, yerr=stds, label=policy, capsize=3)

    short_traces = [t.replace("-troundgrt5m.sqlite", "").replace("-tround.sqlite", "")
                    for t in traces]
    ax.set_xticks(x)
    ax.set_xticklabels(short_traces, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pooling savings")
    ax.set_title("Pooling savings by trace and policy")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    out = os.path.join(out_dir, "savings_by_trace.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure 1 saved: {out}")


# ── Figure 2: Training curves ─────────────────────────────────────────────
def plot_training_curves(rl_dirs, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    any_plot = False

    for d in rl_dirs:
        label = os.path.basename(d.rstrip("/"))
        npz_path = os.path.join(d, "..", "..", "logs", label, "evaluations.npz")
        # Also try direct log dir
        alt_npz = os.path.join("output", "logs", label, "evaluations.npz")
        for p in [npz_path, alt_npz]:
            if os.path.exists(p):
                data = np.load(p)
                timesteps = data["timesteps"]
                results = data["results"]
                means = results.mean(axis=1)
                ax.plot(timesteps, means, label=label, alpha=0.8)
                any_plot = True
                break

    if not any_plot:
        plt.close(fig)
        print("Figure 2: no evaluations.npz found — skipped.")
        return

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Training curves")
    ax.legend(fontsize=8)

    out = os.path.join(out_dir, "training_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure 2 saved: {out}")


# ── Figure 3: Per-MPD load time series ───────────────────────────────────
def plot_timeseries(detail_rows, trace_filter, out_dir):
    # Group by (trace, policy)
    groups = {}
    for r in detail_rows:
        key = (r["trace"], r["policy"])
        groups.setdefault(key, []).append(r)

    if not groups:
        print("Figure 3: no episode_detail.csv data — skipped.")
        return

    traces = sorted(set(r["trace"] for r in detail_rows))
    if trace_filter:
        traces = [t for t in traces if t in trace_filter]

    for trace in traces[:3]:  # limit to avoid too many files
        policies = sorted(set(r["policy"] for r in detail_rows if r["trace"] == trace))
        for policy in policies:
            rows = [r for r in detail_rows if r["trace"] == trace and r["policy"] == policy]
            if not rows:
                continue
            savings_vals = [_float(r["savings"]) for r in rows]
            short_t = trace.replace("-troundgrt5m.sqlite", "").replace("-tround.sqlite", "")
            short_p = policy.replace(":", "_")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(savings_vals, bins=20, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Pooling savings")
            ax.set_ylabel("Count")
            ax.set_title(f"Savings distribution: {short_p} / {short_t}")
            out = os.path.join(out_dir, f"timeseries_{short_t}_{short_p}.png")
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"Figure 3 saved: {out}")


# ── Figure 4: Load variance comparison ───────────────────────────────────
def plot_load_variance(baseline_rows, rl_rows, topology_filter, trace_filter, out_dir):
    all_rows = baseline_rows + rl_rows
    if topology_filter:
        all_rows = [r for r in all_rows if topology_filter in r.get("topology", "")]

    # Filter rows that have non-nan variance
    var_rows = [r for r in all_rows if not np.isnan(_float(r.get("mpd_load_variance_mean", "nan")))]
    if not var_rows:
        print("Figure 4: no mpd_load_variance_mean data — skipped.")
        return

    policies = sorted(set(r["policy"] for r in var_rows))
    traces = sorted(set(r["trace"] for r in var_rows))
    if trace_filter:
        traces = [t for t in traces if t in trace_filter]

    lookup = {}
    for r in var_rows:
        lookup[(r["policy"], r["trace"])] = r

    x = np.arange(len(traces))
    width = 0.8 / max(len(policies), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(traces) * 1.2), 6))

    for pi, policy in enumerate(policies):
        means = [_float(lookup.get((policy, t), {}).get("mpd_load_variance_mean", "nan"))
                 for t in traces]
        ax.bar(x + pi * width - (len(policies) - 1) * width / 2,
               means, width * 0.9, label=policy)

    short_traces = [t.replace("-troundgrt5m.sqlite", "").replace("-tround.sqlite", "")
                    for t in traces]
    ax.set_xticks(x)
    ax.set_xticklabels(short_traces, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean MPD load variance")
    ax.set_title("MPD load variance by trace and policy")
    ax.legend(fontsize=8)

    out = os.path.join(out_dir, "load_variance_by_trace.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure 4 saved: {out}")


# ── Figure 5: LaTeX table ─────────────────────────────────────────────────
def write_latex_table(baseline_rows, topology_filter, out_dir):
    if not baseline_rows:
        print("Figure 5: no baseline data — skipped.")
        return

    rows = baseline_rows
    if topology_filter:
        rows = [r for r in rows if topology_filter in r.get("topology", "")]

    greedy_rows = [r for r in rows if r["policy"] == "greedy"]
    optimal_rows = [r for r in rows if r["policy"] == "optimal"]

    topologies = sorted(set(r["topology"] for r in greedy_rows))
    traces = sorted(set(r["trace"] for r in greedy_rows))

    g_lookup = {(r["topology"], r["trace"]): r for r in greedy_rows}
    o_lookup = {(r["topology"], r["trace"]): r for r in optimal_rows}

    lines = []
    short_tops = [os.path.basename(t).replace(".csv", "") for t in topologies]

    lines.append("\\begin{tabular}{l" + "c" * len(topologies) + ("c" if optimal_rows else "") + "}")
    lines.append("\\hline")
    header = "Trace & " + " & ".join(short_tops)
    if optimal_rows:
        header += " & Gap (opt-greedy)"
    header += " \\\\"
    lines.append(header)
    lines.append("\\hline")

    for trace in traces:
        short_t = trace.replace("-troundgrt5m.sqlite", "").replace("-tround.sqlite", "")
        cells = []
        for topo in topologies:
            r = g_lookup.get((topo, trace))
            if r:
                cells.append(f"{_float(r['savings_mean']):.3f}$\\pm${_float(r['savings_std']):.3f}")
            else:
                cells.append("--")
        if optimal_rows:
            # Average gap across topologies
            gaps = []
            for topo in topologies:
                g = g_lookup.get((topo, trace))
                o = o_lookup.get((topo, trace))
                if g and o:
                    gaps.append(_float(o["savings_mean"]) - _float(g["savings_mean"]))
            if gaps:
                cells.append(f"{np.mean(gaps):.3f}")
            else:
                cells.append("--")
        lines.append(short_t + " & " + " & ".join(cells) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    out = os.path.join(out_dir, "tab_greedy_vs_opt.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Figure 5 saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Plot results from baseline and RL eval CSVs")
    ap.add_argument("--baselines-csv", default="output/baselines/results.csv")
    ap.add_argument("--rl-dirs", nargs="*", default=None,
                    help="Paths to RL eval result dirs. Default: all subdirs of output/rl_evals/")
    ap.add_argument("--out-dir", default="output/figures/")
    ap.add_argument("--traces", nargs="*", default=None,
                    help="Filter to subset of traces")
    ap.add_argument("--topology", default="AG16x6_expander_quads_r5_sym_fixed",
                    help="Filter to one topology (substring match)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Discover RL dirs
    rl_dirs = args.rl_dirs
    if rl_dirs is None:
        rl_root = "output/rl_evals"
        if os.path.isdir(rl_root):
            rl_dirs = [os.path.join(rl_root, d) for d in os.listdir(rl_root)
                       if os.path.isdir(os.path.join(rl_root, d))]
        else:
            rl_dirs = []

    baseline_rows = _load_csv(args.baselines_csv)
    rl_rows = _load_all_rl(rl_dirs)
    detail_rows = _load_rl_detail(rl_dirs)

    print(f"Baseline rows: {len(baseline_rows)}")
    print(f"RL rows: {len(rl_rows)}")
    print(f"Detail rows: {len(detail_rows)}")

    topo_filter = args.topology or ""
    trace_filter = args.traces or []

    plot_savings_by_trace(baseline_rows, rl_rows, topo_filter, trace_filter, args.out_dir)
    plot_training_curves(rl_dirs, args.out_dir)
    plot_timeseries(detail_rows, trace_filter, args.out_dir)
    plot_load_variance(baseline_rows, rl_rows, topo_filter, trace_filter, args.out_dir)
    write_latex_table(baseline_rows, topo_filter, args.out_dir)

    print(f"\nAll figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
