#!/usr/bin/env python3
"""
Reads all TensorBoard event files under output/logs/ and produces:
  - output/analysis/metrics.csv      (tidy, all runs × all metrics × all steps)
  - output/analysis/*.png            (one plot per key metric, all runs overlaid)
  - output/analysis/run_summary.csv  (per-run statistics)

Usage:
  python scripts/analyze_runs.py
  python scripts/analyze_runs.py --runs aug_v4_rewardA aug_v4_rewardA_v2 split721_sac
  python scripts/analyze_runs.py --smooth 10  # rolling average window
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Config ────────────────────────────────────────────────────────────────────

# Metrics to extract and plot (tag → readable label)
METRICS = {
    "eval/pooling_savings_mean": "Pooling Savings (eval mean)",
    "eval/mean_reward":          "Episode Reward (eval)",
    "train/ent_coef":            "Entropy Coef (α)",
    "sac/q1_mean":               "Q1 Mean",
    "sac/q_spread":              "Q Spread |Q1−Q2|",
    "sac/actor_grad_norm":       "Actor Grad Norm",
    "sac/critic_grad_norm":      "Critic Grad Norm",
    "train/critic_loss":         "Critic Loss",
    "train/actor_loss":          "Actor Loss",
    "env/pooling_savings":       "Pooling Savings (train env)",
    "time/fps":                  "Throughput (fps)",
}

# Runs to label nicely in plots
RUN_LABELS = {
    "aug_v4_rewardA":      "run5 (244fps, rewardA)",
    "split721_sac":        "run6 (22fps, current reward)",
    "aug_v4_rewardA_v2":   "run7 (52fps, rewardA)",
    "aug_v3_rewardA":      "aug_v3_rewardA",
    "aug_v2_rewardA":      "aug_v2_rewardA",
    "aug_all_traces":      "aug_all_traces",
}

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

# ── Load ──────────────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> pd.DataFrame:
    """Load all scalar metrics from a run directory into a tidy DataFrame."""
    rows = []
    # Find the deepest SAC_N subdirectory (SB3 nests logs one level deeper)
    event_dirs = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not event_dirs:
        return pd.DataFrame()

    for event_file in event_dirs:
        ea = EventAccumulator(str(event_file.parent), size_guidance={"scalars": 0})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ea.Reload()

        available = ea.Tags().get("scalars", [])
        for tag in METRICS:
            if tag not in available:
                continue
            for event in ea.Scalars(tag):
                rows.append({
                    "step":   event.step,
                    "metric": tag,
                    "value":  event.value,
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_all_runs(log_root: Path, run_filter=None) -> pd.DataFrame:
    all_dfs = []
    for run_dir in sorted(log_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_filter and run_id not in run_filter:
            continue
        df = load_run(run_dir)
        if df.empty:
            print(f"  [skip] {run_id} — no scalar events found")
            continue
        df["run"] = run_id
        all_dfs.append(df)
        print(f"  [ok]   {run_id}: {len(df)} data points, "
              f"{df['metric'].nunique()} metrics, "
              f"max_step={df['step'].max():,}")

    if not all_dfs:
        print("No data found.")
        sys.exit(1)
    return pd.concat(all_dfs, ignore_index=True)


# ── Plot ──────────────────────────────────────────────────────────────────────

def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window, min_periods=1, center=True).mean()


def plot_metric(df: pd.DataFrame, tag: str, label: str,
                out_path: Path, window: int = 1):
    sub = df[df["metric"] == tag].copy()
    if sub.empty:
        return

    runs = sub["run"].unique()
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, run_id in enumerate(sorted(runs)):
        color = COLORS[i % len(COLORS)]
        run_label = RUN_LABELS.get(run_id, run_id)
        rdf = sub[sub["run"] == run_id].sort_values("step")
        y = smooth(rdf["value"], window)
        ax.plot(rdf["step"], y, label=run_label, color=color, linewidth=1.5)
        if window > 1:
            ax.plot(rdf["step"], rdf["value"], color=color, alpha=0.2, linewidth=0.5)

    if tag == "eval/pooling_savings_mean":
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, label="greedy baseline (0)")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = tag.replace("/", "_") + ".png"
    fig.savefig(out_path / fname, dpi=150)
    plt.close(fig)
    print(f"  saved: {fname}")


def plot_dashboard(df: pd.DataFrame, out_path: Path, window: int = 1):
    """Single-figure dashboard with all key metrics."""
    tags = [t for t in METRICS if not df[df["metric"] == t].empty]
    n = len(tags)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    axes = axes.flatten()

    runs = sorted(df["run"].unique())
    color_map = {r: COLORS[i % len(COLORS)] for i, r in enumerate(runs)}

    for ax_idx, tag in enumerate(tags):
        ax = axes[ax_idx]
        sub = df[df["metric"] == tag]
        for run_id in runs:
            rdf = sub[sub["run"] == run_id].sort_values("step")
            if rdf.empty:
                continue
            y = smooth(rdf["value"], window)
            label = RUN_LABELS.get(run_id, run_id)
            ax.plot(rdf["step"], y, label=label,
                    color=color_map[run_id], linewidth=1.2)
        if tag == "eval/pooling_savings_mean":
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(METRICS[tag], fontsize=9)
        ax.set_xlabel("Steps", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(runs), 4), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    # Hide unused axes
    for i in range(len(tags), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Training Run Comparison — All Metrics", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: dashboard.png")


# ── Summary stats ─────────────────────────────────────────────────────────────

def run_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id in sorted(df["run"].unique()):
        rdf = df[df["run"] == run_id]
        row = {"run": run_id}

        # Pooling savings: best and final
        sv = rdf[rdf["metric"] == "eval/pooling_savings_mean"].sort_values("step")
        if not sv.empty:
            row["savings_best"]  = sv["value"].max()
            row["savings_final"] = sv["value"].iloc[-1]
            row["savings_mean"]  = sv["value"].mean()

        # Entropy collapse
        ec = rdf[rdf["metric"] == "train/ent_coef"].sort_values("step")
        if not ec.empty:
            row["ent_coef_final"] = ec["value"].iloc[-1]
            row["ent_coef_min"]   = ec["value"].min()

        # Throughput
        fps = rdf[rdf["metric"] == "time/fps"]
        if not fps.empty:
            row["fps_median"] = fps["value"].median()

        # Max steps reached
        row["max_steps"] = int(rdf["step"].max())

        rows.append(row)

    return pd.DataFrame(rows).set_index("run")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="output/logs")
    ap.add_argument("--out-dir", default="output/analysis")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Filter to specific run IDs (default: all)")
    ap.add_argument("--smooth", type=int, default=5,
                    help="Rolling average window for plots (default: 5)")
    args = ap.parse_args()

    log_root = _REPO / args.log_dir
    out_path = _REPO / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from {log_root} ...")
    df = load_all_runs(log_root, run_filter=args.runs)

    print(f"\nLoaded {len(df):,} data points across {df['run'].nunique()} runs.")

    # Save tidy CSV
    csv_path = out_path / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved tidy data → {csv_path}")

    # Summary table
    summary = run_summary(df)
    summary_path = out_path / "run_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nRun summary:\n{summary.to_string()}\n")
    print(f"Saved → {summary_path}")

    # Individual metric plots
    print("\nGenerating plots ...")
    for tag, label in METRICS.items():
        plot_metric(df, tag, label, out_path, window=args.smooth)

    # Dashboard
    plot_dashboard(df, out_path, window=args.smooth)

    print(f"\nAll outputs in: {out_path}/")


if __name__ == "__main__":
    main()
