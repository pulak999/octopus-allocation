"""
explore_lifetimes_arrivals.py — Deep-dive on fig 2 (negative lifetimes)
and fig 4 (arrival rate spike/ramp) across ALL trace pickles.

Run from repo root:
    python3 explore_lifetimes_arrivals.py

Produces:
  Console:  Per-trace breakdown of negative lifetimes + HOTFIX catch rate
  Fig A:    Lifetime distributions per trace (small multiples)
  Fig B:    Arrival rate per trace — raw counts, NOT cumulative
  Fig C:    Zoom on trace-start spike: VMs with start_time == trace_min_start
  Fig D:    Negative-lifetime VMs: are they long-lived or a specific SKU?
"""

import datetime
import os
import glob
import collections
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# When run from the tests directory, we still want to look up traces relative
# to the repo root (one level above this file's parent).
REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = REPO_ROOT / "data" / "traces"

# -------------------------------------------------------------------------
# Auto-discover all trace pickles
# -------------------------------------------------------------------------
pkl_paths = sorted(glob.glob(str(TRACE_DIR / "*.pkl")))
if not pkl_paths:
    raise FileNotFoundError(f"No .pkl files found in {TRACE_DIR}")

trace_names = [os.path.basename(p).replace(".pkl", "") for p in pkl_paths]
print(f"Found {len(trace_names)} traces:")
for n in trace_names:
    print(f"  {n}")

import sys

# Allow running this script directly (without installing the package) by
# making sure the repo root is on sys.path before importing octopus.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from octopus.data import load_trace


# =========================================================================
# Load all traces + extract per-VM stats
# =========================================================================
# Per-trace data stored here
trace_data = {}  # name -> dict of arrays

for name, path in zip(trace_names, pkl_paths):
    print(f"\nLoading {name} ...")
    all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz = load_trace(
        name, trace_dir=str(TRACE_DIR)
    )
    vms = list(all_vms.values())
    n = len(vms)

    start_times = np.array([vm.start_time for vm in vms], dtype=object)
    end_times   = np.array([vm.end_time   for vm in vms], dtype=object)
    lifetimes_hr = np.array([
        (vm.end_time - vm.start_time).total_seconds() / 3600.0
        for vm in vms
    ])
    rss1 = np.array([vm.rss[1] for vm in vms])

    # Trace time boundaries
    trace_min_start = min(vm.start_time for vm in vms)
    trace_max_end   = max(vm.end_time   for vm in vms)

    # --- Negative lifetime breakdown ---
    neg_mask  = lifetimes_hr < 0
    zero_mask = lifetimes_hr == 0
    pos_mask  = lifetimes_hr > 0

    n_neg  = neg_mask.sum()
    n_zero = zero_mask.sum()
    n_pos  = pos_mask.sum()

    # Among negative: how many have start_time == trace_min_start?
    # (i.e. "already running at trace start, end_time from wrong epoch")
    if n_neg > 0:
        neg_at_trace_start = sum(
            1 for vm in vms
            if (vm.end_time - vm.start_time).total_seconds() < 0
            and vm.start_time == trace_min_start
        )
    else:
        neg_at_trace_start = 0

    # Among negative: what are their rss[1] values?
    rss1_neg = rss1[neg_mask]

    # --- Does HOTFIX actually catch all negative-lifetime VMs? ---
    # HOTFIX catches vm_e < 0, meaning to_tick(end_time) - pod_start_ts < 0.
    # For a VM with end_time < start_time, if the whole trace is the pod window,
    # pod_start_ts = to_tick(trace_min_start), and vm_e = to_tick(end_time) - pod_start_ts.
    # A VM that started at trace_min_start with end_time 4 years earlier will have
    # vm_e hugely negative → caught.
    # But a VM that started AFTER trace_min_start with a slightly negative lifetime
    # might have vm_e >= 0 depending on the pod window → NOT caught.
    # We flag these as "potentially uncaught".
    potentially_uncaught = 0
    for vm in vms:
        lt = (vm.end_time - vm.start_time).total_seconds()
        if lt < 0:
            # If end_time is still after trace_min_start, to_tick(end_time) could be >= 0
            if vm.end_time >= trace_min_start:
                potentially_uncaught += 1

    # --- Arrival rate (true counts, not cumulative) ---
    arrivals_by_hour = collections.Counter()
    for vm in vms:
        h = vm.start_time.replace(minute=0, second=0, microsecond=0)
        arrivals_by_hour[h] += 1

    # Spike: VMs whose start_time == trace_min_start (already-running VMs)
    n_spike = sum(1 for vm in vms if vm.start_time == trace_min_start)
    frac_spike = n_spike / n if n > 0 else 0

    trace_data[name] = {
        "vms": vms,
        "n": n,
        "lifetimes_hr": lifetimes_hr,
        "rss1": rss1,
        "rss1_neg": rss1_neg,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "n_pos": n_pos,
        "neg_at_trace_start": neg_at_trace_start,
        "potentially_uncaught": potentially_uncaught,
        "trace_min_start": trace_min_start,
        "trace_max_end": trace_max_end,
        "arrivals_by_hour": arrivals_by_hour,
        "n_spike": n_spike,
        "frac_spike": frac_spike,
    }

    print(f"  Total VMs: {n}")
    print(f"  Negative lifetimes: {n_neg} ({100*n_neg/n:.1f}%)")
    print(f"    → start_time == trace_min_start: {neg_at_trace_start}")
    print(f"    → potentially NOT caught by HOTFIX: {potentially_uncaught}")
    print(f"  Zero lifetimes: {n_zero} ({100*n_zero/n:.1f}%)")
    print(f"  Positive lifetimes: {n_pos} ({100*n_pos/n:.1f}%)")
    print(f"  Arrival spike (start==trace_start): {n_spike} ({100*frac_spike:.1f}%)")
    print(f"  Trace window: {trace_min_start} → {trace_max_end}")


# =========================================================================
# Fig A: Lifetime distributions per trace (small multiples, log scale)
# Positive lifetimes only — separate panel for negative count
# =========================================================================
n_traces = len(trace_names)
ncols = min(4, n_traces)
nrows = (n_traces + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
fig.suptitle("Fig A: Positive lifetime distributions per trace (log scale)", fontsize=13)
axes_flat = np.array(axes).flatten()

for idx, name in enumerate(trace_names):
    ax = axes_flat[idx]
    d = trace_data[name]
    pos = d["lifetimes_hr"][d["lifetimes_hr"] > 0]
    ax.hist(pos, bins=60, color="darkorange", edgecolor="none")
    ax.set_xscale("log")
    ax.set_title(f"{name[:20]}\n"
                 f"neg={d['n_neg']} ({100*d['n_neg']/d['n']:.0f}%)  "
                 f"spike={d['n_spike']} ({100*d['frac_spike']:.0f}%)",
                 fontsize=8)
    ax.set_xlabel("Lifetime (hr)", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(labelsize=6)

for ax in axes_flat[n_traces:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("figA_lifetimes_per_trace.png", dpi=150)
print("\nSaved figA_lifetimes_per_trace.png")


# =========================================================================
# Fig B: Arrival rate per trace — raw hourly counts
# Plotted WITHOUT the spike hour to show true diurnal structure
# =========================================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
fig.suptitle("Fig B: Hourly VM arrivals per trace (spike hour excluded)", fontsize=13)
axes_flat = np.array(axes).flatten()

for idx, name in enumerate(trace_names):
    ax = axes_flat[idx]
    d = trace_data[name]
    ab = d["arrivals_by_hour"]
    spike_hour = d["trace_min_start"].replace(minute=0, second=0, microsecond=0)

    # Exclude the spike hour
    hours   = sorted(h for h in ab if h != spike_hour)
    counts  = [ab[h] for h in hours]

    ax.plot(hours, counts, linewidth=0.7, color="mediumpurple")
    ax.set_title(name[:25], fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(labelsize=6)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.set_ylabel("Arrivals/hr", fontsize=7)

for ax in axes_flat[n_traces:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("figB_arrival_rate_no_spike.png", dpi=150)
print("Saved figB_arrival_rate_no_spike.png")


# =========================================================================
# Fig C: Zoom on trace-start spike — are already-running VMs clustered by SKU?
# Plot rss[1] distribution of spike VMs vs non-spike VMs
# =========================================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
fig.suptitle("Fig C: rss[1] — spike VMs vs non-spike VMs per trace", fontsize=13)
axes_flat = np.array(axes).flatten()

for idx, name in enumerate(trace_names):
    ax = axes_flat[idx]
    d = trace_data[name]
    spike_start = d["trace_min_start"]

    rss1_spike    = np.array([vm.rss[1] for vm in d["vms"] if vm.start_time == spike_start])
    rss1_nonspike = np.array([vm.rss[1] for vm in d["vms"] if vm.start_time != spike_start])

    bins = np.logspace(
        np.log10(max(512, min(d["rss1"].min(), 512))),
        np.log10(d["rss1"].max() + 1),
        50
    )
    if len(rss1_spike) > 0:
        ax.hist(rss1_spike,    bins=bins, alpha=0.6, label=f"spike (n={len(rss1_spike)})",    color="crimson",   edgecolor="none")
    if len(rss1_nonspike) > 0:
        ax.hist(rss1_nonspike, bins=bins, alpha=0.6, label=f"normal (n={len(rss1_nonspike)})", color="steelblue", edgecolor="none")
    ax.set_xscale("log")
    ax.set_title(name[:25], fontsize=8)
    ax.legend(fontsize=6)
    ax.set_xlabel("rss[1] MB", fontsize=7)
    ax.tick_params(labelsize=6)

for ax in axes_flat[n_traces:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("figC_spike_vs_normal_rss1.png", dpi=150)
print("Saved figC_spike_vs_normal_rss1.png")


# =========================================================================
# Fig D: Negative-lifetime VMs — rss[1] and how far negative (hours)
# =========================================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
fig.suptitle("Fig D: Negative-lifetime VMs — depth and rss[1] per trace", fontsize=13)
axes_flat = np.array(axes).flatten()

for idx, name in enumerate(trace_names):
    ax = axes_flat[idx]
    d = trace_data[name]
    neg_mask = d["lifetimes_hr"] < 0
    n_neg = neg_mask.sum()

    if n_neg == 0:
        ax.text(0.5, 0.5, "No negative lifetimes", transform=ax.transAxes, ha="center", fontsize=9)
        ax.set_title(name[:25], fontsize=8)
        continue

    neg_lt = d["lifetimes_hr"][neg_mask]  # these are negative hours

    # Scatter: how negative (abs hours) vs rss[1]
    ax.scatter(np.abs(neg_lt), d["rss1"][neg_mask], alpha=0.3, s=8, color="crimson")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|lifetime| hours", fontsize=7)
    ax.set_ylabel("rss[1] MB", fontsize=7)
    ax.set_title(f"{name[:20]}\nn_neg={n_neg}", fontsize=8)
    ax.tick_params(labelsize=6)

for ax in axes_flat[n_traces:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("figD_negative_lifetimes_detail.png", dpi=150)
print("Saved figD_negative_lifetimes_detail.png")

print("\nDone. Figures: figA, figB, figC, figD")