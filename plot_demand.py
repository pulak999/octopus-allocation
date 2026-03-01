#!/usr/bin/env python3
"""
plot_demand.py  –  VM memory demand time-series for every datacenter trace.

Outputs (saved to doc/v1/figs/):
  demand_overview.pdf   –  2×5 grid: one panel per datacenter, showing
                            per-server demand (thin grey) + cluster total (blue)
                            + cluster peak (red dashed)
  demand_diurnal.pdf    –  mean ± 1-std of hourly-averaged demand for each DC
  peak_to_mean.pdf      –  bar chart: ratio of cluster-peak to cluster-mean
                            across all ten DCs (motivates pooling)

Usage:
  python3 plot_demand.py          # run from the octopus-allocation/ directory
"""

import os
import pickle
import datetime
import numpy as np
import matplotlib

# ── VM class must be defined before unpickling ────────────────────────────────
class VM:
    def __init__(self):
        self.vm_id         = -1
        self.node_id       = -1
        self.cores         = -1
        self.memory        = -1
        self.nic           = -1
        self.rss           = list()
        self.ssd           = -1
        self.max_mem_ts    = list()
        self.avg_mem_ts    = list()
        self.max_cpu_ts    = list()
        self.avg_cpu_ts    = list()
        self.start_time    = datetime.datetime(2024, 1, 1, 1, 1)
        self.end_time      = datetime.datetime(2020, 1, 1, 1, 1)
        self.instance_role = ""
        self.subscription_id = ""
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

# ── Config ───────────────────────────────────────────────────────────────────
TRACE_DIR = "traces"
FIG_DIR   = "doc/v1/figs"
MEM_IDX   = 1          # index into VM.rss / machine_sz for memory (GB)
TICK_MIN  = 5          # trace resolution in minutes

CLUSTER_FILES = [
    "AMS20PrdApp19-tround.sqlite.pkl",
    "LON23PrdApp01-troundgrt5m.sqlite.pkl",
    "BLAPrdApp19-troundgrt5m.sqlite.pkl",
    "LVL01PrdApp05-troundgrt5m.sqlite.pkl",
    "BN9PrdApp18-troundgrt5m.sqlite.pkl",
    "SG2PrdApp35-troundgrt5m.sqlite.pkl",
    "DSM08PrdApp05-troundgrt5m.sqlite.pkl",
    "SYD21PrdApp07-troundgrt5m.sqlite.pkl",
    "YTO21PrdApp05-troundgrt5m.sqlite.pkl",
    "DUB24PrdApp09-troundgrt5m.sqlite.pkl",
]

# Short labels for axis titles
DC_LABELS = {
    "AMS20": "Amsterdam (AMS)",
    "LON23": "London (LON)",
    "BLA":   "Bangalore (BLA)",
    "LVL01": "Louisville (LVL)",
    "BN9":   "BN9",
    "SG2":   "Singapore (SG2)",
    "DSM08": "Des Moines (DSM)",
    "SYD21": "Sydney (SYD)",
    "YTO21": "Toronto (YTO)",
    "DUB24": "Dublin (DUB)",
}

def label_of(fname):
    key = fname.split("PrdApp")[0]
    return DC_LABELS.get(key, key)

os.makedirs(FIG_DIR, exist_ok=True)

# ── Data loading ─────────────────────────────────────────────────────────────

def load_trace(fname):
    path = os.path.join(TRACE_DIR, fname)
    with open(path, "rb") as f:
        all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz = pickle.load(f)
    return all_vms, node_to_vms, node_to_machine, machine_sz


def compute_demand(all_vms, node_to_vms, node_to_machine, machine_sz):
    """
    Returns
    -------
    timestamps  : list[datetime]  – one per tick
    per_node    : dict[node_id -> np.ndarray]  – GB demanded at each tick
    cluster_total : np.ndarray   – sum over all nodes
    node_peak   : dict[node_id -> float]       – per-node peak
    node_capacity : dict[node_id -> float]     – physical memory (GB)
    """
    # find global time range
    t_min = datetime.datetime.max
    t_max = datetime.datetime.min
    for vm in all_vms.values():
        if vm.start_time < t_min:
            t_min = vm.start_time
        if vm.end_time > t_max:
            t_max = vm.end_time

    # align to 5-minute boundary
    base = t_min.replace(second=0, microsecond=0)
    base -= datetime.timedelta(minutes=base.minute % TICK_MIN)
    delta = datetime.timedelta(minutes=TICK_MIN)

    def to_tick(t):
        return int((t - base).total_seconds() // (TICK_MIN * 60))

    n_ticks = to_tick(t_max) + 2

    per_node   = {}
    node_capacity = {}

    for node, vmkeys in node_to_vms.items():
        arr = np.zeros(n_ticks, dtype=np.float32)
        machine = node_to_machine[node]
        cap = float(np.asarray(machine_sz[machine], dtype=float)[MEM_IDX])
        node_capacity[node] = cap

        for vk in vmkeys:
            vm  = all_vms[vk]
            mem = float(np.asarray(vm.rss, dtype=float)[MEM_IDX])
            if mem <= 0:
                continue
            t0 = max(0, to_tick(vm.start_time))
            t1 = min(n_ticks, to_tick(vm.end_time) + 1)
            if t1 > t0:
                arr[t0:t1] += mem

        per_node[node] = arr

    if not per_node:
        return [], {}, np.array([]), {}, {}

    timestamps    = [base + i * delta for i in range(n_ticks)]
    cluster_total = sum(per_node.values())
    node_peak     = {n: float(a.max()) for n, a in per_node.items()}

    return timestamps, per_node, cluster_total, node_peak, node_capacity


# ── Load all traces ───────────────────────────────────────────────────────────
print("Loading traces …")
traces = {}
for fname in CLUSTER_FILES:
    lbl = label_of(fname)
    print(f"  {lbl}")
    all_vms, node_to_vms, node_to_machine, machine_sz = load_trace(fname)
    ts, per_node, total, node_peak, node_cap = compute_demand(
        all_vms, node_to_vms, node_to_machine, machine_sz
    )
    traces[lbl] = dict(
        ts=ts, per_node=per_node, total=total,
        node_peak=node_peak, node_cap=node_cap,
    )
print("Done loading.\n")


# ── Plot 1: demand_overview  (2×5 grid) ───────────────────────────────────────
print("Plotting demand_overview …")

fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=False)
axes_flat = axes.flatten()

for ax, (lbl, d) in zip(axes_flat, traces.items()):
    ts    = d["ts"]
    total = d["total"]
    pn    = d["per_node"]

    if len(ts) == 0:
        ax.set_title(lbl, fontsize=9)
        continue

    t_arr = np.array(ts)
    # Downsample display to at most 2000 points for clarity
    step = max(1, len(ts) // 2000)
    t_ds = t_arr[::step]

    # Per-server thin grey lines (sample up to 30 servers for legibility)
    sampled_nodes = list(pn.keys())[:30]
    for node in sampled_nodes:
        arr = pn[node]
        # normalise by node capacity so all servers are on the same scale
        cap = d["node_cap"].get(node, 1.0) or 1.0
        ax.plot(t_ds, (arr[::step] / cap) * 100,
                color="silver", lw=0.4, alpha=0.6, zorder=1)

    # Cluster total demand normalised by total physical memory
    total_cap = sum(d["node_cap"].values()) or 1.0
    norm_total = (total[::step] / total_cap) * 100
    ax.plot(t_ds, norm_total, color="#1f77b4", lw=1.4, label="Cluster total", zorder=3)

    # Peak line
    peak_pct = float(total.max() / total_cap) * 100
    ax.axhline(peak_pct, color="#d62728", lw=1.0, ls="--", label=f"Peak {peak_pct:.1f}%", zorder=4)

    # Mean line
    mean_pct = float(total[total > 0].mean() / total_cap) * 100 if total.max() > 0 else 0
    ax.axhline(mean_pct, color="#2ca02c", lw=1.0, ls=":", label=f"Mean {mean_pct:.1f}%", zorder=4)

    ax.set_title(lbl, fontsize=9, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Memory demand (% of total)", fontsize=7)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.6)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

fig.suptitle(
    "Per-server (grey) and cluster-aggregate (blue) memory demand — 10 Azure datacenters\n"
    "Red dashed = cluster peak; green dotted = cluster mean; y-axis = % of total physical DRAM",
    fontsize=10, y=1.01
)
fig.tight_layout()
out = os.path.join(FIG_DIR, "demand_overview.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"  → {out}")
plt.close(fig)


# ── Plot 2: diurnal pattern ───────────────────────────────────────────────────
print("Plotting diurnal …")

fig, axes = plt.subplots(2, 5, figsize=(22, 7), sharey=False)
axes_flat = axes.flatten()

TICKS_PER_DAY = (24 * 60) // TICK_MIN   # 288

for ax, (lbl, d) in zip(axes_flat, traces.items()):
    total = d["total"]
    total_cap = sum(d["node_cap"].values()) or 1.0
    norm = (total / total_cap) * 100

    n_full_days = len(norm) // TICKS_PER_DAY
    if n_full_days < 2:
        ax.set_title(lbl, fontsize=9)
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        continue

    daily = norm[:n_full_days * TICKS_PER_DAY].reshape(n_full_days, TICKS_PER_DAY)
    hours = np.arange(TICKS_PER_DAY) * TICK_MIN / 60

    mean_d = daily.mean(axis=0)
    std_d  = daily.std(axis=0)

    ax.fill_between(hours, mean_d - std_d, mean_d + std_d,
                    alpha=0.25, color="#1f77b4", label="±1 std")
    ax.plot(hours, mean_d, color="#1f77b4", lw=1.5, label="Mean")
    for day_i in range(n_full_days):
        ax.plot(hours, daily[day_i], color="silver", lw=0.4, alpha=0.5)

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("Hour of day", fontsize=7)
    ax.set_ylabel("Memory demand (%)", fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.set_title(lbl, fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="lower right", framealpha=0.6)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

fig.suptitle(
    "Diurnal memory demand patterns — mean (blue) ± 1 std, individual days in grey\n"
    "y-axis = % of total physical DRAM; x-axis = hour of day (UTC)",
    fontsize=10, y=1.01
)
fig.tight_layout()
out = os.path.join(FIG_DIR, "demand_diurnal.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"  → {out}")
plt.close(fig)


# ── Plot 3: peak-to-mean bar chart ────────────────────────────────────────────
print("Plotting peak-to-mean …")

labels, ratios, peaks, means = [], [], [], []

for lbl, d in traces.items():
    total     = d["total"]
    total_cap = sum(d["node_cap"].values()) or 1.0
    norm      = total / total_cap * 100
    if norm.max() == 0:
        continue
    peak = float(norm.max())
    mean = float(norm[norm > 0].mean())
    labels.append(lbl.split("(")[-1].rstrip(")") if "(" in lbl else lbl)
    ratios.append(peak / mean if mean > 0 else 0)
    peaks.append(peak)
    means.append(mean)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- left: peak-to-mean ratio ---
colors = ["#d62728" if r > 1.4 else "#1f77b4" for r in ratios]
bars = ax1.bar(labels, ratios, color=colors, edgecolor="white", linewidth=0.8)
ax1.axhline(1.0, color="black", lw=0.8, ls="--")
ax1.set_ylabel("Peak / Mean demand ratio", fontsize=10)
ax1.set_title("Peak-to-mean ratio per datacenter\n(red = ratio > 1.4)", fontsize=10)
ax1.set_ylim(0, max(ratios) * 1.2)
for bar, r in zip(bars, ratios):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{r:.2f}", ha="center", va="bottom", fontsize=8)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8)
ax1.grid(axis="y", lw=0.4, alpha=0.5)

# --- right: peak vs mean absolute (% of total DRAM) ---
x = np.arange(len(labels))
w = 0.35
ax2.bar(x - w/2, peaks, w, label="Peak demand (%)", color="#d62728", alpha=0.85, edgecolor="white")
ax2.bar(x + w/2, means, w, label="Mean demand (%)", color="#1f77b4", alpha=0.85, edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
ax2.set_ylabel("Memory demand (% of total DRAM)", fontsize=10)
ax2.set_title("Peak vs. mean cluster memory demand\n(y-axis = % of total physical DRAM)", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis="y", lw=0.4, alpha=0.5)

fig.tight_layout()
out = os.path.join(FIG_DIR, "peak_to_mean.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"  → {out}")
plt.close(fig)

print("\nAll figures saved to", FIG_DIR)
