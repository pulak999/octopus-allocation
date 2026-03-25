from __future__ import annotations

# Allow running this file directly without installing the package.
# (When run as `python tests/visualising_data.py`, Python doesn't automatically add the repo root to sys.path.)
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from octopus.data import load_trace

all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz = load_trace("AMS20PrdApp19-tround.sqlite")

# Look at one VM
vm = next(iter(all_vms.values()))
print("vm_id:", vm.vm_id)
print("node_id:", vm.node_id)
print("memory:", vm.memory)
print("rss:", vm.rss)
print("rss len:", len(vm.rss))
print("max_mem_ts len:", len(vm.max_mem_ts))
print("avg_mem_ts len:", len(vm.avg_mem_ts))
print("max_mem_ts[:5]:", vm.max_mem_ts[:5])
print("start_time:", vm.start_time)
print("end_time:", vm.end_time)

# Look at machine_sz
print("\nmachine_sz sample:", list(machine_sz.items())[:3])
print("vm_type_sz sample:", list(vm_type_sz.items())[:3])

# node_to_vms structure
first_node = next(iter(node_to_vms))
print("\nfirst_node:", first_node)
print("VMs on that node:", len(node_to_vms[first_node]))

"""
explore_trace.py — Octopus trace diagnostic + plotting script.

Run from the octopus-allocation repo root:
    python3 explore_trace.py

Produces:
  1. Console: HOTFIX survival rate at baseline and under scaling
  2. Figure 1: rss[1] distribution (log scale)
  3. Figure 2: VM lifetime distribution
  4. Figure 3: Per-node VM count and total memory load
  5. Figure 4: Arrival rate over time (how many VMs start per tick)
  6. Figure 5: HOTFIX survival rate vs. rss scale factor
  7. Figure 6: rss[1] vs other available signals (rss[0/2/3], memory field)
"""

import datetime
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---- adjust this path if needed ----------------------------------------
TRACE = "AMS20PrdApp19-tround.sqlite"
TRACE_DIR = "data/traces"
# ------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octopus.data import load_trace, VM

print(f"Loading {TRACE} ...")
all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz = load_trace(
    TRACE, trace_dir=TRACE_DIR
)
vms = list(all_vms.values())
print(f"  Total VMs: {len(vms)}")
print(f"  Total nodes: {len(node_to_vms)}")
print(f"  Machine types: {len(machine_sz)}")


# =========================================================================
# Helper: simulate HOTFIX filter for a given rss[1] scale factor
# Returns (n_surviving, n_total, events_count)
# =========================================================================
def hotfix_survival(vms_dict, node_to_vms_dict, node_to_machine_dict,
                    machine_sz_dict, scale=1.0, mem_idx=1):
    """Run the _build_events HOTFIX logic on ALL nodes (not a pod).
    Returns fraction of VMs that survive the physical-memory filter."""

    # Find global time range
    n_start = datetime.datetime.max
    n_end = datetime.datetime.min
    for vm in vms_dict.values():
        if vm.start_time < n_start:
            n_start = vm.start_time
        if vm.end_time > n_end:
            n_end = vm.end_time

    base_time = datetime.datetime(
        n_start.year, n_start.month, n_start.day,
        n_start.hour, n_start.minute
    )

    def to_tick(t):
        return int((t - base_time).total_seconds() // 300)

    pod_start_ts = to_tick(n_start)
    pod_end_ts = to_tick(n_end)
    pod_dur = pod_end_ts - pod_start_ts + 1

    vmkey_to_skip = set()

    for node, vmkeys in node_to_vms_dict.items():
        node_alloc = [[] for _ in range(pod_dur)]
        node_dealloc = np.zeros(pod_dur, dtype=np.float64)
        node_cap = float(np.asarray(
            machine_sz_dict[node_to_machine_dict[node]], dtype=float
        )[mem_idx])

        for vmkey in vmkeys:
            vm = vms_dict[vmkey]
            vm_s = to_tick(vm.start_time) - pod_start_ts
            vm_e = to_tick(vm.end_time) - pod_start_ts
            if vm_e < 0:
                vmkey_to_skip.add(vmkey)
                continue
            mem = float(np.asarray(vm.rss, dtype=float)[mem_idx]) * scale
            node_alloc[vm_s].append((vmkey, vm_e + 1, mem))
            if vm_e + 1 < pod_dur:
                node_dealloc[vm_e + 1] += mem

        cur = 0.0
        for ts in range(pod_dur):
            cur -= node_dealloc[ts]
            if cur < 0:
                cur = 0.0
            for vmkey, end_ts, mem in node_alloc[ts]:
                cur += mem
                if cur > node_cap:
                    cur -= mem
                    vmkey_to_skip.add(vmkey)
                    if end_ts < pod_dur:
                        node_dealloc[end_ts] -= mem

    n_total = len(vms_dict)
    n_skip = len(vmkey_to_skip)
    return n_total - n_skip, n_total


# =========================================================================
# Console: baseline HOTFIX survival
# =========================================================================
surviving, total = hotfix_survival(all_vms, node_to_vms, node_to_machine, machine_sz, scale=1.0)
print(f"\nHOTFIX survival (scale=1.0): {surviving}/{total} = {100*surviving/total:.1f}%")
for s in [0.5, 0.8, 1.2, 1.5, 2.0]:
    surv, _ = hotfix_survival(all_vms, node_to_vms, node_to_machine, machine_sz, scale=s)
    print(f"  scale={s:.1f}: {surv}/{total} = {100*surv/total:.1f}%")


# =========================================================================
# Extract raw arrays from all VMs
# =========================================================================
rss0 = np.array([vm.rss[0] for vm in vms])
rss1 = np.array([vm.rss[1] for vm in vms])   # THE signal env uses
rss2 = np.array([vm.rss[2] for vm in vms])
rss3 = np.array([vm.rss[3] for vm in vms])
memory_field = np.array([vm.memory for vm in vms])

# Lifetimes in hours
lifetimes_hr = np.array([
    (vm.end_time - vm.start_time).total_seconds() / 3600.0
    for vm in vms
])

# Per-node total load (sum of rss[1] across all VMs on that node)
node_load = {}
for node, vmkeys in node_to_vms.items():
    node_load[node] = sum(all_vms[k].rss[1] for k in vmkeys)

# Arrival times (binned to hourly)
arrivals_by_hour = collections.Counter()
for vm in vms:
    h = vm.start_time.replace(minute=0, second=0, microsecond=0)
    arrivals_by_hour[h] += 1
sorted_hours = sorted(arrivals_by_hour.keys())
arrival_counts = [arrivals_by_hour[h] for h in sorted_hours]


# =========================================================================
# Figure 1: rss[1] distribution
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Figure 1: rss[1] (env memory signal) distribution", fontsize=13)

ax = axes[0]
ax.hist(rss1, bins=80, color="steelblue", edgecolor="none")
ax.set_xlabel("rss[1] (MB)")
ax.set_ylabel("VM count")
ax.set_title("Linear scale")

ax = axes[1]
ax.hist(rss1[rss1 > 0], bins=80, color="steelblue", edgecolor="none")
ax.set_xscale("log")
ax.set_xlabel("rss[1] (MB, log scale)")
ax.set_ylabel("VM count")
ax.set_title("Log scale")

plt.tight_layout()
plt.savefig("fig1_rss1_distribution.png", dpi=150)
print("\nSaved fig1_rss1_distribution.png")


# =========================================================================
# Figure 2: VM lifetime distribution
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Figure 2: VM lifetime distribution", fontsize=13)

ax = axes[0]
ax.hist(lifetimes_hr, bins=80, color="darkorange", edgecolor="none")
ax.set_xlabel("Lifetime (hours)")
ax.set_ylabel("VM count")
ax.set_title("Linear scale")

ax = axes[1]
lt_pos = lifetimes_hr[lifetimes_hr > 0]
ax.hist(lt_pos, bins=80, color="darkorange", edgecolor="none")
ax.set_xscale("log")
ax.set_xlabel("Lifetime (hours, log scale)")
ax.set_ylabel("VM count")
ax.set_title("Log scale")

plt.tight_layout()
plt.savefig("fig2_lifetimes.png", dpi=150)
print("Saved fig2_lifetimes.png")


# =========================================================================
# Figure 3: Per-node VM count and total memory load
# =========================================================================
counts_per_node = [len(vmkeys) for vmkeys in node_to_vms.values()]
loads_per_node = list(node_load.values())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Figure 3: Per-node statistics", fontsize=13)

ax = axes[0]
ax.hist(counts_per_node, bins=40, color="mediumseagreen", edgecolor="none")
ax.set_xlabel("VM count per node")
ax.set_ylabel("Node count")
ax.set_title("VMs per node")

ax = axes[1]
ax.hist(loads_per_node, bins=40, color="mediumseagreen", edgecolor="none")
ax.set_xlabel("Total rss[1] load per node (MB)")
ax.set_ylabel("Node count")
ax.set_title("Aggregate memory load per node")

plt.tight_layout()
plt.savefig("fig3_per_node.png", dpi=150)
print("Saved fig3_per_node.png")


# =========================================================================
# Figure 4: VM arrival rate over time
# =========================================================================
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(sorted_hours, arrival_counts, color="mediumpurple", linewidth=0.8)
ax.set_xlabel("Time")
ax.set_ylabel("VM arrivals per hour")
ax.set_title("Figure 4: VM arrival rate over trace window")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("fig4_arrival_rate.png", dpi=150)
print("Saved fig4_arrival_rate.png")


# =========================================================================
# Figure 5: HOTFIX survival rate vs. rss scale factor
# =========================================================================
scales = np.linspace(0.3, 3.0, 25)
survival_rates = []
print("\nComputing HOTFIX survival curve (this may take ~30s) ...")
for s in scales:
    surv, tot = hotfix_survival(all_vms, node_to_vms, node_to_machine, machine_sz, scale=s)
    survival_rates.append(100.0 * surv / tot)
    print(f"  scale={s:.2f}  survival={survival_rates[-1]:.1f}%")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(scales, survival_rates, marker="o", markersize=4, color="crimson")
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="baseline (scale=1)")
ax.set_xlabel("rss[1] scale factor")
ax.set_ylabel("VM survival rate (%)")
ax.set_title("Figure 5: HOTFIX VM survival rate vs. rss[1] scale factor")
ax.legend()
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
plt.savefig("fig5_hotfix_survival.png", dpi=150)
print("Saved fig5_hotfix_survival.png")


# =========================================================================
# Figure 6: rss fields compared — are the other indices useful?
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Figure 6: All rss fields + memory field compared", fontsize=13)

for ax, data, label, color in zip(
    axes.flat,
    [rss0, rss1, rss2, rss3],
    ["rss[0]", "rss[1]  ← env uses this", "rss[2]", "rss[3]"],
    ["steelblue", "steelblue", "steelblue", "steelblue"],
):
    pos = data[data > 0]
    if len(pos) == 0:
        ax.text(0.5, 0.5, "all zeros", transform=ax.transAxes, ha="center")
    else:
        ax.hist(pos, bins=60, color=color, edgecolor="none")
        ax.set_xscale("log")
        p50 = np.percentile(pos, 50)
        p95 = np.percentile(pos, 95)
        ax.axvline(p50, color="red", linestyle="--", linewidth=1, label=f"p50={p50:.0f}")
        ax.axvline(p95, color="orange", linestyle="--", linewidth=1, label=f"p95={p95:.0f}")
        ax.legend(fontsize=8)
    ax.set_title(label)
    ax.set_xlabel("Value (log scale)")
    ax.set_ylabel("VM count")

plt.tight_layout()
plt.savefig("fig6_rss_fields.png", dpi=150)
print("Saved fig6_rss_fields.png")

print("\nDone. Check the 6 figures in your working directory.")
print(f"\nSummary stats for rss[1]:")
print(f"  min={rss1.min():.0f}  p5={np.percentile(rss1,5):.0f}  "
      f"p50={np.percentile(rss1,50):.0f}  p95={np.percentile(rss1,95):.0f}  "
      f"max={rss1.max():.0f}  (MB)")
print(f"Summary stats for lifetime (hours):")
print(f"  min={lifetimes_hr.min():.2f}  p5={np.percentile(lifetimes_hr,5):.1f}  "
      f"p50={np.percentile(lifetimes_hr,50):.1f}  p95={np.percentile(lifetimes_hr,95):.1f}  "
      f"max={lifetimes_hr.max():.1f}")