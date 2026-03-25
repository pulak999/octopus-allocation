"""
explore_episode_density.py — How many events does a typical pod episode produce?

Run from repo root:
    python3 explore_episode_density.py

For each trace, samples N_PODS random pod assignments and reports:
  - Distribution of event counts per episode (baseline scale=1.0)
  - How event count drops as scale increases (to calibrate MIN_EVENTS guard)

Produces:
  Console: per-trace summary stats
  Fig E:   Event count distribution per trace (small multiples)
  Fig F:   Median event count vs. scale factor (to pick MIN_EVENTS threshold)
"""

from __future__ import annotations

# Allow running this file directly without installing the package.
# (When run as `python3 diagnose_episodes.py`, Python doesn't automatically
# add `octopus-allocation/` to sys.path.)
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # octopus-allocation/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
import sys
import glob
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octopus.data import load_trace

TRACE_DIR = REPO_ROOT / "data" / "traces"
POD_SIZE   = 16          # matches current env default
N_PODS     = 50          # pod samples per trace
MEM_IDX    = 1           # rss[1]
RANDOM_SEED = 42

# Bad VM filters (from our earlier analysis)
BAD_END_BEFORE   = datetime.datetime(2021, 1, 1)   # catches default end_time=2020
BAD_START_AFTER  = datetime.datetime(2024, 1, 1)   # catches default start_time=2024-01-01

# -------------------------------------------------------------------------
def is_bad_vm(vm):
    """True if VM has a known data-quality issue (should be excluded)."""
    if vm.end_time < BAD_END_BEFORE:          # default end_time
        return True
    if vm.start_time >= BAD_START_AFTER:      # default start_time
        return True
    if vm.end_time <= vm.start_time:          # zero or negative lifetime
        return True
    return False


# -------------------------------------------------------------------------
def simulate_episode(all_vms, node_to_vms, node_to_machine, machine_sz,
                     pod_nodes, scale=1.0):
    """
    Mimics _build_events from env.py for a given pod assignment.
    Returns number of events that survive HOTFIX.
    """
    # Time range across pod
    n_start = datetime.datetime.max
    n_end   = datetime.datetime.min
    any_vm  = False
    for node in pod_nodes:
        for vmkey in node_to_vms.get(node, []):
            vm = all_vms[vmkey]
            if is_bad_vm(vm):
                continue
            any_vm = True
            if vm.start_time < n_start:
                n_start = vm.start_time
            if vm.end_time > n_end:
                n_end = vm.end_time

    if not any_vm:
        return 0

    base_time = datetime.datetime(
        n_start.year, n_start.month, n_start.day,
        n_start.hour, n_start.minute
    )

    def to_tick(t):
        return int((t - base_time).total_seconds() // 300)

    pod_start_ts = to_tick(n_start)
    pod_end_ts   = to_tick(n_end)
    pod_dur      = pod_end_ts - pod_start_ts + 1
    if pod_dur <= 0:
        return 0

    vmkey_to_skip = set()

    for node in pod_nodes:
        node_alloc   = [[] for _ in range(pod_dur)]
        node_dealloc = np.zeros(pod_dur, dtype=np.float64)
        node_cap     = float(
            np.asarray(machine_sz[node_to_machine[node]], dtype=float)[MEM_IDX]
        )

        for vmkey in node_to_vms.get(node, []):
            vm = all_vms[vmkey]
            if is_bad_vm(vm):
                vmkey_to_skip.add(vmkey)
                continue
            vm_s = to_tick(vm.start_time) - pod_start_ts
            vm_e = to_tick(vm.end_time)   - pod_start_ts
            if vm_e < 0:
                vmkey_to_skip.add(vmkey)
                continue
            mem = float(np.asarray(vm.rss, dtype=float)[MEM_IDX]) * scale
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

    # Count surviving events
    n_events = 0
    for node in pod_nodes:
        for vmkey in node_to_vms.get(node, []):
            if vmkey in vmkey_to_skip:
                continue
            vm = all_vms[vmkey]
            mem = float(np.asarray(vm.rss, dtype=float)[MEM_IDX]) * scale
            if mem > 0:
                n_events += 1

    return n_events


# =========================================================================
# Load traces + run diagnostics
# =========================================================================
pkl_paths = sorted(glob.glob(str(TRACE_DIR / "*.pkl")))
trace_names = [os.path.basename(p).replace(".pkl", "") for p in pkl_paths]
print(
    f"Found {len(trace_names)} traces in {TRACE_DIR}. "
    f"Sampling {N_PODS} pods each at pod_size={POD_SIZE}.\n"
)

if not trace_names:
    print("No trace pickles found. Expected *.pkl files under the trace directory above.")
    sys.exit(1)

scales_to_test = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

# Results: trace_name -> {scale -> list of event counts}
results = {}

for name in trace_names:
    print(f"Loading {name} ...")
    all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz = load_trace(
        name, trace_dir=str(TRACE_DIR)
    )

    # Only use clean nodes (nodes that have at least one good VM)
    clean_nodes = []
    for node, vmkeys in node_to_vms.items():
        good = [k for k in vmkeys if not is_bad_vm(all_vms[k])]
        if good:
            clean_nodes.append(int(node))

    print(f"  Clean nodes: {len(clean_nodes)} / {len(node_to_vms)}")

    rng = random.Random(RANDOM_SEED)
    results[name] = {}

    for scale in scales_to_test:
        counts = []
        for _ in range(N_PODS):
            pod_nodes = rng.sample(clean_nodes, min(POD_SIZE, len(clean_nodes)))
            n_ev = simulate_episode(
                all_vms, node_to_vms, node_to_machine, machine_sz,
                pod_nodes, scale=scale
            )
            counts.append(n_ev)
            rng = random.Random(rng.randint(0, 2**31))  # advance rng

        results[name][scale] = counts

    # Console summary at baseline
    baseline = results[name][1.0]
    print(f"  Event counts at scale=1.0:")
    print(f"    min={min(baseline)}  p10={int(np.percentile(baseline,10))}  "
          f"p50={int(np.percentile(baseline,50))}  "
          f"p90={int(np.percentile(baseline,90))}  max={max(baseline)}")
    print(f"    Episodes with <10 events: {sum(1 for c in baseline if c < 10)}/{N_PODS}")
    print(f"    Episodes with <50 events: {sum(1 for c in baseline if c < 50)}/{N_PODS}")
    print()


# =========================================================================
# Fig E: Event count distribution per trace at baseline (scale=1.0)
# =========================================================================
n_traces = len(trace_names)
ncols = min(4, n_traces)
nrows = (n_traces + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
fig.suptitle(f"Fig E: Episode event count distribution per trace\n"
             f"(scale=1.0, pod_size={POD_SIZE}, {N_PODS} samples)", fontsize=12)
axes_flat = np.array(axes).flatten()

for idx, name in enumerate(trace_names):
    ax = axes_flat[idx]
    counts = results[name][1.0]
    p50 = int(np.percentile(counts, 50))
    p10 = int(np.percentile(counts, 10))
    ax.hist(counts, bins=20, color="steelblue", edgecolor="none")
    ax.axvline(p50, color="red",    linestyle="--", linewidth=1.2, label=f"p50={p50}")
    ax.axvline(p10, color="orange", linestyle="--", linewidth=1.2, label=f"p10={p10}")
    ax.legend(fontsize=7)
    ax.set_title(name[:28], fontsize=8)
    ax.set_xlabel("Events per episode", fontsize=7)
    ax.set_ylabel("Pod count", fontsize=7)
    ax.tick_params(labelsize=6)

for ax in axes_flat[n_traces:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig("figE_event_counts.png", dpi=150)
print("Saved figE_event_counts.png")


# =========================================================================
# Fig F: Median event count vs scale factor (one line per trace)
# Use this to pick MIN_EVENTS threshold
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title(f"Fig F: Median episode event count vs. rss[1] scale factor\n"
             f"(pod_size={POD_SIZE}, {N_PODS} samples per trace)", fontsize=12)

cmap = plt.cm.tab10
for idx, name in enumerate(trace_names):
    medians = [np.median(results[name][s]) for s in scales_to_test]
    ax.plot(scales_to_test, medians, marker="o", markersize=4,
            label=name[:20], color=cmap(idx / len(trace_names)))

# Draw a candidate MIN_EVENTS threshold line
candidate_min = 20
ax.axhline(candidate_min, color="black", linestyle=":", linewidth=1.2,
           label=f"candidate MIN_EVENTS={candidate_min}")

ax.set_xlabel("rss[1] scale factor")
ax.set_ylabel("Median event count")
ax.legend(fontsize=7, loc="upper right")
ax.set_xticks(scales_to_test)
plt.tight_layout()
plt.savefig("figF_events_vs_scale.png", dpi=150)
print("Saved figF_events_vs_scale.png")

print("\nDone.")