#!/usr/bin/env python3
"""
Single-panel savings plot: old reward (collapse) vs new reward (stable).
Plain language labels — no SAC jargon.
Output: run6_vs_run7.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REPO = HERE.parent.parent.parent.parent

OLD_LOG = REPO / "output/logs/split721_sac/SAC_1"
NEW_LOG = REPO / "output/logs/aug_v4_rewardA_v2/SAC_1"

C_OLD  = "#C0392B"
C_NEW  = "#2E75B6"
C_ZERO = "#888888"

def load_tag(log_path, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step  for e in events], dtype=float) / 1e6
    vals  = np.array([e.value for e in events], dtype=float)
    return steps, vals

old_s, old_v = load_tag(OLD_LOG, "eval/pooling_savings_mean")
new_s, new_v = load_tag(NEW_LOG, "eval/pooling_savings_mean")

old_v *= 100
new_v *= 100

fig, ax = plt.subplots(figsize=(7.5, 3.6))

ax.axhline(0, color=C_ZERO, lw=1.2, ls="--", zorder=0, label="break-even (0 %)")

ax.plot(old_s, old_v, color=C_OLD, lw=2.2,
        label="Old reward — policy collapsed")
ax.plot(new_s, new_v, color=C_NEW, lw=2.2,
        label="New reward — stable, near break-even")

# shade the collapse region
ax.fill_between(old_s, old_v, 0,
                where=(old_v < 0), color=C_OLD, alpha=0.12)

# annotate collapse
ax.annotate("agent stops\nexploring here",
            xy=(old_s[np.argmin(old_v[:8])], old_v[:8].min()),
            xytext=(0.22, -95),
            fontsize=9, color=C_OLD, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=C_OLD, lw=1.0))

# annotate end of old reward run
ax.annotate("training\nstopped",
            xy=(old_s[-1], old_v[-1]),
            xytext=(old_s[-1] + 0.18, old_v[-1] - 18),
            fontsize=9, color=C_OLD, ha="left",
            arrowprops=dict(arrowstyle="-|>", color=C_OLD, lw=0.9))

ax.text(new_s[-1] + 0.04, new_v[-1] + 1.5,
        f"+{new_v[-1]:.1f}%", color=C_NEW, fontsize=10,
        va="bottom", fontweight="bold")

ax.set_xlabel(r"training steps ($\times 10^6$)", fontsize=11)
ax.set_ylabel("pooling savings  (%)", fontsize=11)
ax.set_xlim(left=0)
ax.set_ylim(top=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)
ax.legend(fontsize=10, frameon=False, loc="center right")

fig.tight_layout()
out = HERE / "run6_vs_run7.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
