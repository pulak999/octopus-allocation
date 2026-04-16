#!/usr/bin/env python3
"""
Plot the savings plateau for the new (departure-aware) reward.
Shows the ceiling clearly — no improvement after ~100k steps.
Output: plateau.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REPO = HERE.parent.parent.parent.parent

NEW_LOG = REPO / "output/logs/aug_v4_rewardA_v2/SAC_1"

C_NEW    = "#2E75B6"
C_ORANGE = "#E67E22"
C_ZERO   = "#888888"

def load_tag(log_path, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step  for e in events], dtype=float) / 1e6
    vals  = np.array([e.value for e in events], dtype=float)
    return steps, vals

s, v = load_tag(NEW_LOG, "eval/pooling_savings_mean")
v = v * 100

fig, ax = plt.subplots(figsize=(7.5, 3.6))

ax.axhline(0, color=C_ZERO, lw=1.2, ls="--", zorder=0)
ax.plot(s, v, color=C_NEW, lw=2.2, zorder=3)

# find approximate plateau onset (~100k steps = 0.1M)
plateau_onset_idx = np.searchsorted(s, 0.12)
plateau_val = float(np.median(v[plateau_onset_idx:]))

# shade plateau region
ax.fill_between(s[plateau_onset_idx:], v[plateau_onset_idx:],
                plateau_val - 0.3, color=C_ORANGE, alpha=0.12, zorder=1)
ax.axhline(plateau_val, xmin=s[plateau_onset_idx]/s[-1],
           color=C_ORANGE, lw=1.4, ls="--", zorder=2)

# vertical marker at plateau onset
ax.axvline(s[plateau_onset_idx], color=C_ORANGE, lw=1.0, ls=":", zorder=2)
ax.text(s[plateau_onset_idx] + 0.03, plateau_val + 0.4,
        f"plateau at $\\approx${plateau_val:.1f}%",
        color=C_ORANGE, fontsize=10, va="bottom")
ax.text(s[plateau_onset_idx] + 0.03, plateau_val - 0.5,
        "no improvement for next 1.9M steps",
        color=C_ORANGE, fontsize=9, va="top")

# label onset
ax.text(s[plateau_onset_idx], -0.8,
        f"100k steps", color=C_ORANGE, fontsize=9,
        ha="center", va="top")

# break-even label
ax.text(s[-1] + 0.02, 0.2, "break-even\n(0 %)",
        color=C_ZERO, fontsize=9, va="bottom")

ax.set_xlabel(r"training steps ($\times 10^6$)", fontsize=11)
ax.set_ylabel("pooling savings  (%)", fontsize=11)
ax.set_xlim(left=0)
ax.set_ylim(bottom=v.min() - 1, top=v.max() + 2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)

fig.tight_layout()
out = HERE / "plateau.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
