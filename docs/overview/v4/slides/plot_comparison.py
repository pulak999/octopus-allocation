#!/usr/bin/env python3
"""
Generate bar chart (greedy vs RL) for slides-v4 slide 2a.

Reads the existing 50-pod detail CSVs for AMS20 and LVL01.
Produces a grouped bar chart with individual pod dots overlaid.
Output: comparison_bar.pdf (for \includegraphics in LaTeX).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).parent
REPO = HERE.parent.parent.parent.parent  # octopus-allocation root

AMS20 = REPO / "output/evals/comparison/run7_AMS20_detail.csv"
LVL01 = REPO / "output/evals/comparison/run7_LVL01_detail.csv"

# ── colours ───────────────────────────────────────────────────────────
C_GREEDY = "#C0392B"
C_RL     = "#2E75B6"
C_BREAK  = "#888888"

# ── load data ─────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    greedy = df[df["policy"] == "greedy"]["pooling_ratio"].values
    rl     = df[df["policy"] == "rl"]["pooling_ratio"].values
    return greedy, rl

ams_greedy, ams_rl = load(AMS20)
lvl_greedy, lvl_rl = load(LVL01)

# ── layout ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.8))

# group positions
groups   = ["AMS20", "LVL01"]
x_greedy = np.array([0.0, 2.6])
x_rl     = np.array([0.7, 3.3])
bar_w    = 0.55

data = {
    "AMS20-greedy": (x_greedy[0], ams_greedy, C_GREEDY),
    "AMS20-rl":     (x_rl[0],     ams_rl,     C_RL),
    "LVL01-greedy": (x_greedy[1], lvl_greedy, C_GREEDY),
    "LVL01-rl":     (x_rl[1],     lvl_rl,     C_RL),
}

# y-axis cap for greedy LVL01 (it's ~13, would dwarf everything)
Y_CAP = 5.5
AXIS_MAX = 5.8

for key, (x, vals, colour) in data.items():
    mean = vals.mean()
    # bar height capped
    bar_h = min(mean, Y_CAP)
    ax.bar(x, bar_h, width=bar_w, color=colour, alpha=0.82, zorder=2)

    # individual pod dots (jittered x, capped y)
    jitter = np.random.default_rng(42).uniform(-0.13, 0.13, len(vals))
    capped = np.clip(vals, 0, Y_CAP)
    ax.scatter(x + jitter, capped, color=colour, s=14, alpha=0.55,
               zorder=3, linewidths=0)

    # label: if capped, show true mean with arrow
    label = f"{mean:.2f}"
    if mean > Y_CAP:
        ax.annotate(
            label,
            xy=(x, Y_CAP),
            xytext=(x, Y_CAP + 0.18),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=colour,
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.0),
        )
    else:
        ax.text(x, bar_h + 0.08, label, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=colour)

# ── break-even line ───────────────────────────────────────────────────
ax.axhline(1.0, color=C_BREAK, linestyle="--", linewidth=1.2, zorder=1)
ax.text(4.15, 1.04, "break-even (1.0)", color=C_BREAK, fontsize=9, va="bottom")

# ── group labels ──────────────────────────────────────────────────────
for i, (label, xg, xr) in enumerate(zip(groups, x_greedy, x_rl)):
    ax.text((xg + xr) / 2.0, -0.55, label, ha="center", va="top",
            fontsize=12, fontweight="bold")

# ── axes ──────────────────────────────────────────────────────────────
ax.set_xlim(-0.45, 4.15)
ax.set_ylim(0, AXIS_MAX)
ax.set_ylabel("pooling ratio", fontsize=11)
ax.set_xticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# y ticks at integers only
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.tick_params(axis="y", labelsize=10)

# ── legend ────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=C_GREEDY, alpha=0.82, label="Greedy"),
    mpatches.Patch(color=C_RL,     alpha=0.82, label="RL (run 7)"),
]
ax.legend(handles=patches, fontsize=10, frameon=False,
          loc="upper right", bbox_to_anchor=(0.98, 0.98))

# ── caption note ─────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "Each dot is one pod mapping (n = 50 random seeds). "
         "Greedy LVL01 bar capped at 5; true mean shown.",
         ha="center", fontsize=8, color="#555555")

plt.tight_layout(rect=[0, 0.04, 1, 1])

out = HERE / "comparison_bar.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
