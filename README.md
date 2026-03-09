# Octopus Allocation

RL-based memory pooling for CXL (Compute Express Link) in cloud datacenters.

## Structure

```
octopus-allocation/
├── scripts/           # Entry points
│   ├── train.py       # Train SAC/PPO on memory-pooling env
│   ├── evaluate.py    # Evaluate greedy vs RL policies
│   ├── plot_demand.py # VM demand time-series for paper figures
│   └── plot_memory_pooling_sweep.py
├── octopus/           # Core library (env, baselines, topology, data)
├── data/
│   ├── traces/       # Azure VM trace pickles (.pkl)
│   └── topologies/   # CXL topology CSVs
├── output/            # Runtime artifacts (not in git)
│   ├── checkpoints/   # Trained models
│   ├── logs/          # Training logs
│   └── plots/         # Generated figures
├── doc/
│   ├── v1/            # LaTeX report + slides
│   ├── context/       # Reference PDFs and slides
│   └── plans/         # Planning docs (plan-v1.md, plan-v2.md)
├── notebooks/         # Jupyter notebooks
└── Makefile          # venv, plots, pdf
```

## Quick start

```bash
make venv
source venv/bin/activate
python scripts/train.py
python scripts/evaluate.py --policy greedy rl
make plots   # Regenerate doc/v1/figs/
make pdf     # Build LaTeX report
```
