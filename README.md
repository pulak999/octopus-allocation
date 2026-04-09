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
pip install -e .
python scripts/train.py
python scripts/evaluate.py --policy greedy rl
make plots   # Regenerate doc/v1/figs/
make pdf     # Build LaTeX report
```

## Training with a 7/2/1 trace split

The project has 10 canonical traces (see `scripts/eval_rl.py` / `scripts/eval_baselines.py`):

- `AMS20PrdApp19-tround.sqlite`
- `BLAPrdApp19-troundgrt5m.sqlite`
- `BN9PrdApp18-troundgrt5m.sqlite`
- `DSM08PrdApp05-troundgrt5m.sqlite`
- `DUB24PrdApp09-troundgrt5m.sqlite`
- `LON23PrdApp01-troundgrt5m.sqlite`
- `LVL01PrdApp05-troundgrt5m.sqlite`
- `SG2PrdApp35-troundgrt5m.sqlite`
- `SYD21PrdApp07-troundgrt5m.sqlite`
- `YTO21PrdApp05-troundgrt5m.sqlite`

Use this pattern for your desired split:

- 7 traces for training (`--traces`)
- 2 traces for test-time evaluation
- 1 trace for eval during training (`--eval-trace`)

Best-practice note:

- Split by whole trace, not by samples within a trace.
- Keep validation/eval separate from final test traces.
- With only 10 traces, do not treat a single 7/2/1 partition as definitive. For stronger results, repeat the experiment across multiple trace holdout splits and report mean/std across splits.

### 1) Set your split

Recommended first split from the repo's existing trace characterization artifacts (`output/trace_characterization.csv`, `data/splits/train_val_traces.json`, `data/splits/test_traces.sealed.json`):

- `YTO21PrdApp05-troundgrt5m.sqlite` as eval/validation because it is closest to the median workload across the characterization metrics.
- `LVL01PrdApp05-troundgrt5m.sqlite` and `AMS20PrdApp19-tround.sqlite` as test because they represent opposite workload extremes (bursty-short-lived vs steady-long-lived / memory-heavy).
- The remaining 7 traces as train.

Treat this as a good first split for iteration, not a universal canonical benchmark split.

```bash
# Recommended first split from repo metadata
TRAIN_TRACES=(
  BLAPrdApp19-troundgrt5m.sqlite
  BN9PrdApp18-troundgrt5m.sqlite
  DSM08PrdApp05-troundgrt5m.sqlite
  DUB24PrdApp09-troundgrt5m.sqlite
  LON23PrdApp01-troundgrt5m.sqlite
  SG2PrdApp35-troundgrt5m.sqlite
  SYD21PrdApp07-troundgrt5m.sqlite
)
TEST_TRACES=(
  LVL01PrdApp05-troundgrt5m.sqlite
  AMS20PrdApp19-tround.sqlite
)
EVAL_TRACE=YTO21PrdApp05-troundgrt5m.sqlite
```

### 2) Train RL on the 7 training traces

```bash
python scripts/train_rl.py \
  --run-id split721_sac \
  --algo sac \
  --traces "${TRAIN_TRACES[@]}" \
  --eval-trace "$EVAL_TRACE" \
  --topology data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv \
  --total-timesteps 500000 \
  --n-envs 8 \
  --device auto
```

Notes:

- `--traces` cycles across the provided training traces when building parallel envs.
- Checkpoints/logs for this run go under `output/checkpoints/split721_sac` and `output/logs/split721_sac`.
- Best model path: `output/checkpoints/split721_sac/best_model.zip`.

### 3) Compare RL vs greedy on the 2 held-out test traces

```bash
for T in "${TEST_TRACES[@]}"; do
  echo "=== $T ==="
  python scripts/evaluate.py \
    --trace "$T" \
    --policy greedy rl \
    --model output/checkpoints/split721_sac/best_model \
    --n-iter 50 \
    --out-dir output/evals/split721 \
    --out-prefix "test_${T%.sqlite}"
done
```

This prints per-trace summary stats and also saves plottable CSVs:

- `output/evals/split721/test_<trace>_summary.csv`
- `output/evals/split721/test_<trace>_detail.csv`
