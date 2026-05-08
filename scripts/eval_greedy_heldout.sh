#!/usr/bin/env bash
# Evaluate greedy baseline on held-out traces (LVL01, AMS20).
# Run from repo root:
#   bash scripts/eval_greedy_heldout.sh

set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

OUT_DIR="output/evals/heldout_greedy"
N_ITER=50

TEST_TRACES=(
  "LVL01PrdApp05-troundgrt5m.sqlite"
  "AMS20PrdApp19-tround.sqlite"
)

for T in "${TEST_TRACES[@]}"; do
  echo "=== $T (greedy) ==="
  python scripts/evaluate.py \
    --trace "$T" \
    --policy greedy \
    --n-iter "$N_ITER" \
    --out-dir "$OUT_DIR" \
    --out-prefix "greedy_${T%.sqlite}"
done

echo ""
echo "Saved outputs to $OUT_DIR/"
