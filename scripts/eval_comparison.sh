#!/usr/bin/env bash
# Evaluate run7 (aug_v4_rewardA_v2) vs greedy on both held-out test traces.
# Run from repo root: bash scripts/eval_comparison.sh

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

MODEL=output/checkpoints/aug_v4_rewardA_v2/best_model
OUT=output/evals/comparison
N=50

echo "=== LVL01 ==="
python scripts/evaluate.py \
  --trace LVL01PrdApp05-troundgrt5m.sqlite \
  --policy greedy rl \
  --model "$MODEL" \
  --n-iter $N \
  --out-dir "$OUT" \
  --out-prefix run7_LVL01

echo ""
echo "=== AMS20 ==="
python scripts/evaluate.py \
  --trace AMS20PrdApp19-tround.sqlite \
  --policy greedy rl \
  --model "$MODEL" \
  --n-iter $N \
  --out-dir "$OUT" \
  --out-prefix run7_AMS20

echo ""
echo "=== Run6 (split721_sac) — LVL01 ==="
python scripts/evaluate.py \
  --trace LVL01PrdApp05-troundgrt5m.sqlite \
  --policy greedy rl \
  --model output/checkpoints/split721_sac/best_model \
  --n-iter $N \
  --out-dir "$OUT" \
  --out-prefix run6_LVL01

echo ""
echo "=== Run6 (split721_sac) — AMS20 ==="
python scripts/evaluate.py \
  --trace AMS20PrdApp19-tround.sqlite \
  --policy greedy rl \
  --model output/checkpoints/split721_sac/best_model \
  --n-iter $N \
  --out-dir "$OUT" \
  --out-prefix run6_AMS20

echo ""
echo "Results saved to $OUT/"
