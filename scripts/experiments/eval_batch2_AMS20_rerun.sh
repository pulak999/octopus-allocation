#!/usr/bin/env bash
# Re-eval R1/R2/R3/current on AMS20 sequentially — results were overwritten by the LVL01 batch.
set -euo pipefail

cd "$(dirname "$0")/../.."
source venv/bin/activate

TRACE="AMS20PrdApp19-tround"
NITER=20
LOGDIR="output/logs"
mkdir -p "$LOGDIR"

for RUN in ablation_R1_v1 ablation_R2_v1 ablation_R3_v1 ablation_current_v1; do
    echo "[rerun-AMS20] Starting $RUN ..."
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_rl.py \
        --run-id "$RUN" --n-iter "$NITER" --traces "$TRACE" \
        > "$LOGDIR/eval_${RUN}_AMS20.log" 2>&1
    echo "[rerun-AMS20] $RUN done"
done

echo "[rerun-AMS20] All finished."
