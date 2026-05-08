#!/usr/bin/env bash
# Re-run wave 2 R4/R5 on LVL01 after the track_vm_allocs Python-loop fix.
# current already finished (44 min); kill PIDs 2176850/2176851 before running.
set -euo pipefail

cd "$(dirname "$0")"
source venv/bin/activate

TRACE="LVL01PrdApp05-troundgrt5m"
NITER=20
LOGDIR="output/logs"
mkdir -p "$LOGDIR"

echo "[rerun-LVL01-wave2] Starting R4 on cuda:0 ..."
CUDA_VISIBLE_DEVICES=0 python scripts/eval_rl.py \
    --run-id ablation_R4_v1 --n-iter "$NITER" --traces "$TRACE" \
    > "$LOGDIR/eval_R4_v1_LVL01.log" 2>&1 &
PID_R4=$!

echo "[rerun-LVL01-wave2] Starting R5 on cuda:1 ..."
CUDA_VISIBLE_DEVICES=1 python scripts/eval_rl.py \
    --run-id ablation_R5_v1 --n-iter "$NITER" --traces "$TRACE" \
    > "$LOGDIR/eval_R5_v1_LVL01.log" 2>&1 &
PID_R5=$!

echo "[rerun-LVL01-wave2] Launched (PIDs: R4=$PID_R4  R5=$PID_R5)"
echo "[rerun-LVL01-wave2] Logs: $LOGDIR/eval_{R4,R5}_v1_LVL01.log"
echo "[rerun-LVL01-wave2] Waiting ..."

wait $PID_R4 && echo "[rerun-LVL01-wave2] R4 done"
wait $PID_R5 && echo "[rerun-LVL01-wave2] R5 done"

echo "[rerun-LVL01-wave2] All finished."
