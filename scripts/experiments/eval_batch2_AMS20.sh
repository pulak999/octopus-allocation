#!/usr/bin/env bash
# Eval batch 2 — held-out trace: AMS20PrdApp19-tround
# Wave 1: R1/R2/R3 in parallel, wave 2: R4/R5/current in parallel.
set -euo pipefail

cd "$(dirname "$0")/../.."
source venv/bin/activate

TRACE="AMS20PrdApp19-tround"
NITER=20
LOGDIR="output/logs"
mkdir -p "$LOGDIR"

# --- Wave 1: R1 / R2 / R3 ---
echo "[batch-eval-AMS20] Wave 1: Starting R1 on cuda:0 ..."
CUDA_VISIBLE_DEVICES=0 python scripts/eval_rl.py \
    --run-id ablation_R1_v1 --n-iter "$NITER" --traces "$TRACE" \
    > "$LOGDIR/eval_R1_v1_AMS20.log" 2>&1 &
PID_R1=$!

echo "[batch-eval-AMS20] Wave 1: Starting R2 on cuda:1 ..."
CUDA_VISIBLE_DEVICES=1 python scripts/eval_rl.py \
    --run-id ablation_R2_v1 --n-iter "$NITER" --traces "$TRACE" \
    > "$LOGDIR/eval_R2_v1_AMS20.log" 2>&1 &
PID_R2=$!

echo "[batch-eval-AMS20] Wave 1: Starting R3 on cuda:2 ..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_rl.py \
    --run-id ablation_R3_v1 --n-iter "$NITER" --traces "$TRACE" \
    > "$LOGDIR/eval_R3_v1_AMS20.log" 2>&1 &
PID_R3=$!

echo "[batch-eval-AMS20] Wave 1 launched (PIDs: R1=$PID_R1  R2=$PID_R2  R3=$PID_R3)"
echo "[batch-eval-AMS20] Logs: $LOGDIR/eval_{R1,R2,R3}_v1_AMS20.log"
echo "[batch-eval-AMS20] Waiting for wave 1 ..."

wait $PID_R1 && echo "[batch-eval-AMS20] R1 done"
wait $PID_R2 && echo "[batch-eval-AMS20] R2 done"
wait $PID_R3 && echo "[batch-eval-AMS20] R3 done"

# --- Wave 2: R4 / R5 / current ---
echo "[batch-eval-AMS20] Wave 2: Starting R4 on cuda:0 ..."
CUDA_VISIBLE_DEVICES=0 python scripts/eval_rl.py \
    --run-id ablation_R4_v1 --n-iter "$NITER" --traces "$TRACE" \
    > "$LOGDIR/eval_R4_v1_AMS20.log" 2>&1 &
PID_R4=$!

echo "[batch-eval-AMS20] Wave 2: Starting R5 on cuda:1 ..."
CUDA_VISIBLE_DEVICES=1 python scripts/eval_rl.py \
    --run-id ablation_R5_v1 --n-iter "$NITER" --traces "$TRACE" \
    > "$LOGDIR/eval_R5_v1_AMS20.log" 2>&1 &
PID_R5=$!

echo "[batch-eval-AMS20] Wave 2: Starting current on cuda:2 ..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_rl.py \
    --run-id ablation_current_v1 --n-iter "$NITER" --traces "$TRACE" \
    > "$LOGDIR/eval_current_v1_AMS20.log" 2>&1 &
PID_CUR=$!

echo "[batch-eval-AMS20] Wave 2 launched (PIDs: R4=$PID_R4  R5=$PID_R5  current=$PID_CUR)"
echo "[batch-eval-AMS20] Logs: $LOGDIR/eval_{R4,R5,current}_v1_AMS20.log"
echo "[batch-eval-AMS20] Waiting for wave 2 ..."

wait $PID_R4 && echo "[batch-eval-AMS20] R4 done"
wait $PID_R5 && echo "[batch-eval-AMS20] R5 done"
wait $PID_CUR && echo "[batch-eval-AMS20] current done"

echo "[batch-eval-AMS20] All finished."
