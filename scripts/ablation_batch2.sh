#!/usr/bin/env bash
# Ablation batch 2: R4 (cuda:0), R5 (cuda:1), current (cuda:2)
# Run from repo root: bash scripts/ablation_batch2.sh
# Assumes batch1 has already run (event cache already built).
# Waits for all three to finish before exiting.

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

TOTAL=500000
N_ENVS=32
CKPT_FREQ=10000
EVAL_FREQ=5000

COMMON="--total-timesteps $TOTAL --n-envs $N_ENVS --skip-hotfix --no-wandb \
        --checkpoint-freq $CKPT_FREQ --eval-freq $EVAL_FREQ \
        --augmentation --multi-trace"

echo "[batch2] Starting R4 on cuda:0 ..."
CUDA_VISIBLE_DEVICES=0 python scripts/train_rl.py \
    --run-id ablation_R4_v1 --reward-variant R4 \
    --device cuda:0 --eval-device cuda:0 \
    $COMMON \
    > output/logs/ablation_R4_v1.log 2>&1 &
PID_R4=$!

echo "[batch2] Starting R5 on cuda:1 ..."
CUDA_VISIBLE_DEVICES=1 python scripts/train_rl.py \
    --run-id ablation_R5_v1 --reward-variant R5 \
    --device cuda:0 --eval-device cuda:0 \
    $COMMON \
    > output/logs/ablation_R5_v1.log 2>&1 &
PID_R5=$!

echo "[batch2] Starting current on cuda:2 ..."
CUDA_VISIBLE_DEVICES=2 python scripts/train_rl.py \
    --run-id ablation_current_v1 --reward-variant current \
    --device cuda:0 --eval-device cuda:0 \
    $COMMON \
    > output/logs/ablation_current_v1.log 2>&1 &
PID_CUR=$!

echo "[batch2] All three launched (PIDs: R4=$PID_R4  R5=$PID_R5  current=$PID_CUR)"
echo "[batch2] Logs: output/logs/ablation_{R4,R5,current}_v1.log"
echo "[batch2] Waiting for all to finish ..."

wait $PID_R4  && echo "[batch2] R4 done"      || echo "[batch2] R4 FAILED (exit $?)"
wait $PID_R5  && echo "[batch2] R5 done"      || echo "[batch2] R5 FAILED (exit $?)"
wait $PID_CUR && echo "[batch2] current done" || echo "[batch2] current FAILED (exit $?)"

echo "[batch2] All ablations complete."
echo "[batch2] Evaluate all 6 runs:"
echo "  python scripts/eval_rl.py \\"
echo "    --run-id ablation_R1_v1 ablation_R2_v1 ablation_R3_v1 \\"
echo "           ablation_R4_v1 ablation_R5_v1 ablation_current_v1 \\"
echo "    --n-iter 50"
