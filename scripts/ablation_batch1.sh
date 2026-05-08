#!/usr/bin/env bash
# Ablation batch 1: R1 (cuda:0), R2 (cuda:1), R3 (cuda:2)
# Run from repo root: bash scripts/ablation_batch1.sh
# Pre-warms the event cache serially first (avoids race on first write),
# then launches all three in parallel and waits for them to finish.

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

TOTAL=200000
N_ENVS=32
CKPT_FREQ=500
EVAL_FREQ=1000

COMMON="--total-timesteps $TOTAL --n-envs $N_ENVS --skip-hotfix --no-wandb \
        --checkpoint-freq $CKPT_FREQ --eval-freq $EVAL_FREQ \
        --augmentation --multi-trace"

# ── Pre-warm event cache (serial, ~380s) ─────────────────────────────────────
# All 3 parallel jobs share the same cache key (skip_hotfix=True + augmentation).
# Pre-warming prevents a non-atomic write race when all 3 hit a cold cache at once.
echo "[batch1] Pre-warming event cache (all 10 traces × 128 seeds, skip_hotfix=True) ..."
CUDA_VISIBLE_DEVICES=0 python scripts/train_rl.py \
    --run-id _prewarm_cache \
    --reward-variant current \
    --device cuda:0 --eval-device cuda:0 \
    --total-timesteps 1 --n-envs 1 \
    --skip-hotfix --no-wandb \
    --augmentation --multi-trace \
    > output/logs/_prewarm_cache.log 2>&1
echo "[batch1] Cache ready."

# ── Launch three ablations in parallel ───────────────────────────────────────
echo "[batch1] Starting R1 on cuda:0 ..."
CUDA_VISIBLE_DEVICES=0 python scripts/train_rl.py \
    --run-id ablation_R1_v1 --reward-variant R1 \
    --device cuda:0 --eval-device cuda:0 \
    $COMMON \
    > output/logs/ablation_R1_v1.log 2>&1 &
PID_R1=$!

echo "[batch1] Starting R2 on cuda:1 ..."
CUDA_VISIBLE_DEVICES=1 python scripts/train_rl.py \
    --run-id ablation_R2_v1 --reward-variant R2 \
    --device cuda:0 --eval-device cuda:0 \
    $COMMON \
    > output/logs/ablation_R2_v1.log 2>&1 &
PID_R2=$!

echo "[batch1] Starting R3 on cuda:2 ..."
CUDA_VISIBLE_DEVICES=2 python scripts/train_rl.py \
    --run-id ablation_R3_v1 --reward-variant R3 \
    --device cuda:0 --eval-device cuda:0 \
    $COMMON \
    > output/logs/ablation_R3_v1.log 2>&1 &
PID_R3=$!

echo "[batch1] All three launched (PIDs: R1=$PID_R1  R2=$PID_R2  R3=$PID_R3)"
echo "[batch1] Logs: output/logs/ablation_{R1,R2,R3}_v1.log"
echo "[batch1] Waiting for all to finish ..."

wait $PID_R1 && echo "[batch1] R1 done" || echo "[batch1] R1 FAILED (exit $?)"
wait $PID_R2 && echo "[batch1] R2 done" || echo "[batch1] R2 FAILED (exit $?)"
wait $PID_R3 && echo "[batch1] R3 done" || echo "[batch1] R3 FAILED (exit $?)"

echo "[batch1] Complete. Run batch2 next:"
echo "  bash scripts/ablation_batch2.sh"
