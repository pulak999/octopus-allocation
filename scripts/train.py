#!/usr/bin/env python3
"""
Train SAC (or PPO) on the Octopus CXL memory-pooling environment.

Usage
-----
  # Activate venv first, then:
  python train.py                                      # defaults: SAC, AG16x6, AMS20
  python train.py --algo ppo --total-timesteps 2000000
  python train.py --device cuda:0                      # pin to first Titan
"""

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from octopus.data import load_trace, load_topology
from octopus.env import OctopusMemPoolEnv


# ── helpers ──────────────────────────────────────────────────────────────
def _make_env(trace_data, M, seed):
    """Factory that returns a zero-arg callable for vec-env construction."""
    all_vms, node_to_vms, node_to_machine, _vm_type_sz, machine_sz = trace_data

    def _init():
        return OctopusMemPoolEnv(
            all_vms=all_vms,
            node_to_vms=node_to_vms,
            node_to_machine=node_to_machine,
            machine_sz=machine_sz,
            M=M,
            seed=seed,
        )

    return _init


# ── main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Train RL for Octopus memory allocation"
    )
    ap.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ap.add_argument(
        "--trace",
        default="AMS20PrdApp19-tround.sqlite",
        help="Training trace (cluster name stem)",
    )
    ap.add_argument(
        "--eval-trace",
        default="LON23PrdApp01-troundgrt5m.sqlite",
        help="Validation trace",
    )
    ap.add_argument(
        "--topology",
        default="data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv",
    )
    ap.add_argument("--total-timesteps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", default="output/checkpoints")
    ap.add_argument("--log-dir", default="output/logs")
    ap.add_argument(
        "--device",
        default="auto",
        help="Torch device: 'auto', 'cpu', 'cuda:0', etc.",
    )
    ap.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Save a checkpoint every N timesteps",
    )
    ap.add_argument(
        "--eval-freq",
        type=int,
        default=20_000,
        help="Evaluate every N timesteps",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Override defaults for much faster wall-clock training "
        "(CPU, smaller net, batched gradient steps, less eval/ckpt I/O). "
        "Typically 10-30× faster for this env's tiny obs space.",
    )
    args = ap.parse_args()

    # ── --fast overrides ──────────────────────────────────────────────
    if args.fast:
        args.checkpoint_freq = 50_000
        args.eval_freq = 100_000

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading training trace: {args.trace} …")
    trace_data = load_trace(args.trace)

    M, num_hosts, num_pools = load_topology(args.topology)
    max_deg = max(sum(row) for row in M)
    print(
        f"Topology: {num_hosts} hosts, {num_pools} MPDs, "
        f"max_degree={max_deg}"
    )

    # ── Environments ─────────────────────────────────────────────────────
    # SAC is off-policy → one env suffices; keeps memory usage low.
    train_env = DummyVecEnv([_make_env(trace_data, M, seed=args.seed)])

    print(f"Loading eval trace: {args.eval_trace} …")
    eval_trace_data = load_trace(args.eval_trace)
    eval_env = DummyVecEnv(
        [_make_env(eval_trace_data, M, seed=args.seed + 10_000)]
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_dir,
        name_prefix=f"octopus_{args.algo}",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=1 if args.fast else 3,
        deterministic=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    net_arch = [128, 128] if args.fast else [256, 256]
    common_kw = dict(
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device,
        seed=args.seed,
        policy_kwargs=dict(net_arch=net_arch),
    )

    if args.algo == "sac":
        # --fast: batch gradient work and shrink buffer
        sac_kw = dict(
            learning_rate=3e-4,
            buffer_size=300_000 if args.fast else 1_000_000,
            learning_starts=1_000,
            batch_size=512 if args.fast else 256,
            gamma=0.999,
            tau=0.005,
            ent_coef="auto",
            train_freq=16 if args.fast else 1,
            gradient_steps=1 if args.fast else 1,
        )
        model = SAC("MlpPolicy", train_env, **sac_kw, **common_kw)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            **common_kw,
        )

    # ── Train ────────────────────────────────────────────────────────────
    print(
        f"\nTraining {args.algo.upper()} for "
        f"{args.total_timesteps:,} timesteps …\n"
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # ── Save final model ─────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, f"octopus_{args.algo}_final")
    model.save(final_path)
    print(f"\n✓  Final model saved → {final_path}")


if __name__ == "__main__":
    main()