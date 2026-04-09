#!/usr/bin/env python3
"""
Train SAC (or PPO) on the Octopus CXL memory-pooling environment.

Usage
-----
  # Activate venv first, then:
  python train.py                                      # defaults: SAC, AG16x6, AMS20
  python train.py --algo ppo --total-timesteps 2000000
  python train.py --device cuda:0                      # pin to first Titan

  From repo root: pip install -e .   # optional; avoids manual PYTHONPATH
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os
import random

import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from octopus.data import load_trace, load_topology, precompute_pod_events
from octopus.env import OctopusMemPoolEnv
from octopus.topology import generate_pod_to_nodes, expand_M_to_all_nodes


# ── Pooling savings callback ──────────────────────────────────────────────
class PoolingSavingsCallback(BaseCallback):
    """After every *eval_freq* timesteps, run pooling_simulation on the eval
    trace and log eval/pooling_savings_mean, eval/pooling_savings_std, and
    eval/mpd_load_variance_mean."""

    def __init__(self, eval_trace_data, M, max_degree, eval_freq, n_iter=10, verbose=0):
        super().__init__(verbose)
        self._eval_trace_data = eval_trace_data
        self._M = M
        self._max_degree = max_degree
        self._eval_freq = eval_freq
        self._n_iter = n_iter
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self._eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        self._run_eval()
        return True

    def _run_eval(self):
        from scripts.evaluate import pooling_simulation, make_rl_alloc_cb

        all_vms, node_to_vms, node_to_machine, _, machine_sz = self._eval_trace_data
        savings_list = []
        variance_list = []

        for i in range(self._n_iter):
            pod_to_nodes = generate_pod_to_nodes(node_to_vms, len(self._M), 99999 + i)
            node_to_M, expanded_M = expand_M_to_all_nodes(self._M, pod_to_nodes)

            rl_cb = make_rl_alloc_cb(self.model, self._max_degree)

            ratio = pooling_simulation(
                node_to_M, expanded_M, rl_cb,
                all_vms, node_to_vms, node_to_machine, machine_sz,
            )
            savings_list.append(1.0 - ratio)

        savings_arr = np.array(savings_list)
        self.logger.record("eval/pooling_savings_mean", float(np.mean(savings_arr)))
        self.logger.record("eval/pooling_savings_std", float(np.std(savings_arr)))
        if self.verbose:
            print(
                f"  [PoolingSavings @ {self.num_timesteps}] "
                f"mean={np.mean(savings_arr):.4f}  std={np.std(savings_arr):.4f}"
            )


# ── helpers ──────────────────────────────────────────────────────────────
def _make_env(trace_data, M, seed, variance_lambda=0.5, precomputed_events=None):
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
            variance_lambda=variance_lambda,
            precomputed_events=precomputed_events,
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
        "--variance-lambda",
        type=float,
        default=0.5,
        help="Weight on MPD load variance penalty in reward (0 = no penalty)",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Override defaults for much faster wall-clock training "
        "(n_envs=32, SubprocVecEnv, smaller net, batched gradient steps, "
        "less eval/ckpt I/O).",
    )
    ap.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments. >1 uses SubprocVecEnv. "
        "PPO benefits most; SAC also improves with scaled train_freq.",
    )
    ap.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="SAC gradient steps per update (default: matches --n-envs). "
        "Override for finer control.",
    )
    args = ap.parse_args()

    # ── Deterministic seeding ─────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── --fast overrides ──────────────────────────────────────────────
    if args.fast:
        args.checkpoint_freq = 50_000
        args.eval_freq = 100_000
        if args.n_envs == 1:           # only override if user didn't set it
            args.n_envs = 32

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

    # ── Precompute pod events (avoids per-reset O(n_vms) work) ───────────
    n_envs = args.n_envs
    # Estimate seeds needed: each env increments episode_count independently;
    # 500 episodes per env is generous for a 500k-step run.
    _max_episodes = 500
    train_seeds = [args.seed + i + env_idx
                   for env_idx in range(n_envs)
                   for i in range(_max_episodes)]
    print(f"Precomputing pod events for {len(train_seeds)} seeds "
          f"({n_envs} envs × {_max_episodes} episodes) …")
    precomputed = precompute_pod_events(trace_data, M, train_seeds)
    print(f"  ✓ {len(precomputed)} entries cached")

    # ── Environments ─────────────────────────────────────────────────────
    env_fns = [
        _make_env(trace_data, M, seed=args.seed + i,
                  variance_lambda=args.variance_lambda,
                  precomputed_events=precomputed)
        for i in range(n_envs)
    ]
    if n_envs > 1:
        train_env = SubprocVecEnv(env_fns, start_method="fork")
        print(f"Using SubprocVecEnv with {n_envs} workers")
    else:
        train_env = DummyVecEnv(env_fns)

    print(f"Loading eval trace: {args.eval_trace} …")
    eval_trace_data = load_trace(args.eval_trace)
    eval_env = DummyVecEnv(
        [_make_env(eval_trace_data, M, seed=args.seed + 10_000, variance_lambda=args.variance_lambda)]
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
    pooling_cb = PoolingSavingsCallback(
        eval_trace_data=eval_trace_data,
        M=M,
        max_degree=max_deg,
        eval_freq=args.eval_freq,
        n_iter=5 if args.fast else 10,
        verbose=1,
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
        # Scale train_freq and gradient_steps with n_envs so the
        # update-to-data ratio stays constant as we add more workers.
        _train_freq = max(n_envs, 16) if args.fast else n_envs if n_envs > 1 else 1
        _grad_steps = args.gradient_steps if args.gradient_steps is not None else _train_freq
        sac_kw = dict(
            learning_rate=3e-4,
            buffer_size=300_000 if args.fast else 1_000_000,
            learning_starts=1_000,
            batch_size=512 if (args.fast or n_envs > 1) else 256,
            gamma=0.999,
            tau=0.005,
            ent_coef="auto",
            train_freq=_train_freq,
            gradient_steps=_grad_steps,
        )
        print(f"SAC: train_freq={_train_freq}, gradient_steps={_grad_steps}, "
              f"batch_size={sac_kw['batch_size']}")
        model = SAC("MlpPolicy", train_env, **sac_kw, **common_kw)
    else:
        # PPO: n_steps per env is fixed; total rollout = n_envs * n_steps.
        # Scale batch_size so minibatch count stays ~constant.
        _ppo_batch = min(n_envs * 64, 2048)
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=_ppo_batch,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            **common_kw,
        )
        print(f"PPO: n_envs={n_envs}, effective_batch={n_envs * 2048}, "
              f"minibatch={_ppo_batch}")

    # ── Train ────────────────────────────────────────────────────────────
    print(
        f"\nTraining {args.algo.upper()} for "
        f"{args.total_timesteps:,} timesteps …\n"
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb, pooling_cb],
        progress_bar=True,
    )

    # ── Save final model ─────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, f"octopus_{args.algo}_final")
    model.save(final_path)
    print(f"\n✓  Final model saved → {final_path}")


if __name__ == "__main__":
    main()