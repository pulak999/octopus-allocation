#!/usr/bin/env python3
"""
Train SAC (or PPO) on the Octopus CXL memory-pooling environment.

All artifacts are namespaced under --run-id so multiple runs can coexist.

Usage
-----
  python scripts/train_rl.py --run-id v0_fast --fast --total-timesteps 5000
  python scripts/train_rl.py --run-id v1_lam05 --variance-lambda 0.5 --total-timesteps 500000
"""

import argparse
import json
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
from stable_baselines3.common.vec_env import DummyVecEnv

from octopus.data import load_trace, load_topology
from octopus.env import OctopusMemPoolEnv
from octopus.topology import generate_pod_to_nodes, expand_M_to_all_nodes


# ── Pooling savings callback ──────────────────────────────────────────────
class PoolingSavingsCallback(BaseCallback):
    """After every *eval_freq* timesteps, run pooling_simulation on the eval
    trace and log eval/pooling_savings_mean and eval/pooling_savings_std."""

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


# ── Augmentation logging callback ─────────────────────────────────────────
class AugmentationLogCallback(BaseCallback):
    """Log augmentation parameter diversity during training."""

    def __init__(self, log_freq=500, verbose=0):
        super().__init__(verbose)
        self._scales = []
        self._log_freq = log_freq

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "aug_params" in info:
                self._scales.append(info["aug_params"]["scale"])

        if len(self._scales) >= self._log_freq:
            scales = np.array(self._scales[-self._log_freq:])
            self.logger.record("aug/scale_mean", float(scales.mean()))
            self.logger.record("aug/scale_std", float(scales.std()))
            self.logger.record("aug/scale_min", float(scales.min()))
            self.logger.record("aug/scale_max", float(scales.max()))
            self._scales = []
        return True


# ── helpers ──────────────────────────────────────────────────────────────
def _make_env(trace_data, M, seed, variance_lambda=0.5, skip_hotfix=False,
              aug_config=None, trace_pool=None):
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
            skip_hotfix=skip_hotfix,
            aug_config=aug_config,
            trace_pool=trace_pool,
        )

    return _init


# ── main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Train RL for Octopus memory allocation (namespaced by --run-id)"
    )
    ap.add_argument("--run-id", required=True,
                    help="Unique run identifier, e.g. v0_fast or v1_lam05")
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
        "(CPU, smaller net, batched gradient steps, less eval/ckpt I/O).",
    )
    ap.add_argument(
        "--skip-hotfix",
        action="store_true",
        help="Skip the HOTFIX VM filter in event construction. "
        "Use for CXL pooling training where per-node DRAM caps don't apply.",
    )

    # ── Augmentation args ─────────────────────────────────────────────
    aug_group = ap.add_argument_group("augmentation")
    aug_group.add_argument("--augmentation", action="store_true",
                           help="Enable episode-level data augmentation")
    aug_group.add_argument("--aug-scale-lo", type=float, default=0.7,
                           help="Lower bound of memory scale factor (default: 0.7)")
    aug_group.add_argument("--aug-scale-hi", type=float, default=1.2,
                           help="Upper bound of memory scale factor (default: 1.2)")
    aug_group.add_argument("--aug-scale-dist", choices=["triangular", "uniform"],
                           default="triangular",
                           help="Scale factor distribution (default: triangular)")
    aug_group.add_argument("--aug-memory-noise", type=float, default=0.0,
                           help="Per-VM memory noise sigma (0.0 = disabled, 0.03 = ±3%%)")
    aug_group.add_argument("--aug-jitter", type=int, default=0,
                           help="Max arrival tick jitter (0 = disabled, 1-2 recommended)")
    aug_group.add_argument("--aug-lifetime-noise", type=float, default=0.0,
                           help="Lifetime noise fraction (0.0 = disabled, 0.05 = ±5%%)")
    aug_group.add_argument("--aug-link-failures", type=float, default=0.0,
                           help="Link failure ratio (0.0 = disabled, 0.01-0.05 recommended)")
    aug_group.add_argument("--multi-trace", action="store_true",
                           help="Sample trace randomly each episode")
    aug_group.add_argument("--aug-traces", nargs="*", default=None,
                           help="Subset of traces for multi-trace (default: all 10)")

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

    # ── Namespaced output dirs ────────────────────────────────────────
    save_dir = os.path.join("output", "checkpoints", args.run_id)
    log_dir = os.path.join("output", "logs", args.run_id)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ── Save config ───────────────────────────────────────────────────
    import datetime
    config = vars(args).copy()
    config["timestamp"] = datetime.datetime.utcnow().isoformat()
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved → {config_path}")

    # ── Load data ─────────────────────────────────────────────────────
    print(f"Loading training trace: {args.trace} …")
    trace_data = load_trace(args.trace)

    M, num_hosts, num_pools = load_topology(args.topology)
    max_deg = max(sum(row) for row in M)
    print(f"Topology: {num_hosts} hosts, {num_pools} MPDs, max_degree={max_deg}")

    # ── Build augmentation config ─────────────────────────────────────
    aug_config = None
    trace_pool = None
    if args.augmentation:
        from octopus.augmentation import AugmentationConfig
        aug_config = AugmentationConfig(
            scale_range=(args.aug_scale_lo, args.aug_scale_hi),
            scale_distribution=args.aug_scale_dist,
            memory_noise_sigma=args.aug_memory_noise,
            arrival_jitter_ticks=args.aug_jitter,
            lifetime_noise_frac=args.aug_lifetime_noise,
            link_failure_ratio=args.aug_link_failures,
            multi_trace=args.multi_trace,
            enabled=True,
        )
        print(f"Augmentation: scale=[{args.aug_scale_lo}, {args.aug_scale_hi}] "
              f"noise_sigma={args.aug_memory_noise} jitter={args.aug_jitter} "
              f"lifetime={args.aug_lifetime_noise} link_fail={args.aug_link_failures} "
              f"multi_trace={args.multi_trace}")

        # Load multi-trace pool if requested
        if args.multi_trace:
            from octopus.data import load_trace as _lt
            trace_names = args.aug_traces or [
                "AMS20PrdApp19-tround.sqlite",
                "BLAPrdApp19-troundgrt5m.sqlite",
                "BN9PrdApp18-troundgrt5m.sqlite",
                "DSM08PrdApp05-troundgrt5m.sqlite",
                "DUB24PrdApp09-troundgrt5m.sqlite",
                "LON23PrdApp01-troundgrt5m.sqlite",
                "LVL01PrdApp05-troundgrt5m.sqlite",
                "SG2PrdApp35-troundgrt5m.sqlite",
                "SYD21PrdApp07-troundgrt5m.sqlite",
                "YTO21PrdApp05-troundgrt5m.sqlite",
            ]
            trace_pool = []
            for tn in trace_names:
                if not tn.endswith(".sqlite"):
                    tn = tn + ".sqlite"
                print(f"  Loading trace for pool: {tn} …")
                trace_pool.append(_lt(tn))
            print(f"  Loaded {len(trace_pool)} traces into pool")

    # ── Environments ──────────────────────────────────────────────────
    train_env = DummyVecEnv([_make_env(trace_data, M, seed=args.seed,
                                       variance_lambda=args.variance_lambda,
                                       skip_hotfix=args.skip_hotfix,
                                       aug_config=aug_config,
                                       trace_pool=trace_pool)])

    print(f"Loading eval trace: {args.eval_trace} …")
    eval_trace_data = load_trace(args.eval_trace)
    eval_env = DummyVecEnv(
        [_make_env(eval_trace_data, M, seed=args.seed + 10_000,
                   variance_lambda=args.variance_lambda,
                   skip_hotfix=args.skip_hotfix)]
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=save_dir,
        name_prefix=f"octopus_{args.algo}",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
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

    # ── Model ─────────────────────────────────────────────────────────
    net_arch = [128, 128] if args.fast else [256, 256]
    common_kw = dict(
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device,
        seed=args.seed,
        policy_kwargs=dict(net_arch=net_arch),
    )

    if args.algo == "sac":
        sac_kw = dict(
            learning_rate=3e-4,
            buffer_size=300_000 if args.fast else 1_000_000,
            learning_starts=1_000,
            batch_size=512 if args.fast else 256,
            gamma=0.999,
            tau=0.005,
            ent_coef="auto",
            train_freq=16 if args.fast else 1,
            gradient_steps=1,
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

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nTraining {args.algo.upper()} [{args.run_id}] for "
          f"{args.total_timesteps:,} timesteps …\n")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb, pooling_cb] + (
            [AugmentationLogCallback(log_freq=500)] if aug_config else []
        ),
        progress_bar=True,
    )

    # ── Save final model ──────────────────────────────────────────────
    final_path = os.path.join(save_dir, f"octopus_{args.algo}_final")
    model.save(final_path)
    print(f"\n✓  Final model saved → {final_path}")


if __name__ == "__main__":
    main()
