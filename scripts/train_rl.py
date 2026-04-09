#!/usr/bin/env python3
"""
Train SAC (or PPO) on the Octopus CXL memory-pooling environment.

All artifacts are namespaced under --run-id so multiple runs can coexist.

Usage
-----
  python scripts/train_rl.py --run-id v0_fast --fast --total-timesteps 5000
  python scripts/train_rl.py --run-id v1_lam05 --variance-lambda 0.5 --total-timesteps 500000

  From repo root you can also: pip install -e .
"""

# Allow `python scripts/train_rl.py` without PYTHONPATH (see pyproject.toml).
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from octopus.data import load_trace, load_topology, to_arrays
from octopus.env import OctopusMemPoolEnv
from octopus.topology import generate_pod_to_nodes, expand_M_to_all_nodes


def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _eval_worker_main(cmd_q, res_q, policy_cpu, eval_trace_data, M,
                      max_degree, n_iter, obs_variant, lookahead_window,
                      device_str):
    """Persistent eval worker — runs in a spawned subprocess.

    Waits on cmd_q for ('eval', state_dict_cpu, snap_step) commands.
    Runs n_iter × pooling_simulation and puts (mean, std, snap_step) on res_q.
    Exits cleanly on ('stop',).
    """
    import datetime as _dt
    import numpy as np
    import torch
    from octopus.topology import generate_pod_to_nodes, expand_M_to_all_nodes
    from scripts.evaluate import pooling_simulation

    device = torch.device(device_str)
    policy = policy_cpu.to(device)
    policy.set_training_mode(False)

    _obs_dim = max_degree * 6 + 2 if obs_variant != "current" else max_degree * 2 + 4
    _obs_buf = torch.zeros(1, _obs_dim, dtype=torch.float32, device=device)

    all_vms, node_to_vms, node_to_machine, _, machine_sz = eval_trace_data
    track = obs_variant != "current"

    def _make_alloc_cb(mhd_to_hosts_exp, Q_j_exp):
        """Build one RL inference closure for a single pod mapping iteration."""
        def _rl_alloc_cb(cxl_mem, mhd_list, cur_cxl_mem_vec, ctx):
            n_acc = len(mhd_list)
            num_mhd = ctx["num_mhd"]
            norm = ctx["pod_rss_mem"] if ctx["pod_rss_mem"] > 0 else 1.0
            tick = ctx["tick"]

            if obs_variant == "current":
                loads = np.zeros(max_degree, dtype=np.float32)
                for idx, mhd in enumerate(mhd_list):
                    loads[idx] = float(cur_cxl_mem_vec[mhd]) / norm
                mask = np.zeros(max_degree, dtype=np.float32)
                mask[:n_acc] = 1.0
                vm_norm = np.float32(cxl_mem / norm)
                peak_norm = np.float32(float(np.max(cur_cxl_mem_vec)) / norm)
                abs_time = ctx["base_time"] + _dt.timedelta(
                    minutes=(tick + ctx["pod_start_ts"]) * 5
                )
                hour = abs_time.hour + abs_time.minute / 60.0
                hour_sin = np.float32(np.sin(2.0 * np.pi * hour / 24.0))
                hour_cos = np.float32(np.cos(2.0 * np.pi * hour / 24.0))
                obs = np.concatenate(
                    [loads, mask, np.array([vm_norm, peak_norm, hour_sin, hour_cos])]
                ).astype(np.float32)
            else:
                mpd_vm_allocs = ctx["mpd_vm_allocs"]
                host_cxl_load = ctx["host_cxl_load"]
                W = lookahead_window
                t_plus_W = tick + W
                obs = np.zeros(max_degree * 6 + 2, dtype=np.float32)
                for k, mhd in enumerate(mhd_list):
                    dj = sj = 0.0
                    for dt, mem in mpd_vm_allocs[mhd]:
                        if dt <= t_plus_W:
                            dj += mem * (1.0 - (dt - tick) / W)
                        else:
                            sj += mem
                    base_idx = k * 6
                    obs[base_idx]     = float(cur_cxl_mem_vec[mhd]) / norm
                    obs[base_idx + 1] = dj / norm
                    obs[base_idx + 2] = sj / norm
                    obs[base_idx + 3] = 1.0
                    _mhd_to_hosts = ctx.get("mhd_to_hosts", mhd_to_hosts_exp)
                    _Q_j = ctx.get("Q_j", Q_j_exp)
                    P_j = float(sum(host_cxl_load[h] for h in _mhd_to_hosts[mhd]))
                    obs[base_idx + 4] = P_j / norm
                    obs[base_idx + 5] = float(_Q_j[mhd])
                obs[-2] = float(np.max(cur_cxl_mem_vec)) / norm
                obs[-1] = float(cxl_mem) / norm

            _obs_buf[0].copy_(torch.from_numpy(obs))
            with torch.no_grad():
                action = policy._predict(_obs_buf, deterministic=True).cpu().numpy()[0]

            raw = action[:n_acc].astype(np.float64)
            raw = raw - raw.max()
            exp_raw = np.exp(raw)
            proportions = exp_raw / (exp_raw.sum() + 1e-12)
            alloc_arr = np.zeros(num_mhd, dtype=np.float64)
            for idx, mhd in enumerate(mhd_list):
                alloc_arr[mhd] = proportions[idx] * cxl_mem
            return alloc_arr

        return _rl_alloc_cb

    while True:
        cmd = cmd_q.get()
        if cmd[0] == "stop":
            break
        if cmd[0] == "eval":
            _, state_dict_cpu, snap_step = cmd
            policy.load_state_dict(
                {k: v.to(device) for k, v in state_dict_cpu.items()}
            )
            policy.set_training_mode(False)

            savings_list = []
            for i in range(n_iter):
                pod_to_nodes = generate_pod_to_nodes(node_to_vms, len(M), 99999 + i)
                node_to_M, expanded_M = expand_M_to_all_nodes(M, pod_to_nodes)

                mhd_to_hosts_exp = None
                Q_j_exp = None
                if obs_variant != "current":
                    num_mhd_exp = len(expanded_M[0])
                    mhd_to_hosts_exp = {j: [] for j in range(num_mhd_exp)}
                    h_to_mhds = {}
                    for h, row in enumerate(expanded_M):
                        h_to_mhds[h] = [j for j, v in enumerate(row) if v]
                        for j in h_to_mhds[h]:
                            mhd_to_hosts_exp[j].append(h)
                    Q_j_exp = np.zeros(num_mhd_exp, dtype=np.float32)
                    for j in range(num_mhd_exp):
                        hosts = mhd_to_hosts_exp[j]
                        if hosts:
                            inv_deg = [1.0 / len(h_to_mhds[h]) for h in hosts
                                       if h_to_mhds[h]]
                            if inv_deg:
                                Q_j_exp[j] = float(np.mean(inv_deg))

                rl_cb = _make_alloc_cb(mhd_to_hosts_exp, Q_j_exp)
                ratio = pooling_simulation(
                    node_to_M, expanded_M, rl_cb,
                    all_vms, node_to_vms, node_to_machine, machine_sz,
                    track_vm_allocs=track,
                )
                savings_list.append(1.0 - ratio)

            arr = np.array(savings_list)
            res_q.put((float(np.mean(arr)), float(np.std(arr)), snap_step))


# ── Pooling savings callback ──────────────────────────────────────────────
class PoolingSavingsCallback(BaseCallback):
    """After every *eval_freq* timesteps, run pooling_simulation on the eval
    trace and log eval/pooling_savings_mean and eval/pooling_savings_std."""

    def __init__(self, eval_trace_data, M, max_degree, eval_freq, n_iter=10,
                 obs_variant="current", mhd_to_hosts=None, Q_j=None,
                 lookahead_window=200, eval_device=None, verbose=0):
        super().__init__(verbose)
        self._eval_trace_data = eval_trace_data
        self._M = M
        self._max_degree = max_degree
        self._eval_freq = eval_freq
        self._n_iter = n_iter
        self._obs_variant = obs_variant
        self._mhd_to_hosts = mhd_to_hosts
        self._Q_j = Q_j
        self._lookahead_window = lookahead_window
        self._last_eval_step = 0
        # async worker state
        self._worker = None
        self._cmd_q = None
        self._res_q = None
        self._pending_step = -1
        self._eval_device = eval_device  # e.g. "cuda:1"; None = mirror training device

    def on_training_start(self, locals_, globals_) -> None:
        import copy
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._cmd_q = ctx.Queue()
        self._res_q = ctx.Queue()
        policy_cpu = copy.deepcopy(self.model.policy).cpu()
        device_str = self._eval_device or str(next(self.model.policy.parameters()).device)
        self._worker = ctx.Process(
            target=_eval_worker_main,
            args=(self._cmd_q, self._res_q, policy_cpu,
                  self._eval_trace_data, self._M, self._max_degree,
                  self._n_iter, self._obs_variant, self._lookahead_window,
                  device_str),
            daemon=True,
        )
        self._worker.start()
        _log(f"[PoolingSavings] async worker started on {device_str}")

    def _on_step(self) -> bool:
        import queue
        # 1. Detect worker crash early
        if self._worker is not None and self._worker.exitcode is not None:
            _log(f"[PoolingSavings] WARNING: worker exited with code "
                 f"{self._worker.exitcode} — eval results may be missing")
            return True

        # 2. Collect any completed result (non-blocking)
        if self._res_q is not None:
            try:
                mean, std, snap_step = self._res_q.get_nowait()
                self.logger.record("eval/pooling_savings_mean", mean)
                self.logger.record("eval/pooling_savings_std", std)
                if self.verbose:
                    _log(f"  [PoolingSavings @ {snap_step}] "
                         f"mean={mean:.4f}  std={std:.4f}  "
                         f"(collected at {self.num_timesteps})")
                self._pending_step = -1
            except queue.Empty:
                pass

        # 3. Dispatch new eval if due
        if self.num_timesteps - self._last_eval_step >= self._eval_freq:
            self._last_eval_step = self.num_timesteps
            sd_cpu = {k: v.cpu() for k, v in self.model.policy.state_dict().items()}
            self._cmd_q.put(("eval", sd_cpu, self.num_timesteps))
            self._pending_step = self.num_timesteps
            if self.verbose:
                _log(f"  [PoolingSavings @ {self.num_timesteps}] eval dispatched (async)")
        return True

    def on_training_end(self) -> None:
        import queue
        if self._worker is None:
            return
        self._cmd_q.put(("stop",))
        self._worker.join(timeout=300)
        # Drain any result that finished just before shutdown
        try:
            mean, std, snap_step = self._res_q.get_nowait()
            self.logger.record("eval/pooling_savings_mean", mean)
            self.logger.record("eval/pooling_savings_std", std)
            _log(f"  [PoolingSavings @ {snap_step}] final result collected on shutdown")
        except queue.Empty:
            pass


# ── SAC diagnostics callback ─────────────────────────────────────────────
class SACDiagnosticsCallback(BaseCallback):
    """Log SAC internals: entropy coefficient, Q-values, gradient norms."""

    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self._log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self._log_freq != 0:
            return True

        model = self.model
        # Entropy coefficient (alpha)
        if hasattr(model, "ent_coef_tensor"):
            ent_coef = model.ent_coef_tensor.exp().item()
            self.logger.record("sac/ent_coef", ent_coef)

        # Q-value statistics from the last critic forward pass
        if hasattr(model, "critic") and hasattr(model, "replay_buffer"):
            try:
                replay = model.replay_buffer
                if replay.size() > model.batch_size:
                    data = replay.sample(min(256, model.batch_size))
                    with torch.no_grad():
                        q1, q2 = model.critic(data.observations, data.actions)
                    self.logger.record("sac/q1_mean", float(q1.mean()))
                    self.logger.record("sac/q2_mean", float(q2.mean()))
                    self.logger.record("sac/q1_std", float(q1.std()))
                    self.logger.record("sac/q_spread", float((q1 - q2).abs().mean()))
            except Exception:
                pass  # skip if buffer not ready

        # Gradient norms
        for name, net in [("actor", getattr(model, "actor", None)),
                          ("critic", getattr(model, "critic", None))]:
            if net is None:
                continue
            total_norm = 0.0
            for p in net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.logger.record(f"sac/{name}_grad_norm", total_norm)

        return True


# ── Environment metrics callback ─────────────────────────────────────────
class EnvMetricsCallback(BaseCallback):
    """Log environment-specific metrics from the info dict at episode end."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "pooling_ratio" in info:
                self.logger.record("env/pooling_ratio", info["pooling_ratio"])
                self.logger.record("env/pooling_savings", info.get("pooling_savings", 0.0))
            if "max_peak" in info:
                self.logger.record("env/max_peak", info["max_peak"])
            if "num_events" in info:
                self.logger.record("env/num_events", info["num_events"])
        return True


# ── Augmentation logging callback ─────────────────────────────────────────
class AugmentationLogCallback(BaseCallback):
    """Log augmentation parameter diversity during training."""

    def __init__(self, log_freq=500, verbose=0):
        super().__init__(verbose)
        self._scales = []
        self._link_failures = 0
        self._trace_ids = []
        self._log_freq = log_freq

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "aug_params" in info:
                params = info["aug_params"]
                self._scales.append(params["scale"])
                if params.get("fail_ratio", 0.0) > 0:
                    self._link_failures += 1
            if "trace_id" in info:
                self._trace_ids.append(info["trace_id"])

        if len(self._scales) >= self._log_freq:
            scales = np.array(self._scales[-self._log_freq:])
            self.logger.record("aug/scale_mean", float(scales.mean()))
            self.logger.record("aug/scale_std", float(scales.std()))
            self.logger.record("aug/scale_min", float(scales.min()))
            self.logger.record("aug/scale_max", float(scales.max()))
            self.logger.record("aug/link_failures_applied", self._link_failures)
            if self._trace_ids:
                self.logger.record("aug/unique_traces", len(set(self._trace_ids)))
            self._scales = []
            self._link_failures = 0
            self._trace_ids = []
        return True


# ── helpers ──────────────────────────────────────────────────────────────
def _make_env(trace_arrays, M, seed, variance_lambda=0.5, skip_hotfix=False,
              aug_config=None, trace_pool=None, reward_variant="current",
              lookahead_window=200, reward_lambda=0.2):
    def _init():
        return OctopusMemPoolEnv(
            trace_arrays=trace_arrays,
            M=M,
            seed=seed,
            variance_lambda=variance_lambda,
            skip_hotfix=skip_hotfix,
            aug_config=aug_config,
            trace_pool=trace_pool,
            reward_variant=reward_variant,
            lookahead_window=lookahead_window,
            reward_lambda=reward_lambda,
        )

    return _init


# ── main ─────────────────────────────────────────────────────────────────
def main():
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

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
    ap.add_argument("--train-freq", type=int, default=None,
                    help="SAC train_freq: gradient update every N vec_env steps. "
                         "Default: 16 if --fast, else 1. Higher = faster but less sample-efficient.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        default="auto",
        help="Torch device: 'auto', 'cpu', 'cuda:0', etc.",
    )
    ap.add_argument(
        "--eval-device",
        default="cuda:1",
        help="Torch device for the async eval worker (default: cuda:1). "
             "Pass '' to mirror --device.",
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
    ap.add_argument("--n-envs", type=int, default=1,
                    help="Number of parallel environments (1=DummyVecEnv, >1=SubprocVecEnv)")
    ap.add_argument(
        "--traces", nargs="+", default=None,
        help="Multiple training traces (cycled across envs). Overrides --trace.",
    )
    ap.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training; run random policy for --total-timesteps steps and report mean reward.",
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
    ap.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    ap.add_argument(
        "--wandb-project",
        default="cxl-memory-pooling",
        help="W&B project name (default: cxl-memory-pooling)",
    )
    ap.add_argument(
        "--reward-variant",
        choices=["current", "A", "B"],
        default="current",
        help="Reward function variant: 'current' (Δpeak+λVar), 'A' (local proj. peak), "
             "'B' (A + global stress term) (default: current)",
    )
    ap.add_argument(
        "--lookahead-window",
        type=int,
        default=200,
        help="Lookahead window W (ticks) for D_j departure relief (default: 200)",
    )
    ap.add_argument(
        "--reward-lambda",
        type=float,
        default=0.2,
        help="λ weight for reward B's global stress term (default: 0.2)",
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

    # ── W&B init ───────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_id,
            config=vars(args),
            sync_tensorboard=True,
        )
        _log(f"W&B run: {wandb_run.url}")

    # ── Save config ───────────────────────────────────────────────────
    config = vars(args).copy()
    config["wandb"] = not args.no_wandb
    config["timestamp"] = datetime.datetime.utcnow().isoformat()
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    _log(f"Config saved → {config_path}")

    # ── Load data ─────────────────────────────────────────────────────
    _log(f"Loading training trace: {args.trace} …")
    trace_data = to_arrays(load_trace(args.trace))

    M, num_hosts, num_pools = load_topology(args.topology)
    max_deg = max(sum(row) for row in M)
    _log(f"Topology: {num_hosts} hosts, {num_pools} MPDs, max_degree={max_deg}")

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
        _log(f"Augmentation: scale=[{args.aug_scale_lo}, {args.aug_scale_hi}] "
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
            normalized = [tn if tn.endswith(".sqlite") else tn + ".sqlite"
                          for tn in trace_names]

            _log(f"  Loading {len(normalized)} traces into pool …")
            trace_pool = []
            for stem in normalized:
                _log(f"    loading {stem} …")
                trace_pool.append(to_arrays(_lt(stem)))
            _log(f"  Loaded {len(trace_pool)} traces into pool")

    # ── Load multiple traces if --traces given ──────────────────────
    all_trace_data = []
    if args.traces:
        for tn in args.traces:
            if not tn.endswith(".sqlite"):
                tn = tn + ".sqlite"
            _log(f"Loading training trace: {tn} …")
            all_trace_data.append(to_arrays(load_trace(tn)))
        _log(f"  {len(all_trace_data)} training traces loaded")
    else:
        all_trace_data.append(trace_data)

    # ── Precompute topology-derived structures for new obs variants ────
    mhd_to_hosts = None
    Q_j = None
    if args.reward_variant != "current":
        import numpy as _np
        num_mhd = len(M[0])
        mhd_to_hosts = {j: [] for j in range(num_mhd)}
        host_to_mhds_tmp = {}
        for h, row in enumerate(M):
            host_to_mhds_tmp[h] = [j for j, v in enumerate(row) if v]
            for j in host_to_mhds_tmp[h]:
                mhd_to_hosts[j].append(h)
        Q_j = _np.zeros(num_mhd, dtype=_np.float32)
        for j in range(num_mhd):
            hosts = mhd_to_hosts[j]
            if hosts:
                inv_deg = [1.0 / len(host_to_mhds_tmp[h]) for h in hosts if host_to_mhds_tmp[h]]
                if inv_deg:
                    Q_j[j] = float(_np.mean(inv_deg))
        _log(f"Reward variant: {args.reward_variant} — mhd_to_hosts/Q_j precomputed")

    # ── Environments ──────────────────────────────────────────────────
    n_envs = args.n_envs
    env_fns = []
    for i in range(n_envs):
        td = all_trace_data[i % len(all_trace_data)]
        env_fns.append(_make_env(td, M, seed=args.seed + i,
                                 variance_lambda=args.variance_lambda,
                                 skip_hotfix=args.skip_hotfix,
                                 aug_config=aug_config,
                                 trace_pool=trace_pool,
                                 reward_variant=args.reward_variant,
                                 lookahead_window=args.lookahead_window,
                                 reward_lambda=args.reward_lambda))
    if n_envs > 1:
        train_env = SubprocVecEnv(env_fns)
        _log(f"SubprocVecEnv: {n_envs} parallel environments")
    else:
        train_env = DummyVecEnv(env_fns)

    _log(f"Loading eval trace: {args.eval_trace} …")
    eval_trace_raw  = load_trace(args.eval_trace)   # tuple — for PoolingSavingsCallback
    eval_trace_data = to_arrays(eval_trace_raw)     # arrays — for DummyVecEnv
    eval_env = DummyVecEnv(
        [_make_env(eval_trace_data, M, seed=args.seed + 10_000,
                   variance_lambda=args.variance_lambda,
                   skip_hotfix=args.skip_hotfix,
                   reward_variant=args.reward_variant,
                   lookahead_window=args.lookahead_window,
                   reward_lambda=args.reward_lambda)]
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    # SB3 callback freqs are in _on_step calls; with n_envs, each call = n_envs timesteps
    eff_ckpt_freq = max(1, args.checkpoint_freq // n_envs)
    eff_eval_freq = max(1, args.eval_freq // n_envs)

    checkpoint_cb = CheckpointCallback(
        save_freq=eff_ckpt_freq,
        save_path=save_dir,
        name_prefix=f"octopus_{args.algo}",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eff_eval_freq,
        n_eval_episodes=1 if args.fast else 3,
        deterministic=True,
    )
    pooling_cb = PoolingSavingsCallback(
        eval_trace_data=eval_trace_raw,
        M=M,
        max_degree=max_deg,
        eval_freq=args.eval_freq,
        n_iter=5 if args.fast else 10,
        obs_variant=args.reward_variant,
        mhd_to_hosts=mhd_to_hosts,
        Q_j=Q_j,
        lookahead_window=args.lookahead_window,
        eval_device=args.eval_device or None,
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
            train_freq=args.train_freq if args.train_freq is not None else (16 if args.fast else 1),
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

    # ── Build callback list ──────────────────────────────────────────
    callbacks = [checkpoint_cb, eval_cb, pooling_cb, EnvMetricsCallback()]
    if args.algo == "sac":
        callbacks.append(SACDiagnosticsCallback(log_freq=1000))
    if aug_config:
        callbacks.append(AugmentationLogCallback(log_freq=500))
    if wandb_run is not None:
        callbacks.append(WandbCallback(
            model_save_path=save_dir,
            model_save_freq=args.checkpoint_freq,
            verbose=0,
        ))

    # ── No-train mode: random policy baseline ───────────────────────
    if args.no_train:
        _log(f"--no-train: running random policy for {args.total_timesteps:,} steps …")
        obs = train_env.reset()
        rewards = []
        ep_reward = 0.0
        for step in range(args.total_timesteps):
            action = [train_env.action_space.sample() for _ in range(n_envs)]
            obs, reward, done, info = train_env.step(action)
            ep_reward += reward[0]
            if done[0]:
                rewards.append(ep_reward)
                ep_reward = 0.0
        if ep_reward != 0.0:
            rewards.append(ep_reward)
        rewards_arr = np.array(rewards) if rewards else np.array([0.0])
        _log(f"Random policy baseline ({len(rewards)} episodes):")
        _log(f"  mean reward: {rewards_arr.mean():.4f}")
        _log(f"  std reward:  {rewards_arr.std():.4f}")
        _log(f"  min reward:  {rewards_arr.min():.4f}")
        _log(f"  max reward:  {rewards_arr.max():.4f}")
        # Save results
        import json as _json
        result = {
            "run_id": args.run_id,
            "mode": "random_baseline",
            "total_timesteps": args.total_timesteps,
            "n_episodes": len(rewards),
            "mean_reward": float(rewards_arr.mean()),
            "std_reward": float(rewards_arr.std()),
            "min_reward": float(rewards_arr.min()),
            "max_reward": float(rewards_arr.max()),
        }
        result_path = os.path.join(save_dir, "random_baseline.json")
        with open(result_path, "w") as f:
            _json.dump(result, f, indent=2)
        _log(f"  Results saved → {result_path}")
        train_env.close()
        eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()
        return

    # ── Train ─────────────────────────────────────────────────────────
    _log(f"Training {args.algo.upper()} [{args.run_id}] for "
         f"{args.total_timesteps:,} timesteps …")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # ── Save final model ──────────────────────────────────────────────
    final_path = os.path.join(save_dir, f"octopus_{args.algo}_final")
    model.save(final_path)
    _log(f"Final model saved → {final_path}")

    train_env.close()
    eval_env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
