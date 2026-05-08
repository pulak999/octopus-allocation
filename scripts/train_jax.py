#!/usr/bin/env python3
"""Train a minimal JAX SAC agent on Octopus JAX env (current reward variant)."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from octopus.data import load_topology, load_trace, to_arrays
from octopus.jax.env import make_episode, reset_fn, step_fn
from octopus.jax.sac import ReplayBuffer, create_sac_state, update_sac


DEFAULT_TRACE = "AMS20PrdApp19-tround.sqlite"
DEFAULT_TOPOLOGY = "data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--reward-variant", default="current", choices=["current"])
    p.add_argument("--lookahead-window", type=int, default=200)
    p.add_argument("--reward-lambda", type=float, default=0.2)
    p.add_argument("--trace", default=DEFAULT_TRACE)
    p.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--train-freq", type=int, default=16)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--checkpoint-freq", type=int, default=10_000)
    p.add_argument("--log-freq", type=int, default=1_000)
    return p.parse_args()


def save_checkpoint(save_dir: str, step: int, sac_state, cfg: dict):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"step_{step}.pkl"), "wb") as f:
        pickle.dump(
            {
                "actor_params": sac_state.actor.params,
                "critic_params": sac_state.critic.params,
                "target_critic_params": sac_state.target_critic_params,
                "log_alpha": np.array(sac_state.log_alpha),
                "step": step,
            },
            f,
        )
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    args = parse_args()
    t0 = time.time()
    print(f"[jax] backend={jax.default_backend()} devices={jax.devices()}")

    raw = load_trace(args.trace)
    ta = to_arrays(raw)
    M, _, _ = load_topology(args.topology)
    M = np.asarray(M, dtype=np.int32)

    sc = make_episode(
        seed=args.seed,
        trace_arrays=ta,
        M=M,
        reward_variant=args.reward_variant,
        lookahead_window=args.lookahead_window,
        reward_lambda=args.reward_lambda,
        skip_hotfix=False,
    )
    obs_dim = sc.max_degree * 2 + 4
    action_dim = sc.max_degree
    print(f"[jax] obs_dim={obs_dim} action_dim={action_dim} num_events={sc.num_events}")

    state0, obs0 = reset_fn(sc)
    batched_state = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None], (args.n_envs,) + x.shape), state0
    )
    obs_batch = np.broadcast_to(np.array(obs0)[None], (args.n_envs, obs_dim)).copy()

    def _single_step(s, a):
        return step_fn(s, a, sc)

    batched_step = jax.jit(jax.vmap(_single_step, in_axes=(0, 0)))

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    sac_state = create_sac_state(
        init_key,
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
    )
    replay = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, capacity=args.buffer_size)
    np_rng = np.random.default_rng(args.seed + 123)

    save_dir = os.path.join("output", "checkpoints", args.run_id)
    cfg = {
        "trainer": "jax",
        "run_id": args.run_id,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "total_timesteps": args.total_timesteps,
        "reward_variant": args.reward_variant,
        "lookahead_window": args.lookahead_window,
        "reward_lambda": args.reward_lambda,
        "trace": args.trace,
        "topology": args.topology,
        "backend": jax.default_backend(),
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "alpha_lr": args.alpha_lr,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "effective_env_steps_per_update": args.n_envs * args.train_freq,
        "learning_starts": args.learning_starts,
    }
    save_checkpoint(save_dir, 0, sac_state, cfg)

    metrics_last = {}
    steps = 0
    while steps < args.total_timesteps:
        if steps < args.learning_starts:
            actions_np = np_rng.standard_normal((args.n_envs, action_dim)).astype(np.float32)
        else:
            key, act_key = jax.random.split(key)
            obs_j = jnp.asarray(obs_batch, dtype=jnp.float32)
            mean, log_std = sac_state.actor.apply_fn(sac_state.actor.params, obs_j)
            std = jnp.exp(log_std)
            noise = jax.random.normal(act_key, mean.shape)
            actions = jnp.tanh(mean + std * noise)
            actions_np = np.array(actions, dtype=np.float32)

        batched_state, next_obs_j, rewards_j, dones_j, _ = batched_step(
            batched_state, jnp.asarray(actions_np)
        )
        next_obs = np.array(next_obs_j)
        rewards = np.array(rewards_j, dtype=np.float32)
        dones = np.array(dones_j, dtype=np.float32)

        replay.add_batch(obs_batch, actions_np, rewards, next_obs, dones)

        obs_batch = next_obs
        steps += args.n_envs

        if bool(np.all(dones > 0.5)):
            batched_state = jax.tree.map(
                lambda x: jnp.broadcast_to(x[None], (args.n_envs,) + x.shape), state0
            )
            obs_batch = np.broadcast_to(np.array(obs0)[None], (args.n_envs, obs_dim)).copy()

        if steps >= args.learning_starts and steps % args.train_freq == 0 and replay.size >= args.batch_size:
            for _ in range(args.gradient_steps):
                batch = replay.sample(args.batch_size, np_rng)
                sac_state, metrics, key = update_sac(
                    sac_state,
                    batch,
                    key,
                    gamma=args.gamma,
                    tau=args.tau,
                )
                metrics_last = {k: float(v) for k, v in metrics.items()}

        if steps % args.log_freq == 0:
            fps = steps / max(time.time() - t0, 1e-6)
            base = f"[jax] step={steps}/{args.total_timesteps} fps={fps:.1f} replay={replay.size}"
            if metrics_last:
                base += (
                    f" critic={metrics_last['critic_loss']:.4f}"
                    f" actor={metrics_last['actor_loss']:.4f}"
                    f" alpha={metrics_last['alpha']:.4f}"
                    f" gN(c/a/alpha)="
                    f"{metrics_last['critic_grad_norm']:.3f}/"
                    f"{metrics_last['actor_grad_norm']:.3f}/"
                    f"{metrics_last['alpha_grad_norm']:.3f}"
                )
            print(base, flush=True)

        if steps % args.checkpoint_freq == 0:
            save_checkpoint(save_dir, steps, sac_state, cfg)

    save_checkpoint(save_dir, steps, sac_state, cfg)
    print(f"[jax] done: steps={steps} elapsed={time.time()-t0:.1f}s save_dir={save_dir}")


if __name__ == "__main__":
    main()

