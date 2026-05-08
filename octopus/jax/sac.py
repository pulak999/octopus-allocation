"""Minimal JAX/Flax SAC components for Octopus JAX trainer."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import train_state


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
ALPHA_TX = optax.adam(3e-4)


class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.relu(nn.Dense(self.hidden_dim)(obs))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class Critic(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray, act: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([obs, act], axis=-1)

        q1 = nn.relu(nn.Dense(self.hidden_dim)(x))
        q1 = nn.relu(nn.Dense(self.hidden_dim)(q1))
        q1 = nn.Dense(1)(q1).squeeze(-1)

        q2 = nn.relu(nn.Dense(self.hidden_dim)(x))
        q2 = nn.relu(nn.Dense(self.hidden_dim)(q2))
        q2 = nn.Dense(1)(q2).squeeze(-1)
        return q1, q2


@struct.dataclass
class SACState:
    actor: train_state.TrainState
    critic: train_state.TrainState
    target_critic_params: dict
    log_alpha: jnp.ndarray
    alpha_opt_state: optax.OptState


@struct.dataclass
class Batch:
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, actions, rewards, next_obs, dones):
        n = int(obs.shape[0])
        idx = (np.arange(n) + self.ptr) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.dones[idx] = dones
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        idx = rng.integers(0, self.size, size=batch_size)
        return Batch(
            obs=jnp.asarray(self.obs[idx]),
            actions=jnp.asarray(self.actions[idx]),
            rewards=jnp.asarray(self.rewards[idx]),
            next_obs=jnp.asarray(self.next_obs[idx]),
            dones=jnp.asarray(self.dones[idx]),
        )


def _sample_action_from_dist(mean: jnp.ndarray, log_std: jnp.ndarray, key: jax.Array):
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    pre_tanh = mean + std * eps
    action = jnp.tanh(pre_tanh)
    log_prob = -0.5 * (
        ((pre_tanh - mean) / (std + 1e-8)) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
    )
    log_prob = jnp.sum(log_prob, axis=-1)
    correction = jnp.sum(jnp.log(1.0 - action**2 + 1e-6), axis=-1)
    log_prob = log_prob - correction
    return action, log_prob


def _tree_l2_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    sq = [jnp.sum(jnp.square(x)) for x in leaves]
    return jnp.sqrt(jnp.sum(jnp.stack(sq)))


def create_sac_state(
    key: jax.Array,
    obs_dim: int,
    action_dim: int,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    hidden_dim: int = 256,
) -> SACState:
    actor_model = Actor(action_dim=action_dim, hidden_dim=hidden_dim)
    critic_model = Critic(hidden_dim=hidden_dim)
    key, k1, k2 = jax.random.split(key, 3)
    obs_dummy = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    act_dummy = jnp.zeros((1, action_dim), dtype=jnp.float32)

    actor_params = actor_model.init(k1, obs_dummy)
    critic_params = critic_model.init(k2, obs_dummy, act_dummy)

    actor_state = train_state.TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(actor_lr),
    )
    critic_state = train_state.TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=optax.adam(critic_lr),
    )
    alpha_tx = optax.adam(alpha_lr)
    log_alpha = jnp.array(0.0, dtype=jnp.float32)
    alpha_opt_state = alpha_tx.init(log_alpha)

    return SACState(
        actor=actor_state,
        critic=critic_state,
        target_critic_params=critic_params,
        log_alpha=log_alpha,
        alpha_opt_state=alpha_opt_state,
    )


@jax.jit
def select_actions(
    actor_params: dict,
    actor_apply_fn,
    obs: jnp.ndarray,
    key: jax.Array,
    deterministic: bool = False,
):
    mean, log_std = actor_apply_fn(actor_params, obs)
    if deterministic:
        return jnp.tanh(mean), key
    key, sub = jax.random.split(key)
    actions, _ = _sample_action_from_dist(mean, log_std, sub)
    return actions, key


@jax.jit
def update_sac(
    state: SACState,
    batch: Batch,
    key: jax.Array,
    *,
    gamma: float = 0.999,
    tau: float = 0.005,
    target_entropy: float | None = None,
) -> tuple[SACState, dict, jax.Array]:
    if target_entropy is None:
        target_entropy = -float(batch.actions.shape[-1])

    key, next_key, actor_key, alpha_key = jax.random.split(key, 4)

    alpha = jnp.exp(state.log_alpha)

    def critic_loss_fn(critic_params):
        next_mean, next_log_std = state.actor.apply_fn(state.actor.params, batch.next_obs)
        next_actions, next_logp = _sample_action_from_dist(next_mean, next_log_std, next_key)
        tq1, tq2 = state.critic.apply_fn(state.target_critic_params, batch.next_obs, next_actions)
        min_tq = jnp.minimum(tq1, tq2) - alpha * next_logp
        target_q = batch.rewards + (1.0 - batch.dones) * gamma * min_tq

        q1, q2 = state.critic.apply_fn(critic_params, batch.obs, batch.actions)
        loss = jnp.mean((q1 - target_q) ** 2 + (q2 - target_q) ** 2)
        return loss, (q1.mean(), q2.mean())

    (critic_loss, (q1_mean, q2_mean)), critic_grads = jax.value_and_grad(
        critic_loss_fn, has_aux=True
    )(state.critic.params)
    critic_grad_norm = _tree_l2_norm(critic_grads)
    critic_state = state.critic.apply_gradients(grads=critic_grads)

    def actor_loss_fn(actor_params):
        mean, log_std = state.actor.apply_fn(actor_params, batch.obs)
        actions, logp = _sample_action_from_dist(mean, log_std, actor_key)
        q1, q2 = critic_state.apply_fn(critic_state.params, batch.obs, actions)
        min_q = jnp.minimum(q1, q2)
        loss = jnp.mean(alpha * logp - min_q)
        return loss, logp.mean()

    (actor_loss, logp_mean), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        state.actor.params
    )
    actor_grad_norm = _tree_l2_norm(actor_grads)
    actor_state = state.actor.apply_gradients(grads=actor_grads)

    def alpha_loss_fn(log_alpha):
        alpha_val = jnp.exp(log_alpha)
        mean, log_std = actor_state.apply_fn(actor_state.params, batch.obs)
        _a, logp = _sample_action_from_dist(mean, log_std, alpha_key)
        return jnp.mean(alpha_val * (-logp - target_entropy))

    alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(state.log_alpha)
    alpha_grad_norm = jnp.abs(alpha_grads)
    updates, alpha_opt_state = ALPHA_TX.update(alpha_grads, state.alpha_opt_state, state.log_alpha)
    log_alpha = optax.apply_updates(state.log_alpha, updates)

    target_critic_params = optax.incremental_update(
        critic_state.params, state.target_critic_params, tau
    )

    new_state = SACState(
        actor=actor_state,
        critic=critic_state,
        target_critic_params=target_critic_params,
        log_alpha=log_alpha,
        alpha_opt_state=alpha_opt_state,
    )
    metrics = {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "alpha": jnp.exp(log_alpha),
        "q1_mean": q1_mean,
        "q2_mean": q2_mean,
        "logp_mean": logp_mean,
        "critic_grad_norm": critic_grad_norm,
        "actor_grad_norm": actor_grad_norm,
        "alpha_grad_norm": alpha_grad_norm,
    }
    return new_state, metrics, key

