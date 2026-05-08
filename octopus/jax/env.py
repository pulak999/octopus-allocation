"""
JAX environment for Octopus CXL memory pooling — pure-function design.

_mpd_dt/_mpd_mem/_mpd_n are NOT in OctopusState.  D_j and S_j are derived
from dealloc_buf via einsum: dealloc_buf[t, j] = total memory from VMs on MPD j
departing at tick t, so summing over t with appropriate weights is mathematically
identical to the Numba compute_D_S_j kernel.  This avoids MAX_ACTIVE_VMS overflow
(empirical max: 2007 total VM allocs per MPD per episode).

Static shapes: MAX_TICKS=2304, MAX_EVENTS=4096.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import random as _random

import chex
import jax
import jax.numpy as jnp
import numpy as np

MAX_TICKS: int = 2304
MAX_EVENTS: int = 4096  # empirical max 2407; assert in make_episode

_REWARD_CURRENT = 0
_REWARD_A = 1
_REWARD_B = 2

_TICK_SECONDS = 300  # 5 minutes per tick


@chex.dataclass
class OctopusState:
    """Mutable per-step simulation state; all fields are JAX arrays."""
    mpd_load: chex.Array          # (num_mhd,) float — current CXL load per MPD
    dealloc_buf: chex.Array       # (MAX_TICKS, num_mhd) float — scheduled MPD deallocations
    host_load: chex.Array         # (pod_size,) float — current CXL load per host
    host_dealloc_buf: chex.Array  # (MAX_TICKS, pod_size) float
    max_peak: chex.Array          # () float — running episode peak MPD load
    event_idx: chex.Array         # () int32 — index into static events array
    last_depart_tick: chex.Array  # () int32 — last tick processed for departures
    key: chex.Array               # (2,) uint32 — JAX PRNG key (reserved for future use)


@dataclasses.dataclass(frozen=True)
class StaticConfig:
    """Episode-static config; values are fixed per compiled JIT kernel."""
    # Event array: [tick, pod_id, vm_mem_gb, dealloc_tick], float32
    # Rows beyond num_events are sentinel rows with tick=-1 and all zeros.
    events: np.ndarray        # (MAX_EVENTS, 4) float32

    # Topology: padded with -1 beyond n_accessible[h] / n_connected[j]
    host_to_mhds: np.ndarray  # (pod_size, max_degree) int32
    n_accessible: np.ndarray  # (pod_size,) int32 — valid MPD count per host
    mhd_to_hosts: np.ndarray  # (num_mhd, max_conn) int32
    n_connected: np.ndarray   # (num_mhd,) int32 — connected host count per MPD

    Q_j: np.ndarray           # (num_mhd,) float32 — neighbor scarcity (static per topology)

    pod_dram: float           # total pod DRAM in GB (normalisation constant)
    num_events: int           # actual event count (≤ MAX_EVENTS)

    reward_variant: int       # _REWARD_CURRENT=0, _REWARD_A=1, _REWARD_B=2
    lookahead_window: int     # W — ticks ahead for D_j/S_j
    reward_lambda: float      # λ for reward B global term
    variance_lambda: float    # λ_var for reward current load-variance term

    pod_size: int
    num_mhd: int
    max_degree: int           # max MPDs accessible from any host
    max_conn: int             # max hosts connected to any MPD

    base_time_sec: int        # seconds since _EPOCH; for hour-of-day obs feature
    pod_start_ts: int         # episode's first tick offset (for absolute time)

    def __hash__(self) -> int:
        # JAX requires static args to be hashable.  Use object identity so each
        # StaticConfig instance compiles its own JIT kernel.  Within a training
        # loop, keep the same StaticConfig object across all steps of one episode.
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


# ---------------------------------------------------------------------------
# D_j / S_j helpers
# ---------------------------------------------------------------------------

def _compute_D_S_j(
    dealloc_buf: chex.Array,
    last_depart_tick,
    tick,
    W: int,
) -> tuple[chex.Array, chex.Array]:
    """D_j and S_j from dealloc_buf — no per-VM slot arrays needed.

    Proof of equivalence with Numba compute_D_S_j:
      dealloc_buf[t, j] = Σ_{VMs v on j with dealloc_tick_v = t} mem_v
      D_j[j] = Σ_{t ∈ (last_depart_tick, tick+W]} buf[t,j] · (1 − (t−tick)/W)
             = Σ_v mem_v · (1 − (dealloc_tick_v − tick)/W)  for active VMs with dt_v ≤ tick+W
      S_j[j] = Σ_{t > tick+W} buf[t,j]

    Parameters
    ----------
    dealloc_buf : (MAX_TICKS, num_mhd)
    last_depart_tick : scalar int — entries at t ≤ this have already been applied
    tick : scalar int — current event tick (usually equals last_depart_tick)
    W : int — lookahead window
    """
    dtype = dealloc_buf.dtype
    t = jnp.arange(MAX_TICKS, dtype=jnp.int32)

    near = (t > last_depart_tick) & (t <= tick + W)
    weight = jnp.where(
        near,
        jnp.ones(MAX_TICKS, dtype=dtype) - (t.astype(dtype) - jnp.asarray(tick, dtype)) / jnp.asarray(W, dtype),
        jnp.zeros(MAX_TICKS, dtype=dtype),
    )
    D_j = jnp.einsum("t,tj->j", weight, dealloc_buf)

    far = t > tick + W
    S_j = jnp.einsum("t,tj->j", far.astype(dtype), dealloc_buf)
    return D_j, S_j


def _compute_D_j(
    dealloc_buf: chex.Array,
    last_depart_tick,
    tick,
    W: int,
) -> chex.Array:
    """D_j only — used in step() before allocation (reward A/B pre-alloc term)."""
    dtype = dealloc_buf.dtype
    t = jnp.arange(MAX_TICKS, dtype=jnp.int32)
    near = (t > last_depart_tick) & (t <= tick + W)
    weight = jnp.where(
        near,
        jnp.ones(MAX_TICKS, dtype=dtype) - (t.astype(dtype) - jnp.asarray(tick, dtype)) / jnp.asarray(W, dtype),
        jnp.zeros(MAX_TICKS, dtype=dtype),
    )
    return jnp.einsum("t,tj->j", weight, dealloc_buf)


# ---------------------------------------------------------------------------
# JAX-side helpers (all pure, JIT-compilable)
# ---------------------------------------------------------------------------

def _process_departures(state: OctopusState, next_tick) -> OctopusState:
    """Subtract dealloc_buf entries from (last_depart_tick, next_tick] and advance pointer.

    When next_tick <= last_depart_tick, this is a no-op (mask is all False).
    """
    dtype = state.dealloc_buf.dtype
    t = jnp.arange(MAX_TICKS, dtype=jnp.int32)
    mask = (t > state.last_depart_tick) & (t <= next_tick)
    mask_f = mask.astype(dtype)

    delta_mpd  = jnp.einsum("t,tj->j", mask_f, state.dealloc_buf)
    delta_host = jnp.einsum("t,tp->p", mask_f, state.host_dealloc_buf)

    new_last = jnp.where(
        next_tick > state.last_depart_tick,
        next_tick,
        state.last_depart_tick,
    ).astype(jnp.int32)

    return state.replace(
        mpd_load=jnp.maximum(state.mpd_load - delta_mpd, 0.0),
        host_load=jnp.maximum(state.host_load - delta_host, 0.0),
        last_depart_tick=new_last,
    )


def _get_obs_fn(state: OctopusState, static_config: StaticConfig) -> chex.Array:
    """Build the observation vector for the current event (JIT-compilable).

    Returns zero vector when episode is done (event_idx >= num_events).
    Mirrors OctopusMemPoolEnv._get_obs() exactly.
    """
    md = static_config.max_degree
    num_events = static_config.num_events
    done = state.event_idx >= num_events

    safe_idx = jnp.minimum(state.event_idx, num_events - 1)
    events_jax = jnp.asarray(static_config.events)
    row = events_jax[safe_idx]
    tick = row[0].astype(jnp.int32)
    node_id = row[1].astype(jnp.int32)
    vm_mem = row[2]

    norm = jnp.asarray(static_config.pod_dram, dtype=state.mpd_load.dtype)
    norm_safe = norm + jnp.asarray(1e-12, dtype=norm.dtype)

    host_to_mhds_jax = jnp.asarray(static_config.host_to_mhds)
    accessible = host_to_mhds_jax[node_id]  # (max_degree,) int32
    valid_mask = (accessible >= 0)
    safe_accessible = jnp.where(valid_mask, accessible, 0)

    if static_config.reward_variant == _REWARD_CURRENT:
        obs_dim = md * 2 + 4
        # Accessible MPD loads normalised, padded
        loads = jnp.where(valid_mask, state.mpd_load[safe_accessible] / norm_safe,
                          jnp.zeros(md, dtype=state.mpd_load.dtype))
        mask_f = valid_mask.astype(jnp.float32)
        vm_norm = (vm_mem / norm_safe).astype(jnp.float32)
        peak_norm = (jnp.max(state.mpd_load) / norm_safe).astype(jnp.float32)

        # Hour-of-day features
        # Compute time features in float64 under x64 mode to match Gymnasium's
        # datetime truncation precisely; then cast down to float32 obs.
        time_dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
        abs_sec = (
            jnp.asarray(static_config.base_time_sec, dtype=time_dtype)
            + (tick.astype(time_dtype) + jnp.asarray(static_config.pod_start_ts, dtype=time_dtype))
            * jnp.asarray(_TICK_SECONDS, dtype=time_dtype)
        )
        hour = (
            (abs_sec % jnp.asarray(86400.0, dtype=time_dtype))
            / jnp.asarray(3600.0, dtype=time_dtype)
        )
        hour_sin = jnp.sin(2.0 * jnp.pi * hour / 24.0).astype(jnp.float32)
        hour_cos = jnp.cos(2.0 * jnp.pi * hour / 24.0).astype(jnp.float32)

        obs = jnp.concatenate([
            loads.astype(jnp.float32),
            mask_f,
            jnp.array([vm_norm, peak_norm, hour_sin, hour_cos], dtype=jnp.float32),
        ])
    else:
        obs_dim = md * 6 + 2
        W = static_config.lookahead_window
        D_j, S_j = _compute_D_S_j(state.dealloc_buf, state.last_depart_tick, tick, W)

        # Per-MPD features (vectorized over max_degree)
        c_j = jnp.where(valid_mask, state.mpd_load[safe_accessible] / norm_safe, 0.0)
        d_j = jnp.where(valid_mask, D_j[safe_accessible] / norm_safe, 0.0)
        s_j = jnp.where(valid_mask, S_j[safe_accessible] / norm_safe, 0.0)
        mask_f = valid_mask.astype(jnp.float32)

        # P_j: sum host_load over connected hosts for each accessible MPD
        mhd_to_hosts_jax = jnp.asarray(static_config.mhd_to_hosts)
        conn_block = mhd_to_hosts_jax[safe_accessible]      # (max_degree, max_conn) int32
        conn_valid = conn_block >= 0
        safe_conn = jnp.where(conn_valid, conn_block, 0)    # (max_degree, max_conn) int32
        host_gathered = state.host_load[safe_conn]           # (max_degree, max_conn) float
        P_j_raw = jnp.sum(jnp.where(conn_valid, host_gathered, 0.0), axis=1)  # (max_degree,)
        p_j = jnp.where(valid_mask, P_j_raw / norm_safe, 0.0)

        q_j_all = jnp.asarray(static_config.Q_j, dtype=state.mpd_load.dtype)
        q_j = jnp.where(valid_mask, q_j_all[safe_accessible], 0.0)

        # Stack: (max_degree, 6) → flatten → (max_degree*6,)
        per_mpd = jnp.stack([c_j, d_j, s_j, mask_f.astype(c_j.dtype), p_j, q_j], axis=1)
        global_peak = (jnp.max(state.mpd_load) / norm_safe).astype(jnp.float32)
        global_vm   = (vm_mem / norm_safe).astype(jnp.float32)
        obs = jnp.concatenate([
            per_mpd.reshape(-1).astype(jnp.float32),
            jnp.array([global_peak, global_vm], dtype=jnp.float32),
        ])

    return jnp.where(done, jnp.zeros(obs_dim, dtype=jnp.float32), obs)


def reset_fn(static_config: StaticConfig) -> tuple[OctopusState, chex.Array]:
    """Initialise JAX state from a pre-built StaticConfig (episode events already on host).

    Returns (OctopusState, obs).  The host must call make_episode() first to build
    StaticConfig (pod selection + event construction happen on the host).
    """
    dtype = jnp.float32
    state = OctopusState(
        mpd_load=jnp.zeros(static_config.num_mhd, dtype=dtype),
        dealloc_buf=jnp.zeros((MAX_TICKS, static_config.num_mhd), dtype=dtype),
        host_load=jnp.zeros(static_config.pod_size, dtype=dtype),
        host_dealloc_buf=jnp.zeros((MAX_TICKS, static_config.pod_size), dtype=dtype),
        max_peak=jnp.zeros((), dtype=dtype),
        event_idx=jnp.zeros((), dtype=jnp.int32),
        last_depart_tick=jnp.full((), -1, dtype=jnp.int32),
        key=jax.random.PRNGKey(0),
    )

    # Process departures up to the first event's tick (mirrors SB3 reset behaviour)
    if static_config.num_events > 0:
        first_tick = int(static_config.events[0, 0])
        state = _process_departures(state, jnp.asarray(first_tick, dtype=jnp.int32))

    obs = _get_obs_fn(state, static_config)
    return state, obs


def step_fn(
    state: OctopusState,
    action: chex.Array,
    static_config: StaticConfig,
) -> tuple[OctopusState, chex.Array, chex.Array, chex.Array, dict]:
    """Pure JAX step — mirrors OctopusMemPoolEnv.step() exactly.

    JIT-compile with static_argnums=2:
        jit_step = jax.jit(step_fn, static_argnums=2)

    Returns (new_state, obs, reward, done, info).
    info is always an empty dict (use state fields for metrics).
    """
    num_events = static_config.num_events
    done_before = state.event_idx >= num_events

    # Safe gather (clamped so OOB reads return a valid row)
    safe_idx = jnp.minimum(state.event_idx, num_events - 1)
    events_jax = jnp.asarray(static_config.events)
    row = events_jax[safe_idx]
    tick         = row[0].astype(jnp.int32)
    node_id      = row[1].astype(jnp.int32)
    vm_mem       = row[2].astype(state.mpd_load.dtype)
    dealloc_tick_raw = row[3].astype(jnp.int32)
    # Clip OOB dealloc ticks to MAX_TICKS-1 (one past last event tick is still within array)
    dealloc_tick = jnp.clip(dealloc_tick_raw, 0, MAX_TICKS - 1)

    W    = static_config.lookahead_window
    norm = jnp.asarray(static_config.pod_dram, dtype=state.mpd_load.dtype)
    norm_safe = norm + jnp.asarray(1e-12, dtype=norm.dtype)
    md   = static_config.max_degree

    # Accessible MPDs for this host
    host_to_mhds_jax = jnp.asarray(static_config.host_to_mhds)
    accessible  = host_to_mhds_jax[node_id]               # (max_degree,) int32
    valid_mask  = (accessible >= 0)                        # (max_degree,) bool
    safe_acc    = jnp.where(valid_mask, accessible, 0)    # (max_degree,) int32

    # Action → softmax proportions over valid MPDs
    logits = action[:md].astype(state.mpd_load.dtype)
    logits_masked  = jnp.where(valid_mask, logits, jnp.asarray(-1e9, dtype=logits.dtype))
    logits_stable  = logits_masked - jnp.max(logits_masked)
    exp_logits     = jnp.exp(logits_stable) * valid_mask.astype(logits.dtype)
    proportions    = exp_logits / (exp_logits.sum() + jnp.asarray(1e-12, dtype=exp_logits.dtype))
    alloc_per_slot = proportions * vm_mem                  # (max_degree,) — GB per slot

    # D_j pre-allocation (reward A/B only; Python if is fine since reward_variant is static)
    if static_config.reward_variant != _REWARD_CURRENT:
        D_j_pre = _compute_D_j(state.dealloc_buf, state.last_depart_tick, tick, W)
    else:
        D_j_pre = None

    # Old peak (before allocation)
    old_peak = jnp.max(state.mpd_load)

    # --- Apply allocation --------------------------------------------------
    contrib = jnp.where(valid_mask, alloc_per_slot, jnp.zeros_like(alloc_per_slot))

    # VMs whose original dealloc_tick >= MAX_TICKS never depart within the episode.
    # Matching SB3 behaviour: add 0 to dealloc_buf for those VMs so they stay
    # allocated but don't generate a future departure event.
    in_window = (dealloc_tick_raw < MAX_TICKS)
    dealloc_contrib = jnp.where(in_window, contrib, jnp.zeros_like(contrib))

    mpd_load_new = state.mpd_load.at[safe_acc].add(contrib)
    dealloc_buf_new = state.dealloc_buf.at[dealloc_tick, safe_acc].add(dealloc_contrib)

    host_load_new = state.host_load.at[node_id].add(vm_mem)
    host_dealloc_buf_new = state.host_dealloc_buf.at[dealloc_tick, node_id].add(
        jnp.where(in_window, vm_mem, jnp.zeros_like(vm_mem))
    )

    new_peak    = jnp.max(mpd_load_new)
    max_peak_new = jnp.maximum(state.max_peak, new_peak)

    # --- Compute reward ----------------------------------------------------
    rv = static_config.reward_variant   # static → Python if branches resolved at trace time
    finfo_min = jnp.asarray(jnp.finfo(state.mpd_load.dtype).min, dtype=state.mpd_load.dtype)

    if rv == _REWARD_CURRENT:
        fair_share    = norm_safe / jnp.asarray(static_config.num_mhd, dtype=norm.dtype)
        mpd_loads_norm = mpd_load_new / fair_share
        load_variance  = jnp.var(mpd_loads_norm)
        reward = (
            -(new_peak - old_peak) / fair_share
            - jnp.asarray(static_config.variance_lambda, dtype=state.mpd_load.dtype) * load_variance
        )
    elif rv == _REWARD_A:
        chat_plus = jnp.where(
            valid_mask,
            (mpd_load_new[safe_acc] - D_j_pre[safe_acc]) / norm_safe,
            finfo_min,
        )
        reward = -jnp.max(chat_plus)
    else:  # _REWARD_B
        chat_plus = jnp.where(
            valid_mask,
            (mpd_load_new[safe_acc] - D_j_pre[safe_acc]) / norm_safe,
            finfo_min,
        )
        reward_A = -jnp.max(chat_plus)

        # Reachable mask over all num_mhd MPDs for this host
        mpd_ids  = jnp.arange(static_config.num_mhd, dtype=jnp.int32)
        reachable = jnp.any(
            (accessible[:, None] == mpd_ids[None, :]) & valid_mask[:, None],
            axis=0,
        )  # (num_mhd,) bool
        unreachable = ~reachable
        global_vals = jnp.where(
            unreachable,
            (mpd_load_new - D_j_pre) / norm_safe,
            finfo_min,
        )
        has_unreach  = jnp.any(unreachable)
        global_term  = jnp.where(has_unreach, jnp.max(global_vals), jnp.zeros((), dtype=state.mpd_load.dtype))
        reward = reward_A - jnp.asarray(static_config.reward_lambda, dtype=state.mpd_load.dtype) * global_term

    # --- Advance event index and process departures -----------------------
    new_event_idx = state.event_idx + 1
    done = new_event_idx >= num_events

    # Next event tick (clamped so we don't read OOB at episode end)
    safe_next = jnp.minimum(new_event_idx, num_events - 1)
    next_tick  = events_jax[safe_next, 0].astype(jnp.int32)
    next_tick  = jnp.where(done, tick, next_tick)           # no-op if done

    # Process departures only when moving to a later tick
    should_dep   = (~done) & (next_tick > tick)
    depart_target = jnp.where(should_dep, next_tick, state.last_depart_tick)

    new_state = state.replace(
        mpd_load=mpd_load_new,
        dealloc_buf=dealloc_buf_new,
        host_load=host_load_new,
        host_dealloc_buf=host_dealloc_buf_new,
        max_peak=max_peak_new,
        event_idx=new_event_idx,
    )
    new_state = _process_departures(new_state, depart_target)

    obs = _get_obs_fn(new_state, static_config)

    # Zero out reward and obs if the episode was already done before this call
    reward_out = jnp.where(done_before, jnp.zeros((), dtype=jnp.float32), reward.astype(jnp.float32))
    obs_out    = jnp.where(done_before, jnp.zeros_like(obs), obs)

    return new_state, obs_out, reward_out, done, {}


# ---------------------------------------------------------------------------
# Vectorised rollout (Chunk C)
# ---------------------------------------------------------------------------

def rollout_fn(
    state: OctopusState,
    actions: chex.Array,
    static_config: StaticConfig,
) -> tuple[OctopusState, tuple]:
    """lax.scan rollout for a single env over H steps.

    Parameters
    ----------
    state   : OctopusState — initial state
    actions : (H, max_degree) float32 — pre-generated action sequence
    static_config : StaticConfig — episode constants (captured at JIT time)

    Returns
    -------
    final_state : OctopusState
    transitions : (obs, reward, done) each with leading H axis
    """
    def body(carry: OctopusState, action: chex.Array):
        new_state, obs, reward, done, _ = step_fn(carry, action, static_config)
        return new_state, (obs, reward, done)

    return jax.lax.scan(body, state, actions)


def make_rollout_fn(static_config: StaticConfig):
    """Factory: returns a JIT+vmap'd rollout function closed over static_config.

    The returned function signature:
        fn(batched_state, batched_actions) -> (batched_final_state, transitions)

    where:
        batched_state   : OctopusState with leading n_envs axis (stack reset_fn outputs)
        batched_actions : (n_envs, H, max_degree) float32
        transitions     : tuple of (obs, reward, done) each (n_envs, H, ...)

    Call once per StaticConfig; reuse across all rollout chunks in a training loop.
    """
    def _single(state: OctopusState, actions: chex.Array):
        return rollout_fn(state, actions, static_config)

    return jax.jit(jax.vmap(_single))


# ---------------------------------------------------------------------------
# Host-side episode builders (pure Python / NumPy — not JAX)
# ---------------------------------------------------------------------------

def _host_generate_pod(seed: int, trace_arrays, pod_size: int):
    """Select pod_size nodes using the same stdlib-random shuffle as SB3.

    Returns (pod_indices, pod_dram) where pod_indices are indices into trace_arrays.
    """
    n_nodes = len(trace_arrays.node_ids)
    _random.seed(seed)
    indices = list(range(n_nodes))
    _random.shuffle(indices)
    pod_indices = indices[:pod_size]
    pod_dram = float(trace_arrays.node_dram[pod_indices].sum())
    return pod_indices, pod_dram


def _host_build_events(
    trace_arrays,
    pod_indices: list[int],
    pod_size: int,
    skip_hotfix: bool = False,
) -> tuple[list, int, float, _dt.datetime, int]:
    """Build sorted VM arrival events for the given pod — mirrors SB3 _build_events.

    Returns (events, pod_dur, pod_dram, base_time, pod_start_ts) where:
        events : list of (tick, node_in_pod_id, vm_mem_gb, dealloc_tick) tuples
        pod_dur : int — episode duration in ticks
        pod_dram : float — total pod DRAM in GB
        base_time : datetime — epoch for hour-of-day features
        pod_start_ts : int — first tick offset (always 0 in practice)
    """
    from octopus.data import _EPOCH
    ta = trace_arrays

    pod_dram = float(ta.node_dram[pod_indices].sum())

    # Gather VM ranges per pod node
    node_vm_ranges = []
    for k, ni in enumerate(pod_indices):
        lo = int(ta.node_offsets[ni])
        hi = int(ta.node_offsets[ni + 1])
        if hi > lo:
            node_vm_ranges.append((k, ni, lo, hi))

    if not node_vm_ranges:
        base_time = _dt.datetime(2024, 1, 1)
        return [], 0, pod_dram, base_time, 0

    all_vm_idx = np.concatenate([ta.vm_ptrs[lo:hi] for _, _, lo, hi in node_vm_ranges])
    min_start_sec = int(ta.vm_start[all_vm_idx].min())

    base_sec = min_start_sec - (min_start_sec % 60)
    TICK = 300

    def to_tick(t_sec: int) -> int:
        return int((t_sec - base_sec) // TICK)

    pod_start_ts = to_tick(min_start_sec)
    max_end_sec  = int(ta.vm_end[all_vm_idx].max())
    pod_end_ts   = to_tick(max_end_sec)
    pod_dur      = pod_end_ts - pod_start_ts + 1
    base_time    = _EPOCH + _dt.timedelta(seconds=base_sec)

    if pod_dur <= 0:
        return [], 0, pod_dram, base_time, pod_start_ts

    skip_set: set[int] = set()
    if not skip_hotfix:
        for k, ni, lo, hi in node_vm_ranges:
            vms = ta.vm_ptrs[lo:hi]
            node_cap = float(ta.node_dram[ni])
            node_alloc = [[] for _ in range(pod_dur)]
            node_dealloc = np.zeros(pod_dur, dtype=np.float64)
            for vm_i in vms:
                vm_i = int(vm_i)
                vm_s = to_tick(int(ta.vm_start[vm_i])) - pod_start_ts
                vm_e = to_tick(int(ta.vm_end[vm_i]))   - pod_start_ts
                if vm_e < 0:
                    skip_set.add(vm_i)
                    continue
                mem = float(ta.vm_mem[vm_i])
                node_alloc[max(vm_s, 0)].append((vm_i, vm_e + 1, mem))
                if vm_e + 1 < pod_dur:
                    node_dealloc[vm_e + 1] += mem
            cur = 0.0
            for ts in range(pod_dur):
                cur -= node_dealloc[ts]
                if cur < 0:
                    cur = 0.0
                for vm_i, end_ts, mem in node_alloc[ts]:
                    cur += mem
                    if cur > node_cap:
                        cur -= mem
                        skip_set.add(vm_i)
                        if end_ts < pod_dur:
                            node_dealloc[end_ts] -= mem

    events: list[tuple] = []
    for k, ni, lo, hi in node_vm_ranges:
        for vm_i in ta.vm_ptrs[lo:hi]:
            vm_i = int(vm_i)
            if vm_i in skip_set:
                continue
            vm_s = to_tick(int(ta.vm_start[vm_i])) - pod_start_ts
            vm_e = to_tick(int(ta.vm_end[vm_i]))   - pod_start_ts
            if vm_e < 0:
                continue
            mem = float(ta.vm_mem[vm_i])
            if mem <= 0:
                continue
            events.append((vm_s, k, mem, vm_e + 1))

    events.sort(key=lambda e: (e[0], e[1], -e[2]))
    return events, pod_dur, pod_dram, base_time, pod_start_ts


def build_static_config(
    events: list[tuple],
    pod_dram: float,
    base_time: _dt.datetime,
    pod_start_ts: int,
    M,                      # (pod_size, num_mhd) array-like (adjacency matrix)
    reward_variant: str,
    lookahead_window: int,
    reward_lambda: float,
    variance_lambda: float = 0.5,
) -> StaticConfig:
    """Convert host-side episode data into a JAX-compatible StaticConfig.

    Parameters
    ----------
    events : list of (tick, node_in_pod_id, vm_mem_gb, dealloc_tick) tuples
    M : (pod_size, num_mhd) adjacency matrix (0 or 1)
    """
    from octopus.data import _EPOCH

    assert reward_variant in ("current", "A", "B"), \
        f"reward_variant must be 'current', 'A', or 'B', got {reward_variant!r}"
    rv_int = {"current": _REWARD_CURRENT, "A": _REWARD_A, "B": _REWARD_B}[reward_variant]

    M_np = np.array(M, dtype=np.int32)
    pod_size = M_np.shape[0]
    num_mhd  = M_np.shape[1]

    # Build topology arrays
    host_to_mhds_list = [
        [j for j in range(num_mhd) if M_np[h, j] != 0]
        for h in range(pod_size)
    ]
    max_degree = max((len(v) for v in host_to_mhds_list), default=1)
    n_accessible = np.array([len(v) for v in host_to_mhds_list], dtype=np.int32)
    host_to_mhds_np = np.full((pod_size, max_degree), -1, dtype=np.int32)
    for h, mhds in enumerate(host_to_mhds_list):
        host_to_mhds_np[h, :len(mhds)] = mhds

    mhd_to_hosts_list: list[list[int]] = [[] for _ in range(num_mhd)]
    for h, mhds in enumerate(host_to_mhds_list):
        for j in mhds:
            mhd_to_hosts_list[j].append(h)
    max_conn = max((len(v) for v in mhd_to_hosts_list), default=1)
    n_connected = np.array([len(v) for v in mhd_to_hosts_list], dtype=np.int32)
    mhd_to_hosts_np = np.full((num_mhd, max_conn), -1, dtype=np.int32)
    for j, hosts in enumerate(mhd_to_hosts_list):
        mhd_to_hosts_np[j, :len(hosts)] = hosts

    # Q_j: mean(1/deg(h)) for connected hosts
    Q_j = np.zeros(num_mhd, dtype=np.float32)
    for j in range(num_mhd):
        hosts = mhd_to_hosts_list[j]
        if hosts:
            inv_degs = [1.0 / len(host_to_mhds_list[h]) for h in hosts
                        if len(host_to_mhds_list[h]) > 0]
            if inv_degs:
                Q_j[j] = float(np.mean(inv_degs))

    # Pad events to (MAX_EVENTS, 4) with sentinel rows (tick = -1)
    assert len(events) <= MAX_EVENTS, \
        f"num_events={len(events)} exceeds MAX_EVENTS={MAX_EVENTS}"
    events_np = np.full((MAX_EVENTS, 4), -1.0, dtype=np.float32)
    if events:
        events_arr = np.array(events, dtype=np.float32)
        events_np[:len(events)] = events_arr

    base_time_sec = int((base_time - _EPOCH).total_seconds())

    return StaticConfig(
        events=events_np,
        host_to_mhds=host_to_mhds_np,
        n_accessible=n_accessible,
        mhd_to_hosts=mhd_to_hosts_np,
        n_connected=n_connected,
        Q_j=Q_j,
        pod_dram=pod_dram,
        num_events=len(events),
        reward_variant=rv_int,
        lookahead_window=lookahead_window,
        reward_lambda=reward_lambda,
        variance_lambda=variance_lambda,
        pod_size=pod_size,
        num_mhd=num_mhd,
        max_degree=max_degree,
        max_conn=max_conn,
        base_time_sec=base_time_sec,
        pod_start_ts=pod_start_ts,
    )


def make_episode(
    seed: int,
    trace_arrays,
    M,
    *,
    reward_variant: str = "current",
    lookahead_window: int = 200,
    reward_lambda: float = 0.2,
    variance_lambda: float = 0.5,
    skip_hotfix: bool = False,
    precomputed: dict | None = None,
) -> StaticConfig:
    """Build a StaticConfig for one episode (host-side; pure Python/NumPy).

    Pod selection uses stdlib random.seed(seed) so that the seed → pod mapping
    is identical to OctopusMemPoolEnv.reset(seed=seed).

    Parameters
    ----------
    precomputed : optional dict {seed → (events, pod_dur, pod_dram, base_time, pod_start_ts)}
        If provided and seed is found, skip pod generation and event building.
    """
    M_np = np.array(M, dtype=np.int32)
    pod_size = M_np.shape[0]

    if precomputed is not None and seed in precomputed:
        events, _pod_dur, pod_dram, base_time, pod_start_ts = precomputed[seed]
    else:
        pod_indices, pod_dram = _host_generate_pod(seed, trace_arrays, pod_size)
        events, _pod_dur, pod_dram, base_time, pod_start_ts = _host_build_events(
            trace_arrays, pod_indices, pod_size, skip_hotfix=skip_hotfix
        )

    return build_static_config(
        events=events,
        pod_dram=pod_dram,
        base_time=base_time,
        pod_start_ts=pod_start_ts,
        M=M_np,
        reward_variant=reward_variant,
        lookahead_window=lookahead_window,
        reward_lambda=reward_lambda,
        variance_lambda=variance_lambda,
    )
