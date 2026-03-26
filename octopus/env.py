"""
Gymnasium environment for Octopus CXL memory pooling.

Each episode replays VM arrivals for a single pod (randomly assigned from
the trace) over the full ~14-day trace window.  The agent allocates each
VM's CXL memory across the host's accessible MPDs.

**No migration**: once memory is allocated to an MPD it stays until the
VM terminates.

NOTE (CXL fraction): the current implementation puts 100 % of each VM's
``rss[1]`` memory on CXL, matching the notebook.  In production, only
~65 % of DRAM is on MPDs — flag this for a future parameter.
"""

from __future__ import annotations

import datetime as _dt
import random as _random
from datetime import timedelta

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class OctopusMemPoolEnv(gym.Env):
    """RL environment for CXL memory allocation in the Octopus architecture.

    Observation (size = 2 * max_degree + 4)
    ----------------------------------------
    - Accessible MPD loads (normalised), padded to *max_degree*
    - Accessibility mask (1 = valid, 0 = padding)
    - VM memory request (normalised)
    - Current global peak MPD load (normalised)
    - sin(hour-of-day), cos(hour-of-day)

    Action (size = max_degree)
    --------------------------
    Continuous logits in [-1, 1].  Softmax over valid (unmasked) entries
    yields allocation proportions; multiply by VM request to get GB.

    Reward
    ------
    ``-(peak_after_alloc - peak_before_alloc)``
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        all_vms,
        node_to_vms,
        node_to_machine,
        machine_sz,
        M,
        seed: int | None = None,
        variance_lambda: float = 0.5,
        skip_hotfix: bool = False,
        precomputed_events: dict | None = None,
        aug_config=None,
        trace_pool=None,
        reward_variant: str = "current",
        lookahead_window: int = 200,
        reward_lambda: float = 0.2,
    ):
        super().__init__()

        # Store trace data (read-only, shared across episodes)
        self.all_vms = all_vms
        self.node_to_vms = node_to_vms
        self.node_to_machine = node_to_machine
        self.machine_sz = machine_sz

        # Per-pod topology (base — never mutated by augmentation)
        self.M = [list(row) for row in M]          # keep as nested list
        self._M_np = np.array(M, dtype=np.int32)   # numpy copy for augmentation
        self.pod_size = len(M)
        self.num_mhd = len(M[0])
        self.mem_idx = 1                            # index into rss / machine_sz

        # Pre-compute per-host accessible MPD lists & max degree
        self.host_to_mhds: dict[int, list[int]] = {}
        for h in range(self.pod_size):
            self.host_to_mhds[h] = [
                j for j in range(self.num_mhd) if M[h][j] != 0
            ]
        self.max_degree = max(len(v) for v in self.host_to_mhds.values())
        assert self.max_degree > 0, "Topology has a host with no accessible MPDs"

        # Reward variant config
        assert reward_variant in ("current", "A", "B"), \
            f"reward_variant must be 'current', 'A', or 'B', got {reward_variant!r}"
        self.reward_variant = reward_variant
        self.lookahead_window = lookahead_window
        self.reward_lambda = reward_lambda

        # Spaces
        if reward_variant != "current":
            obs_dim = self.max_degree * 6 + 2
        else:
            obs_dim = self.max_degree * 2 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.max_degree,), dtype=np.float32
        )

        self.variance_lambda = variance_lambda
        self.skip_hotfix = skip_hotfix

        # Topology-derived: mhd_to_hosts and Q_j (recomputed after link failures)
        self._recompute_topology_derived()

        # Optional precomputed event cache {seed -> (events, pod_dur, pod_dram, base_time, pod_start_ts)}
        self._precomputed_events = precomputed_events

        # Augmentation
        self.aug_config = aug_config
        self._aug_rng = np.random.default_rng()

        # Multi-trace: list of (all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz)
        self.trace_pool = trace_pool

        # Seed management
        self._init_seed = seed
        self._seed = seed if seed is not None else 0
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)

        # Determine pod-assignment seed
        if seed is not None:
            self._seed = seed
        else:
            self._seed = (self._init_seed or 0) + self._episode_count
        self._episode_count += 1

        # 0. Multi-trace: swap to a random trace if enabled
        if (self.aug_config is not None and self.aug_config.multi_trace
                and self.trace_pool):
            trace_idx = int(self._aug_rng.integers(0, len(self.trace_pool)))
            self._switch_trace(self.trace_pool[trace_idx])

        # 1 & 2. Pod assignment + event timeline
        if self._precomputed_events is not None and self._seed in self._precomputed_events:
            self._load_precomputed(self._seed)
        else:
            self._generate_pod(self._seed)
            self._build_events()

        # 2b. Apply augmentation if configured
        self._current_aug_params = None
        if self.aug_config is not None and self.aug_config.enabled:
            self._apply_augmentation()

        # Reset host_to_mhds to base topology (augmentation may override below)
        if self._current_aug_params is None:
            # No augmentation — ensure host_to_mhds matches base topology
            for h in range(self.pod_size):
                self.host_to_mhds[h] = [
                    j for j in range(self.num_mhd) if self.M[h][j] != 0
                ]

        # 3. Simulation state
        self.cur_cxl_mem_vec = np.zeros(self.num_mhd, dtype=np.float64)
        self.dealloc_events = np.zeros(
            (self.pod_dur, self.num_mhd), dtype=np.float64
        )
        # Per-MPD active VM list: mpd_vm_allocs[j] = [(dealloc_tick, mem_gb), ...]
        self.mpd_vm_allocs: list[list[tuple[int, float]]] = [
            [] for _ in range(self.num_mhd)
        ]
        # Per-host CXL load tracking (for P_j neighborhood pressure)
        pod_size = len(self.pod_nodes) if hasattr(self, "pod_nodes") else self.pod_size
        self.cur_host_cxl_load = np.zeros(pod_size, dtype=np.float64)
        self.host_dealloc_events = np.zeros((self.pod_dur, pod_size), dtype=np.float64)
        self.max_peak = 0.0
        self.event_idx = 0
        self._last_depart_tick = -1                  # nothing processed yet
        self._cached_D_j: np.ndarray | None = None  # cached from last step(), used in _get_obs()

        # 4. Process departures up to first event's tick
        if self.events:
            first_tick = self.events[0][0]
            self._process_departures_through(first_tick)

        obs = self._get_obs()
        info = {
            "pod_seed": self._seed,
            "num_events": len(self.events),
        }
        if self._current_aug_params is not None:
            info["aug_params"] = self._current_aug_params
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        if self.event_idx >= len(self.events):
            return self._get_obs(), 0.0, True, False, {}

        tick, node_in_pod_id, vm_mem, dealloc_tick = self.events[self.event_idx]

        # --- Convert action → allocation --------------------------------
        mhd_list = self.host_to_mhds[node_in_pod_id]
        n_acc = len(mhd_list)

        raw = np.asarray(action[:n_acc], dtype=np.float64)
        raw = raw - raw.max()                        # numerical stability
        exp_raw = np.exp(raw)
        proportions = exp_raw / (exp_raw.sum() + 1e-12)

        alloc_gb = proportions * vm_mem              # GB per accessible MPD

        # --- Compute D_j before allocation (used by reward A/B and obs) -
        if self.reward_variant != "current":
            D_j_pre = self._compute_D_j(tick)
        else:
            D_j_pre = None

        # --- Apply allocation (no migration) ----------------------------
        old_peak = float(np.max(self.cur_cxl_mem_vec))

        for idx, mhd in enumerate(mhd_list):
            self.cur_cxl_mem_vec[mhd] += alloc_gb[idx]
            # Track per-MPD active VMs (for D_j / S_j computation)
            if alloc_gb[idx] > 0:
                self.mpd_vm_allocs[mhd].append((dealloc_tick, float(alloc_gb[idx])))

        # Track per-host CXL load (for P_j neighborhood pressure)
        self.cur_host_cxl_load[node_in_pod_id] += vm_mem

        # Schedule deallocation
        if dealloc_tick < self.pod_dur:
            for idx, mhd in enumerate(mhd_list):
                self.dealloc_events[dealloc_tick, mhd] += alloc_gb[idx]
            self.host_dealloc_events[dealloc_tick, node_in_pod_id] += vm_mem

        # --- Reward ------------------------------------------------------
        new_peak = float(np.max(self.cur_cxl_mem_vec))
        if self.reward_variant == "current":
            fair_share = self.pod_dram / self.num_mhd if self.num_mhd > 0 else 1.0
            mpd_loads = self.cur_cxl_mem_vec / (fair_share + 1e-12)
            load_variance = float(np.var(mpd_loads))
            reward = -(new_peak - old_peak) / (fair_share + 1e-12) - self.variance_lambda * load_variance
        else:
            norm = self.pod_dram if self.pod_dram > 0 else 1.0
            # ĉ_j+(t) = (c_j_post - D_j_pre) / D_pod  for j ∈ N(i)
            chat_plus = [
                (self.cur_cxl_mem_vec[mhd] - D_j_pre[mhd]) / norm
                for mhd in mhd_list
            ]
            reward_A = -float(max(chat_plus))
            if self.reward_variant == "A":
                reward = reward_A
            else:  # "B"
                # ĉ_j(t) = (c_j_pre - D_j_pre) / D_pod  for j ∉ N(i)
                # c_j_pre = cur_cxl_mem_vec[j] - alloc_gb for j ∈ N(i), unchanged for j ∉ N(i)
                mhd_set = set(mhd_list)
                unreachable = [j for j in range(self.num_mhd) if j not in mhd_set]
                if unreachable:
                    global_term = max(
                        (self.cur_cxl_mem_vec[j] - D_j_pre[j]) / norm
                        for j in unreachable
                    )
                else:
                    global_term = 0.0
                reward = reward_A - self.reward_lambda * float(global_term)
        self.max_peak = max(self.max_peak, new_peak)

        # --- Advance to next event --------------------------------------
        self.event_idx += 1
        done = self.event_idx >= len(self.events)

        if not done:
            next_tick = self.events[self.event_idx][0]
            if next_tick > tick:
                self._process_departures_through(next_tick)

        # --- Info -------------------------------------------------------
        info = {
            "peak": new_peak,
            "max_peak": self.max_peak,
            "vm_mem": vm_mem,
            "tick": tick,
        }
        if done:
            pooling_ratio = (
                (self.max_peak * self.num_mhd / self.pod_dram)
                if self.pod_dram > 0
                else 0.0
            )
            info["pooling_ratio"] = pooling_ratio
            info["pooling_savings"] = 1.0 - pooling_ratio

        return self._get_obs(), float(reward), done, False, info

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _generate_pod(self, seed: int):
        """Randomly select *pod_size* nodes for this episode's pod."""
        _random.seed(seed)
        node_list = [int(nid) for nid in self.node_to_vms.keys()]
        _random.shuffle(node_list)

        self.pod_nodes = node_list[: self.pod_size]
        self.node_to_pod_id = {n: i for i, n in enumerate(self.pod_nodes)}

        # Pod DRAM capacity
        self.pod_dram = 0.0
        for node in self.pod_nodes:
            cap = np.asarray(
                self.machine_sz[self.node_to_machine[node]], dtype=float
            )
            self.pod_dram += float(cap[self.mem_idx])

    # ------------------------------------------------------------------
    def _build_events(self):
        """Build a sorted list of VM arrival events for the current pod.

        Also applies the notebook's HOTFIX — filtering VMs that would
        overflow per-node physical memory.
        """
        # --- Time range --------------------------------------------------
        n_start = _dt.datetime.max
        n_end = _dt.datetime.min
        any_vm = False

        for node in self.pod_nodes:
            for vmkey in self.node_to_vms.get(node, []):
                vm = self.all_vms[vmkey]
                any_vm = True
                if vm.start_time < n_start:
                    n_start = vm.start_time
                if vm.end_time > n_end:
                    n_end = vm.end_time

        if not any_vm:
            self.events: list[tuple] = []
            self.pod_dur = 0
            self.base_time = _dt.datetime(2024, 1, 1)
            self.trace_start = self.base_time
            return

        self.base_time = _dt.datetime(
            n_start.year, n_start.month, n_start.day,
            n_start.hour, n_start.minute,
        )
        self.trace_start = n_start

        def to_tick(t):
            return int((t - self.base_time).total_seconds() // 300)

        pod_start_ts = to_tick(n_start)
        pod_end_ts = to_tick(n_end)
        self.pod_dur = pod_end_ts - pod_start_ts + 1
        self._pod_start_ts = pod_start_ts

        if self.pod_dur <= 0:
            self.events = []
            return

        # --- HOTFIX: filter VMs that would exceed per-node DRAM ----------
        vmkey_to_skip: set = set()

        if not self.skip_hotfix:
            for node in self.pod_nodes:
                node_alloc = [[] for _ in range(self.pod_dur)]
                node_dealloc = np.zeros(self.pod_dur, dtype=np.float64)
                node_cap = float(
                    np.asarray(
                        self.machine_sz[self.node_to_machine[node]], dtype=float
                    )[self.mem_idx]
                )

                for vmkey in self.node_to_vms.get(node, []):
                    vm = self.all_vms[vmkey]
                    vm_s = to_tick(vm.start_time) - pod_start_ts
                    vm_e = to_tick(vm.end_time) - pod_start_ts
                    if vm_e < 0:
                        vmkey_to_skip.add(vmkey)
                        continue
                    mem = float(np.asarray(vm.rss, dtype=float)[self.mem_idx])
                    node_alloc[vm_s].append((vmkey, vm_e + 1, mem))
                    if vm_e + 1 < self.pod_dur:
                        node_dealloc[vm_e + 1] += mem

                cur = 0.0
                for ts in range(self.pod_dur):
                    cur -= node_dealloc[ts]
                    if cur < 0:
                        cur = 0.0
                    for vmkey, end_ts, mem in node_alloc[ts]:
                        cur += mem
                        if cur > node_cap:
                            cur -= mem
                            vmkey_to_skip.add(vmkey)
                            if end_ts < self.pod_dur:
                                node_dealloc[end_ts] -= mem

        # --- Collect events -----------------------------------------------
        events: list[tuple] = []
        for node in self.pod_nodes:
            pod_id = self.node_to_pod_id[node]
            for vmkey in self.node_to_vms.get(node, []):
                if vmkey in vmkey_to_skip:
                    continue
                vm = self.all_vms[vmkey]
                vm_s = to_tick(vm.start_time) - pod_start_ts
                vm_e = to_tick(vm.end_time) - pod_start_ts
                if vm_e < 0:
                    continue  # VM ends before pod window — always skip
                mem = float(np.asarray(vm.rss, dtype=float)[self.mem_idx])
                if mem <= 0:
                    continue
                events.append((vm_s, pod_id, mem, vm_e + 1))

        # Deterministic ordering: tick → host → descending size
        events.sort(key=lambda e: (e[0], e[1], -e[2]))
        self.events = events

    # ------------------------------------------------------------------
    def _process_departures_through(self, tick: int):
        """Subtract deallocation events from ``_last_depart_tick + 1`` to
        *tick* (inclusive), then update ``_last_depart_tick``."""
        start = self._last_depart_tick + 1
        end = min(tick, self.pod_dur - 1)
        for t in range(start, end + 1):
            self.cur_cxl_mem_vec -= self.dealloc_events[t, :]
            self.cur_host_cxl_load -= self.host_dealloc_events[t, :]
        # Numerical safety (guard against tiny drift below zero)
        np.maximum(self.cur_cxl_mem_vec, 0.0, out=self.cur_cxl_mem_vec)
        np.maximum(self.cur_host_cxl_load, 0.0, out=self.cur_host_cxl_load)
        # Purge expired VM entries from per-MPD lists
        for j in range(self.num_mhd):
            if self.mpd_vm_allocs[j]:
                self.mpd_vm_allocs[j] = [
                    (dt, m) for dt, m in self.mpd_vm_allocs[j] if dt > tick
                ]
        self._last_depart_tick = tick

    # ------------------------------------------------------------------
    def _load_precomputed(self, seed: int):
        """Restore episode state from a precomputed cache entry."""
        events, pod_dur, pod_dram, base_time, pod_start_ts = self._precomputed_events[seed]
        self.events = events
        self.pod_dur = pod_dur
        self.pod_dram = pod_dram
        self.base_time = base_time
        self._pod_start_ts = pod_start_ts
        self.trace_start = base_time

        # Reconstruct host_to_mhds and node mapping from events so _get_obs works.
        # We still need pod_nodes / node_to_pod_id for topology info — regenerate them
        # cheaply (no event-building cost).
        self._generate_pod(seed)

    # ------------------------------------------------------------------
    def _compute_D_j(self, tick: int) -> np.ndarray:
        """D_j(t, W): time-weighted departure relief per MPD.

        D_j = sum over active VMs on MPD j departing within W steps of
              mem(v) * (1 - (end(v) - t) / W).

        Only considers VMs already in mpd_vm_allocs (pre-allocation for
        the current event). All entries have dealloc_tick > tick (purged
        by _process_departures_through).
        """
        D = np.zeros(self.num_mhd, dtype=np.float64)
        W = self.lookahead_window
        t_plus_W = tick + W
        for j in range(self.num_mhd):
            for dt, mem in self.mpd_vm_allocs[j]:
                if dt <= t_plus_W:
                    D[j] += mem * (1.0 - (dt - tick) / W)
        return D

    # ------------------------------------------------------------------
    def _recompute_topology_derived(self):
        """Build mhd_to_hosts (inverse of host_to_mhds) and Q_j (neighbor scarcity).

        Must be called after host_to_mhds is updated — at init and after link failures.

        Q_j = mean(1/deg(h) for h in mhd_to_hosts[j])
        A high Q_j means the hosts sharing MPD j have few alternatives, so MPD j
        is strategically scarce.
        """
        self.mhd_to_hosts: dict[int, list[int]] = {j: [] for j in range(self.num_mhd)}
        for h, mhds in self.host_to_mhds.items():
            for j in mhds:
                self.mhd_to_hosts[j].append(h)

        self.Q_j = np.zeros(self.num_mhd, dtype=np.float32)
        for j in range(self.num_mhd):
            hosts = self.mhd_to_hosts[j]
            if hosts:
                inv_degrees = [
                    1.0 / len(self.host_to_mhds[h])
                    for h in hosts
                    if len(self.host_to_mhds[h]) > 0
                ]
                if inv_degrees:
                    self.Q_j[j] = float(np.mean(inv_degrees))

    # ------------------------------------------------------------------
    def _apply_augmentation(self):
        """Apply augmentation transforms to the current episode's events and topology.

        Called from reset() after event construction. Modifies self.events and
        self.host_to_mhds in place. Retries up to max_resample_attempts if
        augmented episode has too few events.
        """
        from octopus.augmentation import sample_augmentation_params, apply_augmentation

        for _attempt in range(self.aug_config.max_resample_attempts):
            aug_params = sample_augmentation_params(self.aug_config, self._aug_rng)
            aug_events, aug_M = apply_augmentation(
                list(self.events), self._M_np, aug_params, self._aug_rng
            )
            if len(aug_events) >= self.aug_config.min_events:
                self.events = aug_events
                self._current_aug_params = aug_params
                # Recompute host_to_mhds from augmented topology
                for h in range(self.pod_size):
                    self.host_to_mhds[h] = [
                        j for j in range(self.num_mhd) if aug_M[h, j] != 0
                    ]
                self._recompute_topology_derived()
                return

        # All attempts failed guard — use unaugmented episode, reset topology
        self._current_aug_params = {
            "scale": 1.0, "noise_sigma": 0.0, "jitter": 0,
            "lifetime_frac": 0.0, "fail_ratio": 0.0,
        }
        for h in range(self.pod_size):
            self.host_to_mhds[h] = [
                j for j in range(self.num_mhd) if self.M[h][j] != 0
            ]
        self._recompute_topology_derived()

    # ------------------------------------------------------------------
    def _switch_trace(self, trace_data):
        """Swap trace data for multi-trace augmentation.

        trace_data: tuple (all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz)
        Precomputed event cache is NOT used with multi-trace (events built fresh).
        """
        self.all_vms = trace_data[0]
        self.node_to_vms = trace_data[1]
        self.node_to_machine = trace_data[2]
        self.machine_sz = trace_data[4]

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build the observation vector for the current event."""
        if self.event_idx >= len(self.events):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        tick, node_in_pod_id, vm_mem, _ = self.events[self.event_idx]
        mhd_list = self.host_to_mhds[node_in_pod_id]
        n_acc = len(mhd_list)
        norm = self.pod_dram if self.pod_dram > 0 else 1.0

        if self.reward_variant == "current":
            # --- Original 2*max_degree+4 obs ----------------------------
            loads = np.zeros(self.max_degree, dtype=np.float32)
            for idx, mhd in enumerate(mhd_list):
                loads[idx] = float(self.cur_cxl_mem_vec[mhd]) / norm

            mask = np.zeros(self.max_degree, dtype=np.float32)
            mask[:n_acc] = 1.0

            vm_norm = np.float32(vm_mem / norm)
            peak_norm = np.float32(float(np.max(self.cur_cxl_mem_vec)) / norm)

            abs_time = self.base_time + timedelta(
                minutes=(tick + self._pod_start_ts) * 5
            )
            hour = abs_time.hour + abs_time.minute / 60.0
            hour_sin = np.float32(np.sin(2.0 * np.pi * hour / 24.0))
            hour_cos = np.float32(np.cos(2.0 * np.pi * hour / 24.0))

            return np.concatenate(
                [loads, mask, np.array([vm_norm, peak_norm, hour_sin, hour_cos])]
            ).astype(np.float32)

        else:
            # --- New 6*max_degree+2 obs (variants A and B) --------------
            # D_j and S_j for all MPDs at current tick
            W = self.lookahead_window
            t_plus_W = tick + W
            D_j = np.zeros(self.num_mhd, dtype=np.float64)
            S_j = np.zeros(self.num_mhd, dtype=np.float64)
            for j in range(self.num_mhd):
                for dt, mem in self.mpd_vm_allocs[j]:
                    if dt <= t_plus_W:
                        D_j[j] += mem * (1.0 - (dt - tick) / W)
                    else:
                        S_j[j] += mem

            obs = np.zeros(self.max_degree * 6 + 2, dtype=np.float32)
            for k, mhd in enumerate(mhd_list):
                base_idx = k * 6
                obs[base_idx]     = float(self.cur_cxl_mem_vec[mhd]) / norm  # c_j/D_pod
                obs[base_idx + 1] = float(D_j[mhd]) / norm                   # D_j/D_pod
                obs[base_idx + 2] = float(S_j[mhd]) / norm                   # S_j/D_pod
                obs[base_idx + 3] = 1.0                                        # mask
                # P_j: sum of cur_host_cxl_load for connected hosts
                P_j = float(sum(
                    self.cur_host_cxl_load[h] for h in self.mhd_to_hosts[mhd]
                ))
                obs[base_idx + 4] = P_j / norm                                # P_j/D_pod
                obs[base_idx + 5] = float(self.Q_j[mhd])                      # Q_j

            # Global features
            obs[-2] = float(np.max(self.cur_cxl_mem_vec)) / norm  # global peak / D_pod
            obs[-1] = float(vm_mem) / norm                          # vm_mem / D_pod
            return obs
