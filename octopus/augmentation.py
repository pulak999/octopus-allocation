"""
Episode-level data augmentation for Octopus CXL memory pooling.

Transforms are applied during env.reset() after event construction.
They do not touch the step logic, reward, observation space, or action space.

Safe augmentation ranges (from data_aug.tex HOTFIX survival analysis):
  scale [0.7, 1.2] — beyond 1.2, excessive VM dropout produces sparse episodes.
  memory_noise_sigma <= 0.03 — must be << scale range width to avoid overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AugmentationConfig:
    """Episode-level augmentation configuration.

    Each field controls one augmentation knob. Set to None/0/False to disable.
    """

    # Memory scaling: uniform factor applied to all VM rss[1] in the episode
    scale_range: Tuple[float, float] = (0.7, 1.2)
    scale_distribution: str = "triangular"  # "triangular" or "uniform"

    # Per-VM memory noise: multiplicative noise on top of episode-level scale
    # sigma=0.03 means +/-3%. Must be << scale_range width.
    memory_noise_sigma: float = 0.0

    # Arrival jitter: max tick shift (+/-) applied to VM start times
    arrival_jitter_ticks: int = 0  # 0 = disabled; 1-2 recommended

    # Lifetime perturbation: fractional noise on positive-lifetime VMs only
    lifetime_noise_frac: float = 0.0  # 0.0 = disabled; 0.05 = +/-5%

    # Topology perturbation: fraction of CXL links to randomly remove
    link_failure_ratio: float = 0.0  # 0.0 = disabled; 0.01-0.05 recommended

    # Multi-trace: whether to sample traces randomly each episode
    multi_trace: bool = False

    # Validity guards
    min_events: int = 100  # reject episodes with fewer events
    max_resample_attempts: int = 5  # retry augmentation if guards fail

    # Master switch
    enabled: bool = True


def sample_augmentation_params(
    config: AugmentationConfig, rng: np.random.Generator
) -> dict:
    """Sample concrete augmentation parameters from config ranges.

    Returns a dict of parameters for one episode:
        scale, noise_sigma, jitter, lifetime_frac, fail_ratio
    """
    if not config.enabled:
        return {
            "scale": 1.0,
            "noise_sigma": 0.0,
            "jitter": 0,
            "lifetime_frac": 0.0,
            "fail_ratio": 0.0,
        }

    # Memory scale
    lo, hi = config.scale_range
    if config.scale_distribution == "triangular":
        scale = float(rng.triangular(lo, 1.0, hi))
    else:
        scale = float(rng.uniform(lo, hi))

    return {
        "scale": scale,
        "noise_sigma": config.memory_noise_sigma,
        "jitter": config.arrival_jitter_ticks,
        "lifetime_frac": config.lifetime_noise_frac,
        "fail_ratio": config.link_failure_ratio,
    }


# ---------------------------------------------------------------------------
# Transform functions
# ---------------------------------------------------------------------------

def scale_memory(events: list, factor: float) -> list:
    """Scale all VM memory requests by a uniform factor.

    events: list of (tick, host_id, vm_mem, dealloc_tick)
    Returns new list with vm_mem *= factor.
    """
    if factor == 1.0:
        return events
    return [(tick, host, mem * factor, dealloc) for tick, host, mem, dealloc in events]


def add_memory_noise(events: list, sigma: float, rng: np.random.Generator) -> list:
    """Add per-VM multiplicative noise to memory values.

    sigma: noise magnitude (0.03 = +/-3%)
    Applied after episode-level scaling.
    """
    if sigma <= 0.0:
        return events
    noisy = []
    for tick, host, mem, dealloc in events:
        noise = 1.0 + rng.uniform(-sigma, sigma)
        noisy.append((tick, host, mem * max(noise, 0.01), dealloc))
    return noisy


def jitter_arrivals(
    events: list, max_shift: int, rng: np.random.Generator
) -> list:
    """Perturb VM arrival ticks by +/-max_shift while preserving non-negative ticks.

    Dealloc tick is shifted by the same amount to preserve lifetime.
    """
    if max_shift == 0:
        return events
    jittered = []
    for tick, host, mem, dealloc in events:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        new_tick = max(0, tick + shift)
        new_dealloc = max(new_tick + 1, dealloc + shift)
        jittered.append((new_tick, host, mem, new_dealloc))
    return jittered


def perturb_lifetimes(
    events: list, frac: float, rng: np.random.Generator
) -> list:
    """Perturb VM lifetimes by +/-frac fraction.

    Only applied to VMs with positive lifetime (dealloc > tick).
    """
    if frac <= 0.0:
        return events
    perturbed = []
    for tick, host, mem, dealloc in events:
        lifetime = dealloc - tick
        if lifetime > 0:
            noise = rng.uniform(-frac, frac)
            new_lifetime = max(1, int(lifetime * (1.0 + noise)))
            perturbed.append((tick, host, mem, tick + new_lifetime))
        else:
            perturbed.append((tick, host, mem, dealloc))
    return perturbed


def inject_link_failures(
    M: np.ndarray, ratio: float, rng: np.random.Generator
) -> np.ndarray:
    """Randomly remove CXL links from the topology matrix.

    M: (num_hosts, num_mpds) adjacency matrix (numpy array)
    ratio: fraction of links to remove

    Safety: never disconnect a host from ALL its MPDs.
    Returns a copy of M with some links zeroed out.
    """
    if ratio <= 0.0:
        return M

    M_aug = M.copy()
    rows, cols = np.nonzero(M_aug)
    n_links = len(rows)
    if n_links == 0:
        return M_aug

    n_remove = max(1, int(n_links * ratio))

    candidates = rng.permutation(n_links)
    removed = 0
    for idx in candidates:
        if removed >= n_remove:
            break
        r, c = rows[idx], cols[idx]
        if M_aug[r, :].sum() <= 1:
            continue
        M_aug[r, c] = 0
        removed += 1

    return M_aug


def apply_augmentation(
    events: list,
    M_np: np.ndarray,
    aug_params: dict,
    rng: np.random.Generator,
) -> tuple:
    """Apply all augmentation transforms to an event list.

    Parameters
    ----------
    events : list of (tick, host_id, vm_mem, dealloc_tick)
    M_np : numpy array, shape (num_hosts, num_mpds)
    aug_params : dict from sample_augmentation_params()
    rng : numpy random Generator

    Returns
    -------
    (augmented_events, augmented_M_np)
    Events are re-sorted after transforms.
    """
    # 1. Scale memory (episode-level)
    events = scale_memory(events, aug_params["scale"])

    # 2. Per-VM memory noise
    if aug_params["noise_sigma"] > 0:
        events = add_memory_noise(events, aug_params["noise_sigma"], rng)

    # 3. Jitter arrivals
    if aug_params["jitter"] > 0:
        events = jitter_arrivals(events, aug_params["jitter"], rng)

    # 4. Perturb lifetimes
    if aug_params["lifetime_frac"] > 0:
        events = perturb_lifetimes(events, aug_params["lifetime_frac"], rng)

    # 5. Re-sort: (tick, host, -mem) for deterministic ordering
    events.sort(key=lambda e: (e[0], e[1], -e[2]))

    # 6. Link failures (topology perturbation)
    M_aug = inject_link_failures(M_np, aug_params["fail_ratio"], rng)

    return events, M_aug
