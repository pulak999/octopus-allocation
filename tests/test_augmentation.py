"""Unit and integration tests for the data augmentation pipeline."""

import numpy as np
import pytest

from octopus.augmentation import (
    AugmentationConfig,
    add_memory_noise,
    apply_augmentation,
    inject_link_failures,
    jitter_arrivals,
    perturb_lifetimes,
    sample_augmentation_params,
    scale_memory,
)


# ---------------------------------------------------------------------------
# 8a. scale_memory correctness
# ---------------------------------------------------------------------------
def test_scale_memory():
    events = [(0, 0, 100.0, 10), (0, 1, 200.0, 20), (5, 0, 50.0, 15)]
    scaled = scale_memory(events, 0.8)
    assert scaled[0][2] == pytest.approx(80.0)
    assert scaled[1][2] == pytest.approx(160.0)
    assert scaled[2][2] == pytest.approx(40.0)
    # Ticks and hosts unchanged
    assert all(
        s[0] == e[0] and s[1] == e[1] and s[3] == e[3]
        for s, e in zip(scaled, events)
    )


def test_scale_memory_identity():
    events = [(0, 0, 100.0, 10), (1, 1, 200.0, 20)]
    assert scale_memory(events, 1.0) is events  # no copy when factor=1


# ---------------------------------------------------------------------------
# 8b. jitter preserves non-negative ticks
# ---------------------------------------------------------------------------
def test_jitter_nonnegative_ticks():
    events = [(0, 0, 100.0, 10), (1, 0, 100.0, 11)]
    rng = np.random.default_rng(42)
    jittered = jitter_arrivals(events, max_shift=2, rng=rng)
    assert all(e[0] >= 0 for e in jittered)
    assert all(e[3] > e[0] for e in jittered)  # dealloc > arrival


def test_jitter_zero_is_noop():
    events = [(0, 0, 100.0, 10), (5, 1, 200.0, 20)]
    rng = np.random.default_rng(42)
    assert jitter_arrivals(events, max_shift=0, rng=rng) is events


@pytest.mark.parametrize("seed", range(10))
def test_jitter_preserves_positive_lifetime(seed):
    """Jittered events must always have dealloc > tick."""
    events = [(0, 0, 100.0, 1), (0, 1, 50.0, 2), (3, 0, 75.0, 4)]
    rng = np.random.default_rng(seed)
    jittered = jitter_arrivals(events, max_shift=3, rng=rng)
    for tick, _h, _m, dealloc in jittered:
        assert dealloc > tick


# ---------------------------------------------------------------------------
# 8c. lifetime perturbation skips invalid VMs
# ---------------------------------------------------------------------------
def test_lifetime_perturb_skips_negative():
    events = [(0, 0, 100.0, -5), (0, 1, 100.0, 20)]
    rng = np.random.default_rng(42)
    perturbed = perturb_lifetimes(events, frac=0.1, rng=rng)
    assert perturbed[0][3] == -5  # negative lifetime unchanged
    assert perturbed[1][3] > 0  # positive lifetime stays positive


def test_lifetime_perturb_zero_is_noop():
    events = [(0, 0, 100.0, 10)]
    rng = np.random.default_rng(42)
    assert perturb_lifetimes(events, frac=0.0, rng=rng) is events


def test_lifetime_perturb_minimum_one():
    """Even with large noise, lifetime must be >= 1."""
    events = [(0, 0, 100.0, 1)]  # lifetime = 1
    rng = np.random.default_rng(42)
    for _ in range(100):
        perturbed = perturb_lifetimes(events, frac=0.99, rng=rng)
        assert perturbed[0][3] >= 1  # tick + max(1, ...) >= 1


# ---------------------------------------------------------------------------
# 8d. link failure safety
# ---------------------------------------------------------------------------
def test_link_failure_no_disconnect():
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    rng = np.random.default_rng(42)
    M_aug = inject_link_failures(M, ratio=0.5, rng=rng)
    assert all(M_aug[i, :].sum() >= 1 for i in range(M.shape[0]))


def test_link_failure_zero_is_noop():
    M = np.array([[1, 1], [1, 1]])
    rng = np.random.default_rng(42)
    M_aug = inject_link_failures(M, ratio=0.0, rng=rng)
    assert M_aug is M


def test_link_failure_removes_links():
    M = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    rng = np.random.default_rng(42)
    M_aug = inject_link_failures(M, ratio=0.25, rng=rng)
    assert M_aug.sum() < M.sum()
    assert all(M_aug[i, :].sum() >= 1 for i in range(M.shape[0]))


@pytest.mark.parametrize("seed", range(20))
def test_link_failure_safety_stress(seed):
    """Stress test: even with high ratio, every host keeps >= 1 link."""
    M = np.eye(5, dtype=int)  # each host has exactly 1 link
    rng = np.random.default_rng(seed)
    M_aug = inject_link_failures(M, ratio=0.9, rng=rng)
    # Cannot remove any links since each row has exactly 1
    assert np.array_equal(M_aug, M)


# ---------------------------------------------------------------------------
# Per-VM memory noise
# ---------------------------------------------------------------------------
def test_memory_noise_zero_is_noop():
    events = [(0, 0, 100.0, 10)]
    rng = np.random.default_rng(42)
    assert add_memory_noise(events, sigma=0.0, rng=rng) is events


def test_memory_noise_changes_values():
    events = [(0, 0, 100.0, 10), (1, 1, 200.0, 20)]
    rng = np.random.default_rng(42)
    noisy = add_memory_noise(events, sigma=0.05, rng=rng)
    # Values should differ from originals
    assert noisy[0][2] != events[0][2] or noisy[1][2] != events[1][2]
    # But should be within bounds
    for orig, n in zip(events, noisy):
        assert n[2] > 0
        assert abs(n[2] / orig[2] - 1.0) <= 0.05 + 1e-9


# ---------------------------------------------------------------------------
# AugmentationConfig + sampling
# ---------------------------------------------------------------------------
def test_sample_params_disabled():
    cfg = AugmentationConfig(enabled=False)
    rng = np.random.default_rng(42)
    p = sample_augmentation_params(cfg, rng)
    assert p["scale"] == 1.0
    assert p["noise_sigma"] == 0.0
    assert p["jitter"] == 0
    assert p["lifetime_frac"] == 0.0
    assert p["fail_ratio"] == 0.0


def test_sample_params_triangular_in_range():
    cfg = AugmentationConfig(scale_range=(0.7, 1.2), scale_distribution="triangular")
    rng = np.random.default_rng(42)
    for _ in range(100):
        p = sample_augmentation_params(cfg, rng)
        assert 0.7 <= p["scale"] <= 1.2


def test_sample_params_uniform_in_range():
    cfg = AugmentationConfig(scale_range=(0.8, 1.1), scale_distribution="uniform")
    rng = np.random.default_rng(42)
    for _ in range(100):
        p = sample_augmentation_params(cfg, rng)
        assert 0.8 <= p["scale"] <= 1.1


# ---------------------------------------------------------------------------
# apply_augmentation pipeline
# ---------------------------------------------------------------------------
def test_apply_augmentation_sorts_events():
    events = [(5, 0, 100.0, 15), (0, 1, 200.0, 10), (0, 0, 50.0, 8)]
    M = np.array([[1, 1], [1, 1]])
    rng = np.random.default_rng(42)
    params = {"scale": 1.0, "noise_sigma": 0.0, "jitter": 0,
              "lifetime_frac": 0.0, "fail_ratio": 0.0}
    aug_events, aug_M = apply_augmentation(events, M, params, rng)
    # Events should be sorted by (tick, host, -mem)
    for i in range(len(aug_events) - 1):
        a, b = aug_events[i], aug_events[i + 1]
        assert (a[0], a[1], -a[2]) <= (b[0], b[1], -b[2])


def test_apply_augmentation_full_pipeline():
    events = [(0, 0, 100.0, 10), (1, 1, 200.0, 20), (5, 0, 50.0, 15)]
    M = np.array([[1, 1, 0], [0, 1, 1]])
    rng = np.random.default_rng(42)
    params = {"scale": 0.9, "noise_sigma": 0.02, "jitter": 1,
              "lifetime_frac": 0.05, "fail_ratio": 0.3}
    aug_events, aug_M = apply_augmentation(list(events), M, params, rng)
    # Basic checks
    assert len(aug_events) == len(events)
    assert aug_M.shape == M.shape
    # Memory should be scaled
    for e in aug_events:
        assert e[2] > 0
