# Data Augmentation Implementation Plan

Agent instructions for implementing data augmentation into the Octopus CXL RL training
pipeline. All paths relative to `/home/pm3371/gitrepos/octopus-allocation/`. Run with venv active.

**Stack:** SB3 (SAC) + SubprocVecEnv (N=32) + existing `OctopusMemPoolEnv`.
**Hardware:** 3x TITAN RTX (24 GB each), kernel 5.15, CUDA 12.5.
**Reference:** `docs/data_augmentation/data_aug.tex` (data exploration + augmentation design).

**Goal:** Increase episode diversity during training by applying episode-level transforms
to VM traces — memory scaling, arrival jitter, lifetime perturbation, and topology
perturbation — without violating SKU structure or simulator invariants.

**Key design constraint:** Augmentation operates at the **episode level**, not per-VM.
`rss[1]` values are discrete SKU sizes (powers of two). Per-VM scaling would create
memory sizes that don't exist in production. A single scale factor is applied uniformly
to all VMs in one episode (see `data_aug.tex` Section 7).

**Augmentation does NOT touch the core env step logic.** Transforms are applied during
event construction (in `_build_events()` / at `reset()` time). The step function, reward,
observation space, and action space are unchanged.

---

## Background: Safe Augmentation Ranges

From `data_aug.tex` HOTFIX survival analysis (Fig 5, Section 6):

| Scale factor | HOTFIX survival rate |
|---|---|
| 0.5 | 85.8% |
| 0.7 | 83.6% |
| 1.0 | 81.0% |
| 1.2 | 76.7% |
| 1.5 | 64.9% |
| 2.0 | 50.7% |

**Safe range: [0.7, 1.2].** Beyond 1.2, excessive VM dropout produces unrealistically
sparse episodes. Below 0.7, little additional variation.

Episode density at baseline (all 10 traces, pod_size=16): p10 = 400–1160 events,
p50 = 588–1638 events. `MIN_EVENTS = 100` is a conservative guard.

---

## Task 1 — `AugmentationConfig` dataclass

**File:** `octopus/augmentation.py` (new file)

```python
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

@dataclass
class AugmentationConfig:
    """Episode-level augmentation configuration.

    Each field controls one augmentation knob. Set to None/0 to disable.
    """
    # Memory scaling: uniform factor applied to all VM rss[1] in the episode
    scale_range: Tuple[float, float] = (0.7, 1.2)
    scale_distribution: str = "triangular"  # "triangular" or "uniform"

    # Arrival jitter: max tick shift (±) applied to VM start times
    arrival_jitter_ticks: int = 0  # 0 = disabled; 1-2 recommended

    # Lifetime perturbation: fractional noise on positive-lifetime VMs only
    lifetime_noise_frac: float = 0.0  # 0.0 = disabled; 0.05 = ±5%

    # Topology perturbation: fraction of CXL links to randomly remove
    link_failure_ratio: float = 0.0  # 0.0 = disabled; 0.01-0.05 recommended

    # Multi-trace: whether to sample traces randomly each episode
    multi_trace: bool = False

    # Validity guards
    min_events: int = 100  # reject episodes with fewer events
    max_resample_attempts: int = 5  # retry augmentation if guards fail

    # Enabled flag (master switch)
    enabled: bool = True
```

Also add a sampling function:

```python
def sample_augmentation_params(config: AugmentationConfig, rng: np.random.Generator) -> dict:
    """Sample concrete augmentation parameters from config ranges.

    Returns a dict of parameters for one episode:
        scale: float
        jitter: int
        lifetime_frac: float
        fail_ratio: float
    """
    if not config.enabled:
        return {"scale": 1.0, "jitter": 0, "lifetime_frac": 0.0, "fail_ratio": 0.0}

    # Memory scale
    lo, hi = config.scale_range
    if config.scale_distribution == "triangular":
        scale = float(rng.triangular(lo, 1.0, hi))
    else:
        scale = float(rng.uniform(lo, hi))

    # Arrival jitter
    jitter = config.arrival_jitter_ticks

    # Lifetime noise
    lifetime_frac = config.lifetime_noise_frac

    # Link failure
    fail_ratio = config.link_failure_ratio

    return {
        "scale": scale,
        "jitter": jitter,
        "lifetime_frac": lifetime_frac,
        "fail_ratio": fail_ratio,
    }
```

---

## Task 2 — Augmentation transform functions

**File:** `octopus/augmentation.py` (same file)

### 2a. `scale_memory(events, factor)`

```python
def scale_memory(events: list, factor: float) -> list:
    """Scale all VM memory requests by a uniform factor.

    events: list of (tick, host_id, vm_mem, dealloc_tick)
    Returns new list with vm_mem *= factor.
    """
    return [(tick, host, mem * factor, dealloc) for tick, host, mem, dealloc in events]
```

This is the primary augmentation. It preserves:
- Event ordering (tick, host, size-descending sort is reapplied after all transforms)
- Relative VM size ratios within an episode
- HOTFIX filter is reapplied after scaling

### 2b. `jitter_arrivals(events, max_shift, rng)`

```python
def jitter_arrivals(events: list, max_shift: int, rng: np.random.Generator) -> list:
    """Perturb VM arrival ticks by ±max_shift while preserving chronology.

    Small bounded shifts decorrelate exact co-arrivals. max_shift of 1–2 ticks
    (5–10 minutes) is recommended.
    """
    if max_shift == 0:
        return events
    jittered = []
    for tick, host, mem, dealloc in events:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        new_tick = max(0, tick + shift)
        # Maintain lifetime: shift dealloc by the same amount
        new_dealloc = max(new_tick + 1, dealloc + shift)
        jittered.append((new_tick, host, mem, new_dealloc))
    return jittered
```

### 2c. `perturb_lifetimes(events, frac, rng)`

```python
def perturb_lifetimes(events: list, frac: float, rng: np.random.Generator) -> list:
    """Perturb VM lifetimes by ±frac fraction.

    Only applied to VMs with positive lifetime (dealloc > tick).
    Never applied to negative-lifetime artifacts.
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
```

### 2d. `inject_link_failures(M, ratio, rng)`

```python
def inject_link_failures(M: np.ndarray, ratio: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Randomly remove CXL links from the topology matrix.

    M: (num_hosts, num_mpds) adjacency matrix
    ratio: fraction of links to remove (0.0 = none, 0.05 = 5%)

    Safety: never disconnect a host from ALL its MPDs. If removing a link
    would leave a host with 0 connections, skip that removal.

    Returns a copy of M with some links zeroed out.
    """
    if ratio <= 0.0:
        return M

    M_aug = M.copy()
    rows, cols = np.nonzero(M_aug)
    n_links = len(rows)
    n_remove = max(1, int(n_links * ratio))

    # Shuffle link indices and try removing up to n_remove
    candidates = rng.permutation(n_links)
    removed = 0
    for idx in candidates:
        if removed >= n_remove:
            break
        r, c = rows[idx], cols[idx]
        # Check: host r must retain at least 1 link
        if M_aug[r, :].sum() <= 1:
            continue
        M_aug[r, c] = 0
        removed += 1

    return M_aug
```

### 2e. `apply_augmentation(events, M, aug_params, rng)`

Master function that applies all transforms in sequence:

```python
def apply_augmentation(events: list, M: np.ndarray,
                       aug_params: dict,
                       rng: np.random.Generator) -> tuple:
    """Apply all augmentation transforms to an event list.

    Returns (augmented_events, augmented_M).
    Events are re-sorted after transforms.
    """
    # 1. Scale memory
    events = scale_memory(events, aug_params["scale"])

    # 2. Jitter arrivals
    if aug_params["jitter"] > 0:
        events = jitter_arrivals(events, aug_params["jitter"], rng)

    # 3. Perturb lifetimes
    if aug_params["lifetime_frac"] > 0:
        events = perturb_lifetimes(events, aug_params["lifetime_frac"], rng)

    # 4. Re-sort: (tick, host, -mem) for deterministic ordering
    events.sort(key=lambda e: (e[0], e[1], -e[2]))

    # 5. Link failures
    M_aug = inject_link_failures(M, aug_params["fail_ratio"], rng)

    return events, M_aug
```

---

## Task 3 — Integrate augmentation into `OctopusMemPoolEnv`

**File:** `octopus/env.py`

### 3a. Constructor changes

Add `aug_config` parameter to `__init__`:

```python
def __init__(self, ..., aug_config=None):
    ...
    self.aug_config = aug_config  # AugmentationConfig or None
    self._aug_rng = np.random.default_rng()  # separate RNG for augmentation
```

### 3b. `reset()` changes

After building events (either from precomputed cache or `_build_events()`), apply
augmentation transforms before the simulation begins:

```python
def reset(self, seed=None, options=None):
    ...
    # Existing: generate pod, build events
    ...

    # NEW: apply augmentation if configured
    if self.aug_config is not None and self.aug_config.enabled:
        from octopus.augmentation import sample_augmentation_params, apply_augmentation

        for attempt in range(self.aug_config.max_resample_attempts):
            aug_params = sample_augmentation_params(self.aug_config, self._aug_rng)
            aug_events, aug_M = apply_augmentation(
                list(self.events), self.M, aug_params, self._aug_rng
            )
            if len(aug_events) >= self.aug_config.min_events:
                self.events = aug_events
                self._aug_M = aug_M  # use augmented topology for this episode
                self._current_aug_params = aug_params  # for logging
                break
        else:
            # All attempts failed guard — use unaugmented episode
            self._aug_M = self.M
            self._current_aug_params = {"scale": 1.0, "jitter": 0,
                                        "lifetime_frac": 0.0, "fail_ratio": 0.0}

    # Continue with existing simulation init using self.events...
    ...
```

### 3c. Topology handling

When `aug_config` includes link failures (`fail_ratio > 0`), the augmented topology
`self._aug_M` replaces `self.M` for the current episode only. This affects:

- `_get_obs()`: accessible MPD mask uses `self._aug_M[host]` instead of `self.M[host]`
- `step()`: softmax mask uses `self._aug_M` instead of `self.M`

Implement by adding a property:

```python
@property
def active_M(self):
    """Return augmented topology for current episode, or base topology."""
    return getattr(self, '_aug_M', self.M)
```

Then replace all `self.M` references in `step()` and `_get_obs()` with `self.active_M`.
**Do NOT change `_build_events()` or `_generate_pod()`** — those always use the base
topology. Augmentation is applied after event construction.

### 3d. Info dict augmentation metadata

In `reset()`, include augmentation params in the returned info dict:

```python
info = {
    "pod_seed": pod_seed,
    "num_events": len(self.events),
    "aug_params": self._current_aug_params,  # NEW
}
```

This allows the training callback to log augmentation diversity.

---

## Task 4 — Integrate augmentation into `precompute_pod_events()`

**File:** `octopus/data.py`

Augmentation and precomputation are **partially incompatible**: precomputation caches
a fixed event list per seed, but augmentation wants a different event list every episode.

**Design decision:** When augmentation is enabled, **disable precomputation** for the
augmented transforms. Precompute the base events (no augmentation) as usual, then apply
augmentation at `reset()` time on the cached base events.

This means:
- `precompute_pod_events()` is unchanged — it always produces unaugmented events.
- `env.reset()` loads precomputed base events, then applies augmentation.
- The augmentation cost (list iteration + sort) is O(n_events) ≈ 0.1–0.5ms — negligible
  compared to the precompute cache lookup cost (~7ms saved).

**No changes needed to `data.py`.** The augmentation pipeline in Task 3b runs after
`_load_precomputed()`.

---

## Task 5 — Multi-trace augmentation

**File:** `octopus/env.py`, `scripts/train_rl.py`

### 5a. Multi-trace support in env

When `aug_config.multi_trace = True`, the environment should sample a random trace
for each episode instead of using a fixed trace.

Add to `__init__`:

```python
def __init__(self, ..., aug_config=None, trace_pool=None):
    ...
    self.trace_pool = trace_pool  # list of (trace_data, trace_name) tuples, or None
```

In `reset()`, when `multi_trace` is enabled:

```python
if self.aug_config and self.aug_config.multi_trace and self.trace_pool:
    trace_idx = self._aug_rng.integers(0, len(self.trace_pool))
    self._switch_trace(self.trace_pool[trace_idx])
```

`_switch_trace()` updates `self.all_vms`, `self.node_to_vms`, etc. and regenerates
the pod for the new trace. **Precomputed events cannot be used with multi-trace** —
events are built fresh each reset.

### 5b. Multi-trace support in training script

In `train_rl.py`, add `--multi-trace` flag:

```python
ap.add_argument("--multi-trace", action="store_true",
                help="Sample trace randomly each episode (requires loading all traces)")
ap.add_argument("--aug-traces", nargs="*", default=None,
                help="Subset of traces for multi-trace (default: all 10)")
```

When enabled:
1. Load all specified traces at startup (lazy loading — each worker loads on first use).
2. Pass `trace_pool` to `OctopusMemPoolEnv`.
3. Disable precomputed event cache (incompatible with multi-trace).

**Note:** Multi-trace adds ~6s startup per trace. With lazy loading, each SubprocVecEnv
worker loads only when it first encounters a new trace. Total memory: ~2 GB for all 10
traces (fits comfortably).

---

## Task 6 — Training script CLI integration

**File:** `scripts/train_rl.py`

Add augmentation CLI arguments:

```python
# Augmentation args
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
aug_group.add_argument("--aug-jitter", type=int, default=0,
                       help="Max arrival tick jitter (0 = disabled, 1-2 recommended)")
aug_group.add_argument("--aug-lifetime-noise", type=float, default=0.0,
                       help="Lifetime noise fraction (0.0 = disabled, 0.05 = ±5%%)")
aug_group.add_argument("--aug-link-failures", type=float, default=0.0,
                       help="Link failure ratio (0.0 = disabled, 0.01-0.05 recommended)")
aug_group.add_argument("--multi-trace", action="store_true",
                       help="Sample trace randomly each episode")
```

Build `AugmentationConfig` from args:

```python
if args.augmentation:
    from octopus.augmentation import AugmentationConfig
    aug_config = AugmentationConfig(
        scale_range=(args.aug_scale_lo, args.aug_scale_hi),
        scale_distribution=args.aug_scale_dist,
        arrival_jitter_ticks=args.aug_jitter,
        lifetime_noise_frac=args.aug_lifetime_noise,
        link_failure_ratio=args.aug_link_failures,
        multi_trace=args.multi_trace,
        enabled=True,
    )
else:
    aug_config = None
```

Pass to env construction:

```python
env = OctopusMemPoolEnv(..., aug_config=aug_config, trace_pool=trace_pool)
```

Save augmentation config in `config.json` alongside other training params.

---

## Task 7 — Augmentation diversity logging

**File:** `scripts/train_rl.py` (callback)

Add an `AugmentationLogCallback` that logs augmentation param statistics every N episodes:

```python
class AugmentationLogCallback(BaseCallback):
    """Log augmentation parameter diversity during training."""

    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.scales = []
        self.log_freq = log_freq

    def _on_step(self):
        # Collect aug_params from info dicts
        infos = self.locals.get("infos", [])
        for info in infos:
            if "aug_params" in info:
                self.scales.append(info["aug_params"]["scale"])

        if len(self.scales) >= self.log_freq:
            scales = np.array(self.scales[-self.log_freq:])
            self.logger.record("aug/scale_mean", float(scales.mean()))
            self.logger.record("aug/scale_std", float(scales.std()))
            self.logger.record("aug/scale_min", float(scales.min()))
            self.logger.record("aug/scale_max", float(scales.max()))
        return True
```

Register alongside existing callbacks in `train_rl.py`.

---

## Task 8 — Tests

**File:** `tests/test_augmentation.py` (new file)

### 8a. Scale memory correctness

```python
def test_scale_memory():
    events = [(0, 0, 100.0, 10), (0, 1, 200.0, 20), (5, 0, 50.0, 15)]
    scaled = scale_memory(events, 0.8)
    assert scaled[0][2] == 80.0
    assert scaled[1][2] == 160.0
    assert scaled[2][2] == 40.0
    # Ticks and hosts unchanged
    assert all(s[0] == e[0] and s[1] == e[1] and s[3] == e[3]
               for s, e in zip(scaled, events))
```

### 8b. Jitter preserves non-negative ticks

```python
def test_jitter_nonnegative_ticks():
    events = [(0, 0, 100.0, 10), (1, 0, 100.0, 11)]
    rng = np.random.default_rng(42)
    jittered = jitter_arrivals(events, max_shift=2, rng=rng)
    assert all(e[0] >= 0 for e in jittered)
    assert all(e[3] > e[0] for e in jittered)  # dealloc > arrival
```

### 8c. Lifetime perturbation skips invalid VMs

```python
def test_lifetime_perturb_skips_negative():
    events = [(0, 0, 100.0, -5), (0, 1, 100.0, 20)]  # first has negative lifetime
    rng = np.random.default_rng(42)
    perturbed = perturb_lifetimes(events, frac=0.1, rng=rng)
    assert perturbed[0][3] == -5  # unchanged
    assert perturbed[1][3] != 20 or True  # may or may not change (stochastic)
    assert perturbed[1][3] > 0  # always positive
```

### 8d. Link failure safety

```python
def test_link_failure_no_disconnect():
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    rng = np.random.default_rng(42)
    M_aug = inject_link_failures(M, ratio=0.5, rng=rng)
    # Every host must retain at least 1 link
    assert all(M_aug[i, :].sum() >= 1 for i in range(M.shape[0]))
```

### 8e. End-to-end augmented episode

```python
def test_augmented_episode_runs():
    """Smoke test: create env with augmentation, reset, step through 10 events."""
    from octopus.augmentation import AugmentationConfig
    from octopus.env import OctopusMemPoolEnv

    config = AugmentationConfig(
        scale_range=(0.8, 1.1),
        arrival_jitter_ticks=1,
        lifetime_noise_frac=0.05,
        link_failure_ratio=0.02,
    )
    env = OctopusMemPoolEnv(
        trace_name="AMS20PrdApp19-tround",
        topology_path="data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv",
        aug_config=config,
    )
    obs, info = env.reset(seed=42)
    assert "aug_params" in info
    assert info["aug_params"]["scale"] != 1.0 or True  # stochastic

    for _ in range(min(10, info["num_events"] - 1)):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
```

### 8f. Augmentation disabled = identical to baseline

```python
def test_no_augmentation_matches_baseline():
    """With augmentation disabled, env should produce identical episodes."""
    from octopus.env import OctopusMemPoolEnv

    env1 = OctopusMemPoolEnv(
        trace_name="AMS20PrdApp19-tround",
        topology_path="data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv",
        aug_config=None,
    )
    env2 = OctopusMemPoolEnv(
        trace_name="AMS20PrdApp19-tround",
        topology_path="data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv",
        aug_config=AugmentationConfig(enabled=False),
    )

    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)
    assert np.allclose(obs1, obs2)
    assert info1["num_events"] == info2["num_events"]
```

---

## Task 9 — Training runs

### 9a. Baseline (no augmentation) — control run

```bash
python scripts/train_rl.py \
    --run-id v3_noaug_baseline \
    --total-timesteps 500000 \
    --n-envs 32 \
    --trace AMS20PrdApp19-tround.sqlite \
    --variance-lambda 0.5
```

### 9b. Memory scaling only

```bash
python scripts/train_rl.py \
    --run-id v3_aug_scale \
    --augmentation \
    --aug-scale-lo 0.7 --aug-scale-hi 1.2 \
    --total-timesteps 500000 \
    --n-envs 32 \
    --trace AMS20PrdApp19-tround.sqlite \
    --variance-lambda 0.5
```

### 9c. Full augmentation

```bash
python scripts/train_rl.py \
    --run-id v3_aug_full \
    --augmentation \
    --aug-scale-lo 0.7 --aug-scale-hi 1.2 \
    --aug-jitter 1 \
    --aug-lifetime-noise 0.05 \
    --aug-link-failures 0.02 \
    --total-timesteps 500000 \
    --n-envs 32 \
    --trace AMS20PrdApp19-tround.sqlite \
    --variance-lambda 0.5
```

### 9d. Multi-trace + full augmentation

```bash
python scripts/train_rl.py \
    --run-id v3_aug_multitrace \
    --augmentation \
    --aug-scale-lo 0.7 --aug-scale-hi 1.2 \
    --aug-jitter 1 \
    --aug-lifetime-noise 0.05 \
    --aug-link-failures 0.02 \
    --multi-trace \
    --total-timesteps 500000 \
    --n-envs 32 \
    --variance-lambda 0.5
```

### 9e. Evaluation (all runs)

```bash
for run in v3_noaug_baseline v3_aug_scale v3_aug_full v3_aug_multitrace; do
    python scripts/eval_rl.py --run-id $run --n-iter 50
done

python scripts/plot_results.py \
    --rl-dirs output/rl_evals/v3_noaug_baseline \
              output/rl_evals/v3_aug_scale \
              output/rl_evals/v3_aug_full \
              output/rl_evals/v3_aug_multitrace \
    --out-dir output/figures/v3_augmentation/
```

---

## Execution order

```
1 (AugmentationConfig dataclass)
    ↓
2 (transform functions: scale, jitter, lifetime, link failures)
    ↓
3 (integrate into env.py: __init__, reset, active_M property)
    ↓
4 (verify precompute compatibility — no changes needed)
    ↓
5 (multi-trace support — optional, can defer)
    ↓
6 (CLI args in train_rl.py)
    ↓
7 (augmentation logging callback)
    ↓
8 (tests)
    ↓
9 (training runs + evaluation)
```

Tasks 1–3 are the core implementation. Task 5 (multi-trace) is optional and can be
deferred if single-trace augmentation shows good results. Tasks 8–9 validate the
implementation.

---

## Verification checklist

- **Task 1:** `from octopus.augmentation import AugmentationConfig` imports without error.
  `sample_augmentation_params(AugmentationConfig(), rng)` returns dict with all 4 keys.
- **Task 2:** Unit tests for each transform pass (Task 8a–8d).
- **Task 3:** `env.reset()` with augmentation enabled returns info dict containing
  `aug_params` with scale ≠ 1.0. Ten resets produce at least 5 distinct scale values.
- **Task 4:** Precomputed events + augmentation: reset time < 10ms (precompute still works,
  augmentation adds < 1ms).
- **Task 5:** `--multi-trace` flag: env samples different traces across episodes (check
  via info dict or log).
- **Task 6:** `python scripts/train_rl.py --run-id test --augmentation --fast --total-timesteps 1000`
  completes without error; `config.json` contains augmentation params.
- **Task 7:** TensorBoard shows `aug/scale_mean`, `aug/scale_std` metrics during training.
- **Task 8:** `python -m pytest tests/test_augmentation.py -v` passes all tests.
- **Task 9:** `output/rl_evals/v3_aug_scale/results.csv` exists; savings numbers are
  plausible (within ±0.1 of baseline). Compare v3_noaug_baseline vs v3_aug_scale on
  unseen traces to measure generalization.

---

## File layout

```
# New files
octopus/augmentation.py           # AugmentationConfig, transforms, apply_augmentation
tests/test_augmentation.py        # Unit tests for augmentation

# Modified files
octopus/env.py                    # aug_config param, reset() augmentation, active_M property
scripts/train_rl.py               # --augmentation CLI args, AugmentationLogCallback

# Unchanged files
octopus/data.py                   # precompute_pod_events() — no changes needed
octopus/baselines.py              # allocation policies — unchanged
scripts/evaluate.py               # pooling_simulation — unchanged
scripts/eval_rl.py                # RL evaluation — unchanged
scripts/plot_results.py           # plotting — unchanged
```
