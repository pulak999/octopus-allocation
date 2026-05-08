# Reward Ablation Plan — R1 through R5, Pipeline Correctness Tests

Implements R1, R4, R5 (R4+PBRS+Sub) from `docs/training/v4/reward-shaping.tex`,
renames the existing `"A"` / `"B"` variants to `"R2"` / `"R3"` throughout,
and adds a correctness test suite motivated by the greedy double-counting bug.
R6 (optimal-gap shaping) is **gated** — tracked at the bottom of this doc.

---

## Naming map

| `reward_variant` string | Paper name | Status |
|---|---|---|
| `"current"` | Run-6 reward (Δpeak + λVar) | Keep as-is (regression baseline) |
| `"A"` | — | **Rename → `"R2"`** (keep `"A"` as a deprecated alias) |
| `"B"` | — | **Rename → `"R3"`** (keep `"B"` as a deprecated alias) |
| `"R1"` | R1 — single worst MPD, no departure awareness | **Add** |
| `"R2"` | R2 — localized group, departure-aware | **Add as canonical name for "A"** |
| `"R3"` | R3 — global group + λ · unreachable peak | **Add as canonical name for "B"** |
| `"R4"` | R4 — sparse terminal | **Add** |
| `"R5"` | R4 + PBRS + sub-episodes | **Add** |

### Alias strategy

Keep `"A"` and `"B"` valid in the `assert` and argparse `choices` so that
existing checkpoints can still be evaluated with `eval_rl.py`. Internally,
treat `"A"` as `"R2"` and `"B"` as `"R3"` by normalising in `__init__`:

```python
_VARIANT_ALIASES = {"A": "R2", "B": "R3"}
self.reward_variant = _VARIANT_ALIASES.get(reward_variant, reward_variant)
```

### Fresh-run naming convention

Each new run uses `--run-id ablation_<variant>_v1`, e.g.:
```
--run-id ablation_R1_v1
--run-id ablation_R2_v1   # replaces exp_rewardA_v4
--run-id ablation_R3_v1   # replaces exp_rewardB_v4
--run-id ablation_R4_v1
--run-id ablation_R5_v1
```

W&B run name, checkpoint dir, and `config.json` all inherit from `run_id`.
No old results are reused.

---

## Obs-dim mapping (unchanged logic, extended)

| Variants | Obs dim | Obs features |
|---|---|---|
| `"current"`, `"R1"` | `2 * max_degree + 4` | loads, mask, vm_mem, peak, hour_sin/cos |
| `"R2"`, `"R3"`, `"R4"`, `"R5"` | `6 * max_degree + 2` | c_j, D_j, S_j, mask, P_j, Q_j per MPD + global peak + vm_mem |

R1 deliberately uses the simpler obs (no departure features) to isolate scope
as the controlled floor of the ablation.
R4 and R5 use the richer obs so the critic can learn departure-aware features
even from a sparse terminal signal.

---

## Chunk 0 — Rename A → R2, B → R3

**Files:** `octopus/env.py`, `scripts/train_rl.py`, `scripts/eval_rl.py`

### octopus/env.py

1. Add alias normalisation at the top of `__init__`, before the assert:
   ```python
   _VARIANT_ALIASES = {"A": "R2", "B": "R3"}
   reward_variant = _VARIANT_ALIASES.get(reward_variant, reward_variant)
   ```
2. Extend assert to accept all valid strings:
   ```python
   assert reward_variant in ("current", "R1", "R2", "R3", "R4", "R5"), ...
   ```
3. Replace `reward_variant == "A"` with `== "R2"` and `== "B"` with `== "R3"`
   everywhere in `step()` and `_get_obs()` (the internal logic is unchanged).
4. Update obs-dim branch:
   ```python
   _SIMPLE_OBS = {"current", "R1"}
   obs_dim = self.max_degree * 2 + 4 if reward_variant in _SIMPLE_OBS else self.max_degree * 6 + 2
   ```

### scripts/train_rl.py

1. Add `"R1"`, `"R2"`, `"R3"`, `"R4"`, `"R5"` to `choices` in `--reward-variant`.
   Keep `"A"`, `"B"` as valid choices (deprecated).
2. Change the `if args.reward_variant != "current"` precompute block condition to:
   ```python
   if args.reward_variant not in ("current", "R1", "A"):
   ```
   (R1 and "current" use simple obs, so they don't need `mhd_to_hosts` / `Q_j`.)
3. Update the help string for `--reward-variant` to list all valid choices.

### scripts/eval_rl.py

Same `choices` extension and obs-path condition as `train_rl.py`.

---

## Chunk 1 — Add R1

**Files:** `octopus/env.py`

In `step()` reward block, add after the existing variants:

```python
elif self.reward_variant == "R1":
    norm = self.pod_dram if self.pod_dram > 0 else 1.0
    reward = -new_peak / norm
```

`new_peak` is already computed above the reward block. No new state or obs
changes — R1 uses the `"current"` obs path set up in Chunk 0.

---

## Chunk 2 — Add R4 (sparse terminal)

**Files:** `octopus/env.py`

### Hoist pooling_savings computation

Currently `pooling_savings` is only computed inside the `if done:` info block
(line ~322). Move it before the reward block so R4 can read it:

```python
# compute once; used by both reward and info
_pooling_ratio = (
    (self.max_peak * self.num_mhd / self.pod_dram)
    if self.pod_dram > 0 else 0.0
)
_pooling_savings = 1.0 - _pooling_ratio
```

Then in the reward block:

```python
elif self.reward_variant == "R4":
    reward = _pooling_savings if done else 0.0
```

And in the `info` block: replace the inline formula with the pre-computed vars.

---

## Chunk 3 — Add R5 (R4 + PBRS + Sub-episodes)

**Files:** `octopus/env.py`, `scripts/train_rl.py`

### New `__init__` parameters

```python
sub_episode_len: int = 576,    # ticks ≈ 2 days at 5-min resolution
pbrs_gamma: float = 0.99,
```

### env.py — reset()

Add after `self.max_peak = 0.0`:

```python
self._sub_peak = 0.0
self._sub_step = 0
self._prev_potential = 0.0    # Φ(s) = −max_j c_j / D_pod = 0 at episode start
```

### env.py — step()

```python
elif self.reward_variant == "R5":
    norm = self.pod_dram if self.pod_dram > 0 else 1.0
    self._sub_peak = max(self._sub_peak, new_peak)
    self._sub_step += 1

    # PBRS: F = γ·Φ(s') − Φ(s),  Φ(s) = −max_j c_j / D_pod
    cur_potential = -new_peak / norm
    pbrs = self.pbrs_gamma * cur_potential - self._prev_potential
    self._prev_potential = cur_potential

    at_boundary = (self._sub_step % self.sub_episode_len == 0)
    if at_boundary or done:
        sub_ratio = (self._sub_peak * self.num_mhd / self.pod_dram
                     if self.pod_dram > 0 else 0.0)
        sub_savings = 1.0 - sub_ratio
        reward = sub_savings + pbrs
        self._sub_peak = 0.0    # reset peak tracker only; MPD loads carry over
    else:
        reward = pbrs
```

### scripts/train_rl.py

Add to argparse:
```
--sub-episode-len  INT    default 576
--pbrs-gamma       FLOAT  default 0.99
```

Pass both through `make_env` lambda → `OctopusMemPoolEnv`.
Include `"R5"` in the rich-obs precompute condition.

---

## Chunk 4 — Pipeline correctness test suite

**File:** `tests/test_pipeline_correctness.py` (new)

This test file is motivated by the greedy double-counting bug: a mutation in
`greedy_alloc` plus a redundant add in `pooling_simulation` produced ~10×
inflated ratios that went undetected. The tests below assert invariants that
would have caught that class of bug.

### Helper: deterministic env rollout

A shared helper `_run_episode(env, policy_fn)` steps through a full episode
with a caller-supplied `policy_fn(obs) -> action` and returns
`(rewards, infos, trajectory)` where `trajectory` is a list of
`(tick, mhd_list, alloc_gb, cur_cxl_mem_vec_copy)` snapshots.

### Test 1 — Mass conservation

After every step, the sum of `cur_cxl_mem_vec` must equal the sum of GB from
all VMs that have arrived and not yet departed:

```python
# tracked externally via (alloc_gb, dealloc_tick) pairs
live_alloc = sum(gb for gb, dt in allocations if dt > current_tick)
assert np.sum(env.cur_cxl_mem_vec) == pytest.approx(live_alloc, abs=1e-6)
```

Catches double-allocation (the greedy bug class) and under-deallocation.

### Test 2 — Deallocation clears memory exactly

Manually construct an episode with one VM allocating 10 GB to a known MPD,
departing at tick T. Advance through tick T. Assert:
- `cur_cxl_mem_vec[j] == 0.0` (for the MPD that held the VM)
- `_mpd_n[j] == 0`

### Test 3 — pooling_ratio formula

At episode end, verify:
```python
expected = env.max_peak * env.num_mhd / env.pod_dram
assert info["pooling_ratio"] == pytest.approx(expected, abs=1e-9)
assert info["pooling_savings"] == pytest.approx(1.0 - expected, abs=1e-9)
```

### Test 4 — Eval pipeline vs env parity (highest priority)

Run the same deterministic greedy policy through:
1. `OctopusMemPoolEnv.step()` loop (Gymnasium env)
2. `pooling_simulation(..., alloc_cb=_greedy_alloc_cb)` (evaluate.py)

Assert `pooling_ratio` matches to `1e-4`. This is the exact class of bug that
corrupted the greedy baseline — if there is any state divergence between the
two code paths, this test will catch it.

Uses `greedy_alloc` (now pure after the fix) as the shared policy.

### Test 5 — R4 intermediate rewards are zero

Run a full R4 episode. Assert every intermediate reward is `0.0`. Assert
terminal reward equals `info["pooling_savings"]` to `1e-9`.

### Test 6 — R5 sub-peak resets, MPD loads do not

Inject a controlled VM arriving at tick 0, departing at tick 1000 (beyond sub-
episode boundary). At the sub-episode boundary:
- `env._sub_peak == 0.0` (reset)
- `env.cur_cxl_mem_vec.sum() > 0.0` (load still present, VM still live)

### Test 7 — R5 PBRS is policy-invariant (same optimal policy)

Run two R5 episodes with the same seed and a uniform policy. Compute cumulative
reward with PBRS and without. Confirm that the sub-episode savings component is
identical and the PBRS term sums to approximately `γ·Φ(s_T) - Φ(s_0)` (the
telescoping identity, which holds exactly at episode end).

### Test 8 — Obs reflects actual env state

After each step, re-derive what the obs should be from raw env fields and
compare to what `_get_obs()` returns:
- `obs[k*6]` for accessible MPD `k` == `cur_cxl_mem_vec[mhd] / pod_dram`
- `obs[-2]` == `cur_cxl_mem_vec.max() / pod_dram`
- `obs[-1]` == `vm_mem / pod_dram`

### Test 9 — greedy_alloc mutation regression

Confirm the Chunk-0 fix holds: calling `greedy_alloc` with a `cur_cxl_mem_vec`
copy does not mutate the original. (Locks in the fix against future regression.)

```python
orig = cur.copy()
greedy_alloc(cxl_mem, mhd_list, cur.copy())
assert np.allclose(cur, orig)
```

### Test 10 — R1 reward non-positive and bounded

Every R1 reward satisfies `-1.0 ≤ reward ≤ 0.0` over a full episode with any
random policy.

---

## Run order

```
R1  →  R2 (fresh run, same logic as "A")  →  R3 (fresh run, same logic as "B")
→  R4  →  R5
```

Each: `--total-timesteps 5_000_000`, `--eval-trace LVL01PrdApp05-troundgrt5m`,
primary metric `eval/pooling_savings_mean`.

---

## GATED — R6 (optimal-gap shaping)

**Do not implement until the R1–R5 sweep is complete.**

Requires `find_optimal(node_cxl_arr, M)` from `octopus/optimal.py` (Dinic
max-flow, ~1 ms/call). At 32 envs × 1000-step eval interval the compute cost
is non-trivial and should be benchmarked before enabling.

When unblocked:
- New variant `"R6"` in env.
- New params: `opt_gap_every: int = 20`, `opt_gap_beta: float = 0.2`.
- Cache last `t_star`; recompute every `opt_gap_every` steps.
- `δ_t = (new_peak - t_star) / D_pod`; subtract `β * δ_t` from base reward.
- Applied on top of whichever R1–R5 variant won the primary sweep.
