# Adversarial Reproducibility Review (Must-Fix Now)

Scope:
- `octopus-allocation/plot_demand.py`
- `octopus-allocation/plot_memory_pooling_sweep.py`
- `octopus-allocation/train.py`

## Must-fix issues (top 3)

### A) Wrong activity gate in optimal evaluation
- **File + lines:** `plot_memory_pooling_sweep.py` (`optimal_required_capacity`, around `409`-`415`)
- **Bug/risk:** Active-load detection uses `np.max(diff[ts, :])` (instantaneous change), not `node_cxl` (current load).
- **Why it causes inconsistency/wrongness:** Ticks with sustained load but no event delta are skipped, which can underestimate required capacity and create timing-sensitive results.
- **Concrete fix:** Gate on `node_cxl` (or evaluate each window tick):
  - Replace `if float(np.max(diff[ts, :])) <= 0:` with `if float(np.max(node_cxl)) <= 0:`.

### B) Incomplete deterministic training setup
- **File + lines:** `train.py` (`68`, `72`-`75`, `147`; deterministic backend config missing)
- **Bug/risk:** Seed is passed to SB3, but explicit deterministic PyTorch/CUDA settings are missing.
- **Why it causes inconsistency:** CuDNN/kernel selection can still vary across runs and environments even with a fixed seed.
- **Concrete fix:** Before env/model creation, add:
  - `random.seed(args.seed)`, `np.random.seed(args.seed)`, `torch.manual_seed(args.seed)`, `torch.cuda.manual_seed_all(args.seed)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - Optional strict mode: `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG`.

### C) RL checkpoint selection is timing-dependent
- **File + lines:** `plot_memory_pooling_sweep.py` (`599`-`620`, `602`-`616`)
- **Bug/risk:** Script picks whichever candidate model path exists first (with optional waiting).
- **Why it causes inconsistency:** Same command can evaluate different checkpoints depending on filesystem timing/state.
- **Concrete fix:** Require one explicit `--rl-model` path and fail fast if not present; log resolved absolute path + model hash.

## Priorities

1. **P0:** Fix `optimal_required_capacity` gating (`node_cxl` vs `diff`).
2. **P1:** Add deterministic seeding/backend config in `train.py`.
3. **P1:** Make RL model selection explicit and deterministic in sweep script.
