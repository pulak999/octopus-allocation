## [2026-03-24] Data Augmentation Implementation (plan-v1)

### Features Implemented
- Episode-level data augmentation pipeline with 6 knobs: memory scaling, per-VM noise, arrival jitter, lifetime perturbation, link failures, multi-trace sampling
- Full integration into OctopusMemPoolEnv (reset-time transforms, no step/reward changes)
- CLI args in train_rl.py for all augmentation knobs
- Augmentation logging callback for TensorBoard
- 49 unit + integration tests (all passing)

### Files Changed
| File | What changed |
|------|-------------|
| octopus/augmentation.py | NEW — AugmentationConfig, 6 transform functions, apply_augmentation pipeline |
| octopus/env.py | Added aug_config, trace_pool params; _apply_augmentation() recomputes host_to_mhds; _switch_trace() for multi-trace; aug_params in info dict |
| scripts/train_rl.py | Added augmentation CLI arg group (10 args), AugmentationLogCallback, multi-trace pool loading, conditional callback registration |
| tests/test_augmentation.py | NEW — 49 tests covering all transforms, config sampling, pipeline, and end-to-end env integration |
| code.md | 3-phase code review at commit 24216f8a |
| CLAUDE.md | NEW — project overview, build instructions, conventions |
| ARCH.md | NEW — architecture overview, data flow, augmentation pipeline |
| TODO.md | Updated — all plan-v1 tasks marked complete |

### Functions Written
| Function | File | Description |
|----------|------|-------------|
| scale_memory | octopus/augmentation.py | Uniform episode-level memory scaling |
| add_memory_noise | octopus/augmentation.py | Per-VM multiplicative noise (sigma-bounded) |
| jitter_arrivals | octopus/augmentation.py | Bounded tick shifts preserving non-negative ticks |
| perturb_lifetimes | octopus/augmentation.py | Fractional lifetime noise (positive VMs only) |
| inject_link_failures | octopus/augmentation.py | Random CXL link removal with safety guarantee |
| apply_augmentation | octopus/augmentation.py | Master pipeline: all transforms + re-sort |
| sample_augmentation_params | octopus/augmentation.py | Sample concrete params from config ranges |
| _apply_augmentation | octopus/env.py | Reset-time augmentation with retry logic |
| _switch_trace | octopus/env.py | Swap trace data for multi-trace support |
| AugmentationLogCallback._on_step | scripts/train_rl.py | Log scale distribution stats to TensorBoard |

### Data Structures Created
| Name | File | Description |
|------|------|-------------|
| AugmentationConfig | octopus/augmentation.py | Dataclass: scale_range, memory_noise_sigma, arrival_jitter_ticks, lifetime_noise_frac, link_failure_ratio, multi_trace, min_events, max_resample_attempts, enabled |

### Notes
- Plan's `active_M` property approach was wrong — `step()`/`_get_obs()` use `host_to_mhds`, not `self.M`. Fixed by recomputing `host_to_mhds` from augmented topology in `_apply_augmentation()`.
- Plan's test code used nonexistent `trace_name=` kwarg — fixed to use `load_trace()` + actual constructor.
- Added `memory_noise_sigma` knob (Knob 7 from design-decisions.md) not in original plan.
- Task 9 (training runs) deferred to plan-v2 per user decision.
- `_M_np` (numpy copy of M) added to env for augmentation; base `self.M` (nested list) preserved for compatibility.
