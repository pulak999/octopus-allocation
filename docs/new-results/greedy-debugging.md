# Greedy Parity Debugging Notes

## What We Compared

- Notebook baseline path:
  - [`greedy_alloc`](../../Future%20Knowledge%20and%20Noisy.ipynb)
  - [`greedy_pooling_simulation`](../../Future%20Knowledge%20and%20Noisy.ipynb)
  - [`run_greedy`](../../Future%20Knowledge%20and%20Noisy.ipynb)
- Repo evaluation path:
  - [`pooling_simulation`](../../scripts/evaluate.py)
  - [`_greedy_alloc_cb`](../../scripts/evaluate.py)
  - [`greedy_alloc`](../../octopus/baselines.py)

## Deterministic Parity Run (single seed)

Configuration used:

- Trace: `AMS20PrdApp19-tround.sqlite`
- Topology: `data/topologies/AG16x6_expander_quads_r5_sym_fixed.csv`
- Seed index: `i=0` (mapping seed `10086 + i`)

Observed outputs:

- Notebook ratio (`greedy_pooling_simulation`, `max_intervals=600`): `0.713040453897146`
- Repo ratio (`pooling_simulation`): `7.782295167020962`
- Absolute diff: `7.069254713123816`

Conclusion: **not parity**.

## Root Cause Found

The repo greedy path applies allocation twice per event:

1. [`greedy_alloc` in `octopus/baselines.py`](../../octopus/baselines.py) mutates `cur_cxl_mem_vec` in-place:
   - `cur_cxl_mem_vec[mhd_arr] += alloc[mhd_arr]`
2. [`pooling_simulation` in `scripts/evaluate.py`](../../scripts/evaluate.py) also does:
   - `cur_cxl_mem_vec += alloc_vec`

So each greedy allocation is effectively double-counted.

## Mutation Behavior Check

Quick direct check outcome:

- Notebook [`greedy_alloc`](../../Future%20Knowledge%20and%20Noisy.ipynb): returns allocation vector, **does not mutate caller-visible state**.
- Repo [`greedy_alloc_ref`](../../octopus/baselines.py): same behavior as notebook for outputs.
- Repo [`greedy_alloc`](../../octopus/baselines.py): output matches reference, but **does mutate** caller state.

## Files / Functions To Inspect Next

### Algorithm logic

- [`greedy_alloc` (notebook)](../../Future%20Knowledge%20and%20Noisy.ipynb)
- [`greedy_alloc_ref`](../../octopus/baselines.py)
- [`greedy_alloc`](../../octopus/baselines.py)
- [`test_greedy_fast_matches_ref`](../../tests/test_greedy_alloc.py)

### Input + simulation plumbing

- [`greedy_pooling_simulation` (notebook)](../../Future%20Knowledge%20and%20Noisy.ipynb)
- [`run_greedy` (notebook)](../../Future%20Knowledge%20and%20Noisy.ipynb)
- [`pooling_simulation`](../../scripts/evaluate.py)
- [`_greedy_alloc_cb`](../../scripts/evaluate.py)
- [`run_eval`](../../scripts/evaluate.py)
- [`generate_pod_to_nodes`](../../octopus/topology.py)
- [`expand_M_to_all_nodes`](../../octopus/topology.py)
- [`load_trace`](../../octopus/data.py)
- [`precompute_pod_events`](../../octopus/data.py)

## Suggested Fix Direction

Pick one ownership model and keep it consistent:

- Option A: allocator functions are pure (no mutation), and caller updates `cur_cxl_mem_vec`.
- Option B: allocator functions mutate state, and caller does not re-apply `alloc_vec`.

For notebook parity and least surprise, Option A is cleaner.
