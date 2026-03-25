import numpy as np
import pytest
from octopus.baselines import greedy_alloc_ref, greedy_alloc


def _random_input(seed, n_mhd=6, n_acc=4):
    rng = np.random.default_rng(seed)
    cur = rng.uniform(0, 100, size=n_mhd)
    mhd_list = sorted(rng.choice(n_mhd, size=n_acc, replace=False).tolist())
    cxl_mem = float(rng.uniform(1, 50))
    return cxl_mem, mhd_list, cur


@pytest.mark.parametrize("seed", range(20))
def test_greedy_fast_matches_ref(seed):
    from octopus.baselines import greedy_alloc_ref, greedy_alloc
    cxl_mem, mhd_list, cur = _random_input(seed)
    ref = greedy_alloc_ref(cxl_mem, mhd_list, cur.copy())
    fast = greedy_alloc(cxl_mem, mhd_list, cur.copy())
    assert np.allclose(ref, fast, atol=1e-9), f"seed={seed}: ref={ref} fast={fast}"
