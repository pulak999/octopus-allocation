"""
Optimal (per-tick) CXL memory placement solver.

This ports the exact implementation from `memory-pooling-v0.ipynb`:
  - Split the bipartite host<->MPD graph into connected components
  - For each component, binary search the minimum per-MPD peak load `t`
    such that a feasible flow exists (Dinic max-flow).

The public entrypoint is `find_optimal(node_cxl_arr, M)`, which returns the
optimal *per MPD* peak load (GB) for the given instantaneous host demands.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np


# ---------- Max-flow (Dinic) ----------
class _Dinic:
    __slots__ = ("N", "g", "level", "it")

    def __init__(self, N: int):
        self.N = int(N)
        self.g: List[List[List[float]]] = [[] for _ in range(self.N)]
        self.level = [0] * self.N
        self.it = [0] * self.N

    def _add_edge(self, u: int, v: int, c: float) -> None:
        self.g[u].append([v, float(c), len(self.g[v])])
        self.g[v].append([u, 0.0, len(self.g[u]) - 1])

    def add_edge(self, u: int, v: int, c: float) -> None:
        self._add_edge(u, v, c)

    def bfs(self, s: int, t: int, eps: float = 1e-12) -> bool:
        self.level = [-1] * self.N
        dq: deque[int] = deque([s])
        self.level[s] = 0
        while dq:
            u = dq.popleft()
            for v, cap, _rev in self.g[u]:
                if cap > eps and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    dq.append(v)
        return self.level[t] >= 0

    def dfs(self, u: int, t: int, f: float, eps: float = 1e-12) -> float:
        if u == t:
            return f
        gi = self.g[u]
        for i in range(self.it[u], len(gi)):
            self.it[u] = i
            v, cap, rev = gi[i]
            if cap > eps and self.level[u] + 1 == self.level[v]:
                d = self.dfs(v, t, min(f, cap), eps)
                if d > eps:
                    gi[i][1] -= d
                    self.g[v][rev][1] += d
                    return d
        return 0.0

    def max_flow(self, s: int, t: int) -> float:
        flow = 0.0
        INF = 1e100
        while self.bfs(s, t):
            self.it = [0] * self.N
            while True:
                f = self.dfs(s, t, INF)
                if f <= 1e-12:
                    break
                flow += f
        return flow


# ---------- Connected components on bipartite graph ----------
def _bipartite_components(M_bin: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Parameters
    ----------
    M_bin : np.ndarray[bool], shape (n, m)

    Returns
    -------
    list of (rows_idx, cols_idx) for each connected component.
    """
    n, m = M_bin.shape
    row_to_cols = [np.nonzero(M_bin[i])[0] for i in range(n)]
    col_to_rows = [np.nonzero(M_bin[:, j])[0] for j in range(m)]
    row_seen = np.zeros(n, dtype=bool)
    col_seen = np.zeros(m, dtype=bool)
    comps: List[Tuple[np.ndarray, np.ndarray]] = []

    for r0 in range(n):
        if row_seen[r0]:
            continue
        if row_to_cols[r0].size == 0:
            row_seen[r0] = True
            comps.append((np.array([r0], dtype=int), np.array([], dtype=int)))
            continue

        rows: List[int] = []
        cols: List[int] = []
        rq: deque[int] = deque([r0])
        row_seen[r0] = True
        while rq:
            r = rq.popleft()
            rows.append(r)
            for c in row_to_cols[r]:
                if not col_seen[c]:
                    col_seen[c] = True
                    cols.append(int(c))
                    for rr in col_to_rows[c]:
                        if not row_seen[rr]:
                            row_seen[rr] = True
                            rq.append(int(rr))
        comps.append((np.array(rows, dtype=int), np.array(cols, dtype=int)))

    for c0 in range(m):
        if not col_seen[c0]:
            col_seen[c0] = True
            comps.append((np.array([], dtype=int), np.array([c0], dtype=int)))

    return comps


# ---------- Per-component solver (binary search + flow) ----------
_state_cache: Dict[bytes, Tuple[float, float]] = {}


def _component_peak(
    b_sub: np.ndarray, M_sub: np.ndarray, key_bytes: bytes, iters: int = 30
) -> float:
    """
    Parameters
    ----------
    b_sub : np.ndarray, shape (n_sub,)
        Host demands (GB) for this connected component.
    M_sub : np.ndarray[bool], shape (n_sub, m_sub)
        Adjacency mask for this component.
    key_bytes : bytes
        Topology fingerprint for warm-starting bounds.
    iters : int
        Binary-search iterations.
    """
    n, m = M_sub.shape
    total = float(b_sub.sum())
    if n == 0 or m == 0 or total <= 0.0:
        return 0.0

    deg = M_sub.sum(axis=1)
    if np.any((deg == 0) & (b_sub > 0)):
        return float("inf")

    per_row_lb = np.max(
        np.divide(
            b_sub,
            np.where(deg > 0, deg, 1),
            where=deg > 0,
            out=np.zeros_like(b_sub),
        )
    )
    LB = max(total / m, float(per_row_lb))
    UB = total

    if key_bytes in _state_cache:
        pLB, pUB = _state_cache[key_bytes]
        LB = max(LB, pLB)
        UB = min(UB, pUB)
        if LB > UB:
            LB, UB = pLB, pUB

    # Base graph:
    #   s -> rows (capacity demand),
    #   rows -> cols (INF where edge exists),
    #   cols -> t (capacity = cap, injected per feasibility check).
    s = 0
    first_r = 1
    first_c = 1 + n
    t_sink = 1 + n + m - 1
    baseG = _Dinic(1 + n + m)
    for i in range(n):
        baseG.add_edge(s, first_r + i, float(b_sub[i]))
    INF = 1e100
    for i in range(n):
        u = first_r + i
        for j in np.nonzero(M_sub[i])[0]:
            v = first_c + int(j)
            baseG.add_edge(u, v, INF)

    def feasible(cap: float) -> bool:
        G2 = _Dinic(baseG.N)
        for u in range(baseG.N):
            G2.g[u] = [e.copy() for e in baseG.g[u]]
        for j in range(m):
            G2.add_edge(first_c + j, t_sink, cap)
        return G2.max_flow(s, t_sink) >= total - 1e-8

    lo, hi = float(LB), float(UB)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid

    _state_cache[key_bytes] = (lo, hi)
    return float(hi)


def find_optimal(node_cxl_arr, M) -> float:
    """
    Minimize the maximum per-MPD load subject to the adjacency mask M.

    Parameters
    ----------
    node_cxl_arr : array-like, shape (n,)
        Host demands in GB for this tick.
    M : array-like, shape (n, m)
        Adjacency matrix (0/1) where 1 indicates the host can use that MPD.

    Returns
    -------
    float
        Optimal peak load per MPD (GB).
    """
    b = np.asarray(node_cxl_arr, dtype=float).reshape(-1)
    M_bin = np.asarray(M, dtype=bool)
    n, m = M_bin.shape
    if b.shape[0] != n:
        raise ValueError("Length of node_cxl_arr must match rows of M")

    keep_rows = ~(np.isclose(b, 0.0))
    if keep_rows.any() and not keep_rows.all():
        b = b[keep_rows]
        M_bin = M_bin[keep_rows, :]
        n, m = M_bin.shape

    row_deg = M_bin.sum(axis=1)
    col_deg = M_bin.sum(axis=0)
    row_keep = (row_deg > 0) | (b > 0)
    col_keep = col_deg > 0
    if (not bool(np.all(row_keep))) or (not bool(np.all(col_keep))):
        M_bin = M_bin[row_keep][:, col_keep]
        b = b[row_keep]
        n, m = M_bin.shape

    if n == 0 or m == 0 or float(b.sum()) <= 0.0:
        return 0.0

    comps = _bipartite_components(M_bin)
    t_star = 0.0
    for rows_idx, cols_idx in comps:
        if rows_idx.size == 0:
            continue
        b_sub = b[rows_idx]
        M_sub = M_bin[np.ix_(rows_idx, cols_idx)]
        if float(b_sub.sum()) <= 0.0:
            continue
        key_bytes = M_sub.tobytes()
        t_comp = _component_peak(b_sub, M_sub, key_bytes)
        if t_comp > t_star:
            t_star = float(t_comp)

    return float(t_star)

