"""
Data loading utilities for Octopus CXL memory pooling.

Provides:
  - VM class (must be defined before unpickling trace files)
  - load_trace()  — load a pre-processed Azure VM trace pickle
  - load_topology() — load a topology CSV and return adjacency matrix
  - precompute_pod_events() — precompute per-seed event lists so env.reset() is fast
"""

import datetime
import pickle
import csv
import random as _random
from dataclasses import dataclass

import numpy as np

# Reference epoch for converting datetimes → int64 seconds.
# Traces are from 2024; this gives small positive values comfortably within int32.
_EPOCH = datetime.datetime(2020, 1, 1)


@dataclass
class TraceArrays:
    """Numpy representation of a loaded trace — no Python VM objects.

    Reading these arrays does not touch Python refcounts, so they are safe
    to share across ``fork()``-based subprocesses without triggering
    copy-on-write page copies.

    Layout
    ------
    Per-VM arrays (length n_vms):
        vm_start  int64  seconds since _EPOCH
        vm_end    int64  seconds since _EPOCH
        vm_mem    float32  GB (rss[mem_idx])

    Per-node arrays (length n_nodes):
        node_ids   int32  actual node_id values, in node_to_vms.keys() order
        node_dram  float32  machine_sz[machine][mem_idx]

    CSR structure mapping node → VMs:
        node_offsets  int32  shape (n_nodes+1,)
        vm_ptrs       int32  shape (total_vm_entries,)
        node i owns VMs vm_ptrs[node_offsets[i] : node_offsets[i+1]]
        values are indices into vm_* arrays
    """
    vm_start: np.ndarray
    vm_end: np.ndarray
    vm_mem: np.ndarray
    node_ids: np.ndarray
    node_dram: np.ndarray
    node_offsets: np.ndarray
    vm_ptrs: np.ndarray


def to_arrays(trace_data, mem_idx: int = 1) -> "TraceArrays":
    """Convert ``load_trace()`` output to :class:`TraceArrays`.

    Call this once after loading, before forking subprocesses.  The returned
    object holds all VM data in C-heap numpy buffers — reading them in child
    processes does not trigger copy-on-write.
    """
    all_vms, node_to_vms, node_to_machine, _vm_type_sz, machine_sz = trace_data

    # ── per-VM arrays ────────────────────────────────────────────────────
    all_vmkeys = list(all_vms.keys())
    n_vms = len(all_vmkeys)
    vmkey_to_idx = {k: i for i, k in enumerate(all_vmkeys)}

    vm_start = np.empty(n_vms, dtype=np.int64)
    vm_end   = np.empty(n_vms, dtype=np.int64)
    vm_mem   = np.empty(n_vms, dtype=np.float32)

    for i, vmkey in enumerate(all_vmkeys):
        vm = all_vms[vmkey]
        vm_start[i] = int((vm.start_time - _EPOCH).total_seconds())
        vm_end[i]   = int((vm.end_time   - _EPOCH).total_seconds())
        vm_mem[i]   = float(np.asarray(vm.rss, dtype=float)[mem_idx])

    # ── per-node arrays + CSR ────────────────────────────────────────────
    # Preserve dict-key ordering so that _random.shuffle(range(n_nodes))
    # gives the same pod assignments as the old _random.shuffle(node_list).
    node_ids_list = list(node_to_vms.keys())
    n_nodes = len(node_ids_list)

    node_ids  = np.array([int(nid) for nid in node_ids_list], dtype=np.int32)
    node_dram = np.array([
        float(np.asarray(machine_sz[node_to_machine[nid]], dtype=float)[mem_idx])
        for nid in node_ids_list
    ], dtype=np.float32)

    node_offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    for i, nid in enumerate(node_ids_list):
        node_offsets[i + 1] = node_offsets[i] + len(node_to_vms.get(nid, []))

    total_entries = int(node_offsets[-1])
    vm_ptrs = np.empty(total_entries, dtype=np.int32)
    for i, nid in enumerate(node_ids_list):
        lo = int(node_offsets[i])
        for k, vmkey in enumerate(node_to_vms.get(nid, [])):
            vm_ptrs[lo + k] = vmkey_to_idx[vmkey]

    return TraceArrays(
        vm_start=vm_start,
        vm_end=vm_end,
        vm_mem=vm_mem,
        node_ids=node_ids,
        node_dram=node_dram,
        node_offsets=node_offsets,
        vm_ptrs=vm_ptrs,
    )


# ---------------------------------------------------------------------------
# VM class — must match the structure used when the pickles were created
# ---------------------------------------------------------------------------
class VM:
    """Virtual machine record from an Azure production trace."""

    def __init__(self):
        self.vm_id = -1
        self.node_id = -1
        self.cores = -1
        self.memory = -1
        self.nic = -1
        self.rss = list()
        self.ssd = -1
        self.max_mem_ts = list()
        self.avg_mem_ts = list()
        self.max_cpu_ts = list()
        self.avg_cpu_ts = list()
        self.start_time = datetime.datetime(2024, 1, 1, 1, 1)
        self.end_time = datetime.datetime(2020, 1, 1, 1, 1)
        self.instance_role = ""
        self.subscription_id = ""


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
class _VMUnpickler(pickle.Unpickler):
    """Custom unpickler that maps ``__main__.VM`` → ``octopus.data.VM``.

    The trace pickles were created with ``VM`` defined in ``__main__``.
    When we load them from a different module we need this redirect.
    """

    def find_class(self, module, name):
        if name == "VM":
            return VM
        return super().find_class(module, name)


def load_trace(cluster_name, trace_dir="data/traces"):
    """Load a pre-processed Azure VM trace pickle.

    Parameters
    ----------
    cluster_name : str
        Cluster file stem, e.g. ``"AMS20PrdApp19-tround.sqlite"``.
    trace_dir : str
        Directory containing ``<cluster_name>.pkl`` files.

    Returns
    -------
    all_vms : dict[vm_key -> VM]
    node_to_vms : dict[node_id -> list[vm_key]]
    node_to_machine : dict[node_id -> machine_type]
    vm_type_sz : dict
    machine_sz : dict[machine_type -> list of 4 resource capacities]
    """
    path = f"{trace_dir}/{cluster_name}.pkl"
    with open(path, "rb") as f:
        all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz = (
            _VMUnpickler(f).load()
        )
    return all_vms, node_to_vms, node_to_machine, vm_type_sz, machine_sz


# ---------------------------------------------------------------------------
# Topology loading
# ---------------------------------------------------------------------------
def precompute_pod_events(trace_data, M, seeds, mem_idx=1, skip_hotfix=False):
    """Precompute VM arrival event lists for a set of pod-assignment seeds.

    This runs the same logic as OctopusMemPoolEnv._generate_pod() +
    _build_events(), but offline so that env.reset() can look up the result
    instantly instead of recomputing it.

    Parameters
    ----------
    trace_data : tuple
        As returned by load_trace().
    M : list[list[int]]
        Adjacency matrix (num_hosts × num_pools).
    seeds : iterable[int]
        The set of seeds to precompute.
    mem_idx : int
        Index into rss / machine_sz for memory (default 1).

    Returns
    -------
    cache : dict[int, tuple]
        Mapping seed → (events, pod_dur, pod_dram, base_time, pod_start_ts)
        where events is the sorted list of (tick, pod_id, vm_mem, dealloc_tick).
    """
    import datetime as _dt
    from datetime import timedelta

    all_vms, node_to_vms, node_to_machine, _vm_type_sz, machine_sz = trace_data
    pod_size = len(M)

    cache = {}
    node_list_all = [int(nid) for nid in node_to_vms.keys()]

    for seed in seeds:
        # --- Reproduce _generate_pod ---
        _random.seed(seed)
        shuffled = node_list_all[:]
        _random.shuffle(shuffled)
        pod_nodes = shuffled[:pod_size]
        node_to_pod_id = {n: i for i, n in enumerate(pod_nodes)}

        pod_dram = 0.0
        for node in pod_nodes:
            cap = np.asarray(machine_sz[node_to_machine[node]], dtype=float)
            pod_dram += float(cap[mem_idx])

        # --- Reproduce _build_events ---
        n_start = _dt.datetime.max
        n_end = _dt.datetime.min
        any_vm = False
        for node in pod_nodes:
            for vmkey in node_to_vms.get(node, []):
                vm = all_vms[vmkey]
                any_vm = True
                if vm.start_time < n_start:
                    n_start = vm.start_time
                if vm.end_time > n_end:
                    n_end = vm.end_time

        if not any_vm:
            cache[seed] = ([], 0, pod_dram, _dt.datetime(2024, 1, 1), 0)
            continue

        base_time = _dt.datetime(
            n_start.year, n_start.month, n_start.day,
            n_start.hour, n_start.minute,
        )

        def to_tick(t):
            return int((t - base_time).total_seconds() // 300)

        pod_start_ts = to_tick(n_start)
        pod_end_ts = to_tick(n_end)
        pod_dur = pod_end_ts - pod_start_ts + 1

        if pod_dur <= 0:
            cache[seed] = ([], pod_dur, pod_dram, base_time, pod_start_ts)
            continue

        # HOTFIX: filter VMs that exceed per-node DRAM
        vmkey_to_skip: set = set()

        if not skip_hotfix:
            for node in pod_nodes:
                node_alloc = [[] for _ in range(pod_dur)]
                node_dealloc = np.zeros(pod_dur, dtype=np.float64)
                node_cap = float(
                    np.asarray(machine_sz[node_to_machine[node]], dtype=float)[mem_idx]
                )
                for vmkey in node_to_vms.get(node, []):
                    vm = all_vms[vmkey]
                    vm_s = to_tick(vm.start_time) - pod_start_ts
                    vm_e = to_tick(vm.end_time) - pod_start_ts
                    if vm_e < 0:
                        vmkey_to_skip.add(vmkey)
                        continue
                    mem = float(np.asarray(vm.rss, dtype=float)[mem_idx])
                    node_alloc[vm_s].append((vmkey, vm_e + 1, mem))
                    if vm_e + 1 < pod_dur:
                        node_dealloc[vm_e + 1] += mem
                cur = 0.0
                for ts in range(pod_dur):
                    cur -= node_dealloc[ts]
                    if cur < 0:
                        cur = 0.0
                    for vmkey, end_ts, mem in node_alloc[ts]:
                        cur += mem
                        if cur > node_cap:
                            cur -= mem
                            vmkey_to_skip.add(vmkey)
                            if end_ts < pod_dur:
                                node_dealloc[end_ts] -= mem

        events = []
        for node in pod_nodes:
            pod_id = node_to_pod_id[node]
            for vmkey in node_to_vms.get(node, []):
                if vmkey in vmkey_to_skip:
                    continue
                vm = all_vms[vmkey]
                vm_s = to_tick(vm.start_time) - pod_start_ts
                vm_e = to_tick(vm.end_time) - pod_start_ts
                if vm_e < 0:
                    continue  # VM ends before pod window — always skip
                mem = float(np.asarray(vm.rss, dtype=float)[mem_idx])
                if mem <= 0:
                    continue
                events.append((vm_s, pod_id, mem, vm_e + 1))
        events.sort(key=lambda e: (e[0], e[1], -e[2]))

        cache[seed] = (events, pod_dur, pod_dram, base_time, pod_start_ts)

    return cache


def precompute_pod_events_arrays(trace_arrays, M, seeds, skip_hotfix=False,
                                 cache_dir=None):
    """Precompute VM arrival event lists from a :class:`TraceArrays` object.

    Calls :meth:`OctopusMemPoolEnv._generate_pod` and
    :meth:`OctopusMemPoolEnv._build_events` internally so the cached events
    are bit-for-bit identical to those produced by ``env.reset()``.

    Parameters
    ----------
    trace_arrays : TraceArrays
    M : list[list[int]]
    seeds : iterable[int]
    skip_hotfix : bool
    cache_dir : str | None
        Directory for on-disk cache.  On a hit the result is loaded instantly;
        on a miss it is built then saved.  Cache key is an MD5 fingerprint of
        the trace data + M + seeds + skip_hotfix.

    Returns
    -------
    cache : dict[int, tuple]
        ``{seed → (events, pod_dur, pod_dram, base_time, pod_start_ts)}``
    """
    import hashlib
    import os
    import pickle

    seeds = list(seeds)

    if cache_dir is not None:
        h = hashlib.md5()
        # Fingerprint: first/last 200 VMs of each key array + total count
        for arr in (trace_arrays.vm_start, trace_arrays.vm_end, trace_arrays.vm_mem):
            h.update(arr[:200].tobytes())
            h.update(arr[-200:].tobytes())
            h.update(str(len(arr)).encode())
        h.update(str(M).encode())
        h.update(str(seeds).encode())
        h.update(str(skip_hotfix).encode())
        cache_path = os.path.join(cache_dir, f"{h.hexdigest()}.pkl")
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    from octopus.env import OctopusMemPoolEnv  # deferred — avoids circular import

    env = OctopusMemPoolEnv(
        trace_arrays=trace_arrays, M=M, seed=0, skip_hotfix=skip_hotfix
    )
    result = {}
    for seed in seeds:
        env._generate_pod(seed)
        env._build_events()
        result[seed] = (
            list(env.events),
            env.pod_dur,
            env.pod_dram,
            env.base_time,
            env._pod_start_ts,
        )

    if cache_dir is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    return result


def load_topology(csv_path):
    """Load a topology CSV (``server,pool``) and return adjacency matrix.

    Returns
    -------
    matrix : list[list[int]]
        Adjacency matrix of shape ``[num_hosts, num_pools]``.
    num_hosts : int
    num_pools : int
    """
    with open(csv_path, "r") as fp:
        reader = csv.reader(fp)
        rows = [row for row in reader][1:]  # skip header
    server_dev_tup_list = [(int(row[0]), int(row[1])) for row in rows]

    num_pools = len(set(tup[1] for tup in server_dev_tup_list))
    num_hosts = len(set(tup[0] for tup in server_dev_tup_list))
    matrix = [[0 for _ in range(num_pools)] for _ in range(num_hosts)]
    for host, dev in server_dev_tup_list:
        matrix[host][dev] = 1
    return matrix, num_hosts, num_pools
