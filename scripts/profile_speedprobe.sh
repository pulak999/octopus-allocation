#!/usr/bin/env bash
# Profile a short training run with py-spy using a 3-target protocol
# that disambiguates main-process vs. worker-process bottlenecks.
#
# WHY 3 targets:
#   SubprocVecEnv runs 1 main + N workers. A single `--subprocesses`
#   flamegraph aggregates all of them, so `step_wait` (main idle waiting
#   for workers) looks the same shape as worker env work. Profiling
#   main and one worker separately disambiguates this:
#
#     - main.svg    → step_wait% vs. SAC.train% vs. pickle%
#                       dominant step_wait  → bottleneck is ENV-SIDE
#                       dominant SAC.train  → bottleneck is TRAINING-SIDE
#                       dominant pickle     → bottleneck is IPC
#     - worker.svg  → clean view of one env step loop
#                       (expected hot: _compute_D_j, _get_obs,
#                        _process_departures_through)
#
# Outputs:
#   speedprobe_train.log  — training stdout
#   main.svg              — flamegraph of main training process only
#   worker.svg            — flamegraph of ONE subprocess worker
#   main.dump.txt         — stack snapshot of main at record time
#   worker.dump.txt       — stack snapshot of the worker at record time
#
# Usage:
#   bash scripts/profile_speedprobe.sh [--n-envs N] [--duration SEC]
#
# Defaults: --n-envs 32, --duration 60

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Args ──────────────────────────────────────────────────────────────────────
N_ENVS=32
DURATION=60
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-envs)   N_ENVS="$2";   shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Env ───────────────────────────────────────────────────────────────────────
source venv/bin/activate

if ! command -v py-spy &>/dev/null; then
    echo "[profile] py-spy not found — installing …"
    pip install py-spy --quiet
fi

PY_SPY=$(command -v py-spy)

TRAIN_LOG="speedprobe_train.log"
MAIN_SVG="main.svg"
WORKER_SVG="worker.svg"
MAIN_DUMP="main.dump.txt"
WORKER_DUMP="worker.dump.txt"

# Ready marker + warmup sleep: wait until SAC.learn() has actually started
# (marker), then sleep WARMUP_SECS to get past learning_starts=1000 so the
# profile captures steady-state step()+train(), not replay-buffer prefill.
READY_MARKER="Training SAC"
WARMUP_SECS=45
MAX_WAIT=600   # trace load + spawn of 32 workers can take ~3 min

# Truncate any stale log from a previous run BEFORE grep ever touches it.
# (tee opens its output in truncate mode, but only once it reads the FIFO;
#  grep can race ahead and match old content.)
: > "${TRAIN_LOG}"

echo "[profile] Starting training probe (n_envs=${N_ENVS}, 2 traces) …"
echo "[profile] Log → ${TRAIN_LOG}  (also streamed to screen)"

# ── Launch training in background, tee log to screen ─────────────────────────
# Using process substitution so bash still knows the Python PID (tee as pipe
# would give us tee's PID). > tee <(tee log) pattern: we spawn python with
# stdout redirected through a FIFO read by tee.
#
# Simpler: use `stdbuf -oL -eL` for line-buffered output, and tee the fd.
mkfifo /tmp/speedprobe.$$.fifo
tee "${TRAIN_LOG}" < /tmp/speedprobe.$$.fifo &
TEE_PID=$!
stdbuf -oL -eL python scripts/train_rl.py \
    --run-id speedprobe \
    --reward-variant A \
    --multi-trace \
    --aug-traces AMS20PrdApp19-tround BLAPrdApp19-troundgrt5m \
    --n-envs "${N_ENVS}" \
    --total-timesteps 150000 \
    --no-wandb \
    --eval-freq 50000 \
    --checkpoint-freq 150000 \
    >/tmp/speedprobe.$$.fifo 2>&1 &
TRAIN_PID=$!
# Clean up fifo on exit
trap 'rm -f /tmp/speedprobe.$$.fifo; kill "${TRAIN_PID}" "${TEE_PID}" 2>/dev/null || true' EXIT

echo "[profile] Training PID = ${TRAIN_PID}"

# ── Wait until SAC.learn() has started ────────────────────────────────────────
echo "[profile] Waiting for '${READY_MARKER}' in log (max ${MAX_WAIT}s) …"
WAIT_SECS=0
until grep -q "${READY_MARKER}" "${TRAIN_LOG}" 2>/dev/null; do
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "[profile] ERROR: training process died. Last 40 lines of log:"
        tail -40 "${TRAIN_LOG}"
        exit 1
    fi
    sleep 3
    WAIT_SECS=$(( WAIT_SECS + 3 ))
    if [[ ${WAIT_SECS} -ge ${MAX_WAIT} ]]; then
        echo "[profile] ERROR: '${READY_MARKER}' not seen in ${MAX_WAIT}s"
        kill "${TRAIN_PID}" 2>/dev/null || true
        exit 1
    fi
done
echo "[profile] '${READY_MARKER}' seen after ${WAIT_SECS}s"

# Secondary gate: make sure all N workers are actually spawned before we
# try to attach py-spy. SubprocVecEnv takes a while to bring up 32 workers
# even after the 'Training SAC' line prints.
echo "[profile] Waiting for all ${N_ENVS} workers to spawn …"
# With mp.set_start_method("spawn"), workers are spawned via a helper and are
# NOT direct children of TRAIN_PID. Identify them by their command line instead.
SPAWN_WAIT=0
while : ; do
    n_workers=$( { pgrep -f "multiprocessing\.(spawn|forkserver)" 2>/dev/null || true; } | wc -l)
    if [[ "${n_workers}" -ge "${N_ENVS}" ]]; then
        echo "[profile]   ${n_workers}/${N_ENVS} workers up"
        break
    fi
    sleep 2
    SPAWN_WAIT=$(( SPAWN_WAIT + 2 ))
    if [[ ${SPAWN_WAIT} -ge 600 ]]; then
        echo "[profile] ERROR: only ${n_workers}/${N_ENVS} workers after 600s"
        kill "${TRAIN_PID}" 2>/dev/null || true
        exit 1
    fi
done

echo "[profile] Warming up ${WARMUP_SECS}s to clear learning_starts …"
sleep "${WARMUP_SECS}"

# ── Identify one worker PID ───────────────────────────────────────────────────
# With spawn start method, workers are identified by their command line.
WORKER_PID=$( { pgrep -f "multiprocessing\.(spawn|forkserver)" 2>/dev/null || true; } | head -1)
if [[ -z "${WORKER_PID}" ]]; then
    echo "[profile] ERROR: could not find a worker subprocess under PID ${TRAIN_PID}"
    echo "          (pgrep -P ${TRAIN_PID} returned empty)"
    kill "${TRAIN_PID}" 2>/dev/null || true
    exit 1
fi
echo "[profile] Main PID    = ${TRAIN_PID}"
echo "[profile] Worker PID  = ${WORKER_PID}  (one of ${N_ENVS})"

# ── Helpers: py-spy record / dump (no sudo fallback — ptrace_scope is permissive) ─
# If py-spy reports "Permission denied", check: sysctl kernel.yama.ptrace_scope
# (should be 0 on this box) and that py-spy isn't targeting a non-Python PID.
run_pyspy_record() {
    local pid="$1" out="$2"
    "${PY_SPY}" record \
        -o "${out}" \
        --pid "${pid}" \
        --duration "${DURATION}" \
        --nonblocking
    echo "[profile]   ${out} saved"
}

run_pyspy_dump() {
    local pid="$1" out="$2"
    "${PY_SPY}" dump --pid "${pid}" >"${out}"
}

# ── Profile main and worker in PARALLEL (same window, independent targets) ────
echo "[profile] Recording ${DURATION}s on main and worker in parallel …"
run_pyspy_record "${TRAIN_PID}"  "${MAIN_SVG}"   &
P_MAIN=$!
run_pyspy_record "${WORKER_PID}" "${WORKER_SVG}" &
P_WORK=$!
wait "${P_MAIN}" "${P_WORK}"

# ── Also grab a single-frame stack dump from each (text, easy to inspect) ─────
echo "[profile] Capturing stack dumps …"
run_pyspy_dump "${TRAIN_PID}"  "${MAIN_DUMP}"   || true
run_pyspy_dump "${WORKER_PID}" "${WORKER_DUMP}" || true

# ── Stop training (we have what we need) ──────────────────────────────────────
echo "[profile] Stopping training process ${TRAIN_PID} …"
kill "${TRAIN_PID}" 2>/dev/null || true
wait "${TRAIN_PID}" 2>/dev/null || true

# ── Acceptance-gate sniff test (grep SVG text for key function names) ─────────
# Flamegraph SVGs embed frame names as plain text, so grep works.
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " ACCEPTANCE SNIFF TEST"
echo "═══════════════════════════════════════════════════════════════════════"

check_frame() {
    local file="$1" pattern="$2" label="$3"
    if [[ -f "${file}" ]] && grep -q -E "${pattern}" "${file}" 2>/dev/null; then
        echo "   [FOUND]  ${label}  in  ${file}"
    else
        echo "   [miss]   ${label}  in  ${file}"
    fi
}

echo ""
echo " main.svg — expect step_wait dominant if env is the bottleneck"
check_frame "${MAIN_SVG}" 'step_wait'                       "SubprocVecEnv.step_wait"
check_frame "${MAIN_SVG}" '_recv_bytes|PipeConnection|poll' "pipe/recv (IPC wait)"
check_frame "${MAIN_SVG}" 'optim|optimizer|\.train'         "SAC gradient update"

echo ""
echo " worker.svg — expect the variant-A hot trio"
check_frame "${WORKER_SVG}" '_compute_D_j'                  "_compute_D_j (reward A/B)"
check_frame "${WORKER_SVG}" '_get_obs'                      "_get_obs"
check_frame "${WORKER_SVG}" '_process_departures_through'   "_process_departures_through"
check_frame "${WORKER_SVG}" '_build_events'                 "_build_events (reset path)"

echo ""
echo " Verdict guide:"
echo "   env-side bottleneck  → main has step_wait dominant  +  worker has hot trio"
echo "   training-side        → main has SAC.train dominant (plan §5 will not help)"
echo "   IPC-side             → main has pipe/recv dominant (different fix entirely)"
echo ""

# ── FPS summary from training log ─────────────────────────────────────────────
echo " FPS summary from log:"
grep -oP '(?<=\|\s{4}fps\s+\|\s)\d+' "${TRAIN_LOG}" | tail -20 | \
    awk 'NR==1{mn=$1;mx=$1} {s+=$1; if($1<mn)mn=$1; if($1>mx)mx=$1} END{if(NR>0) printf "   last %d readings: avg %.0f fps  min %d  max %d\n", NR, s/NR, mn, mx}' || true

# ── Full training log (scrollback — useful to see what actually happened) ─────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " FULL TRAINING LOG (speedprobe_train.log)"
echo "═══════════════════════════════════════════════════════════════════════"
cat "${TRAIN_LOG}"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " Stack dumps (snapshot at profile time)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "── main.dump.txt ──────────────────────────────────────────────────────"
[[ -f "${MAIN_DUMP}" ]]   && cat "${MAIN_DUMP}"   || echo "(not captured)"
echo ""
echo "── worker.dump.txt ────────────────────────────────────────────────────"
[[ -f "${WORKER_DUMP}" ]] && cat "${WORKER_DUMP}" || echo "(not captured)"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " Done."
echo "   Train log     : $(pwd)/${TRAIN_LOG}"
echo "   Main SVG      : $(pwd)/${MAIN_SVG}"
echo "   Worker SVG    : $(pwd)/${WORKER_SVG}"
echo "   Main dump     : $(pwd)/${MAIN_DUMP}"
echo "   Worker dump   : $(pwd)/${WORKER_DUMP}"
echo "═══════════════════════════════════════════════════════════════════════"
