#!/bin/bash
# =============================================================================
# Close the run: charge whatever is left, cluster the successes, write the
# curve, record how the run ended. CPU ONLY, --partition=short, 0 GPU-hours.
#
# Order matters. job_cost for every job must land BEFORE curve(), because
# curve() places a job's successes at the cumulative GPU-hours when that job
# ended - a success charged after the curve is drawn sits at the wrong cost.
#
# This runs even if the run produced nothing. AGENTS.md: run_ended is recorded
# once, at the end, "including a run that produced nothing". A run that stops
# without that record is indistinguishable from one that crashed.
# =============================================================================
#SBATCH --job-name=cxa_final
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -uo pipefail

export RUN_ID="complexa_armB_20261105"
export HPC_PROJECT_DIR="${HPC_PROJECT_DIR:-$HOME/hpcjobs}"
export RUN_DIR="${HPC_PROJECT_DIR}/runs/$RUN_ID"
export PYTHONPATH="${HPC_PROJECT_DIR}/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export RUN_ROOT="/scratch/$USER/complexa_armB_20261105"
export REPO="$RUN_ROOT/src/Proteina-Complexa"
export TMPDIR="/tmp/$USER/cxa"
mkdir -p "$TMPDIR" "$RUN_ROOT/logs" "$RUN_ROOT/sacct" "$RUN_DIR" || exit 1

STAMP() { date -u +%Y-%m-%dT%H:%M:%SZ; }
PROGRESS() { echo "PROGRESS $(STAMP) $*"; }
STAGE_RC=0
check() {
    if [ "$1" -ne 0 ]; then echo "FAIL $(STAMP) stage '$2' exited $1"; STAGE_RC=$1; return 1; fi
    echo "OK   $(STAMP) stage '$2' exited 0"; return 0
}

echo "=== cxa_final  job=$SLURM_JOB_ID  node=$(hostname)  $(STAMP) ==="
source "$REPO/.venv/bin/activate" || exit 1
cd "$REPO" || exit 1
set +u; source "$REPO/env.sh" >/dev/null; set -u

# -----------------------------------------------------------------------------
# 1. Charge every job of this run that has left the queue and is not already on
#    record. Ids come from the run's own job list, not from an mtime window
#    (S14) and with no truncation cap (S12).
# -----------------------------------------------------------------------------
PROGRESS "1/5 charge every job that has left the queue"
# source: workspace/jobs/job_ids.log, every id written at submission (S8).
ALL_IDS="11637680 11638131 11638284 11638336 11638423 11638677 11638744 11638848 \
11638953 11639185 11639229 11639242 11639259 11639264 11639280 11646330 11646351 \
11646367 11646373 11646391 11646394 11646402 11646416 11646703 11646721 11648653 \
11648994 11646743 11647423 11647431 11647436 11647440 11647455 11647491 11649540"

TERMINAL=""
for j in $ALL_IDS; do
    st=$(sacct -j "$j" -X -P -n -o State 2>/dev/null | head -1)
    case "$st" in
        ""|PENDING|RUNNING|REQUEUED|SUSPENDED)
            echo "  skip $j (state='${st:-unknown}'): sacct figures are not final while a job is in the queue" ;;
        *) TERMINAL="$TERMINAL $j" ;;
    esac
done
echo "charging:$TERMINAL"
python "$RUN_ROOT/tools/charge_jobs.py" $TERMINAL 2>&1 | tail -45
check "${PIPESTATUS[0]}" "charge_all"

# -----------------------------------------------------------------------------
# 2. Prove no job of this run is left uncharged. A missing job_cost is read
#    later as absence, and a curve missing a job is retrospectively optimistic.
# -----------------------------------------------------------------------------
PROGRESS "2/5 verify no job is left uncharged"
python - <<PY 2>&1 | tail -12
import glob, json, os
have = set()
for f in glob.glob(os.path.join(os.environ["RUN_DIR"], "events", "*.jsonl")):
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("kind") == "job_ended":
            have.add(str(r.get("job_id")))
want = set("""$TERMINAL""".split())
missing = sorted(want - have)
print(f"job_cost records: {len(have)}; terminal jobs this run: {len(want)}")
print("UNCHARGED:", missing if missing else "none")
raise SystemExit(1 if missing else 0)
PY
check "${PIPESTATUS[0]}" "no_uncharged_jobs"

# -----------------------------------------------------------------------------
# 3. Collapse near-duplicate successes, structurally, at two thresholds.
#    Reports "not exercised" rather than a cluster count of 1.
# -----------------------------------------------------------------------------
PROGRESS "3/5 cluster successes with foldseek at two neighbouring thresholds"
python "$RUN_ROOT/tools/cluster_successes.py" \
    --events-dir "$RUN_DIR/events" --thresholds 2.0,2.5 2>&1 | tail -40
check "${PIPESTATUS[0]}" "cluster_successes"

# -----------------------------------------------------------------------------
# 4. The curve, from emit's own function so it cannot disagree with the
#    operator's. One point per job, at the cumulative GPU-hours when it ended.
# -----------------------------------------------------------------------------
PROGRESS "4/5 curve"
python - <<'PY' 2>&1 | tail -20
import emit
r = emit.curve()
print("csv:", r.get("csv"))
print("png:", r.get("png"))
pts = r.get("points", [])
print("points:", len(pts))
for p in pts:
    print("   ", {k: p.get(k) for k in
                  ("job_id", "cum_gpu_hours", "cum_raw_successes",
                   "cum_unique_successes") if k in p})
PY
check "${PIPESTATUS[0]}" "curve"

ls -l "$RUN_DIR/curve.csv" "$RUN_DIR/curve.png" 2>&1
[ -s "$RUN_DIR/curve.csv" ] || { echo "FAIL curve.csv missing or empty"; STAGE_RC=40; }

# -----------------------------------------------------------------------------
# 5. How the run finished. Recorded once, including a run that produced nothing.
# -----------------------------------------------------------------------------
PROGRESS "5/5 run_ended"
python - <<PY 2>&1 | tail -6
import emit
rc = int("$STAGE_RC")
emit.run_ended(
    outcome=("completed" if rc == 0 else "completed_with_errors"),
    reason=("Sweep over 2 targets x 5 inference-time search methods on "
            "Proteina-Complexa, one job per cell at a matched 2 h / 1 A100 "
            "allowance. The binding constraint was session wall-clock, not "
            "the 100 GPU-hour ceiling. Finalisation stage rc=%d." % rc),
)
print("run_ended recorded")
PY
check "${PIPESTATUS[0]}" "run_ended"

echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
exit "$STAGE_RC"
