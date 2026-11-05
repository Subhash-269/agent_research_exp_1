#!/bin/bash
# =============================================================================
# Charge sweep cells that have LEFT THE QUEUE, through emit.job_cost.
# CPU ONLY, --partition=short, 0 GPU-hours.
#
# Only ids that have already finished go in CHARGE_IDS. A job's own sacct
# figures are not final while it runs, so charging a RUNNING job would record a
# cost that is wrong and never corrected. This file is rewritten with the ids
# that have finished each time it is submitted; git history shows which batch
# each run charged.
#
# charge_jobs.py saves each sacct row to $RUN_ROOT/sacct/<id>.psv BEFORE
# deriving anything from it (S11), skips ids already charged so a resubmission
# cannot double-count, and treats a missing sacct row as a hard stop rather
# than a silently skipped record (S12).
# =============================================================================
#SBATCH --job-name=cxa_chg
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

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
STAGE_RC=0
check() {
    if [ "$1" -ne 0 ]; then echo "FAIL $(STAMP) stage '$2' exited $1"; STAGE_RC=$1; return 1; fi
    echo "OK   $(STAMP) stage '$2' exited 0"; return 0
}

echo "=== cxa_chg  job=$SLURM_JOB_ID  node=$(hostname)  $(STAMP) ==="
source "$REPO/.venv/bin/activate" || exit 1
cd "$REPO" || exit 1

# source: workspace/jobs/job_ids.log (S8), restricted to ids whose sacct
# state is terminal at the time this file was written.
CHARGE_IDS="11638131 11638848 11638953 11646703 11648653"

echo "charging: $CHARGE_IDS"
python "$RUN_ROOT/tools/charge_jobs.py" $CHARGE_IDS 2>&1 | tail -40
check "${PIPESTATUS[0]}" "charge_failed_cells"

# The .psv files are the audit trail S11 exists for: the scheduler's stdout on
# disk, not just its exit code. Show that each one is actually there.
echo "--- sacct rows saved to disk ---"
for j in $CHARGE_IDS; do
    f="$RUN_ROOT/sacct/$j.psv"
    if [ -s "$f" ]; then echo "OK   $f  $(wc -c < "$f") bytes"; cat "$f";
    else echo "FAIL $f missing or empty"; STAGE_RC=30; fi
done

echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
exit "$STAGE_RC"
