#!/bin/bash
# =============================================================================
# Charge the six sweep cells that failed on the --steps argparse error.
# CPU ONLY, --partition=short, 0 GPU-hours.
#
# AGENTS.md: "job_cost for every job, including failed ones and ones cancelled
# while pending. A failed allocation is charged in full, and a curve that omits
# it is retrospectively optimistic." These six held an H200 allocation for 3-24
# seconds each before argparse rejected the command line. That is real GPU time
# this run spent and did not get a design for, so it goes on the ledger.
#
# A job's own sacct figures are not final while it runs, so this runs after they
# have all left the queue - confirmed empty by `squeue -u $USER` before submit.
# charge_jobs.py saves each sacct row to $RUN_ROOT/sacct/<id>.psv BEFORE
# deriving anything from it (S11), and treats a missing row as a hard stop
# rather than a skipped record.
# =============================================================================
#SBATCH --job-name=cxa_charge6
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

echo "=== cxa_charge6  job=$SLURM_JOB_ID  node=$(hostname)  $(STAMP) ==="
source "$REPO/.venv/bin/activate" || exit 1
cd "$REPO" || exit 1

# source: the six job ids recorded in workspace/jobs/job_ids.log at submission
# (S8), all of them FAILED on the argparse error in cell_worker.sh.
FAILED_CELLS="11639185 11639229 11639242 11639259 11639264 11639280"

echo "charging: $FAILED_CELLS"
python "$RUN_ROOT/tools/charge_jobs.py" $FAILED_CELLS 2>&1 | tail -40
check "${PIPESTATUS[0]}" "charge_failed_cells"

# The .psv files are the audit trail S11 exists for: the scheduler's stdout on
# disk, not just its exit code. Show that each one is actually there.
echo "--- sacct rows saved to disk ---"
for j in $FAILED_CELLS; do
    f="$RUN_ROOT/sacct/$j.psv"
    if [ -s "$f" ]; then echo "OK   $f  $(wc -c < "$f") bytes"; cat "$f";
    else echo "FAIL $f missing or empty"; STAGE_RC=30; fi
done

echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
exit "$STAGE_RC"
