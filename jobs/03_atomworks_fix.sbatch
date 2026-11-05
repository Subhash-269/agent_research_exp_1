#!/bin/bash
# =============================================================================
# Phase 0, job 3: find an atomworks version whose API the repository actually
# uses, and charge the three GPU/cancelled jobs so far. CPU ONLY, 0 GPU-hours.
#
# What went wrong in 11638423 (gpu/h200, FAILED after 308 s):
#
#   proteinfoundation/utils/pdb_utils.py:51
#     from atomworks.io.utils.selection import AtomSelectionStack
#   ImportError: cannot import name 'AtomSelectionStack'
#
# Observed: the installed atomworks is 3.0.4 (logs/pip_freeze.txt), and its
# atomworks.io.utils.selection does not export that name. pdb_utils.py:554 also
# calls AtomSelectionStack.from_contig(target_spec), so both the class and that
# constructor have to exist.
#
# Observed: neither pyproject.toml, env/docker/Dockerfile nor any CI workflow
# in the repository pins atomworks. env/build_uv_env.sh:174 installs it as
#   uv pip install "atomworks[ml,openbabel,dev]" || echo "Warning: ..."
# unpinned, and with a || that turns a failed install into a warning.
#
# Not observed: which version the repository was developed against. Nothing in
# the tree records it. So rather than guess one, this job asks the index what
# versions exist and tests each against the two API surfaces the code needs,
# newest first, stopping at the first that satisfies both.
#
# Doing this on a CPU node matters: 11638423 spent 308 s of H200 time to reach
# an ImportError, which is a dependency question that never needed a GPU.
# =============================================================================
#SBATCH --job-name=cxa_atomworks
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -uo pipefail

export RUN_ID="complexa_armB_20261105"
export HPC_PROJECT_DIR="${HPC_PROJECT_DIR:-$HOME/hpcjobs}"
export RUN_DIR="${HPC_PROJECT_DIR}/runs/$RUN_ID"
export PYTHONPATH="${HPC_PROJECT_DIR}/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export RUN_ROOT="/scratch/$USER/complexa_armB_20261105"
export REPO="$RUN_ROOT/src/Proteina-Complexa"
export HF_HOME="$RUN_ROOT/cache/hf"
export PIP_CACHE_DIR="$RUN_ROOT/cache/pip"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export UV_CACHE_DIR="$RUN_ROOT/cache/uv"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TMPDIR="/tmp/$USER/cxa"
mkdir -p "$TMPDIR" "$RUN_ROOT/logs" "$RUN_ROOT/sacct" "$RUN_DIR" || exit 1
export PATH="$HOME/.local/bin:$PATH"

STAMP() { date -u +%Y-%m-%dT%H:%M:%SZ; }
PROGRESS() { echo "PROGRESS $(STAMP) $*"; }
STAGE_RC=0
check() {
    if [ "$1" -ne 0 ]; then echo "FAIL $(STAMP) stage '$2' exited $1"; STAGE_RC=$1; return 1; fi
    echo "OK   $(STAMP) stage '$2' exited 0"; return 0
}

echo "=== cxa_atomworks  job=$SLURM_JOB_ID  node=$(hostname)  $(STAMP) ==="
source "$REPO/.venv/bin/activate" || exit 1
cd "$REPO" || exit 1

# -----------------------------------------------------------------------------
# 1. Charge every job that has finished and is not yet on record.
#    11638284 was cancelled while PENDING (elapsed 0, never allocated a GPU);
#    it is charged anyway, because a curve that omits it is optimistic.
# -----------------------------------------------------------------------------
PROGRESS "1/5 charge finished jobs"
python "$RUN_ROOT/tools/charge_jobs.py" 11638284 11638336 11638423 2>&1 | tail -20
check "${PIPESTATUS[0]}" "charge_jobs"

# -----------------------------------------------------------------------------
# 2. What does the index offer?
# -----------------------------------------------------------------------------
PROGRESS "2/5 available atomworks versions"
uv pip index versions atomworks 2>&1 | tee "$RUN_ROOT/logs/atomworks_versions.txt" | head -20
check "${PIPESTATUS[0]}" "index_versions"
echo "currently installed: $(uv pip show atomworks 2>/dev/null | grep -i '^Version' || echo unknown)"

# -----------------------------------------------------------------------------
# 3. Test each candidate against the exact API the repository calls.
#    The test is not "does atomworks import" - it is "does the name pdb_utils
#    imports exist, and does the constructor it calls exist". A version that
#    imports but lacks from_contig would fail later, inside a GPU job.
# -----------------------------------------------------------------------------
PROGRESS "3/5 probe versions for AtomSelectionStack.from_contig"
cat > "$TMPDIR/probe_atomworks.py" <<'PY'
import sys
try:
    from atomworks.io.utils.selection import AtomSelectionStack
except Exception as exc:
    print(f"NO  import failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
if not hasattr(AtomSelectionStack, "from_contig"):
    print("NO  AtomSelectionStack exists but has no from_contig")
    sys.exit(2)
import atomworks
print(f"YES atomworks {getattr(atomworks, '__version__', '?')} has "
      f"AtomSelectionStack.from_contig")
PY

# Candidates, newest first. source: the version list printed by the step above;
# this list is read from that output at run time, not hard-coded, so it cannot
# drift from what the index actually offers.
# MAX_CANDIDATES source: chosen by me, no external source. Each attempt is a
# full dependency resolution, so the loop is bounded rather than walking the
# whole release history; if none of the newest few works, that is itself the
# finding and a wider search belongs in its own job.
MAX_CANDIDATES=10
CANDIDATES=$(uv pip index versions atomworks 2>/dev/null \
    | tr ',' '\n' | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | sort -Vr | uniq \
    | head -$MAX_CANDIDATES)
echo "candidates (newest $MAX_CANDIDATES): $(echo "$CANDIDATES" | tr '\n' ' ')"

WORKING=""
for v in $CANDIDATES; do
    echo "--- trying atomworks==$v"
    uv pip install "atomworks[ml,openbabel]==$v" >/dev/null 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then echo "    install rc=$rc, skipping"; continue; fi
    out=$(python "$TMPDIR/probe_atomworks.py" 2>&1)
    echo "    $out"
    case "$out" in
        YES*) WORKING="$v"; break ;;
    esac
done

if [ -z "$WORKING" ]; then
    echo "FAIL no atomworks version on the index provides AtomSelectionStack.from_contig"
    STAGE_RC=10
else
    echo "WORKING atomworks version: $WORKING"
    echo "$WORKING" > "$RUN_ROOT/logs/atomworks_pinned_version.txt"
fi
check "$STAGE_RC" "probe_atomworks"

# -----------------------------------------------------------------------------
# 4. Does the module that failed now import, and does the rest still work?
#    A downgrade can satisfy one import and break another, so both are checked.
# -----------------------------------------------------------------------------
PROGRESS "4/5 import checks after the pin"
if [ -n "$WORKING" ]; then
    python - <<'PY' 2>&1 | tail -20
import sys
import atomworks
print("atomworks", getattr(atomworks, "__version__", "?"))
from atomworks.io.utils.selection import AtomSelectionStack
print("AtomSelectionStack.from_contig:", hasattr(AtomSelectionStack, "from_contig"))
# The exact module that raised in 11638423.
from proteinfoundation.utils import pdb_utils          # noqa: F401
print("proteinfoundation.utils.pdb_utils imports OK")
# The module the generate step actually runs.
from proteinfoundation.proteina import Proteina        # noqa: F401
print("proteinfoundation.proteina imports OK")
import torch, jax
print("torch", torch.__version__, "jax", jax.__version__)
from transformers.models.esm import modeling_esm       # noqa: F401
print("transformers ESM still imports OK")
PY
    check "${PIPESTATUS[0]}" "post_pin_imports"
fi

PROGRESS "4b/5 re-freeze"
uv pip freeze > "$RUN_ROOT/logs/pip_freeze.txt" 2>&1
check $? "uv_pip_freeze"
grep -E "^(atomworks|torch|torchaudio|torchvision|jax|jaxlib|transformers|biotite)" \
    "$RUN_ROOT/logs/pip_freeze.txt"

# -----------------------------------------------------------------------------
# 5. Dry-run the generate entry point far enough to prove the config resolves
#    and the model code imports, without allocating a GPU.
# -----------------------------------------------------------------------------
PROGRESS "5/5 config resolves + generate module imports (no GPU)"
set +u; source "$REPO/env.sh" >/dev/null; set -u
export HF_HOME="$RUN_ROOT/cache/hf"
complexa validate design configs/search_binder_local_pipeline.yaml 2>&1 | tail -25
check "${PIPESTATUS[0]}" "validate_design"

python -c "import proteinfoundation.generate; print('proteinfoundation.generate imports OK')" 2>&1 | tail -5
check "${PIPESTATUS[0]}" "generate_imports"

echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
exit "$STAGE_RC"
