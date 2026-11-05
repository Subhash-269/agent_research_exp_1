#!/bin/bash
# =============================================================================
# Phase 0, job 5: pin dm-haiku to a release that works with the pinned jax,
# then re-run the whole import chain the pipeline needs. CPU ONLY, 0 GPU-hours.
#
# Where this came from. Job 11638744 pinned atomworks successfully - the wheel
# probe read each release's source and found that 3.0.0 through 3.0.4 no longer
# define AtomSelectionStack, while 2.2.1 defines it with from_contig. atomworks
# 2.2.1 is installed and pdb_utils imports. The next import in the chain then
# failed:
#
#   colabdesign/af/alphafold/model/utils.py:23  import haiku as hk
#   haiku/_src/dot.py:44  take_current_trace = jax.core.take_current_trace
#   AttributeError: module 'jax.core' has no attribute 'take_current_trace'
#
# Observed: dm-haiku is unpinned, so the resolver took its newest release,
# which expects a jax.core API that jax 0.4.29 - the version
# env/build_uv_env.sh explicitly pins - does not provide.
#
# This is the third unpinned dependency to break against a pinned one in this
# run (torchaudio, atomworks, dm-haiku). tools/find_working_version.py does the
# search generically from here on: newest first, install with --no-deps so the
# resolver cannot "fix" things by moving jax, and run the real import rather
# than grepping for a symbol.
# =============================================================================
#SBATCH --job-name=cxa_haiku
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

echo "=== cxa_haiku  job=$SLURM_JOB_ID  node=$(hostname)  $(STAMP) ==="
source "$REPO/.venv/bin/activate" || exit 1
cd "$REPO" || exit 1
set +u; source "$REPO/env.sh" >/dev/null; set -u
export HF_HOME="$RUN_ROOT/cache/hf"

PROGRESS "0/5 charge 11638744"
python "$RUN_ROOT/tools/charge_jobs.py" 11638744 2>&1 | tail -6
check "${PIPESTATUS[0]}" "charge_11638744"

PROGRESS "1/5 confirm the atomworks pin from 11638744 is still in place"
python - <<'PY' 2>&1 | tail -6
import atomworks
from atomworks.io.utils.selection import AtomSelectionStack
print("atomworks", getattr(atomworks, "__version__", "?"),
      "from_contig:", hasattr(AtomSelectionStack, "from_contig"))
from proteinfoundation.utils import pdb_utils   # noqa: F401
print("proteinfoundation.utils.pdb_utils imports OK")
PY
check "${PIPESTATUS[0]}" "atomworks_still_ok"

# -----------------------------------------------------------------------------
# 2. Search dm-haiku. The test is the real import chain, ending at the module
#    that raised, not a symbol grep.
# -----------------------------------------------------------------------------
PROGRESS "2/5 find a dm-haiku that works with jax 0.4.29"
python "$RUN_ROOT/tools/find_working_version.py" \
    --package dm-haiku \
    --max-tries 14 \
    --restore-on-failure \
    --test "import jax, haiku; from colabdesign import mk_afdesign_model; print('haiku', haiku.__version__, 'jax', jax.__version__)" \
    2>&1 | tail -30
check "${PIPESTATUS[0]}" "pin_dm_haiku"

# -----------------------------------------------------------------------------
# 3. The whole chain the pipeline actually walks, in one go. Each import here
#    is one that has already broken, or sits directly after one that did.
# -----------------------------------------------------------------------------
PROGRESS "3/5 full import chain"
python - <<'PY' 2>&1 | tail -30
mods = [
    ("torch",                                   "torch"),
    ("jax",                                     "jax"),
    ("haiku",                                   "haiku"),
    ("atomworks",                               "atomworks"),
    ("colabdesign",                             "colabdesign"),
    ("transformers.models.esm.modeling_esm",    None),
    ("proteinfoundation.utils.pdb_utils",       None),
    ("proteinfoundation.search.search_factory", None),
    ("proteinfoundation.proteina",              None),
    ("proteinfoundation.generate",              None),
    ("proteinfoundation.evaluate",              None),
    ("proteinfoundation.analyze",               None),
]
import importlib, sys
bad = []
for name, ver_of in mods:
    try:
        m = importlib.import_module(name)
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        bad.append(name)
        continue
    v = ""
    if ver_of:
        v = " " + str(getattr(importlib.import_module(ver_of), "__version__", "?"))
    print(f"ok   {name}{v}")
if bad:
    print("still broken:", bad)
    sys.exit(1)
print("every module the pipeline imports is importable")
PY
check "${PIPESTATUS[0]}" "full_import_chain"

PROGRESS "4/5 re-freeze"
uv pip freeze > "$RUN_ROOT/logs/pip_freeze.txt" 2>&1
check $? "uv_pip_freeze"
grep -E "^(atomworks|dm-haiku|torch|torchaudio|torchvision|jax|jaxlib|flax|transformers|biotite|colabdesign)" \
    "$RUN_ROOT/logs/pip_freeze.txt"

PROGRESS "5/5 complexa validate design"
complexa validate design configs/search_binder_local_pipeline.yaml 2>&1 | tail -18
check "${PIPESTATUS[0]}" "validate_design"

echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
exit "$STAGE_RC"
