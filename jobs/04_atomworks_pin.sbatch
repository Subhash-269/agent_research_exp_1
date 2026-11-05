#!/bin/bash
# =============================================================================
# Phase 0, job 4: pin atomworks to a version that has the API the repository
# calls. CPU ONLY, --gres=none, 0 GPU-hours.
#
# Why a second attempt: job 11638677's version probe failed at the first step.
# `uv pip index versions` does not exist in this uv build - "error:
# unrecognized subcommand 'index'", exit 2 - so the candidate list came back
# empty and the loop concluded, wrongly, that no version works. The conclusion
# was an artifact of an empty list, not a result. That job's charging stage did
# work, and 11638284/11638336/11638423 are now on the ledger.
#
# The release list is now taken from the PyPI JSON API instead, which this
# session already queried: atomworks is RosettaCommons/atomworks and has 17
# releases, 1.0.0 through 3.0.4. The installed 3.0.4 is the newest, which is
# how an unpinned install landed on it.
#
# This probe does NOT install-and-import each candidate. It unpacks each wheel
# with --no-deps into a throwaway directory and reads the source of
# atomworks/io/utils/selection.py, looking for the class and the constructor
# the repository calls. That is a text check with no dependency resolution and
# no import, so it cannot be confounded by an unrelated broken dependency, and
# it costs seconds per version instead of minutes.
# =============================================================================
#SBATCH --job-name=cxa_awpin
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

echo "=== cxa_awpin  job=$SLURM_JOB_ID  node=$(hostname)  $(STAMP) ==="
source "$REPO/.venv/bin/activate" || exit 1
cd "$REPO" || exit 1

PROGRESS "0/5 charge the job that found the uv bug"
python "$RUN_ROOT/tools/charge_jobs.py" 11638677 2>&1 | tail -6
check "${PIPESTATUS[0]}" "charge_11638677"

# -----------------------------------------------------------------------------
# 1. Release list, straight from the index.
# -----------------------------------------------------------------------------
PROGRESS "1/5 atomworks releases from the PyPI JSON API"
curl -s https://pypi.org/pypi/atomworks/json \
  | python3 -c "
import json,sys
d = json.load(sys.stdin)
print(d['info']['project_urls'].get('repository'))
for v in sorted(d['releases']):
    print(v)
" | tee "$RUN_ROOT/logs/atomworks_releases.txt"
check "${PIPESTATUS[0]}" "pypi_releases"

# -----------------------------------------------------------------------------
# 2. Which release carries AtomSelectionStack.from_contig?
#    Read from the wheel's source, newest first. No import, no dependency
#    resolution: a text check cannot be confounded by an unrelated broken dep.
# -----------------------------------------------------------------------------
PROGRESS "2/5 probe each release's selection.py"
VERSIONS=$(grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' "$RUN_ROOT/logs/atomworks_releases.txt" | sort -Vr)
echo "probing, newest first: $(echo "$VERSIONS" | tr '\n' ' ')"

PROBE_DIR="$TMPDIR/awprobe"
WORKING=""
REPORT="$RUN_ROOT/logs/atomworks_probe.txt"
: > "$REPORT"
for v in $VERSIONS; do
    rm -rf "$PROBE_DIR"; mkdir -p "$PROBE_DIR"
    if ! uv pip install --no-deps --target "$PROBE_DIR" "atomworks==$v" >/dev/null 2>&1; then
        echo "$v  download/unpack failed" | tee -a "$REPORT"
        continue
    fi
    SEL="$PROBE_DIR/atomworks/io/utils/selection.py"
    if [ ! -f "$SEL" ]; then
        echo "$v  no atomworks/io/utils/selection.py in the wheel" | tee -a "$REPORT"
        continue
    fi
    HAS_CLASS=$(grep -c "^class AtomSelectionStack" "$SEL")
    HAS_CTOR=$(grep -c "def from_contig" "$SEL")
    echo "$v  class=$HAS_CLASS from_contig=$HAS_CTOR" | tee -a "$REPORT"
    if [ "$HAS_CLASS" -ge 1 ] && [ "$HAS_CTOR" -ge 1 ]; then
        WORKING="$v"
        break
    fi
done
rm -rf "$PROBE_DIR"

if [ -z "$WORKING" ]; then
    echo "FAIL no atomworks release 1.0.0-3.0.4 defines AtomSelectionStack with from_contig"
    echo "     The repository calls an API that no published atomworks provides."
    STAGE_RC=10
else
    echo "SELECTED atomworks==$WORKING"
    echo "$WORKING" > "$RUN_ROOT/logs/atomworks_pinned_version.txt"
fi
check "$STAGE_RC" "probe_releases"

# -----------------------------------------------------------------------------
# 3. Install the selected version properly, with the extras the build used.
# -----------------------------------------------------------------------------
if [ -n "$WORKING" ]; then
    PROGRESS "3/5 install atomworks[ml,openbabel]==$WORKING"
    uv pip install "atomworks[ml,openbabel]==$WORKING" 2>&1 | tail -25
    check "${PIPESTATUS[0]}" "install_atomworks"

    PROGRESS "3b/5 did that disturb torch, jax or transformers?"
    # A downgrade can satisfy one import and break another. Both directions are
    # checked rather than assumed.
    python - <<'PY' 2>&1 | tail -25
import atomworks
print("atomworks", getattr(atomworks, "__version__", "?"))
from atomworks.io.utils.selection import AtomSelectionStack
print("AtomSelectionStack.from_contig:", hasattr(AtomSelectionStack, "from_contig"))
from proteinfoundation.utils import pdb_utils            # the module that raised
print("proteinfoundation.utils.pdb_utils imports OK")
from proteinfoundation.proteina import Proteina          # noqa: F401
print("proteinfoundation.proteina imports OK")
import proteinfoundation.generate                        # noqa: F401
print("proteinfoundation.generate imports OK")
import torch, jax
print("torch", torch.__version__, "jax", jax.__version__)
from transformers.models.esm import modeling_esm         # noqa: F401
print("transformers ESM imports OK")
import biotite
print("biotite", biotite.__version__)
PY
    check "${PIPESTATUS[0]}" "post_pin_imports"
fi

# -----------------------------------------------------------------------------
# 4. Re-freeze and re-validate.
# -----------------------------------------------------------------------------
PROGRESS "4/5 re-freeze"
uv pip freeze > "$RUN_ROOT/logs/pip_freeze.txt" 2>&1
check $? "uv_pip_freeze"
grep -E "^(atomworks|torch|torchaudio|torchvision|jax|jaxlib|transformers|biotite)" \
    "$RUN_ROOT/logs/pip_freeze.txt"

PROGRESS "5/5 complexa validate design"
set +u; source "$REPO/env.sh" >/dev/null; set -u
export HF_HOME="$RUN_ROOT/cache/hf"
complexa validate design configs/search_binder_local_pipeline.yaml 2>&1 | tail -20
check "${PIPESTATUS[0]}" "validate_design"

echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
exit "$STAGE_RC"
