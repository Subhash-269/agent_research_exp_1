#!/bin/bash
# =============================================================================
# Phase 0, job 2: repair the two stages that failed in job 11637680, and record
# that job's cost. CPU ONLY, --gres=none, 0 GPU-hours.
#
# What failed in 11637680 (observed, not inferred):
#
#  pip_freeze         exit 127. `pip` is not an executable inside a uv-created
#                     venv; uv does not install it. The freeze is taken with
#                     `uv pip freeze` here instead.
#
#  download_community exit 1, "Failed to download ESM2 weights". The traceback
#                     ends at:
#                       torchaudio/lib/_torchaudio.abi3.so:
#                       undefined symbol: torch_library_impl
#                     env/build_uv_env.sh pins torch to 2.7.0+cu126 but leaves
#                     torchvision and torchaudio unpinned, so the resolver took
#                     a torchaudio built against a different torch ABI.
#                     transformers imports torchaudio unconditionally via
#                     audio_utils, so every transformers model class -
#                     including ESM2 - fails to import. Pinning torchaudio to
#                     the matching 2.7.0 build is the fix.
#
# Not observed: whether the resolver would pick the same torchaudio again on a
# rebuild. The pin removes the question rather than answering it.
# =============================================================================
#SBATCH --job-name=cxa_envfix
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
mkdir -p "$TMPDIR" "$RUN_ROOT/logs" "$RUN_DIR" || exit 1
export PATH="$HOME/.local/bin:$PATH"

STAMP() { date -u +%Y-%m-%dT%H:%M:%SZ; }
PROGRESS() { echo "PROGRESS $(STAMP) $*"; }
STAGE_RC=0
check() {
    if [ "$1" -ne 0 ]; then echo "FAIL $(STAMP) stage '$2' exited $1"; STAGE_RC=$1; return 1; fi
    echo "OK   $(STAMP) stage '$2' exited 0"; return 0
}

echo "=== cxa_envfix  job=$SLURM_JOB_ID  node=$(hostname)  $(STAMP) ==="
source "$REPO/.venv/bin/activate" || exit 1
cd "$REPO" || exit 1

# -----------------------------------------------------------------------------
# 1. Charge job 11637680. AGENTS.md: job_cost for EVERY job, including failed
#    ones, and a job's own sacct figures are not final while it runs - so it is
#    charged from a later job, which is this one. Fields come from
#    `sacct -j <id> -X -P -n -o JobID,State,ElapsedRaw,Start,End,Partition,NodeList`
#    and that output is also saved to disk (S11).
# -----------------------------------------------------------------------------
PROGRESS "1/6 charge job 11637680"
SACCT_DIR="$RUN_ROOT/sacct"; mkdir -p "$SACCT_DIR"
sacct -j 11637680 -X -P -n -o JobID,State,ElapsedRaw,Start,End,Partition,NodeList \
    | tee "$SACCT_DIR/11637680.psv"
check "${PIPESTATUS[0]}" "sacct_11637680"

python3 - <<'PY'
import os, emit
line = open(os.environ["RUN_ROOT"] + "/sacct/11637680.psv").read().strip().splitlines()[0]
jid, state, elapsed, start, end, part, node = line.split("|")
with emit.stage("admin", note="charging the phase-0 environment build"):
    emit.job_cost(jid, state=state, elapsed_s=int(elapsed),
                  gpus=0,                    # --gres=none: a CPU job holds no GPU
                  start_utc=start, end_utc=end, partition=part, node=node,
                  log="logs/11637680.out")
print("charged", jid, state, elapsed, "s, 0 GPUs")
PY
check $? "emit_job_cost_11637680"

# -----------------------------------------------------------------------------
# 2. Pin torchaudio to the torch it was built against.
#    source: env/build_uv_env.sh installs torch==2.7.0+cu126 from this index;
#    torchaudio 2.7.0 is the matching release on the same index.
# -----------------------------------------------------------------------------
PROGRESS "2/6 pin torchaudio to 2.7.0+cu126"
uv pip install "torchaudio==2.7.0" "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -20
check "${PIPESTATUS[0]}" "pin_torchaudio"

PROGRESS "2b/6 does transformers import ESM now?"
python - <<'PY' 2>&1 | tail -20
import sys
import torch, torchaudio
print("torch", torch.__version__, "torchaudio", torchaudio.__version__)
from transformers import AutoModel, AutoTokenizer   # noqa: F401
from transformers.models.esm import modeling_esm    # noqa: F401
print("transformers ESM imports OK")
PY
check "${PIPESTATUS[0]}" "transformers_esm_import"

# -----------------------------------------------------------------------------
# 3. Record the resolved dependency set, this time with a command that exists.
# -----------------------------------------------------------------------------
PROGRESS "3/6 uv pip freeze"
uv pip freeze > "$RUN_ROOT/logs/pip_freeze.txt" 2>&1
check $? "uv_pip_freeze"
echo "packages pinned: $(wc -l < "$RUN_ROOT/logs/pip_freeze.txt")"
grep -E "^(torch|torchvision|torchaudio|jax|jaxlib|transformers|colabdesign|alphafold)" \
    "$RUN_ROOT/logs/pip_freeze.txt"

# -----------------------------------------------------------------------------
# 4. Retry the community download now that transformers imports.
# -----------------------------------------------------------------------------
PROGRESS "4/6 complexa download --all (retry)"
complexa download --all 2>&1 | tail -40
check "${PIPESTATUS[0]}" "download_community_retry"
complexa download --status 2>&1 | tail -20

# -----------------------------------------------------------------------------
# 5. R3: exit zero is not sufficient - name the artifacts and check they exist.
# -----------------------------------------------------------------------------
PROGRESS "5/6 artifact checks"
ARTIFACT_RC=0
for f in "$REPO/ckpts/complexa.ckpt" \
         "$REPO/ckpts/complexa_ae.ckpt" \
         "$REPO/.venv/bin/foldseek" \
         "$RUN_ROOT/logs/pip_freeze.txt"; do
    if [ -s "$f" ]; then echo "present: $f ($(stat -c%s "$f") bytes)"
    else echo "ABSENT : $f"; ARTIFACT_RC=1; fi
done
# AF2 parameters are a directory of .npz files, not one file.
ls -la "$REPO/community_models/ckpts/AF2" 2>&1 | head -10
check "$ARTIFACT_RC" "artifacts_present"

# -----------------------------------------------------------------------------
# 6. Dump the target definitions this run will use, so the report can quote
#    them from the repository rather than from PROJECT.md's table.
# -----------------------------------------------------------------------------
PROGRESS "6/6 target definitions"
python - <<'PY' 2>&1 | tee "$RUN_ROOT/logs/targets_used.txt"
import os, yaml
p = os.environ["REPO"] + "/configs/targets/targets_dict.yaml"
cfg = yaml.safe_load(open(p))["target_dict_cfg"]
for k in ("38_TNFalpha", "33_TrkA"):
    t = cfg[k]
    chains = sorted({s.strip()[0] for s in str(t["target_input"]).split(",")})
    print(f"{k}: chains={chains} n_chains={len(chains)} input={t['target_input']} "
          f"hotspots={t['hotspot_residues']} binder_len={t['binder_length']} "
          f"file={t['target_path']}")
PY
check "${PIPESTATUS[0]}" "targets_used"

echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
exit "$STAGE_RC"
