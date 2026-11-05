#!/bin/bash
# =============================================================================
# Phase 0, job 1: build the Proteina-Complexa environment. CPU ONLY.
#
# Runs on --partition=short with --gres=none, so it is charged 0 GPU-hours
# (PROJECT.md "Compute": the ceiling counts allocated GPU time). AGENTS.md:
# "prefer a CPU-only job for any install or verification step, which costs
# zero GPU-hours." S6: large downloads are staged outside the GPU allocation.
#
# Every constant below carries its source (R1).
# =============================================================================
#SBATCH --job-name=cxa_envbuild
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -uo pipefail

# --- identity of this run ----------------------------------------------------
# source: chosen by me, no external source. One RUN_ID for the whole run
# (AGENTS.md, "Recording what the run produces").
export RUN_ID="complexa_armB_20261105"
export HPC_PROJECT_DIR="${HPC_PROJECT_DIR:-$HOME/hpcjobs}"
export RUN_DIR="${HPC_PROJECT_DIR}/runs/$RUN_ID"
export PYTHONPATH="${HPC_PROJECT_DIR}/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# source: chosen by me, no external source. S16 requires a run-unique root that
# does not reuse or extend an earlier run's; /scratch/$USER was surveyed first
# (sacct/S1_quota_evidence.txt) and held no directory of this name.
export RUN_ROOT="/scratch/$USER/complexa_armB_20261105"
export REPO="$RUN_ROOT/src/Proteina-Complexa"

# Caches under this run's own root, NOT the shared /scratch/$USER/{hf,pip,torch}
# left by earlier runs of this project (S16). The connector exports those by
# default; these lines override them deliberately.
export HF_HOME="$RUN_ROOT/cache/hf"
export PIP_CACHE_DIR="$RUN_ROOT/cache/pip"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export UV_CACHE_DIR="$RUN_ROOT/cache/uv"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TMPDIR="/tmp/$USER/cxa"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" \
         "$TMPDIR" "$RUN_ROOT/logs" "$RUN_ROOT/tools" "$RUN_DIR" || exit 1

STAMP() { date -u +%Y-%m-%dT%H:%M:%SZ; }
PROGRESS() { echo "PROGRESS $(STAMP) $*"; }
# S13: every stage's return code is checked, and a failed stage stops the ones
# after it. There is no unconditional `exit 0` anywhere in this script.
STAGE_RC=0
check() {   # check <rc> <stage name>
    if [ "$1" -ne 0 ]; then
        echo "FAIL $(STAMP) stage '$2' exited $1"
        STAGE_RC=$1
        return 1
    fi
    echo "OK   $(STAMP) stage '$2' exited 0"
    return 0
}

echo "=== cxa_envbuild  job=${SLURM_JOB_ID:-none}  node=$(hostname)  $(STAMP) ==="
echo "RUN_ID=$RUN_ID  RUN_DIR=$RUN_DIR  RUN_ROOT=$RUN_ROOT"
echo "--- free space on every volume this job touches (S1) ---"
df -h /home /scratch /tmp

# -----------------------------------------------------------------------------
# 0. Declare the run through emit, before anything else.
# -----------------------------------------------------------------------------
PROGRESS "0/7 emit run_started"
python3 - <<'PY'
import emit
with emit.stage("env", note="CPU-only environment build; charged 0 GPU-hours"):
    emit.write(
        "run_started",
        model="claude-opus-5",
        arm="B",
        harness="Claude Code (VSCode extension), non-interactive",
        # sha256 of AGENTS.md as issued, from `sha256sum AGENTS.md` -> exit 0
        # on the operator's laptop.
        brief_sha256="ccff61af25fa6d688ddf6df2043bdde9a85bf7e08e490213607885d7fc9e7280",
        workspace_root="C:/Users/subha/Desktop/Workspace/Personal/RA_Full_Run/workspace",
        effort="default",
        gpu_hour_ceiling=100.0,        # source: PROJECT.md, "Compute"
        brief_path="workspace/briefs/AGENTS.md",
    )
print("run_started written")
PY
check $? "emit_run_started" || { echo "cannot record this run; refusing to continue"; exit "$STAGE_RC"; }

# -----------------------------------------------------------------------------
# 1. Repository provenance
# -----------------------------------------------------------------------------
PROGRESS "1/7 repo provenance"
cd "$REPO" || exit 1
git rev-parse HEAD | tee "$RUN_ROOT/logs/repo_head.txt"
check $? "repo_head"

# -----------------------------------------------------------------------------
# 2. Build the uv environment exactly as the repository declares it.
#    gpu-env-build skill section 1: read what the repo declares, do not guess.
#    The authoritative installer is env/build_uv_env.sh (full install: torch
#    2.7.0+cu126, JAX 0.4.29, ColabDesign, tmol, rc-foundry).
#    No --root: .env_example resolves UV_VENV to ${LOCAL_CODE_PATH}/.venv, so
#    the venv must sit inside the repo or every tool path in .env is wrong.
# -----------------------------------------------------------------------------
PROGRESS "2/7 build_uv_env.sh (full install, python 3.12) - expect 20-60 min"
export PATH="$HOME/.local/bin:$PATH"
bash env/build_uv_env.sh --python 3.12 2>&1 | tail -250
BUILD_RC=${PIPESTATUS[0]}
check "$BUILD_RC" "build_uv_env"
if [ "$BUILD_RC" -ne 0 ]; then
    echo "environment build failed; not attempting downloads against a broken venv"
    exit "$BUILD_RC"
fi
# R3: exit zero is necessary, not sufficient - check the artifact exists.
if [ ! -x "$REPO/.venv/bin/python" ]; then
    echo "FAIL build_uv_env.sh exited 0 but $REPO/.venv/bin/python is absent"
    exit 3
fi
source "$REPO/.venv/bin/activate" || exit 1
echo "python: $(which python)  $(python --version 2>&1)"

PROGRESS "2b/7 recording the resolved dependency set (gpu-env-build step 7: pin everything)"
pip freeze > "$RUN_ROOT/logs/pip_freeze.txt" 2>&1
check $? "pip_freeze"
wc -l "$RUN_ROOT/logs/pip_freeze.txt"

# -----------------------------------------------------------------------------
# 3. .env. `complexa init` writes it from .env_example with placeholder paths;
#    the placeholders are then replaced with this run's real ones. The values
#    are written here, in a script that is committed, rather than typed on the
#    remote machine, so the configuration is auditable (R8).
# -----------------------------------------------------------------------------
PROGRESS "3/7 complexa init + .env paths"
cd "$REPO" || exit 1
complexa init 2>&1 | tail -30
check "${PIPESTATUS[0]}" "complexa_init"

if [ -f "$REPO/.env" ]; then
    sed -i \
      -e "s|^LOCAL_CODE_PATH=.*|LOCAL_CODE_PATH=$REPO|" \
      -e "s|^LOCAL_DATA_PATH=.*|LOCAL_DATA_PATH=$REPO/assets|" \
      -e "s|^LOCAL_CHECKPOINT_PATH=.*|LOCAL_CHECKPOINT_PATH=$REPO/ckpts|" \
      -e "s|^CLUSTER_USER=.*|CLUSTER_USER=$USER|" \
      "$REPO/.env"
    check $? "env_paths"
    echo "--- .env after rewrite (paths only) ---"
    grep -E "^(LOCAL_|COMMUNITY_|AF2_DIR|ESM_DIR|DATA_PATH|CKPT_PATH|FOLDSEEK_EXEC|UV_VENV)" "$REPO/.env"
else
    echo "FAIL complexa init did not produce $REPO/.env"
    STAGE_RC=4
fi

PROGRESS "3b/7 complexa init uv (generate env.sh for the uv runtime)"
complexa init uv 2>&1 | tail -20
check "${PIPESTATUS[0]}" "complexa_init_uv"

# -----------------------------------------------------------------------------
# 4. Weights. CPU work and bandwidth only - no GPU is held for this (S6).
#    --complexa : the Protein-Target binder model + its autoencoder
#    --all      : community models (ProteinMPNN, LigandMPNN, AF2, ESM2, RF3)
# -----------------------------------------------------------------------------
PROGRESS "4/7 complexa download --complexa"
complexa download --complexa 2>&1 | tail -60
check "${PIPESTATUS[0]}" "download_complexa"

PROGRESS "4b/7 complexa download --all (community models)"
complexa download --all 2>&1 | tail -80
check "${PIPESTATUS[0]}" "download_community"

echo "--- what landed ---"
complexa download --status 2>&1 | tail -40
ls -la "$REPO/ckpts" 2>&1 | head -20
du -sh "$REPO/ckpts" "$REPO/community_models" 2>&1

# -----------------------------------------------------------------------------
# 5. Foldseek. PROJECT.md names Foldseek as the clustering method, and
#    build_uv_env.sh explicitly does not install it.
#    source: env/build_uv_env.sh closing note, which gives this URL family.
#    It is placed at $REPO/.venv/bin/foldseek because that is where .env's
#    UV_FOLDSEEK_EXEC already points.
# -----------------------------------------------------------------------------
PROGRESS "5/7 foldseek"
cd "$RUN_ROOT/tools" || exit 1
if [ ! -x "$RUN_ROOT/tools/foldseek/bin/foldseek" ]; then
    curl -fsSL -o foldseek-linux-avx2.tar.gz https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
    check $? "foldseek_download" && { tar xzf foldseek-linux-avx2.tar.gz; check $? "foldseek_untar"; }
fi
if [ -x "$RUN_ROOT/tools/foldseek/bin/foldseek" ]; then
    cp -f "$RUN_ROOT/tools/foldseek/bin/foldseek" "$REPO/.venv/bin/foldseek"
    check $? "foldseek_install"
    "$REPO/.venv/bin/foldseek" version 2>&1 | head -3
    check $? "foldseek_version"
else
    echo "FAIL foldseek binary absent after download"
    STAGE_RC=5
fi

# -----------------------------------------------------------------------------
# 6. Import check on the CPU node. Catches a broken install here rather than
#    inside a GPU allocation. torch.cuda is expected to be False on a CPU node;
#    that is checked in the GPU attestation job, not this one.
# -----------------------------------------------------------------------------
PROGRESS "6/7 import check (CPU node; CUDA not expected here)"
cd "$REPO" || exit 1
python -c "
import torch, sys
print('torch', torch.__version__, 'cuda_build', torch.version.cuda)
import proteinfoundation; print('proteinfoundation imports OK')
try:
    import jax; print('jax', jax.__version__)
except Exception as e:
    print('jax import failed:', e); sys.exit(7)
" 2>&1 | tail -20
check "${PIPESTATUS[0]}" "import_check"

PROGRESS "6b/7 complexa validate design (config resolves?)"
complexa validate design configs/search_binder_local_pipeline.yaml 2>&1 | tail -40
check "${PIPESTATUS[0]}" "complexa_validate"

# -----------------------------------------------------------------------------
# 7. Summary. The exit code reflects whether every stage succeeded (S13).
# -----------------------------------------------------------------------------
PROGRESS "7/7 done"
echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
echo "venv       : $REPO/.venv"
echo "repo       : $REPO"
echo "ckpts      : $REPO/ckpts"
echo "foldseek   : $REPO/.venv/bin/foldseek"
df -h /home /scratch /tmp
exit "$STAGE_RC"
