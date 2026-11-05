#!/bin/bash
# =============================================================================
# One (target, method) cell of the sweep. Sourced by a per-cell .sbatch that
# sets TARGET, METHOD, SEED, NSAMPLES and CELL_TAG, so there is one copy of
# this logic rather than twenty near-identical job scripts.
#
# Every cell gets the SAME walltime allowance. PROJECT.md: "Methods are matched
# on GPU-hours, not on sample count. One beam search step over N x L candidates
# does far more work than N independent draws." So NSAMPLES is set per method
# to roughly fill the allowance, and the sample count is an OUTPUT of the cell,
# never the thing being equalised.
#
# Measured on the smoke run (job 11638953, 38_TNFalpha, best-of-n, H200):
#   generation  150.46 s for 2 samples   =  75.2 s/sample
#   evaluation  209.51 s for 2 samples   = 104.8 s/sample (1 native + 8 MPNN
#                                          sequences folded per sample)
#   job elapsed 716 s, of which ~356 s is fixed setup: model load, AF2
#   parameter load, Foldseek analysis.
# So best-of-n costs about 180 s per design beyond a ~400 s fixed cost.
# No equivalent measurement exists yet for the three search methods or for
# refinement; their NSAMPLES is set low deliberately, and whatever they produce
# inside the allowance is the measurement.
# =============================================================================
set -uo pipefail

export RUN_ID="complexa_armB_20261105"
export HPC_PROJECT_DIR="${HPC_PROJECT_DIR:-$HOME/hpcjobs}"
export RUN_DIR="${HPC_PROJECT_DIR}/runs/$RUN_ID"
export PYTHONPATH="${HPC_PROJECT_DIR}/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export RUN_ROOT="/scratch/$USER/complexa_armB_20261105"
export REPO="$RUN_ROOT/src/Proteina-Complexa"
export HF_HOME="$RUN_ROOT/cache/hf"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TMPDIR="/tmp/$USER/cxa"
# JAX pre-allocates by default and would starve torch; both are live here.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

: "${TARGET:?cell sbatch must set TARGET}"
: "${METHOD:?cell sbatch must set METHOD}"
: "${SEED:?cell sbatch must set SEED}"
: "${NSAMPLES:?cell sbatch must set NSAMPLES}"
: "${CELL_TAG:?cell sbatch must set CELL_TAG}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
# Minimum seconds that must remain for evaluation to be worth starting.
# source: chosen by me from the smoke run - 209 s of evaluation for 2 samples
# plus roughly 400 s of fixed load, rounded up.
MIN_EVAL_SECONDS=${MIN_EVAL_SECONDS:-900}

# Samples per forward pass. THIS IS A VRAM CONSTANT, not a throughput knob.
# source: measured. The repo default is 16
# (configs/pipeline/binder/binder_generate.yaml:19), and search.max_batch_size
# inherits from it (:49). Job 11646330 ran at that default with nsamples=32 and
# died in generation:
#   torch.OutOfMemoryError: Tried to allocate 4.67 GiB. GPU 0 has a total
#   capacity of 39.49 GiB of which 1.27 GiB is free.
# in concat_pair_feature_factory.py:442, which materialises a pair
# representation of [b, n_extended, n_extended, pair_dim] - quadratic in
# residues and linear in batch. 38_TNFalpha is 438 target residues plus a
# 50-120 residue binder, so the pair tensor is large before any batching.
#
# 2 is the batch the only successful generation in this run used: job 11638953,
# nsamples=2, peak GPU memory.used 12580 MiB of 143771 on an H200, read from
# its own telemetry CSV. 12580 MiB fits an A100-40GB (40960 MiB) with room to
# spare; 8x that does not. Setting it explicitly rather than inheriting the
# default is the point - S4 is "measure VRAM on the smallest unit of work
# before committing to the largest", and the largest was committed to without
# re-measuring. That was the defect.
BATCH_SIZE=${BATCH_SIZE:-2}

RUN_NAME="cxaB_${CELL_TAG}_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RUN_ROOT/logs" "$RUN_ROOT/sacct" "$RUN_DIR" || exit 1

STAMP() { date -u +%Y-%m-%dT%H:%M:%SZ; }
PROGRESS() { echo "PROGRESS $(STAMP) $*"; }
STAGE_RC=0
check() {
    if [ "$1" -ne 0 ]; then echo "FAIL $(STAMP) stage '$2' exited $1"; STAGE_RC=$1; return 1; fi
    echo "OK   $(STAMP) stage '$2' exited 0"; return 0
}

# How long this job may run, from Slurm itself rather than from a guess.
WALL_LIMIT_S=$(squeue -h -j "$SLURM_JOB_ID" -o "%l" | awk -F: '
  {n=split($0,a,"-"); if(n==2){split(a[2],t,":"); print a[1]*86400+t[1]*3600+t[2]*60+t[3]}
   else {split($0,t,":"); if(length(t)==3) print t[1]*3600+t[2]*60+t[3]; else print t[1]*60+t[2]}}')
WALL_LIMIT_S=${WALL_LIMIT_S:-7200}

# ------------------------------------------------------------------ telemetry
TELEM="$RUN_ROOT/logs/telemetry_${SLURM_JOB_ID}.csv"
TELEM_ERR="$RUN_ROOT/logs/telemetry_${SLURM_JOB_ID}.stderr"
HEARTBEAT="$RUN_ROOT/logs/heartbeat_${SLURM_JOB_ID}.log"
SMI_PID=""; HB_PID=""
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
             --format=csv,nounits -l 10 > "$TELEM" 2> "$TELEM_ERR" &
  SMI_PID=$!
else
  echo "WARN nvidia-smi not on PATH; no GPU telemetry will be sampled"
fi
( while true; do echo "$(date -Is) alive load=$(cut -d' ' -f1 /proc/loadavg)"; sleep 30; done ) \
    > "$HEARTBEAT" 2>&1 &
HB_PID=$!
cleanup() {
  for p in $SMI_PID $HB_PID; do
    [ -n "$p" ] || continue
    pkill -P "$p" 2>/dev/null     # children before the parent, or an orphaned
    kill    "$p" 2>/dev/null      # sleep/nvidia-smi keeps the cgroup billable
  done
  wait $SMI_PID $HB_PID 2>/dev/null
  if [ -s "$TELEM_ERR" ]; then
    echo "--- telemetry sampler stderr (captured, not discarded) ---"; cat "$TELEM_ERR"
  fi
}
trap cleanup EXIT TERM INT

echo "=== cell $CELL_TAG  job=$SLURM_JOB_ID  node=$(hostname)  $(STAMP) ==="
echo "target=$TARGET method=$METHOD seed=$SEED nsamples=$NSAMPLES batch_size=$BATCH_SIZE"
echo "run_name=$RUN_NAME  walltime_limit=${WALL_LIMIT_S}s"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

source "$REPO/.venv/bin/activate" || exit 1
cd "$REPO" || exit 1
set +u; source "$REPO/env.sh" >/dev/null; set -u
export HF_HOME="$RUN_ROOT/cache/hf"      # re-assert after .env (S16)

# =============================================================================
# 1. Generate. Run as its own step so that if it overruns, evaluation is
#    skipped deliberately with a reason, rather than Slurm killing the job
#    mid-evaluation and leaving nothing recorded.
# =============================================================================
PROGRESS "1/5 generate  method=$METHOD nsamples=$NSAMPLES"
GEN_START=$SECONDS
# NOTE on argument order. `--steps` is nargs='+' and the hydra overrides are
# positional, so `--steps generate ++run_name=...` makes argparse swallow the
# first override as a second value for --steps and die with
#   invalid choice: '++run_name=...' (choose from generate, filter, ...)
# which is exactly how jobs 11639185/229/242/259/264/280 each failed in under
# 25 s. The flag therefore goes AFTER every positional override.
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name="$RUN_NAME" \
    ++seed=$SEED \
    ++generation.task_name="$TARGET" \
    ++generation.search.algorithm="$METHOD" \
    ++generation.dataloader.dataset.nres.nsamples=$NSAMPLES \
    ++generation.dataloader.batch_size=$BATCH_SIZE \
    ++ckpt_path="$REPO/ckpts" \
    ++autoencoder_ckpt_path="$REPO/ckpts/complexa_ae.ckpt" \
    $EXTRA_OVERRIDES \
    --steps generate \
    --verbose 2>&1 | tail -60
GEN_RC=${PIPESTATUS[0]}
GEN_S=$((SECONDS - GEN_START))
echo "generate wall seconds: $GEN_S"
check "$GEN_RC" "generate"

INF_DIR=$(find "$REPO/inference" -maxdepth 1 -type d -name "*${RUN_NAME}*" | head -1)
NPDB=0
[ -n "$INF_DIR" ] && NPDB=$(find "$INF_DIR" -name "*.pdb" | wc -l)
echo "generated structures: $NPDB  (dir: ${INF_DIR:-none})"

if [ "$NPDB" -eq 0 ]; then
    echo "FAIL generation produced no structures; nothing to evaluate"
    # Not an error to hide: a method that produces nothing inside its allowance
    # is a result, and the job still gets charged for the GPU time it held.
    STAGE_RC=${GEN_RC:-20}
    [ "$STAGE_RC" -eq 0 ] && STAGE_RC=20
    echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
    exit "$STAGE_RC"
fi

# =============================================================================
# 2. Is there time left to evaluate what was generated?
# =============================================================================
REMAIN=$((WALL_LIMIT_S - SECONDS))
echo "seconds elapsed=$SECONDS remaining=$REMAIN  (need >= $MIN_EVAL_SECONDS)"
if [ "$REMAIN" -lt "$MIN_EVAL_SECONDS" ]; then
    echo "FAIL not enough walltime left to evaluate; stopping before Slurm kills us"
    echo "     $NPDB structures were generated and are on disk at $INF_DIR"
    echo "     They are NOT scored, so they contribute no successes."
    STAGE_RC=21
    echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
    exit "$STAGE_RC"
fi

# =============================================================================
# 3. Filter, evaluate, analyze. Return codes checked between stages (S13):
#    a timed-out generation followed by an unconditional evaluate is exactly
#    the failure that makes a job report success over partial samples.
# =============================================================================
PROGRESS "2/5 filter + evaluate + analyze"
EVAL_START=$SECONDS
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name="$RUN_NAME" \
    ++seed=$SEED \
    ++generation.task_name="$TARGET" \
    ++generation.search.algorithm="$METHOD" \
    ++generation.dataloader.dataset.nres.nsamples=$NSAMPLES \
    ++generation.dataloader.batch_size=$BATCH_SIZE \
    ++ckpt_path="$REPO/ckpts" \
    ++autoencoder_ckpt_path="$REPO/ckpts/complexa_ae.ckpt" \
    ++metric.sequence_types="[self,mpnn]" \
    ++metric.num_redesign_seqs=8 \
    ++metric.inverse_folding_model=protein_mpnn \
    ++metric.compute_monomer_metrics=false \
    ++metric.compute_designability=false \
    ++metric.compute_codesignability=false \
    ++metric.compute_esm_metrics=false \
    $EXTRA_OVERRIDES \
    --steps filter evaluate analyze \
    --verbose 2>&1 | tail -80
EVAL_RC=${PIPESTATUS[0]}
echo "evaluate wall seconds: $((SECONDS - EVAL_START))"
check "$EVAL_RC" "evaluate"

# =============================================================================
# 4. Record. Scoped to THIS run_name's directory, so no other run's output and
#    no earlier job of this run can be swept in (S14).
# =============================================================================
PROGRESS "3/5 locate this job's evaluation output"
EVAL_DIR=$(find "$REPO/evaluation_results" -maxdepth 1 -type d -name "*${RUN_NAME}*" | head -1)
echo "EVAL_DIR=${EVAL_DIR:-none}"
if [ -z "$EVAL_DIR" ]; then
    echo "FAIL no evaluation directory for run_name=$RUN_NAME"
    STAGE_RC=22
else
    PROGRESS "4/5 emit designs and per-sequence scores"
    python "$RUN_ROOT/tools/record_results.py" \
        --eval-dir "$EVAL_DIR" --target "$TARGET" --method "$METHOD" --seed "$SEED" \
        --seq-types self,mpnn 2>&1 | tail -20
    check "${PIPESTATUS[0]}" "record_results"
fi

PROGRESS "5/5 cell summary"
echo "cell=$CELL_TAG target=$TARGET method=$METHOD seed=$SEED"
echo "generated=$NPDB generate_s=$GEN_S total_s=$SECONDS"
[ -f "$TELEM" ] && echo "telemetry rows: $(wc -l < "$TELEM")"
[ -f "$TELEM" ] && echo "peak GPU memory.used MiB: $(awk -F, 'NR>1{if($6+0>m)m=$6+0}END{print m}' "$TELEM")"
echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
exit "$STAGE_RC"
