#!/bin/bash
# =============================================================================
# Phase 1: hardware attestation + the first end-to-end pipeline run, on a
# MULTI-CHAIN target.
#
# S10 (PROJECT.md): "Run at least one multi-chain target early, before
# committing budget to the rest." 38_TNFalpha carries three chains (A, B, C) -
# more than any other target in the table - so if a single-chain assumption is
# going to inflate the unique-success count, this is where it shows.
#
# S2's five hardware facts are established at the top of THIS job rather than
# in a separate one: a separate attestation job would cost its own queue wait
# and its own allocation for information this job needs anyway. If any fact is
# missing, or the card is below compute capability 8.0, the job exits before
# the model is loaded.
#
# Deliberately small. This is the job that finds out whether the pipeline runs
# at all and what its output looks like; it is not a measurement of any method.
# =============================================================================
#SBATCH --job-name=cxa_smoke_tnf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -uo pipefail

# ---------------------------------------------------------------- run identity
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
# AGENTS.md: without this JAX pre-allocates the GPU and starves torch. Both
# frameworks are live in this job (torch for the generative model, JAX for the
# AF2 reward and the AF2 refold).
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --- the experimental cell this job runs -------------------------------------
export TARGET="38_TNFalpha" # source: configs/targets/targets_dict.yaml (three chains)
export METHOD="best-of-n"   # source: binder_generate.yaml search.algorithm options
# S15: seeded explicitly, distinct per job, and recorded. Base is the run date.
SEED=$((20261105 + 1))
NSAMPLES=2                  # source: chosen by me, no external source. Smallest
                            # number that still exercises a multi-design harvest.
NREDESIGN=8                 # source: PROJECT.md evaluation step 1, "8 redesigned
                            # binder sequences via ProteinMPNN"
RUN_NAME="cxaB_smoke_tnf_${SLURM_JOB_ID}"

export ATTEST_DIR="$RUN_ROOT/attest"
mkdir -p "$ATTEST_DIR" "$RUN_DIR" "$RUN_ROOT/logs" "$TMPDIR" || exit 1

STAMP() { date -u +%Y-%m-%dT%H:%M:%SZ; }
PROGRESS() { echo "PROGRESS $(STAMP) $*"; }
STAGE_RC=0
check() {
    if [ "$1" -ne 0 ]; then echo "FAIL $(STAMP) stage '$2' exited $1"; STAGE_RC=$1; return 1; fi
    echo "OK   $(STAMP) stage '$2' exited 0"; return 0
}

# ------------------------------------------------------------------ telemetry
TELEM="$RUN_ROOT/logs/telemetry_${SLURM_JOB_ID}.csv"
TELEM_ERR="$RUN_ROOT/logs/telemetry_${SLURM_JOB_ID}.stderr"
HEARTBEAT="$RUN_ROOT/logs/heartbeat_${SLURM_JOB_ID}.log"
SMI_PID=""; HB_PID=""
if command -v nvidia-smi >/dev/null; then
  # The sampler's own stderr goes to its own file and is printed into this log
  # at cleanup. telemetry_demo.sbatch folds it into the CSV, which both
  # corrupts the CSV and hides why an empty one was written.
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
             --format=csv,nounits -l 5 > "$TELEM" 2> "$TELEM_ERR" &
  SMI_PID=$!
else
  echo "WARN nvidia-smi not on PATH; no GPU telemetry will be sampled"
fi
( while true; do echo "$(date -Is) alive load=$(cut -d' ' -f1 /proc/loadavg)"; sleep 15; done ) \
    > "$HEARTBEAT" 2>&1 &
HB_PID=$!
cleanup() {
  for p in $SMI_PID $HB_PID; do
    [ -n "$p" ] || continue
    pkill -P "$p" 2>/dev/null    # CHILDREN first: an orphaned sleep or
    kill    "$p" 2>/dev/null     # nvidia-smi keeps the cgroup alive and Slurm
  done                           # keeps billing after the last line of work
  wait $SMI_PID $HB_PID 2>/dev/null
  if [ -s "$TELEM_ERR" ]; then
    echo "--- telemetry sampler stderr (captured, not discarded) ---"; cat "$TELEM_ERR"
  fi
}
trap cleanup EXIT TERM INT

echo "=== cxa_smoke_tnf  job=$SLURM_JOB_ID  node=$(hostname)  $(STAMP) ==="
echo "target=$TARGET method=$METHOD seed=$SEED nsamples=$NSAMPLES run_name=$RUN_NAME"

# =============================================================================
# 1. S2 - attest the hardware before planning around it. All five facts.
# =============================================================================
PROGRESS "1/8 hardware attestation (S2)"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv \
    | tee "$ATTEST_DIR/nvidia_smi_${SLURM_JOB_ID}.csv"
check "${PIPESTATUS[0]}" "nvidia_smi"

source "$REPO/.venv/bin/activate" || exit 1
cd "$REPO" || exit 1

python - <<'PY' 2>&1 | tee "$ATTEST_DIR/hardware_facts_${SLURM_JOB_ID}.txt"
import json, os, sys, torch
if not torch.cuda.is_available():
    print("FAIL torch reports no CUDA device"); sys.exit(2)
p = torch.cuda.get_device_properties(torch.cuda.current_device())
facts = {
    "gpu_present": True,
    "vendor": "NVIDIA (torch.version.hip is None -> CUDA build, not ROCm)",
    "device_name": p.name,
    "vram_total_mib": p.total_memory // (1024 ** 2),
    "compute_capability": f"{p.major}.{p.minor}",
    "cuda_runtime": torch.version.cuda,
    "torch_version": torch.__version__,
    "bf16_supported": torch.cuda.is_bf16_supported(),
}
print(json.dumps(facts, indent=2))
for k in ("device_name", "vram_total_mib", "compute_capability", "cuda_runtime"):
    if not facts[k]:
        print(f"FAIL hardware fact missing: {k}"); sys.exit(2)
if (p.major, p.minor) < (8, 0):
    print(f"FAIL compute capability {p.major}.{p.minor}: no bf16, no Flash-Attention"); sys.exit(3)
with open(os.environ["ATTEST_DIR"] + f"/hardware_facts_{os.environ['SLURM_JOB_ID']}.json", "w") as fh:
    json.dump(facts, fh, indent=2)
print("attested")
PY
check "${PIPESTATUS[0]}" "torch_attest" || { echo "S2: stop if any fact is absent"; exit "$STAGE_RC"; }

PROGRESS "1b/8 S4 - VRAM and speed at float32 and bfloat16 on this device"
python - <<'PY' 2>&1 | tee "$ATTEST_DIR/precision_probe_${SLURM_JOB_ID}.txt"
import json, os, time, torch
N, REPS = 4096, 20       # source: chosen by me, no external source
dev = torch.device("cuda")
out = {"device": torch.cuda.get_device_properties(0).name, "matmul_n": N, "reps": REPS}
for label, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    a = torch.randn(N, N, device=dev, dtype=dtype); b = torch.randn(N, N, device=dev, dtype=dtype)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(REPS):
        c = a @ b
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    out[label] = {"peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
                  "seconds_per_matmul": round(dt / REPS, 6)}
    del a, b, c
print(json.dumps(out, indent=2))
with open(os.environ["ATTEST_DIR"] + f"/precision_probe_{os.environ['SLURM_JOB_ID']}.json", "w") as fh:
    json.dump(out, fh, indent=2)
PY
check "${PIPESTATUS[0]}" "precision_probe"

PROGRESS "1c/8 jax devices"
python -c "import jax; print('jax', jax.__version__, jax.devices())" 2>&1 | tail -3
check $? "jax_devices"

# =============================================================================
# 2. S10 - what does the multi-chain target actually look like to the pipeline?
#    Asked BEFORE generating, so a single-chain assumption is loud here rather
#    than silent in the success count.
# =============================================================================
PROGRESS "2/8 multi-chain target inspection (S10)"
python - <<'PY' 2>&1 | tee "$RUN_ROOT/logs/target_chains_${SLURM_JOB_ID}.txt"
import os, collections, yaml
repo = os.environ["REPO"]; task = os.environ["TARGET"]
cfg = yaml.safe_load(open(f"{repo}/configs/targets/targets_dict.yaml"))["target_dict_cfg"][task]
print("task:", task)
print("target_input:", cfg["target_input"])
print("hotspots:", cfg["hotspot_residues"])
print("binder_length:", cfg["binder_length"])
path = cfg["target_path"]
if path.startswith("./"):
    path = os.path.join(repo, path[2:])
print("target_path:", path, "exists:", os.path.exists(path))
if os.path.exists(path):
    chains = collections.Counter()
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            chains[line[21]] += 1
    print("chains in the shipped PDB:", dict(chains))
    print("n_chains:", len(chains))
    if len(chains) < 2:
        print("WARNING this target was selected as multi-chain but the file has one chain")
PY
check "${PIPESTATUS[0]}" "target_chains"

# =============================================================================
# 3. Generation + evaluation, end to end.
#    Overrides, each with a reason:
#      sequence_types [self, mpnn]  PROJECT.md step 1 requires BOTH the model's
#                                   own native sequence and the MPNN redesigns,
#                                   reported as separate categories. The repo
#                                   default is [self] only.
#      num_redesign_seqs 8          PROJECT.md step 1: "8 redesigned binder
#                                   sequences". Repo default is 2.
#      inverse_folding_model        PROJECT.md says ProteinMPNN. Repo default is
#        protein_mpnn               soluble_mpnn, which is a different model.
#      compute_monomer_metrics      Monomer designability/codesignability are
#        false                      not among the three numbers PROJECT.md asks
#                                   for, and each one costs GPU time inside the
#                                   100-hour ceiling. Recorded as a deviation.
#      compute_esm_metrics false    Same reason: ESM perplexity is not a
#                                   required metric.
# =============================================================================
PROGRESS "3/8 complexa design (generate -> filter -> evaluate -> analyze)"
START_DESIGN=$(date +%s)
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name="$RUN_NAME" \
    ++seed=$SEED \
    ++generation.task_name="$TARGET" \
    ++generation.search.algorithm="$METHOD" \
    ++generation.dataloader.dataset.nres.nsamples=$NSAMPLES \
    ++generation.search.best_of_n.replicas=1 \
    ++ckpt_path="$REPO/ckpts" \
    ++autoencoder_ckpt_path="$REPO/ckpts/complexa_ae.ckpt" \
    ++metric.sequence_types="[self,mpnn]" \
    ++metric.num_redesign_seqs=$NREDESIGN \
    ++metric.inverse_folding_model=protein_mpnn \
    ++metric.compute_monomer_metrics=false \
    ++metric.compute_designability=false \
    ++metric.compute_codesignability=false \
    ++metric.compute_esm_metrics=false \
    --verbose 2>&1 | tail -150
DESIGN_RC=${PIPESTATUS[0]}
END_DESIGN=$(date +%s)
echo "design wall seconds: $((END_DESIGN - START_DESIGN))"
check "$DESIGN_RC" "complexa_design"

# =============================================================================
# 4. What did it write? Printed in full so the next job can be written against
#    the real layout instead of a guess.
# =============================================================================
PROGRESS "4/8 output layout"
echo "--- ./inference dirs matching this run ---"
find "$REPO/inference" -maxdepth 1 -name "*${RUN_NAME}*" 2>/dev/null
echo "--- ./evaluation_results dirs matching this run ---"
find "$REPO/evaluation_results" -maxdepth 2 -name "*${RUN_NAME}*" 2>/dev/null
echo "--- every sequence_type_stats.json under this run's evaluation output ---"
EVAL_DIR=$(find "$REPO/evaluation_results" -maxdepth 2 -type d -name "*${RUN_NAME}*" 2>/dev/null | head -1)
echo "EVAL_DIR=$EVAL_DIR"
if [ -n "$EVAL_DIR" ]; then
    find "$EVAL_DIR" -name "sequence_type_stats.json" | head -20
    echo "count: $(find "$EVAL_DIR" -name 'sequence_type_stats.json' | wc -l)"
    echo "--- result CSVs ---"
    find "$EVAL_DIR" -name "*.csv" | head -20
else
    echo "no evaluation directory matched run_name=$RUN_NAME"
    STAGE_RC=6
fi

# =============================================================================
# 5. The stats schema, printed before anything is recorded against it.
# =============================================================================
PROGRESS "5/8 stats schema"
if [ -n "$EVAL_DIR" ]; then
    python "$RUN_ROOT/tools/record_results.py" --dump-schema \
        --eval-dir "$EVAL_DIR" --target "$TARGET" --method "$METHOD" --seed $SEED 2>&1 | tail -40
    check "${PIPESTATUS[0]}" "dump_schema"
fi

# =============================================================================
# 6. Record designs and per-sequence scores through emit.
# =============================================================================
PROGRESS "6/8 emit designs and scores"
if [ -n "$EVAL_DIR" ] && [ "$STAGE_RC" -eq 0 ]; then
    python "$RUN_ROOT/tools/record_results.py" \
        --eval-dir "$EVAL_DIR" --target "$TARGET" --method "$METHOD" --seed $SEED \
        --seq-types self,mpnn 2>&1 | tail -30
    check "${PIPESTATUS[0]}" "record_results"
fi

# =============================================================================
# 7. Observed min and max of each metric. PROJECT.md step 4 asks for these so a
#    reader can see that the threshold discriminates rather than passing
#    everything - which is exactly what applying "< 7" to the stored ipAE does.
# =============================================================================
PROGRESS "7/8 metric ranges"
python - <<'PY' 2>&1 | tee "$RUN_ROOT/logs/metric_ranges_${SLURM_JOB_ID}.txt"
import json, os, pathlib
run_dir = pathlib.Path(os.environ["RUN_DIR"]) / "events"
job = os.environ["SLURM_JOB_ID"]
vals = {}
for p in run_dir.glob(f"{job}-*.jsonl"):
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") != "score":
            continue
        for m in ("ipae_stored", "ipae_angstrom", "plddt_0to1", "scrmsd_angstrom"):
            if m in r:
                vals.setdefault(m, []).append(r[m])
if not vals:
    print("no score records from this job")
else:
    for m, xs in vals.items():
        print(f"{m}: n={len(xs)} min={min(xs):.4f} max={max(xs):.4f}")
    print()
    print("Why both ipAE columns are reported: applying the '< 7' threshold to")
    print("the STORED value would pass essentially every design. The Angstrom")
    print("column is stored x 31 and is the one the threshold belongs on.")
PY
check "${PIPESTATUS[0]}" "metric_ranges"

PROGRESS "8/8 done"
echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
[ -f "$TELEM" ] && echo "telemetry rows: $(wc -l < "$TELEM")"
[ -f "$TELEM" ] && echo "peak GPU memory.used (MiB) seen by the sampler: $(awk -F, 'NR>1{if($6+0>m)m=$6+0}END{print m}' "$TELEM")"
exit "$STAGE_RC"
