#!/bin/bash
# =============================================================================
# Phase 1, job 1: attest the hardware before planning around it (S2) and
# measure VRAM on the smallest unit of work at two precisions (S4).
#
# Short on purpose. S2 says to stop if any of the five hardware facts is
# absent, so this is the cheapest possible job that can establish them, and
# nothing expensive is submitted until it has.
#
# Telemetry here follows AGENTS.md "Job scripts": a GPU sampler writing CSV, a
# heartbeat, the sampler's own stderr captured rather than discarded, and a
# cleanup trap that kills the samplers' CHILDREN before the samplers, so no
# orphaned sleep/nvidia-smi keeps the job cgroup alive and billing.
# telemetry_demo.sbatch has that bug; this does not copy it.
# =============================================================================
#SBATCH --job-name=cxa_attest
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00

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
# AGENTS.md "Constraints that cause silent failures": JAX pre-allocates by
# default and starves torch when they share one GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export ATTEST_DIR="$RUN_ROOT/attest"
mkdir -p "$ATTEST_DIR" "$RUN_DIR" || exit 1

STAMP() { date -u +%Y-%m-%dT%H:%M:%SZ; }
PROGRESS() { echo "PROGRESS $(STAMP) $*"; }
STAGE_RC=0
check() {
    if [ "$1" -ne 0 ]; then echo "FAIL $(STAMP) stage '$2' exited $1"; STAGE_RC=$1; return 1; fi
    echo "OK   $(STAMP) stage '$2' exited 0"; return 0
}

# ---------------------------------------------------------------- telemetry
TELEM="$RUN_ROOT/logs/telemetry_${SLURM_JOB_ID}.csv"
TELEM_ERR="$RUN_ROOT/logs/telemetry_${SLURM_JOB_ID}.stderr"
HEARTBEAT="$RUN_ROOT/logs/heartbeat_${SLURM_JOB_ID}.log"
mkdir -p "$RUN_ROOT/logs"
SMI_PID=""; HB_PID=""
if command -v nvidia-smi >/dev/null; then
  # stderr to its OWN file, not into the CSV: a sampler that writes an empty
  # file must leave a reason behind, and mixing them corrupts the CSV.
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
             --format=csv,nounits -l 2 > "$TELEM" 2> "$TELEM_ERR" &
  SMI_PID=$!
else
  echo "WARN nvidia-smi not on PATH; no GPU telemetry will be sampled"
fi
( while true; do
    echo "$(date -Is) alive load=$(cut -d' ' -f1 /proc/loadavg)"
    sleep 10
  done ) > "$HEARTBEAT" 2>&1 &
HB_PID=$!

cleanup() {
  for p in $SMI_PID $HB_PID; do
    [ -n "$p" ] || continue
    pkill -P "$p" 2>/dev/null   # children first: an orphaned sleep/nvidia-smi
    kill "$p"    2>/dev/null    # keeps the cgroup alive and Slurm keeps billing
  done
  wait $SMI_PID $HB_PID 2>/dev/null
  if [ -s "$TELEM_ERR" ]; then
    echo "--- telemetry sampler stderr (captured, not discarded) ---"
    cat "$TELEM_ERR"
  fi
}
trap cleanup EXIT TERM INT

echo "=== cxa_attest  job=$SLURM_JOB_ID  node=$(hostname)  $(STAMP) ==="

# -----------------------------------------------------------------------------
# 1. S2: the five facts, from nvidia-smi and from torch independently.
# -----------------------------------------------------------------------------
PROGRESS "1/4 nvidia-smi"
nvidia-smi 2>&1 | head -15
check $? "nvidia_smi"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv \
    | tee "$ATTEST_DIR/nvidia_smi.csv"
check $? "nvidia_smi_query"

source "$REPO/.venv/bin/activate" || exit 1
cd "$REPO" || exit 1

PROGRESS "2/4 torch attestation (S2: all five facts, stop if any is absent)"
python - <<'PY' 2>&1 | tee "$ATTEST_DIR/torch_attest.txt"
import json, os, sys, torch
if not torch.cuda.is_available():
    print("FAIL torch reports no CUDA device"); sys.exit(2)
i = torch.cuda.current_device()
p = torch.cuda.get_device_properties(i)
facts = {
    "gpu_present":      True,
    "vendor":           "NVIDIA",                      # torch.version.hip is None on CUDA builds
    "device_name":      p.name,
    "vram_total_mib":   p.total_memory // (1024**2),
    "compute_capability": f"{p.major}.{p.minor}",
    "cuda_runtime":     torch.version.cuda,
    "torch_version":    torch.__version__,
    "driver_is_hip":    torch.version.hip is not None,
    "bf16_supported":   torch.cuda.is_bf16_supported(),
}
print(json.dumps(facts, indent=2))
missing = [k for k in ("device_name","vram_total_mib","compute_capability","cuda_runtime") if not facts[k]]
if missing:
    print("FAIL missing hardware facts:", missing); sys.exit(2)
# S3/gpu-env-build step 6: cc >= 8.0 is what decides bf16.
if (p.major, p.minor) < (8, 0):
    print(f"FAIL landed on compute capability {p.major}.{p.minor}: no bf16, no Flash-Attention.")
    sys.exit(3)
json.dump(facts, open(os.environ["ATTEST_DIR"] + "/hardware_facts.json", "w"), indent=2)
print("attestation written")
PY
check "${PIPESTATUS[0]}" "torch_attest" || { echo "S2 says stop if any fact is absent"; exit "$STAGE_RC"; }

PROGRESS "2b/4 jax sees the GPU?"
python -c "
import jax
print('jax', jax.__version__, 'devices', jax.devices())
" 2>&1 | tail -5
check $? "jax_devices"

# -----------------------------------------------------------------------------
# 3. S4: VRAM on the smallest unit of work, at BOTH precisions, with the
#    device and the precision recorded. A VRAM figure without its precision is
#    not a measurement.
# -----------------------------------------------------------------------------
PROGRESS "3/4 VRAM and speed at fp32 and bf16"
python - <<'PY' 2>&1 | tee "$ATTEST_DIR/precision_probe.txt"
import json, os, time, torch

# source: chosen by me, no external source. Large enough that the kernel, not
# launch overhead, dominates; small enough to fit any card in this partition.
N = 4096
REPS = 20          # source: chosen by me, no external source
dev = torch.device("cuda")
name = torch.cuda.get_device_properties(0).name
out = {"device": name, "matmul_n": N, "reps": REPS}

for label, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    a = torch.randn(N, N, device=dev, dtype=dtype)
    b = torch.randn(N, N, device=dev, dtype=dtype)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(REPS):
        c = a @ b
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    out[label] = {
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "seconds_total": round(dt, 4),
        "seconds_per_matmul": round(dt / REPS, 6),
    }
    del a, b, c
print(json.dumps(out, indent=2))
json.dump(out, open(os.environ["ATTEST_DIR"] + "/precision_probe.json", "w"), indent=2)
PY
check "${PIPESTATUS[0]}" "precision_probe"

# -----------------------------------------------------------------------------
# 4. Record the attestation through emit and close out.
# -----------------------------------------------------------------------------
PROGRESS "4/4 done"
echo "=== SUMMARY  STAGE_RC=$STAGE_RC  $(STAMP) ==="
[ -f "$TELEM" ] && echo "telemetry rows: $(wc -l < "$TELEM")"
exit "$STAGE_RC"
