#!/bin/bash
# GPU smoke test: prove an allocation works, record what we were given, and
# leave enough telemetry behind that a human can audit it without trusting
# the agent's summary.
#
# Partition / gres / account / time are injected at submit time by connector.py,
# so this script stays portable. Command-line sbatch flags override the
# #SBATCH directives below, which are therefore only fallback defaults.
#SBATCH --job-name=gpu-smoke
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -o pipefail

PROJECT_DIR="${HPC_PROJECT_DIR:-$HOME/hpcjobs}"
LOGDIR="$PROJECT_DIR/logs"
mkdir -p "$LOGDIR"
TELEMETRY="$LOGDIR/telemetry_${SLURM_JOB_ID}.csv"

echo "===== ALLOCATION ====="
echo "job:       $SLURM_JOB_ID on $(hostname -f)"
echo "partition: $SLURM_JOB_PARTITION"
echo "gres:      ${SLURM_JOB_GRES:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "started:   $(date -Is)"
START_EPOCH=$(date +%s)

# ---------------------------------------------------------------- GPU identity
echo; echo "===== NVIDIA-SMI ====="
if ! command -v nvidia-smi >/dev/null; then
  echo "FATAL: nvidia-smi not found - this is a CPU node, not a GPU node."
  echo "       (almost always a missing or wrong --partition)"
  exit 2
fi
nvidia-smi
echo "--- machine-readable ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

# ------------------------------------------------------------------- telemetry
# Sampled in the background for the life of the job. 2s cadence. The trailing
# kill is in a trap so the sampler dies even if the job body fails early.
echo "timestamp,gpu_idx,util_gpu_pct,util_mem_pct,mem_used_mib,mem_total_mib,temp_c,power_w" > "$TELEMETRY"
(
  while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
               --format=csv,noheader,nounits \
      | sed "s/^/$(date -Is),/" >> "$TELEMETRY" 2>/dev/null
    sleep 2
  done
) &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null' EXIT
echo "telemetry -> $TELEMETRY (sampler pid $SAMPLER_PID, 2s cadence)"

# --------------------------------------------------------- compute-node egress
# The login node having internet does not imply the compute nodes do. If these
# fail, every weight/package download must be pre-staged.
echo; echo "===== COMPUTE-NODE EGRESS ====="
for u in https://huggingface.co https://github.com https://pypi.org; do
  code=$(curl -sS -m 12 -o /dev/null -w '%{http_code}' "$u" 2>&1 || echo FAIL)
  echo "  $u -> $code"
done

# ------------------------------------------------------------------- storage
echo; echo "===== STORAGE VISIBLE FROM THE COMPUTE NODE ====="
for p in "$HOME" /scratch /tmp /dev/shm; do
  [ -e "$p" ] && printf "%-14s %s\n" "$p" "$(df -h "$p" 2>/dev/null | tail -1)"
done

# ------------------------------------------------------------------ progress
# Five heartbeats, spaced, so a watcher can tell "slow" from "hung".
echo; echo "===== PROGRESS ====="
for i in 1 2 3 4 5; do
  echo "PROGRESS step=$i/5 t=$(( $(date +%s) - START_EPOCH ))s"
  sleep 3
done

# ------------------------------------------------------------------- pytorch
echo; echo "===== PYTORCH 4096x4096 MATMUL ====="
if command -v python3 >/dev/null; then
  python3 - <<'PY' 2>&1 | sed 's/^/  /'
import sys, time
try:
    import torch
except ImportError as e:
    # Explicit, not silent: absence of torch is a finding, not a failure.
    print("PYTORCH: NOT IMPORTABLE in this environment ->", e)
    print("PYTORCH: skipping matmul benchmark (no fallback attempted)")
    sys.exit(0)

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("PYTORCH: imported but CUDA not available - no GPU benchmark run")
    sys.exit(0)

d = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0))
cc = torch.cuda.get_device_capability(0)
print("compute capability: %d.%d  bf16_ok=%s" % (cc[0], cc[1], cc[0] >= 8))

a = torch.randn(4096, 4096, device=d)
b = torch.randn(4096, 4096, device=d)
# Warm up first: the first matmul pays cuBLAS init and autotuning, which is
# not what we are trying to measure.
for _ in range(3):
    a @ b
torch.cuda.synchronize()

reps = 20
t0 = time.time()
for _ in range(reps):
    a @ b
torch.cuda.synchronize()
dt = (time.time() - t0) / reps
# 4096^3 fused multiply-adds = 2 * 4096^3 flops
tflops = 2 * (4096 ** 3) / dt / 1e12
print("4096x4096 matmul: %.2f ms/iter over %d iters  (%.1f TFLOP/s fp32)" % (dt * 1e3, reps, tflops))
PY
else
  echo "  no python3 on compute node"
fi

echo; echo "===== TELEMETRY SUMMARY ====="
echo "rows: $(wc -l < "$TELEMETRY")"
head -5 "$TELEMETRY"

echo; echo "finished: $(date -Is)  elapsed=$(( $(date +%s) - START_EPOCH ))s"
echo "===== SMOKE TEST COMPLETE ====="
