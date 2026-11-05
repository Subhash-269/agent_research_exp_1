#!/bin/bash
# Reference pattern for a long-running job that reports on itself.
#
# Long jobs can't be watched interactively, so they must emit their own telemetry:
#   - a GPU sampler writing CSV (machine-readable, feeds cost analysis later)
#   - a heartbeat (distinguishes "hung" from "running but CPU-bound" - a GPU
#     utilisation sampler alone cannot, since both read 0%)
#   - progress lines on stdout at a known cadence, so `tail -f` shows advancement
#
# Copy this skeleton into real jobs; replace the WORKLOAD section.
#SBATCH --job-name=telemetry-demo
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

PROJECT_DIR="${HPC_PROJECT_DIR:-$HOME/hpcjobs}"
LOGDIR="$PROJECT_DIR/logs"
mkdir -p "$LOGDIR"

TELEM="$LOGDIR/telemetry_${SLURM_JOB_ID}.csv"
HEARTBEAT="$LOGDIR/heartbeat_${SLURM_JOB_ID}.log"

echo "node=$(hostname)  job=$SLURM_JOB_ID  gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "telemetry -> $TELEM"

# --- GPU sampler, 2s cadence, CSV so it stays machine-readable ---
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
             --format=csv,nounits -l 2 > "$TELEM" 2>&1 &
  SMI_PID=$!
fi

# --- heartbeat: proves liveness even when the GPU is idle ---
( while true; do
    echo "$(date -Is) step=alive load=$(cut -d' ' -f1 /proc/loadavg) mem_free=$(free -g | awk '/Mem:/{print $7}')G"
    sleep 10
  done ) > "$HEARTBEAT" 2>&1 &
HB_PID=$!

# --- clean up samplers however we exit, including on scancel/timeout ---
cleanup() { kill ${SMI_PID:-} $HB_PID 2>/dev/null; }
trap cleanup EXIT TERM INT

# ============================ WORKLOAD ============================
# Replace this block. Emit a PROGRESS line per unit of work so that
# `tail -f` and automated watchers can report advancement.
TOTAL=8
for i in $(seq 1 $TOTAL); do
  echo "PROGRESS step=$i/$TOTAL t=$(date -Is)"
  openssl speed -seconds 2 sha256 >/dev/null 2>&1   # placeholder CPU work
done
# ==================================================================

cleanup
[ -f "$TELEM" ] && echo "telemetry rows: $(wc -l < "$TELEM")"
echo "DONE $(date -Is)"
