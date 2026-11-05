#!/bin/bash
# Minimal GPU probe: proves an allocation works and records what we were given.
# Deliberately tiny so it clears the queue fast.
#
#   sbatch --partition=$SLURM_PARTITION --gres=$SLURM_GRES \
#          --account=$SLURM_ACCOUNT probe_gpu.sbatch
#
# Partition / gres / account are injected at submit time rather than hardcoded,
# so this script is portable across clusters.
#SBATCH --job-name=gpu-probe
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

PROJECT_DIR="${HPC_PROJECT_DIR:-$HOME/hpcjobs}"

echo "===== ALLOCATION ====="
echo "job:       $SLURM_JOB_ID on $(hostname -f)"
echo "partition: $SLURM_JOB_PARTITION"
echo "gres:      ${SLURM_JOB_GRES:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "started:   $(date -Is)"

echo; echo "===== NVIDIA-SMI ====="
nvidia-smi
echo "--- machine-readable ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

echo; echo "===== COMPUTE-NODE EGRESS ====="
# If these fail, every weight/package download must be pre-staged from the login
# node, and any library that lazily fetches at runtime will die inside a job.
for u in https://huggingface.co https://github.com https://pypi.org; do
  code=$(curl -sS -m 12 -o /dev/null -w '%{http_code}' "$u" 2>&1 || echo FAIL)
  echo "  $u -> $code"
done

echo; echo "===== STORAGE (every volume the runtime touches) ====="
for p in "$HOME" "$PROJECT_DIR" /work /scratch /tmp /dev/shm; do
  [ -e "$p" ] && printf "%-14s %s\n" "$p" "$(df -h "$p" 2>/dev/null | tail -1)"
done

echo; echo "===== DOES A GPU TENSOR ACTUALLY RUN? ====="
# Absence of torch here is informative, not fatal - it just means the
# environment build step hasn't run yet.
if command -v python3 >/dev/null; then
  python3 - <<'PY' 2>&1 | sed 's/^/  /'
try:
    import torch, time
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        d = torch.device("cuda")
        a = torch.randn(4096, 4096, device=d); torch.cuda.synchronize()
        t = time.time(); (a @ a); torch.cuda.synchronize()
        print("4096^2 matmul ok, %.3fs" % (time.time() - t))
        print("device:", torch.cuda.get_device_name(0))
        cc = torch.cuda.get_device_capability(0)
        print("compute capability: %d.%d  bf16_ok=%s" % (cc[0], cc[1], cc[0] >= 8))
except ImportError as e:
    print("no torch in this environment:", e)
PY
else
  echo "  no python3 on compute node"
fi

echo; echo "finished: $(date -Is)"
echo "===== PROBE COMPLETE ====="
