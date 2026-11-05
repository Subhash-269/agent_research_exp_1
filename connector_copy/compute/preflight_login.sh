#!/bin/bash
# Login-node survey: partitions, QOS limits, storage, egress, toolchain.
# Read-only. No GPU and no allocation needed. Run it before planning any sweep:
#   ssh <host> 'bash -s' < preflight_login.sh
# Every check here answers a question that changes how you should schedule work.
echo "===== IDENTITY ====="
echo "host: $(hostname -f)"
echo "user: $(whoami)"
echo "date: $(date -Is)"
echo "os:   $(cat /etc/redhat-release 2>/dev/null || lsb_release -d 2>/dev/null || uname -a)"

echo; echo "===== SCHEDULER / PARTITIONS ====="
command -v sbatch >/dev/null && sbatch --version || echo "NO SLURM"
echo "--- partitions visible to us ---"
sinfo -o "%20P %5a %10l %6D %N" 2>/dev/null | head -40
echo "--- GPU nodes and their gres ---"
sinfo -o "%20P %12G %6D %t" 2>/dev/null | grep -i gpu | head -30
echo "--- your slurm associations / limits ---"
sacctmgr -nP show assoc where user=$USER format=Account,Partition,QOS,GrpTRES,MaxJobs 2>/dev/null
echo "--- QOS table ---"
sacctmgr -nP show qos format=Name,MaxTRESPU,MaxWall,MaxSubmitJobsPU 2>/dev/null | head -20

echo; echo "===== STORAGE (every volume the runtime touches) ====="
for p in "$HOME" /work /scratch /tmp /dev/shm; do
  [ -e "$p" ] && printf "%-12s %s\n" "$p" "$(df -h "$p" 2>/dev/null | tail -1)"
done
echo "--- quotas ---"
quota -s 2>/dev/null || echo "(quota cmd unavailable)"
command -v lfs >/dev/null && lfs quota -hu $USER /work 2>/dev/null

echo; echo "===== EGRESS FROM LOGIN NODE ====="
for u in https://api.anthropic.com/v1/models https://huggingface.co https://github.com; do
  code=$(curl -sS -m 12 -o /dev/null -w '%{http_code}' "$u" 2>&1 || echo FAIL)
  echo "  $u -> $code"
done
echo "proxy vars: HTTPS_PROXY=${HTTPS_PROXY:-unset} HTTP_PROXY=${HTTP_PROXY:-unset}"

echo; echo "===== CONTAINER / MODULE TOOLING ====="
command -v apptainer >/dev/null && apptainer --version || command -v singularity >/dev/null && singularity --version || echo "NO APPTAINER/SINGULARITY"
echo "--- fakeroot build possible? (decides container vs conda strategy) ---"
if command -v apptainer >/dev/null; then
  printf 'Bootstrap: docker\nFrom: alpine:latest\n' > /tmp/_fr_$$.def
  if apptainer build --fakeroot /tmp/_fr_$$.sif /tmp/_fr_$$.def >/tmp/_fr_$$.log 2>&1; then
    echo "FAKEROOT BUILD: OK"
  else
    echo "FAKEROOT BUILD: FAILED -> $(tail -3 /tmp/_fr_$$.log | tr '\n' ' ')"
  fi
  rm -f /tmp/_fr_$$.def /tmp/_fr_$$.sif /tmp/_fr_$$.log
fi
echo "--- relevant modules ---"
module avail 2>&1 | tr ' ' '\n' | grep -iE '^(cuda|anaconda|miniconda|python|apptainer|singularity|gcc)' | sort -u | head -30

echo; echo "===== NODE-LOCAL TOOLING ====="
for c in git python3 conda nvidia-smi rsync; do
  printf "  %-12s %s\n" "$c" "$(command -v $c || echo MISSING)"
done
echo "===== PREFLIGHT COMPLETE ====="
