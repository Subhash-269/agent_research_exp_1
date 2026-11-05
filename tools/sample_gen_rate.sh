#!/bin/bash
# Sample how fast a running cell is producing structures, so the A100 sample
# rate is MEASURED rather than guessed from the H200 smoke run.
# Usage: sample_gen_rate.sh <jobid> <n_samples> <interval_s>
set -u
JOB="$1"; N="${2:-8}"; IV="${3:-180}"
OUT="logs/genrate_${JOB}.tsv"
printf 'utc\telapsed\tn_pdb\n' > "$OUT"
for i in $(seq 1 "$N"); do
  python tools/rx.py "printf '%s\t' \$(date -u +%Y-%m-%dT%H:%M:%SZ); printf '%s\t' \$(squeue -h -j $JOB -o '%M' 2>/dev/null); D=\$(find /scratch/\$USER/complexa_armB_20261105/src/Proteina-Complexa/inference -maxdepth 1 -type d -name '*${JOB}*' | head -1); if [ -n \"\$D\" ]; then find \$D -name '*.pdb' | wc -l; else echo 0; fi" 2>/dev/null \
    | grep -E '^2[0-9]{3}-' >> "$OUT"
  tail -1 "$OUT"
  [ "$i" -lt "$N" ] && sleep "$IV"
done
