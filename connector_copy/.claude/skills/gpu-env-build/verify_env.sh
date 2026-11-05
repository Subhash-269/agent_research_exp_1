#!/usr/bin/env bash
# Verify a GPU environment and emit the attestation in one pass.
#
#   ./verify_env.sh                                  # bare, on the current host
#   apptainer exec --nv -B /scratch:/scratch img.sif ./verify_env.sh
#   ./verify_env.sh --repo /path/to/repo --import proteina_complexa
#
# Reads only. No installs, no network, no mutation of the environment - everything
# that changes state stays in the caller's own commands so it lands in audit.log.
# Writes one file (env_attestation.json by default) and prints the same JSON.
#
# Exit codes let a caller branch without parsing:
#   0  environment usable
#   1  no GPU visible
#   2  torch missing or CUDA unavailable
#   3  GPU fine, but the repo entry point does not import
set -uo pipefail

OUT="env_attestation.json"
REPO=""
IMPORT=""
PRECISION=""

while [ $# -gt 0 ]; do
  case "$1" in
    --out)       OUT="$2"; shift 2 ;;
    --repo)      REPO="$2"; shift 2 ;;
    --import)    IMPORT="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

# ------------------------------------------------------------------ backend guess
# Only a hint for the record; nothing branches on it.
backend="unknown"
if [ -n "${SLURM_JOB_ID:-}" ]; then backend="hpc"
elif [ -n "${RUNPOD_POD_ID:-}" ]; then backend="runpod"
elif [ -f /sys/hypervisor/uuid ] || [ -d /var/lib/cloud ]; then backend="aws"
fi

# -------------------------------------------------------------------- repo commit
repo_commit=""
if [ -n "$REPO" ] && [ -d "$REPO/.git" ]; then
  repo_commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) || repo_commit=""
fi

# ------------------------------------------------------------------------ the GPU
# Asked of the driver rather than of torch, so a torch-less environment still
# produces a usable record instead of nothing.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '{"error":"nvidia-smi not found - no GPU on this host (wrong partition, or a CPU node)","backend":"%s","node":"%s"}\n' \
    "$backend" "$(hostname)"
  exit 1
fi

smi=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
                 --format=csv,noheader,nounits 2>/dev/null | head -1)
gpu_name=$(echo "$smi"   | awk -F', *' '{print $1}')
vram=$(echo "$smi"       | awk -F', *' '{print $2}')
driver=$(echo "$smi"     | awk -F', *' '{print $3}')
cc=$(echo "$smi"         | awk -F', *' '{print $4}')
cc_major=${cc%%.*}
bf16=false; [ "${cc_major:-0}" -ge 8 ] 2>/dev/null && bf16=true

# ---------------------------------------------------------------------- the stack
# One python call does framework, repo import and the matmul, so a broken
# environment fails once rather than three times.
py_json=$(python3 - "$IMPORT" <<'PY' 2>/dev/null
import json, sys, time
out = {"python": None, "torch": None, "torch_cuda": None, "cuda_available": False,
       "device": None, "capability": None, "matmul_4096_ms": None,
       "entrypoint_imports": None, "notes": []}
out["python"] = sys.version.split()[0]
try:
    import torch
except ImportError as e:
    out["notes"].append("torch not importable: %s" % e)
    print(json.dumps(out)); sys.exit(0)

out["torch"] = torch.__version__
out["torch_cuda"] = torch.version.cuda
out["cuda_available"] = bool(torch.cuda.is_available())
if out["cuda_available"]:
    out["device"] = torch.cuda.get_device_name(0)
    c = torch.cuda.get_device_capability(0)
    out["capability"] = "%d.%d" % c
    try:
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        for _ in range(3):          # warm up cuBLAS before timing
            a @ b
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        out["matmul_4096_ms"] = round((time.time() - t0) / 10 * 1e3, 2)
    except Exception as e:
        out["notes"].append("matmul failed: %s" % e)
else:
    out["notes"].append("torch imported but CUDA not available")

# The most common false pass: torch works, the repo does not.
mod = sys.argv[1] if len(sys.argv) > 1 else ""
if mod:
    try:
        __import__(mod)
        out["entrypoint_imports"] = True
    except Exception as e:
        out["entrypoint_imports"] = False
        out["notes"].append("import %s failed: %s" % (mod, e))
print(json.dumps(out))
PY
)
[ -z "$py_json" ] && py_json='{"python":null,"torch":null,"notes":["python3 missing or crashed"]}'

# ----------------------------------------------------------------------- assemble
# Values go through the environment, never spliced into Python source: an empty
# or odd value then yields null instead of a syntax error.
export V_TS="$(date -Is)" V_BACKEND="$backend" V_NODE="$(hostname)" \
       V_JOB="${SLURM_JOB_ID:-}" V_GPU="$gpu_name" V_VRAM="$vram" \
       V_CC="$cc" V_BF16="$bf16" V_DRIVER="$driver" V_COMMIT="$repo_commit" \
       V_PREC="$PRECISION" V_OUT="$OUT" V_PY="$py_json"

python3 <<'PY'
import json, os

def s(key):
    v = os.environ.get(key, "").strip()
    return v or None

def i(key):
    try:
        return int(os.environ.get(key, "").strip())
    except ValueError:
        return None

rec = {
    "timestamp":          s("V_TS"),
    "backend":            s("V_BACKEND"),
    "node":               s("V_NODE"),
    "slurm_job_id":       s("V_JOB"),
    "gpu":                s("V_GPU"),
    "vram_mib":           i("V_VRAM"),
    "compute_capability": s("V_CC"),
    "bf16_supported":     os.environ.get("V_BF16") == "true",
    "driver":             s("V_DRIVER"),
    "repo_commit":        s("V_COMMIT"),
    "precision_used":     s("V_PREC"),
}
try:
    rec.update(json.loads(os.environ.get("V_PY") or "{}"))
except json.JSONDecodeError as e:
    rec.setdefault("notes", []).append("could not parse python probe output: %s" % e)

if rec["precision_used"] is None and not rec["bf16_supported"]:
    rec.setdefault("notes", []).append(
        "cc < 8.0: no bf16. State the precision you actually ran in every reported number.")

with open(os.environ["V_OUT"], "w") as fh:
    json.dump(rec, fh, indent=2)
print(json.dumps(rec, indent=2))
PY

# -------------------------------------------------------------------- exit status
read -r torch_ok entry <<EOF
$(python3 -c "
import json,os
d=json.load(open(os.environ['V_OUT']))
print('1' if d.get('cuda_available') else '0', d.get('entrypoint_imports'))
" 2>/dev/null || echo "0 None")
EOF

echo "written: $OUT" >&2
[ "$torch_ok" = "1" ] || exit 2
[ "$entry" = "False" ] && exit 3
exit 0
