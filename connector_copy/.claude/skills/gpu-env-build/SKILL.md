---
name: gpu-env-build
description: Build a working Python + CUDA environment for a cloned repository on a GPU backend (NEU Explorer HPC, RunPod, or AWS), by reading what that repository itself declares rather than guessing. Use when a repo has been cloned and needs its dependencies installed, when PyTorch or CUDA is missing on a compute node, when choosing between a conda environment and a container image, or when staging model weights before requesting a GPU. Covers Apptainer-vs-Docker, where to build, cache and quota placement, and what to record so a result can be tied to the environment that produced it.
---

# Building a GPU environment for a cloned repo

*The YAML header above is Claude Code's auto-discovery metadata. If your tool does not use it, ignore it and read on — the rest of this file is tool-neutral.*

**Scope.** Getting a repository's code to import and run on a GPU. Not the science, and not job
submission — for reaching a GPU at all, see `compute/AGENTS_gpu_section.md` and `connector.py`.

**This is scaffolding, and it should be declared as such.** It is classified as a survival
guardrail: without it runs die in setup rather than producing numbers. If you use it, say in your
report that environment setup was assisted, so a reader comparing against unassisted runs knows
the difference did not come from the model.

---

## 1. Read what the repo declares. Do not guess.

Never start from a remembered package list. Different repos in the same field pin incompatible
versions of the same libraries — that incompatibility is often the whole reason they are separate
repos. Look, in this order, and stop at the first that is authoritative:

1. **The repo's own installer or CLI.** Many research repos ship `scripts/install.sh`, a `Makefile`
   target, or a CLI subcommand that fetches code and weights. If one exists, it wins — a task
   brief may even require you to use it.
2. **Declared dependencies:** `pyproject.toml`, `requirements*.txt`, `environment.yml`,
   `setup.py` / `setup.cfg`, `poetry.lock`, `uv.lock`, `conda-recipe/`.
3. **`README` / `INSTALL` / `docs/`.** Usually where the CUDA version and the non-obvious ordering
   live (install torch first, then this; build this extension against that).
4. **`Dockerfile`, `.devcontainer/`, and `.github/workflows/`.** Read these for dependency truth
   even where you cannot run them — CI workflows often carry the only install sequence anyone has
   actually verified.
5. **The repo's own tests.** Use them as your verification target in step 6 rather than inventing
   one.

Record which source you used. When two disagree — a `README` saying torch 2.1 and a
`requirements.txt` saying 2.4 — say so and state which you followed.

**Extract specifically:** Python version, CUDA/cuDNN version, the deep-learning framework and its
build (`cu121`, `cu124`, ROCm), any compiled extensions, and whether weights download separately
and how large they are.

---

## 2. Reconcile against the backend you are on

| | Explorer (HPC) | RunPod | AWS |
|---|---|---|---|
| Root | **No** | Yes | Yes |
| Docker | **Never works** | Yes | Yes |
| Container format | Apptainer `.sif` | OCI / Docker | OCI / Docker or AMI |
| System PyTorch | None | In most images | Depends on AMI |
| Cost of a slow build | Patience only | **Billed per second** | **Billed per second** |
| GPU selection | `--gres=gpu:h200:1` | pod `gpu_type` | instance type |

Two consequences worth internalising:

**One dependency set, three artifacts.** A `.sif` will not run on RunPod and an OCI image will not
run on Explorer. Do not maintain two recipes — build or pull the OCI image, then convert:

```bash
apptainer build myenv.sif docker://org/image:tag
```

**On metered backends the build order inverts.** On Explorer a 40-minute image build costs nothing
but time. On RunPod or AWS it is 40 minutes of GPU billing. Never debug a dependency tree while
holding a metered GPU — build elsewhere, or start from a prebuilt image and layer on top.

---

## 3. Choose conda or container

| Use a conda/venv environment | Use a container |
|---|---|
| One stack, short-lived | Several stacks that conflict |
| Repo declares plain pip/conda deps | Repo ships a Dockerfile, or needs system libraries |
| You want to iterate quickly | You need the result reproducible later |

On Explorer both are supported and Research Computing suggests conda first for a plain PyTorch
stack:

```bash
module load miniconda3/25.9.1       # also: anaconda3/2024.06, python/3.13.5
module load cuda/12.8.0             # also: 12.1.1, 12.3.0, 13.2.0
```

Match the CUDA module to what the repo pins **and** to the card you will request — the driver
differs between nodes (CUDA 12.8 on the A100 nodes, 12.3 on some V100 nodes), so a container built
against one may surprise you on the other.

---

## 4. Build

**Redirect every cache off the quota-limited home volume, first, before installing anything:**

```bash
export HF_HOME=/scratch/$USER/hf
export PIP_CACHE_DIR=/scratch/$USER/pip
export TORCH_HOME=/scratch/$USER/torch
```

### Apptainer on Explorer

```bash
# Build on a CPU node. A job holding an idle GPU during setup gets killed.
srun --account=<acct> --partition=short --cpus-per-task=4 --mem=16G --time=04:00:00 --pty /bin/bash

mkdir -p /scratch/$USER/containers /tmp/$USER/aptmp
export APPTAINER_CACHEDIR=/scratch/$USER/containers/cache
export APPTAINER_TMPDIR=/tmp/$USER/aptmp        # node-local disk, not scratch

cd /scratch/$USER/containers
nohup apptainer pull myenv.sif docker://<image>:<tag> > pull.log 2>&1 &
tail -f pull.log                                 # Ctrl-C stops the tail, not the build
```

Three things that look like failures and are not:

- **`APPTAINER_TMPDIR` must be on `/tmp`.** Same image: ~6 minutes on node-local disk versus
  ~39 minutes on scratch. This is the single biggest time lever.
- **`Creating SIF file` prints nothing for 10–40 minutes.** It is not hung. Do not kill it.
- **`WARNING: 'nodev' mount option set on /tmp`** is harmless.

Run with `--nv` or the container sees no GPU, and bind the paths your data lives on:

```bash
apptainer exec --nv -B /scratch:/scratch myenv.sif python your_script.py
```

### Where the image may not survive

`/scratch` is purged monthly and is not backed up. An image built there is temporary. Keep the
build recipe in git so it can be rebuilt, and if the image must persist, put it on a durable
volume (on Explorer that means `/projects`, which a PI has to request).

---

## 5. Stage weights and data separately

Large downloads belong **outside** the GPU allocation and outside any metered window: on Explorer
from a login node or a CPU job, on RunPod/AWS before the GPU work begins or onto a persistent
volume.

Verify completeness by size or checksum before requesting a GPU. A truncated checkpoint fails
deep inside a running job, after the queue wait, which is the most expensive place to discover it.

Compute nodes on Explorer do have outbound internet (PyPI, Hugging Face, GitHub all reachable),
so a missing package *can* be installed inside a job — but that spends GPU time on bandwidth.
Prefer pre-staging.

---

## 6. Verify before you trust it

Prove the environment works, **on the same card class you will use for real work**. Run
`verify_env.sh` from this directory — it checks the environment and writes the step 7
attestation in one pass:

```bash
# bare, inside a GPU allocation
./verify_env.sh --repo /path/to/repo --import <module_name>

# or inside the container you just built
apptainer exec --nv -B /scratch:/scratch myenv.sif \
  ./verify_env.sh --repo /path/to/repo --import <module_name> --precision bf16
```

It reads only: no installs, no network, no mutation. It prints JSON and writes
`env_attestation.json`. Exit codes let you branch without parsing — `0` usable, `1` no GPU
visible, `2` torch missing or CUDA unavailable, `3` GPU fine but the repo entry point does not
import.

That last case is the most common false pass: `torch` works and the repo does not. Pass
`--import` with the repo's top-level module so it gets caught here rather than inside a running
job. Then run the repo's own tests if it has them.

**Compute capability is the number that decides precision.** cc ≥ 8.0 has bfloat16; cc 7.0 (V100)
and 7.5 (T4) do not, and a model that assumes bf16 fails inside the running job rather than at
submit. If you cast to a lower precision to fit in memory, **say so in every measurement you
report** — it changes both speed and, possibly, output quality.

---

## 7. Record what you built

`verify_env.sh` has already written most of this to `env_attestation.json` — GPU model, VRAM,
compute capability, bf16 support, driver, torch and CUDA versions, repo commit, whether the entry
point imports, and a timed matmul. Keep that file **next to the results**, not in chat.

Add by hand the parts a script cannot know:

- the dependency source you followed, and any conflicts you resolved
- image digest or `.sif` path and build date
- the full resolved set (`pip freeze` or `conda env export`) alongside it
- what you could not verify

**Pin everything.** `pip install torch` resolves differently next week. An environment rebuilt
without pins silently changes the experiment, and a result that cannot be tied to the environment
that produced it is not reproducible.

---

## Failure catalogue

| Symptom | Cause |
|---|---|
| `No module named 'torch'` on a compute node | No system PyTorch. Expected — build an env or use a container. |
| `docker: command not found`, or permission errors | Docker needs root and never works on Explorer. Use Apptainer. |
| Container sees no GPU | Missing `--nv`. |
| Container cannot see your files | Missing `-B /scratch:/scratch` or equivalent bind. |
| Job killed during setup | Held an idle GPU. Build on a CPU node. |
| Build takes 40 min instead of 6 | `APPTAINER_TMPDIR` on a network volume. Point it at `/tmp`. |
| Quota errors mid-install | Caches on `/home` (75 GB, fixed, cannot be raised). Redirect them. |
| Image or data vanished | `/scratch` is purged monthly and not backed up. |
| `sbatch` rejects the script | Windows CRLF line endings. `connector.py submit` strips them; by hand use `tr -d '\r'`. |
| Runtime failure about bf16 or Flash-Attention | Landed on cc 7.0/7.5. Request the GPU model explicitly. |
| Job accepted then never runs | Over-quota request. `sbatch` returns success; the job pends on `QOSMaxGRESPerJob` forever. Poll the pending reason. |
| Pod billed far more than the work took | Boot and idle time bill too. Terminate explicitly. |
