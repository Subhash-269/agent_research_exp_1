# GPU execution on the HPC cluster

Append to `AGENTS.md`. Describes how to reach GPUs; says nothing about the science task.

---

## Where you run

You run on the operator's laptop, not on the cluster. The cluster is a batch backend you reach
over SSH. Work is submitted to a scheduler and runs later, without you attached.

`compute/connector.py` is the interface. Configuration lives in `compute/connector_config.json`.

```bash
python connector.py check                      # config + connectivity
python connector.py preflight                  # survey the login node
python connector.py submit <script.sbatch>     # stage + sbatch, records the job id
python connector.py status                     # queue + today's history
python connector.py watch <jobid>              # poll until it leaves the queue
python connector.py logs <jobid>               # job stdout
python connector.py fetch <jobid>              # pull logs + telemetry locally
python connector.py ledger                     # GPU-hours, cumulative, completion order
```

## You drive the full loop

You have two-way access: read and write files on the cluster, submit jobs, poll them, read their
output, edit the code, and resubmit. Iterate until the work is done.

```
write/edit script → submit → poll → read logs → diagnose → fix → resubmit
```

Every action is appended to `compute/audit.log` so the operator can see what you did and
when. Do not suppress or edit that file.

**If the operator asks for the three-step split instead** — you prepare inputs, they submit, you
process results — then stop after preparing and report the exact command for them to run. Default
is the full loop.

## Cluster facts

| | |
|---|---|
| Cluster | NEU Explorer, Rocky Linux 9.7, Slurm 23.11.6 |
| Login node | `explorer-01` — **no GPU**. Never run computation here. |
| GPU partitions | `gpu` (8 h), `gpu-short` (2 h), `gpu-interactive` (2 h) |
| Default partition | `short` — **CPU only**. Always pass `--partition` explicitly. |
| Storage | `/home` and `/scratch`. **There is no `/work`.** |

## Constraints that cause silent failures

**GPU type.** A plain `--gres=gpu:1` can land on a V100 or T4 — compute capability 7.0/7.5, which
has **no bfloat16 and no Flash-Attention kernels**. If the model needs bf16, request
`--gres=gpu:a100:1` or `--gres=gpu:h200:1`. Otherwise the job fails at runtime, after queueing,
not at submit.

**QOS limits, per user.** `gpu` allows 4 concurrent GPUs and **8 submitted jobs**; `gpu-short`
allows 2. Job arrays must be throttled: `--array=0-18%4` runs 19 tasks, 4 at a time.

**Walltime is a hard kill.** Slurm terminates at the limit with no grace period. Anything longer
than the partition limit must checkpoint to disk and be restartable from that checkpoint.

**Quota.** Filesystem free space is not your quota. Check the real limit before downloading
checkpoints, and redirect caches off `/home`:

```bash
export HF_HOME=/scratch/$USER/hf
export PIP_CACHE_DIR=/scratch/$USER/pip
export TORCH_HOME=/scratch/$USER/torch
```

**Line endings.** A script authored on Windows reaches the cluster with CRLF and `sbatch` rejects
it. `connector.py submit` strips this; by hand, use `tr -d '\r'`.

## Environment

Software runs inside an Apptainer image, not a conda environment on the host:

```bash
apptainer exec --nv /path/to/image.sif python your_script.py
```

`--nv` exposes the GPU. Without it the container sees no device.

Compute nodes have outbound internet (PyPI, Hugging Face, GitHub all reachable), so a missing
package can be installed inside a job — but prefer baking it into the image.

## Job scripts

Every job script should:

- carry `#SBATCH` directives before the first command;
- write a `PROGRESS` line per unit of work, so a tail shows advancement;
- sample GPU telemetry to CSV alongside a heartbeat, so a stalled job is distinguishable from a
  CPU-bound one (both read 0% GPU);
- clean up background samplers with `trap cleanup EXIT TERM INT`, so they die on `scancel` too.

`telemetry_demo.sbatch` is the reference skeleton.

## State lives on disk, not in the conversation

A job may queue longer than your session lasts. Write every job ID to a file at submission.
Any later session recovers full status from `sacct -j <id>` alone. Never hold a job ID only in
conversation context.

## Compute accounting

Report GPU-hours as a running total in **completion order**. Charging in score order implies you
knew which runs would succeed before paying for them, which makes any cost curve retrospectively
optimistic.
