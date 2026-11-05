# Smoke-test prompt

Give this to an agent to verify it can drive the cluster end to end. It tests the plumbing only —
no project code, no model, no science.

---

## Task

Verify that you can run GPU work on the Explorer HPC cluster and recover the results, then report
what you found.

You reach the cluster from this machine over SSH. `compute/connector.py` is the interface;
configuration is in `compute/connector_config.json`. Run `python connector.py --help` to see the commands.

Work through these steps in order. If a step fails, say so and stop — do not skip ahead.

### 1. Connectivity

Confirm you can reach the cluster without a password prompt. Report the login node's hostname.

### 2. Survey

Survey the login node. Report:

- GPU types available and how many nodes of each
- partition names and their walltime limits
- your QOS limits — max concurrent GPUs and max submitted jobs
- free space and your actual quota on every volume you could write to
- whether compute nodes can reach the internet

### 3. Write a job

Write a Slurm batch script that:

- requests one GPU for no more than 10 minutes
- prints `nvidia-smi`, the GPU name, VRAM, driver version and compute capability
- prints a `PROGRESS step=N/5` line five times, a few seconds apart
- if PyTorch is importable, multiplies two 4096×4096 matrices on the GPU and reports the time;
  if it is not importable, say so explicitly rather than failing silently
- samples GPU utilisation to a CSV every 2 seconds while it runs

### 4. Submit and recover

Submit it. Record the job ID somewhere on disk before doing anything else. Poll until it leaves
the queue, then retrieve and show:

- the job's stdout
- the telemetry CSV (first few rows)
- `sacct` accounting: state, elapsed, memory used, resources requested

Report how long it waited in the queue versus how long it ran.

### 5. Deliberately fail one job

Submit a second job that requests something impossible — more GPUs than your QOS allows, or a
walltime past the partition limit. Report the exact error. This checks that failures surface
clearly instead of hanging.

### 6. Report

Write a short summary covering:

- what worked, in order
- what failed, with the exact error text
- GPU-hours consumed in total
- anything surprising, including anything that contradicts what step 2 told you

State which numbers you measured yourself and which you read from documentation or inferred. If
you could not complete a step, say which one and why, rather than describing what would have
happened.

---

## Notes for whoever runs this

**Expected runtime:** 10–20 minutes, most of it queue wait.

**Things it should catch:**

| Trap | How it shows up |
|---|---|
| Plain `--gres=gpu:1` lands on a V100 | compute capability reports 7.0, so no bf16 |
| Windows CRLF in the job script | `sbatch: error: Batch script contains DOS line breaks` |
| Forgotten `--account` | `Invalid account` |
| Forgotten `--partition` | job runs on a CPU node, `nvidia-smi` not found |
| QOS ceiling | step 5 should produce `QOSMaxGRESPerUser` or `AssocMaxJobsLimit` |

**What a good run looks like:** every step attempted, failures reported with verbatim error text,
and step 6 distinguishing measured values from inferred ones. An agent that reports success
without showing the `sacct` output has not done step 4.
