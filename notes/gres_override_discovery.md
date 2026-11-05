<!-- version: note-v1  2026-11-05 -->
# The GPU model that ran is not the one the committed script asks for

## What I observed

`jobs/cell_tnf_bon.sbatch` line 13, committed, and byte-identical on the cluster
(`md5sum` both sides = `651868c5a5fe8df6aaa444aa0ddb9cb3`):

```
#SBATCH --gres=gpu:h200:1
```

What the scheduler recorded for the job built from that file:

```
scontrol show job 11646330   -> exit 0
   Partition=gpu   NodeList=d1026
   TresPerNode=gres/gpu:a100:1
   ReqTRES=cpu=12,mem=96G,node=1,billing=112,gres/gpu=1
```

and the job's own first lines:

```
NVIDIA A100-SXM4-40GB, 40960 MiB, 8.0
NVIDIA A100-SXM4-40GB, 40960 MiB, 8.0
```

All eight queued cells are the same — `scontrol` → exit 0 for each:

```
11646330 a100:1   11646391 a100:1
11646351 a100:1   11646394 a100:1
11646367 a100:1   11646402 a100:1
11646373 a100:1   11646416 a100:1
```

## Why

`connector_copy/compute/runners/hpc.py:143`:

```python
f"--gres={opts.get('gres') or self.cfg.get('gres', 'gpu:1')}",
```

`connector.py submit` always puts `--gres=` on the **sbatch command line**, taken
from the submit option if given and otherwise from the connector configuration.
A command-line flag overrides the matching `#SBATCH` directive inside the
script, so the in-script `--gres=gpu:h200:1` never had any effect.

This is observed from the connector's source and from `scontrol`. I did not read
`connector_config.json` to find out what its `gres` value is — AGENTS.md
restricts that file to running the CLI — so **the specific configured value is
not something this session established.** What is established is that the
effective request was `gpu:a100:1` for all eight jobs.

Earlier GPU jobs in this run *did* land on H200 because I passed `--gres` to the
connector explicitly on those submissions (`opts.get('gres')` wins). The
resubmission loop after the argparse fix did not pass it, so the configured
default applied. **That omission is mine.**

## Does it invalidate anything

**No for the method comparison, yes for two other things.**

- *Consistency across cells holds.* All eight cells are `a100:1`, so no method
  is advantaged by faster hardware than another. A GPU-hour on an A100 is not a
  GPU-hour on an H200, but the comparison PROJECT.md asks for is between methods
  at matched GPU-hours, and every cell is on the same card.
- *S3 is satisfied.* The request names a model. It is not a bare `gpu:1`, so the
  failure mode S3 exists for — landing on a V100 or T4 without bf16 — did not
  happen. A100 is compute capability 8.0 and has bf16.
- **The committed script does not describe what ran.** That is the real defect
  here. A later reader checking `cell_tnf_bon.sbatch` would conclude the cell ran
  on an H200. This note and `jobs/job_ids.log` are the correction; the script is
  left as submitted rather than edited afterwards, because editing it now would
  misdescribe history in the opposite direction.
- **The sample counts were sized on H200 throughput.** `nsamples=32` for
  best-of-n comes from job 11638953, measured on an H200: 75.2 s/sample to
  generate, 104.8 s/sample to evaluate, ~400 s fixed. Those numbers do not
  transfer to an A100 and I have **no A100 measurement**. Rather than assume a
  slowdown factor, `tools/sample_gen_rate.sh` samples the running job's
  structure count over time so the A100 rate is measured. The decision on
  whether `nsamples=32` fits a 2 h A100 allowance waits for that measurement.

## Two GPUs visible inside a one-GPU allocation

`nvidia-smi` printed two A100 rows although `AllocTRES` says `gres/gpu=1`. The
H200 smoke job printed one row.

*Hypothesis (unverified from this session): the node's cgroup/device isolation
exposes both devices to the job even though one is allocated.* **Observed:** two
rows from `nvidia-smi --query-gpu`, and `AllocTRES=...,gres/gpu=1`. **Not
observed:** whether the process can actually run work on the second device, or
what `CUDA_VISIBLE_DEVICES` was set to inside the job.

This matters for cost: `job_cost` takes `gpus` from `ReqTRES`, which says 1. If
a job could use two devices, one GPU-hour of billing would understate the
hardware held. I have not established that it can, and the run charges 1 GPU per
job because that is what the scheduler allocated.

## What changes from here

Every subsequent `connector.py submit` passes `--gres` explicitly, so the flag
on the command line and the directive in the script agree and the staged script
remains an honest record of what ran.
