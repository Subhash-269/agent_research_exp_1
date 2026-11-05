<!-- version: agents-v6  frozen 2026-11-05 -->
<!-- Operator note. Environment facts only; the brief itself begins below the `---`.
     Nothing in this file needs editing. Put it, PROJECT.md and a copy of the GPU
     connector together in an empty directory, and start the agent in that directory. -->

**Your workspace root** is a directory named `workspace/` that you create inside the directory
containing this file. Create it before the first unit of work. Write everything there and nowhere
else on this machine; `RUN_REPORT.md`, `run_manifest.json` and `external_reads.log` go at its root.
State the absolute path you resolved in `run_manifest.json`.

That directory must sit **outside any git repository belonging to the project itself**. Prior run
outputs, planning documents and review notes live in such a repository and are not yours to read.
If you need one, R2 applies: log it in `external_reads.log` before you use it. If the directory
containing this file is itself inside the project's repository, stop and report that rather than
working there.

**The GPU connector** is the directory named `agenticAI_with_GPUs` beside this file; its CLI is
`agenticAI_with_GPUs/compute/connector.py`. Determine its provenance yourself rather than expecting
it written here — `git -C <dir> remote get-url origin` and `git -C <dir> rev-parse HEAD` — and
record both, with their exit codes, in `run_manifest.json`.

Work from a copy of the connector whose `audit.log` and `jobs.jsonl` are empty, so they hold your
actions and no earlier run's. Its `connector_config.json` is not in the repository and carries the
operator's cluster identity: **use it as given and do not read it for any purpose other than
running the CLI.** If it is absent, copy `connector_config.example.json`, report that `user` and
`account` are missing, and stop rather than guessing them.

`agenticAI_with_GPUs/.claude/skills/` holds reusable procedures; read `gpu-env-build` before
installing anything.

**The run repository.** If a file named `run_remote.txt` sits beside this one, it holds a single
line: the URL of a private git remote the operator created for this run. Push the workspace there
as described in R8. If there is no such file, commit locally and do not push — its absence is an
instruction, not an oversight to correct.

If any of the above cannot be resolved from the directory you are running in, say exactly what is
missing and stop. Do not substitute a path of your own.


> **Operator checklist.** (1) Put `AGENTS.md`, `PROJECT.md` and a connector copy in an empty
> directory outside the project repository, and start the agent there. Nothing in either file is
> edited. (2) To have the run pushed, add `run_remote.txt` holding the URL of a **private** repo you
> created and have verified you can push to; omit the file to keep the run local. (3) The scope of
> the work is set in `PROJECT.md`, not here. (4) **Bump the version line at the top of any file you
> change, and never change a file mid-run.** Runs are only comparable if they saw the same pair of
> files; lessons from a run go into the next version, not the one being run against. (5) Before
> the run, stage the recording module to the cluster: `python eval/stage_to_cluster.py` from the
> project repository. The agent cannot do this step, because it cannot read that repository.

---

# GPU execution on the HPC cluster

Describes how to reach GPUs; says nothing about the science task.

## Where you run

You run on the operator's laptop, not on the cluster. The cluster is a batch backend you reach over
SSH. Work is submitted to a scheduler and runs later, without you attached.

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
output, edit the code, resubmit. Iterate until the work is done.

```
write/edit script → submit → poll → read logs → diagnose → fix → resubmit
```

Every action is appended to `compute/audit.log` so the operator can see what you did and when. Do
not suppress or edit that file.

**If the operator asks for the three-step split instead** — you prepare inputs, they submit, you
process results — then stop after preparing and report the exact command for them to run. Default is
the full loop.

## Cluster facts

| | |
|---|---|
| Cluster | NEU Explorer, Rocky Linux 9.7, Slurm 23.11.6 |
| Login node | `explorer-01` — **no GPU**. Never run computation here. Heavy imports (torch, jax) are killed with exit 137. |
| GPU partitions | `gpu` (8 h), `gpu-short` (2 h), `gpu-interactive` (2 h) |
| Default partition | `short` — **CPU only**. Always pass `--partition` explicitly. |
| Storage | `/home` and `/scratch`. **There is no `/work`.** |
| QOS, per user | `gpu`: 4 concurrent GPUs, 8 submitted jobs. `gpu-short`: 2. |

## Constraints that cause silent failures

**GPU type.** A plain `--gres=gpu:1` can land on a V100 or T4 — compute capability 7.0/7.5, no
bfloat16, no Flash-Attention kernels. Request `--gres=gpu:a100:1` or `--gres=gpu:h200:1`.

**Walltime is a hard kill.** Slurm terminates at the limit with no grace period. Anything longer
than the partition limit must checkpoint to disk and be restartable from that checkpoint.

**Quota.** Filesystem free space is not your quota. Check the real limit before downloading, and
redirect caches off `/home`:

```bash
export HF_HOME=/scratch/$USER/hf
export PIP_CACHE_DIR=/scratch/$USER/pip
export TORCH_HOME=/scratch/$USER/torch
```

**Line endings.** A Windows-authored script reaches the cluster with CRLF and `sbatch` rejects it.
`connector.py submit` strips this; by hand, use `tr -d '\r'`.

**Output buffering.** Slurm's output file is block-buffered; without `PYTHONUNBUFFERED=1` a `tail`
of a running job shows nothing.

**JAX and torch sharing one GPU.** Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, or JAX's default
pre-allocation starves torch.

## Environment

Software may run inside an Apptainer image rather than a conda environment on the host:

```bash
apptainer exec --nv /path/to/image.sif python your_script.py
```

`--nv` exposes the GPU. Without it the container sees no device.

Compute nodes have outbound internet (PyPI, Hugging Face, GitHub all reachable), so a missing package
can be installed inside a job — but prefer baking it into the image, and prefer a CPU-only job for
any install or verification step, which costs zero GPU-hours.

## Job scripts

Every job script should:

- carry `#SBATCH` directives before the first command;
- write a `PROGRESS` line per unit of work, so a tail shows advancement;
- on a job that requests a GPU, sample telemetry to CSV alongside a heartbeat, so a stalled job is
  distinguishable from a CPU-bound one (both read 0% GPU). **Capture the sampler's own stderr**
  into the log rather than discarding it: a sampler that silently writes an empty file leaves no
  way to find out why;
- where a sampler runs, clean it up with `trap cleanup EXIT TERM INT` — and kill the sampler's
  **children** (`pkill -P "$p"`) before the sampler itself, or orphaned `sleep`/`nvidia-smi`
  processes keep the job cgroup alive and Slurm bills you for minutes after your last line of work.
  This has cost real GPU-hours on this cluster.

`telemetry_demo.sbatch` is the reference skeleton and has the orphaned-child bug; fix it rather than
copying it.

## State lives on disk, not in the conversation

A job may queue longer than your session lasts. Write every job ID to a file at submission. Any
later session recovers full status from `sacct -j <id>` alone — provided you saved its output
(S11). Never hold a job ID only in conversation context.

## Compute accounting

Report GPU-hours as a running total in **completion order**. Charging in score order implies you
knew which runs would succeed before paying for them, which makes any cost curve retrospectively
optimistic.

## Recording what the run produces

Record every design, every score, every clustering decision and every job's cost through `emit`,
as the work happens, and write the curve with it. Do not write your own ledger, results CSV or
curve plotter in its place: a curve assembled any other way cannot be put on the same axis as
another run's.

`emit` is one Python module, standard library only, which the operator stages to the cluster before
the run at `$HPC_PROJECT_DIR/lib/emit.py` (default `$HOME/hpcjobs/lib/`). Its docstring is the
reference; read it there. If it is not there, report that and stop rather than writing a substitute.

Every job script that records declares, before the first Python call:

```bash
export PYTHONPATH="${HPC_PROJECT_DIR:-$HOME/hpcjobs}/lib:${PYTHONPATH:-}"
export RUN_ID="<one id for the whole run>"
export RUN_DIR="${HPC_PROJECT_DIR:-$HOME/hpcjobs}/runs/$RUN_ID"
```

**One `RUN_ID` for the whole run**, not one per job, chosen once and recorded in your report — it
is what groups your jobs into one run, and what the operator uses to collect them. `connector.py
submit` exports only `HPC_PROJECT_DIR`, so a script that omits these fails at the first record.

```python
import emit

emit.write("run_started", model=..., arm=..., harness=..., brief_sha256=..., workspace_root=...)

with emit.stage("inference"):
    emit.design(path, target="TrkA", method="independent", seed=seed)

with emit.stage("evaluation"):
    emit.score(design_id, seq_idx=3, ipae_angstrom=5.8, plddt_0to1=0.934, scrmsd_angstrom=1.12)
    emit.cluster(cluster_id, members, method="tmalign", threshold=2.0, compared_structurally=True)

with emit.stage("admin"):
    emit.job_cost(job_id, state="COMPLETED", elapsed_s=415, gpus=1, start_utc=..., end_utc=...)
    emit.curve()
```

What each call is for, and what it does for you:

| Call | Records | Note |
|---|---|---|
| `write("run_started", ...)` | the run's identity | Once per run, in the first job. `brief_sha256` is the SHA-256 of `AGENTS.md` |
| `design(path, ...)` | one structure | Hashes the file and attributes it to the job from `SLURM_JOB_ID`. Never pass a job id |
| `score(design_id, seq_idx, ...)` | one scored sequence | **Computes** `success` from the rule; you never pass it. Metric names carry units: `ipae_angstrom`, `plddt_0to1`, `scrmsd_angstrom`. `rule_id` defaults to `strict-v1`; `repo-v1` is the `>=` pLDDT variant |
| `cluster(...)` | one clustering decision | `compared_structurally=True` only if structures were actually compared |
| `job_cost(job_id, ...)` | what one job cost | Charged to `job_id`, the job being costed, not the job making the call. Computes GPU-hours itself. Take every field from `sacct -j <id> -X -P -n -o JobID,State,ElapsedRaw,Start,End,Partition,NodeList`, after the job has left the queue |
| `curve()` | `$RUN_DIR/curve.csv` and `curve.png` | The same function the operator's tools use, so your figure and theirs cannot disagree |
| `run_ended(outcome, reason)` | how the run finished | Once, at the end, **including** a run that produced nothing |

**`job_cost` for every job, including failed ones and ones cancelled while pending.** A failed
allocation is charged in full, and a curve that omits it is retrospectively optimistic. A job's own
`sacct` figures are not final while it runs, so record a job's cost after it ends — from a later
job's `admin` stage, or from a short CPU-only job on `--partition=short`, which costs no GPU-hours.

**The curve has one point per job.** `curve()` places a job's successes at the cumulative GPU-hours
when that job ended. A (target, method) pair run as one job gives a one-point curve; run as a
sequence of shorter jobs, each with its own `job_cost`, it gives a curve with shape.

A call with a wrong field name, an unknown metric or a missing required field **raises inside the
job**. That is intended: fix the call, do not catch the error. A missing record is read later as
absence, which is worse than a failed job.

---

## Reporting discipline

These rules govern how you report, not what you compute. They do not change the experiment, the
targets, the baselines or the success criteria. Follow them exactly; they are checked against your
output files after the run, by someone who was not watching you work.

### R0. Declare the run

Before the first unit of work, write `run_manifest.json` in your working directory:

```json
{
  "arm": "B",
  "model": "",
  "effort": "",
  "harness": "",
  "workspace_root": "",
  "started_utc": "",
  "agents_md_version": "",
  "agents_md_sha256": "",
  "project_md_version": "",
  "project_md_sha256": ""
}
```

`workspace_root` is the absolute path of the only directory you will read from or write to.

**Both input files are recorded, each with its version line and its checksum.** They change at
different rates and a run is only interpretable if a later reader knows which pair it saw.
Take each version string from the `<!-- version: ... -->` line at the top of that file.

If you cannot reproduce the operator's checksum, record the operator's value as issued, record
yours beside it with the exact byte range you hashed, and report the discrepancy. Do not
silently substitute either one.

### R1. Every constant carries its source

Any numeric constant in your code, configs or report that you did not measure in this session
carries a `source` naming the file, command or document it came from.

```python
CHECKPOINT_FRACTIONS = [1/16, 1/8, 1/4, 1/2, 1]   # source: chosen by me, no external source
VRAM_CEILING_MIB = 8188                            # source: measured, nvidia-smi this session
```

"Chosen by me, no external source" is a valid and expected answer. Write it. What is not allowed is
silence: a constant with no `source` is a defect whether or not the value is correct. This includes
defaults you set in a file something else reads.

### R2. Stay inside the workspace, and say when you don't

Read nothing outside `workspace_root`. If you must — a system library, a dataset mount, another
run's output — append a line to `external_reads.log` before you use it:

```
<utc>  <absolute path>  <one line: why you needed it>
```

Sibling run directories, prior design documents and planning files are the specific case this rule
exists for. Reading one is permitted and logged; reading one silently is the defect.

### R3. A status claim cites a command and an exit code

Do not write that something succeeded, is running, or is working unless you name the command you ran
and its exit code. The literal form:

```
setup: `bash env/build_uv_env.sh` -> exit 0
```

"It looks like the environment is up" is not a status claim and must not be written as one. If the
last command you ran exited non-zero, your next sentence says so, before any interpretation of it.

**Exit zero is necessary, not sufficient.** A command can succeed at doing half its job — writing
one of the two files it was meant to write, or printing the next step it did not take. A status
claim therefore names the command, its exit code, **and the artifact the command was supposed to
produce**, checked to exist. Where a command's own output says work remains, that is part of its
result: act on it before calling the stage done.

### R4. Do not explain past what you can see

When you state a cause, state what you observed that supports it. If the chain runs outside what
this session could observe, say so and label it a hypothesis:

```
Hypothesis (unverified from this session): the VM was removed at the OS level.
Observed: `wsl -l -v` returns no distro. Not observed: what removed it.
```

A confident causal story that covers a gap in your own visibility is a defect even when it turns out
to be right.

### R5. Show the arithmetic on every derived number

Any figure you calculated rather than read carries its inputs, its units and the operation:

```
GPU-hours = 1262 s x 1 GPU / 3600 = 0.35 GPU-hours
```

Compute totals, per-sample costs, matched-budget allowances and extrapolations are the cases that
matter. A derived figure with no visible arithmetic is a defect.

### R6. Close with what you could not verify

The last section of your report is `## Unverified claims`, listing every statement in the run you
could not check yourself, one line each, with the reason. If there are none, write "None." and say
what you checked to reach that. Do not wait to be asked.

### R7. Three kinds of number, no fourth

Every figure in your report is tagged once:

- **measured** — produced by a command in this session, with the command named
- **inherited** — taken from a file or document, with the file named
- **inferred** — calculated from the other two, with the arithmetic shown per R5

A number that fits none of the three does not belong in the report. Measured figures also carry the
conditions that change them — for a VRAM or timing figure, at minimum the device and the numeric
precision.

### R8. Version the workspace as you go

Initialise a git repository in `workspace_root` before the first unit of work and commit the
initial state. Commit again at every milestone: a script you author or revise, a configuration you
write, a patch you apply to third-party code, a batch of results you fetch. The message names the
stage and says what changed and why.

Author code on this machine and transfer it. **Do not create or edit code directly on the remote
machine.** Where you must change third-party source in place, save the diff into the workspace and
commit it. Compute storage is scratch space and is purged; a patch that exists only there is a
change nobody can audit and you cannot reproduce.

**Commit the two briefs you were issued.** Copy them into the workspace exactly as given and
include them in your first commit. R0 records their checksums, which prove *which* text you ran
against; these copies are that text, so a later reader can read it instead of taking a fingerprint
on trust. If either file changes while the run is in progress, stop and report it.

Mirror third-party source you depend on into the workspace at the exact commit you used, and record
that commit. Exclude what cannot be read usefully or must not be copied: large binary artifacts, built
environments, and any credential or connector configuration. **Design the ignore rules yourself**
from what the workspace actually contains, and state in your report what you excluded and why.

Push to the remote named in `run_remote.txt`, after every commit, so the operator can follow the
run as it happens. Push to that remote and no other.

**Never create a repository, never add a second remote, and never change a repository's
visibility.** Choosing what is published is the operator's decision, not yours. If no remote was
declared, commit locally and do not push — absence of a remote is an instruction, not an
oversight to correct.

The history you push is read by people. Assume anything committed is seen: keep credentials,
tokens and connector configuration out of it, and if you find you have committed one, say so in
your report rather than quietly rewriting history.

Git is bookkeeping, not the work. If a git command fails, record the failure and carry on — never
retry it in a loop, and never let it consume budget.


---

---

## Survival rules

These stay on. They are not about reporting; turn one off and the run does not finish. Every one
is a defect that has already happened on this cluster or in an earlier run.

A project brief may add its own. `PROJECT.md` holds S10, which is specific to that project's
data.

| # | Rule | Why |
|---|---|---|
| S1 | Before downloading large artifacts, establish the real quota on **every** volume the runtime touches, including ones your project directory does not live on — **or record evidence that no limit is readable**, naming each tool tried with its return code, and say what you measured instead. | A WSL2 virtual disk filled `C:` to 105 MB free during a 6.7 GB checkpoint download, crashing the VM, although the project was correctly on `D:`. |
| S2 | Attest the hardware before planning around it: GPU present, vendor, VRAM, compute capability, CUDA runtime. Report all five. Stop if any is absent. | Two runs spent their effort before an Apple-Silicon/CUDA mismatch became decisive. An Apple Silicon GPU is not a CUDA GPU. |
| S3 | Request the GPU **model**, never a bare `--gres=gpu:1`. | On Explorer a bare request can land on a V100 or T4: no bf16, no Flash-Attention. The job then fails deep inside a running job, after queueing, not at submit. The course partition `courses-gpu` has the longest walltime and the weakest cards. |
| S4 | Measure VRAM on the smallest unit of work before committing to the largest, and record the numeric precision you measured at. | Same card class, 2.6× apart on speed and memory, purely from float32 vs bf16. A VRAM figure without its precision is not a measurement. |
| S5 | Verify the harness's own quota before the first unit of work: one cheap call, confirm it returns. | One run executed zero tool calls — an account cap, not a rate limit, and it was indistinguishable until two tiers both failed. |
| S6 | Make long work resumable from disk, and stage large downloads outside the GPU job. | Slurm kills at the walltime limit with no grace period. A 6.7 GB download inside a GPU job spends GPU time on bandwidth. |
| S7 | Compute the GPU-hour ledger from recorded timestamps, not by hand, and report it as a running total in completion order. | A 60× unit error ("21 min = 21 GPU-hours") survived into the group's planning numbers. Completion order matters separately: score order implies foreknowledge. |
| S8 | Write every job ID to disk at submission, before anything else. Recover state from the scheduler, never from conversation context. | A job may queue longer than the session lasts. A `submit` that returns non-zero may still have created a job — the dry run lost a GPU job this way and only found it in `squeue`. Re-check by job name before retrying. |
| S9 | Author job scripts with LF endings, or strip CR before submitting. | A Windows-authored script reaches the cluster with CRLF and `sbatch` rejects it. `connector.py submit` strips it; by hand it does not get stripped. |
| **S11** | **Save the scheduler's output, not just its exit code.** `sacct` state, elapsed, Start and End go into the run folder as files. | The dry run's `audit.log` recorded every `sacct` command and `rc=0` but discarded stdout, and `jobs.jsonl` carried no state or elapsed fields. Every elapsed time in its ledger was unauditable from disk. S7 and S8 are defeated if the underlying data is never captured. |
| **S12** | **No truncation caps in artifact harvesting**, and a missing timestamp is a hard error, not a skipped record. | A `tail -500` on a file-mtime listing dropped 16 of 288 timestamps, and the harvester answered a missing mtime with `continue`, silently dropping a point. Across a sweep it drops results. |
| **S13** | **Check return codes between pipeline stages.** No unconditional `exit 0`. | `set -uo pipefail` without `set -e` and no inter-stage checks means a timed-out generation is followed by filter/evaluate/analyze over partial samples, and the job reports success to Slurm. |
| **S14** | **Harvest artifacts by run identity, not by mtime window.** | `-newermt "-6 hours"` pulled three earlier runs' result CSVs into one job's log, and the parser printed a success table for every block it found with no run filter — a double-count trap for anyone summing logs. |
| **S15** | **Seed every run explicitly and independently, and record the seed.** Do not pool differently-seeded or identically-seeded runs onto one compute curve without saying which. | The dry run left the repo default seed in place for both of its jobs, then pooled them as independent draws; 7 of 8 candidate directory names recurred verbatim across the two. Independence was never established. |
| **S16** | **The remote machine is shared and persistent. Work under a run-unique root you create, and do not reuse or extend another run's.** If you find prior state where you intended to work, report it, say whether you used it, and disclose anything you changed there. | Compute storage outlives runs. A path chosen independently held 21.6 GB of an earlier run of the same project, including its results directory — which S14's harvester would have counted as this run's. Reusing a prebuilt environment also makes a cost curve omit setup it never paid for.
