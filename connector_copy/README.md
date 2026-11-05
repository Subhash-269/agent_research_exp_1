# agenticAI with GPUs

Lets an AI agent reach a GPU on its own: write a job script, submit it, poll it, pull the
results back, cancel it if needed, with no human at the shell. Three backends sit behind one
CLI, so the same commands and the same job scripts work on all of them.

Shared across teams; see [Contributing](#contributing).

**Backend status:** `hpc` (Explorer) works end to end. `runpod` is implemented but its pod
lifecycle is untested. `aws` is a stub — every method raises.

---

## Quick start

Everything runs from inside `compute/`. No pip installs; the `hpc` backend needs only `ssh`
on your PATH.

```bash
cd compute
cp connector_config.example.json connector_config.json
# fill in "user" and "account" for the hpc runner — see below

python connector.py check            # config + connectivity
python connector.py preflight        # survey the cluster (hpc only)
python connector.py submit smoke_gpu.sbatch
python connector.py watch <id>       # poll until it leaves the queue
python connector.py logs <id>        # stdout
python connector.py fetch <id>       # pull logs + telemetry locally
python connector.py ledger           # GPU-hours, cumulative
```

Pick a backend with `--runner hpc|runpod|aws`, or set `"default"` in the config.

Find your Slurm account with:

```bash
sacctmgr show assoc user=$USER format=account%40
```

`connector_config.json` is gitignored — it holds your username and key path. Never commit it.

---

## What is here

| Path | |
|---|---|
| `compute/connector.py` | the CLI |
| `compute/runners/base.py` | the shared `Runner` interface — six methods, plus audit and ledger |
| `compute/runners/hpc.py` | Slurm over SSH |
| `compute/runners/runpod.py` | RunPod API |
| `compute/runners/aws.py` | stub |
| `.claude/skills/` | reusable procedures, one directory each. Not Claude-specific — plain markdown with a short header. Start with `gpu-env-build` for environment setup |
| `compute/RUNBOOK_hpc.md` | Explorer: SSH keys, first job, monitoring, troubleshooting |
| `compute/RUNBOOK_runpod.md` | RunPod setup and usage |
| `compute/SMOKE_TEST_PROMPT.md` | hand this to an agent to verify a backend works |
| `compute/AGENTS_gpu_section.md` | append to your `AGENTS.md` so the agent knows how to reach GPUs |
| `compute/smoke_gpu.sbatch` | GPU probe: identity, telemetry, progress lines, matmul |
| `compute/telemetry_demo.sbatch` | skeleton for long jobs that report on themselves |
| `Claude_Code_Desktop_Manual.docx` | Claude Code Desktop setup |

Two files are written as you work and are gitignored: `jobs.jsonl` records every run id at
submit time, and `audit.log` appends every action taken against a backend. Nothing is held in
process memory, so a later session recovers state from the backend alone.

---

## Four things that will cost you a day

**A plain `--gres=gpu:1` gives you a V100** — compute capability 7.0, no bfloat16. Name the type
you want: `--gres=gpu:h200:1` or `--gres=gpu:a100:1`. A100 and H200 need no special request.

**An over-quota GPU request does not fail — it pends forever.** `sbatch` returns success *and a
job ID*, then the job sits on `Reason=QOSMaxGRESPerJob` indefinitely. "Submitted OK" is not a
success signal. One GPU per job; parallelism comes from array breadth (`--array=0-18%2`).

**There is no PyTorch on the nodes,** and Docker never works there (it needs root). Use
`module load miniconda3` or an Apptainer image.

**`/home` is 75 GB and fixed; `/scratch` is purged monthly and not backed up.** Anything you
leave on scratch, container images included, disappears. `/projects` is the only persistent
option and a PI must request it.

More detail in `compute/RUNBOOK_hpc.md` and `compute/RUNBOOK_runpod.md`.

---

## Adding a backend

Subclass `Runner` in `compute/runners/base.py` and implement six methods: `connect`, `submit`,
`status`, `logs`, `fetch`, `cancel`. Add a `REGISTRY` entry in `runners/__init__.py`. The CLI
never branches on runner, so once those six return real values every command works against your
backend unchanged.

`cost()` is optional but needed for `ledger`. Report GPU-hours in **completion order** —
charging in score order implies you knew which runs would succeed before paying for them, which
makes any cost curve retrospectively optimistic.

Everything a runner does should go through a method on the base class. That is what keeps
`audit.log` complete; a runner that shells out directly breaks the promise that a human can
read what was actually done without trusting the agent's own summary.

Verify with `SMOKE_TEST_PROMPT.md` — hand it to an agent and it should pass unchanged, except
for the Slurm-specific steps.

---

## Contributing

Pull requests to `main`. Keep your own `connector_config.json` out of commits.

If you hit a cluster behaviour that is not documented, add it to the list above that section
is the most useful part of this repo for whoever comes next.

---

## Authors

`compute/` — Venkat Neelraj Nitta (connector, HPC runner), Kinshuk Singh (RunPod runner),
Vishak Shashikumar (RunPod runbook). Desktop manual — Poornima Tagaram.
