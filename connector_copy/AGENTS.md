# Agent guide — agenticAI_with_GPUs

Read by Codex and other agents that honour `AGENTS.md`. `GEMINI.md` points here, and Claude Code
auto-discovers the skills below, so there is one copy of everything.

## Reusable procedures live in `.claude/skills/`

Despite the directory name, these are **not Claude-specific**. Each is a directory containing a
`SKILL.md`: plain markdown, with a short YAML header naming the procedure and saying when to use
it. The header exists so Claude Code can auto-discover the file; if your tool does not use it,
ignore it and read the body.

**List that directory and read whichever matches your task.** Skills are added over time, so the
directory is the index — this file deliberately does not enumerate them.

```bash
ls .claude/skills/*/SKILL.md
head -5 .claude/skills/*/SKILL.md   # the headers say what each one is for
```

Some skills ship helper scripts alongside their `SKILL.md`. Those read state and report; anything
that changes state stays in your own commands, so it lands in `audit.log`.

Skills in this repo are **scaffolding**, and each says so in its own header. If you use one, state
in your report that the step was assisted — otherwise a reader comparing against unassisted runs
will attribute the difference to the model rather than to the help.

## What else is here

| Path | |
|---|---|
| `compute/connector.py` | CLI to reach a GPU: submit a job, poll it, read its logs, pull artifacts back, cancel it |
| `compute/AGENTS_gpu_section.md` | How to reach GPUs on Explorer — append to a task's own `AGENTS.md` |
| `compute/RUNBOOK_hpc.md` | Explorer: SSH keys, first job, monitoring, troubleshooting |
| `compute/RUNBOOK_runpod.md` | RunPod setup and usage |
| `compute/SMOKE_TEST_PROMPT.md` | Verify a backend end to end before trusting it |

## Before you install anything

Read the `gpu-env-build` skill. It is discovery-driven: it tells you to read what the repo you
cloned declares for itself rather than applying a remembered package list, and it carries the
things that have cost this project days — where to build, where caches go, which volumes get
purged, and why a generic GPU request gets you a card with no bfloat16.

## Before you request a GPU

Name the GPU type. On Explorer the order is:

    gpu:h200:1  →  gpu:a100:1  →  gpu:v100-sxm2:1   (fp32 only; say so in any number you report)

A plain `--gres=gpu:1` lands on a V100 (compute capability 7.0, no bf16) and fails inside the
running job rather than at submit.

An over-quota GPU request does **not** fail. `sbatch` returns success and a job ID, then the job
pends on `QOSMaxGRESPerJob` indefinitely. "Submitted OK" is not a success signal — poll the
pending reason.

## Reporting

Report GPU-hours as a running total in completion order, not a single final figure. State the
precision you actually ran. Keep `env_attestation.json` next to the results so a number can be
tied to the environment that produced it.
