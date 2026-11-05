# Gemini guide — agenticAI_with_GPUs

See [AGENTS.md](AGENTS.md). One copy, so every agent reads the same thing.

**Reusable procedures** live in `.claude/skills/` — plain markdown, not Claude-specific. List the
directory and read whichever matches your task. Read `gpu-env-build` before installing anything.

**Before requesting a GPU:** name the type — `gpu:h200:1`, then `gpu:a100:1`, then
`gpu:v100-sxm2:1` (fp32 only). A plain `--gres=gpu:1` gives a V100 with no bfloat16 and fails
inside the running job, not at submit.
