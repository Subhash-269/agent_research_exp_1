<!-- version: plan-v2  written 2026-11-05 before the first GPU job; revised 2026-11-05 after the smoke run, see section 8 -->
# Budget plan: allocating the 100 GPU-hour ceiling

PROJECT.md requires this plan to exist **before the first GPU job** and to be updated as GPU
jobs run. Every number below is tagged per R7 and carries its arithmetic per R5.

`RUN_ID` for the whole run: **`complexa_armB_20261105`**
Run-unique cluster root: `/scratch/nitta.v/complexa_armB_20261105` (S16; nothing pre-existing
there — see `sacct/S1_quota_evidence.txt`).

---

## 1. What the ceiling has to cover

**inherited** (PROJECT.md, "Compute"): ceiling = 100 GPU-hours of *allocated* GPU time,
including setup inside an allocation and every failed attempt. Checkpoint/parameter downloads
staged outside an allocation do not count.

**inherited** (AGENTS.md, "Cluster facts"): partitions `gpu` (8 h walltime), `gpu-short` (2 h),
`gpu-interactive` (2 h). QOS per user: `gpu` = 4 concurrent GPUs / 8 submitted jobs;
`gpu-short` = 2 GPUs. Default partition `short` is CPU-only and costs no GPU-hours.

**measured** (`python connector.py preflight` -> exit 0, output in `notes/preflight_login.txt`):
the `gpu` partition carries a100, h200, v100-pci/sxm and t4 nodes. Per S3 every job requests the
model explicitly (`--gres=gpu:a100:1` or `--gres=gpu:h200:1`), never a bare `gpu:1`.

## 2. What is deliberately kept off the GPU

Everything that can run on a CPU runs on `--partition=short`, which is charged at 0 GPU-hours:

- building the uv environment (`env/build_uv_env.sh`);
- `complexa download` of model checkpoints, AF2 parameters and ESM weights (S6: large
  downloads are staged outside the GPU allocation);
- installing Foldseek;
- every `job_cost` bookkeeping call for jobs that have already ended.

**inferred**: GPU-hours charged for phase 0 = 0 h x 1 GPU / 3600 = **0.00 GPU-hours**, because
no job in it requests a gres. This is the single largest saving available and it is the reason
setup is split out rather than done inside the first GPU job.

## 3. Allocation

| Phase | What | Jobs | Walltime each | GPUs | Planned GPU-hours |
|---|---|---|---|---|---|
| 0 | env build, weight download, Foldseek | 2 CPU jobs | 8 h | 0 | 0.0 |
| 1 | hardware attestation (S2), VRAM at bf16 and fp32 (S4), ipAE-scale verification, multi-chain target exercised first (S10) | 2 GPU jobs | 2 h | 1 | 4.0 |
| 2 | main sweep: 5 methods x 2 targets | 40 GPU jobs | 2 h | 1 | 80.0 |
| — | contingency: requeues, walltime kills, failed allocations | — | — | — | 16.0 |
| | | | | **total** | **100.0** |

**inferred**, phase 1: 2 jobs x 2 h x 1 GPU = 4.0 GPU-hours.
**inferred**, phase 2: 5 methods x 2 targets x 4 jobs x 2 h x 1 GPU = 80.0 GPU-hours.
**inferred**, total committed = 0.0 + 4.0 + 80.0 = 84.0 GPU-hours, leaving
100.0 - 84.0 = 16.0 GPU-hours of contingency.

### Why 4 jobs per cell rather than 1

**inherited** (AGENTS.md, "Recording what the run produces"): `curve()` puts one point per job at
the cumulative GPU-hours when that job ended. A (target, method) cell run as a single 8 h job
yields a one-point curve. Run as 4 x 2 h jobs, each with its own `job_cost`, the same spend
yields a 4-point curve with shape — which is what "methods must be comparable at any point along
the curve, not only at the end" requires. The 2 h unit also matches the `gpu-short` walltime, so
a cell can fall back to that partition when `gpu` is congested.

### Matched budget

Each (method, target) cell gets **8.0 GPU-hours** (4 jobs x 2 h x 1 GPU), identical across all
five methods on a given target. Methods are matched on GPU-hours, not on sample count: one
beam-search update over N x L candidates does far more work than N independent draws, so sample
counts are an output of the cell, never an input to it.

## 4. Methods (all five, from the repository's own search implementations)

| PROJECT.md method | repo setting (`configs/pipeline/binder/binder_generate.yaml`) |
|---|---|
| Best-of-N sampling | `search.algorithm=best-of-n` |
| Beam search | `search.algorithm=beam-search`, `beam_width=4`, `n_branch=4`, `step_checkpoints=[0,100,200,300,400]` |
| Feynman–Kac steering | `search.algorithm=fk-steering`, `beam_width=4`, `n_branch=4`, `temperature=0.1` |
| Monte Carlo Tree Search | `search.algorithm=mcts` |
| Generate and Hallucinate, **repo-internal variant** | `refinement.algorithm=sequence_hallucination` |

**inherited** (`binder_generate.yaml`): beam search and FK steering already ship at N=4, L=4 with
`step_checkpoints: [0, 100, 200, 300, 400]`, i.e. K=100 steps between updates over 400 denoising
steps — exactly the hyperparameters PROJECT.md specifies. Nothing is invented here.

**FK steering inverse temperature.** PROJECT.md asks for beta = 10. The repo exposes
`fk_steering.temperature: 0.1`. 1 / 0.1 = 10, so the shipped value already is beta = 10 if that
field is a temperature rather than an inverse temperature. **This is checked against the source
before the sweep runs, not assumed** — see `notes/` for the verification, and the report states
which it turned out to be.

**MCTS constants, which PROJECT.md leaves to me:**

```
MCTS_N_SIMULATIONS   = 20    # source: repository default, binder_generate.yaml:mcts.n_simulations
MCTS_EXPLORATION_C   = 1.0   # source: repository default, binder_generate.yaml:mcts.exploration_constant
MCTS_EXPLORATION_P   = 0.5   # source: repository default, binder_generate.yaml:mcts.exploration_prob
```

Chosen as the repository's own defaults rather than values of my own, so that MCTS is not
advantaged or handicapped by a hyperparameter I picked while the other four methods run at
their shipped settings.

**Reward.** `reward_model.af2folding.reward_weights.i_pae = -1.0`
(**inherited**, `binder_generate.yaml`) — the negative of interface pAE, which is what PROJECT.md
specifies as the scalar search signal.

## 5. Targets

Two of the twenty-three, chosen so the multi-chain path is exercised first (S10):

| Cell target | repo `task_name` | chains in `target_input` | binder length | why |
|---|---|---|---|---|
| TNF-alpha | `38_TNFalpha` | **three** (A, B, C) | 50–120 | the multi-chain failure case S10 exists for, and the most chains of any target in the table |
| TrkA | `33_TrkA` | one (X) | 50–120 | single-chain control at the same binder-length range, so a method's behaviour on one chain and three is comparable |

**Seeds (S15).** Every job passes an explicit `seed` and records it. Seeds are distinct per job
and per cell; no two jobs share one, and the report says which jobs were pooled onto which curve.

```
SEED_BASE = 20261105   # source: chosen by me, no external source (the run date, for legibility)
```
Job seed = `SEED_BASE + 1000 * cell_index + job_index`, which is a distinct integer per job.

## 6. Under-resourced, stated up front

**21 of the 23 targets in PROJECT.md get zero GPU-hours in this plan.** They are under-resourced,
not attempted and not failed. **inferred**: covering all 23 targets at the same matched allowance
would need 5 methods x 23 targets x 8.0 GPU-hours = 920.0 GPU-hours, which is
920.0 / 100.0 = 9.2x the ceiling. Two targets is what the ceiling buys at an allowance large
enough for a cell to produce any successes at all.

The hallucination baselines (BindCraft, BoltzDesign1, AlphaDesign) are **deferred by PROJECT.md**
and are not installed, not run and not reported. They are deferred because their environments
cannot be built inside this budget — not because they were tried and failed.

## 7. Stop rule

The ceiling is 100 GPU-hours of allocated time. The ledger is recomputed from `sacct` after every
job (S7, S11). When cumulative GPU-hours reach 100.0 the run stops and reports where it got to.
No extension is requested inside the run.

---

## Revision log

| date (UTC) | change |
|---|---|
| 2026-11-05 | v1 written before the first GPU job. |
| 2026-11-05 | v2: section 8 added. Sweep cut from 40 jobs / 80.0 GPU-hours to 10 jobs / 20.0 GPU-hours. The binding constraint is wall-clock, not the ceiling. |


---

## 8. Revision after the smoke run: what the sweep actually is

Section 3 planned 40 sweep jobs and 80.0 GPU-hours. **That is not what was submitted**, and the
difference is large enough that reading section 3 alone would misdescribe the run.

### What changed

| | planned (v1) | submitted (v2) |
|---|---|---|
| sweep jobs | 40 | 10 |
| jobs per (target, method) cell | 4 | 1 |
| walltime each | 2 h | 2 h |
| **inferred** sweep GPU-hours | 40 x 2 x 1 = **80.0** | 10 x 2 x 1 = **20.0** |
| points per cell on the curve | 4 | **1** |

### Why

**measured** (`squeue -u $USER` -> exit 0, `logs/squeue_resume2.txt`): the QOS permits **4
concurrent GPUs and 8 submitted jobs**. 40 jobs of 2 h at 4 at a time is
**inferred**: 40 x 2 / 4 = 20.0 hours of wall-clock, before any queueing. The eight cells
submitted on 2026-11-05 all sat `PENDING (Priority)` behind other users' work, so 20.0 hours is
a floor, not an estimate.

The ceiling was never going to bind. **The limit on this run is session wall-clock.** v1 did not
say that, and it should have: a plan that allocates 84.0 of 100.0 GPU-hours implies the ceiling
is the scarce resource, and here it is not. Cumulative spend at the time of this revision is
0.3184 GPU-hours, **0.32% of the ceiling**.

### What this costs, stated plainly

Section 3's own argument for 4 jobs per cell was that `curve()` places one point per job, so a
cell run as a single job yields **a one-point curve with no shape**. Cutting to 1 job per cell
gives up exactly that. The consequence is concrete and must not be glossed:

- Each (target, method) cell contributes **one point**, at the cumulative GPU-hours when that
  job ended. Comparing methods *along* the curve is therefore **not supported by this run**;
  only the endpoints are comparable.
- PROJECT.md asks for methods comparable at any point along the curve. **This run does not
  deliver that**, and the report says so rather than presenting 10 points as a curve.

The alternative — 4 x 2 h per cell for 2 cells only — would have bought curve shape for 2 of the
10 (target, method) pairs and nothing for the other 8. Covering all 10 pairs once was chosen
instead, because PROJECT.md's primary question is a comparison *across the five methods*, which
needs every method present. **This is my choice, and the trade is recorded here so a reader can
disagree with it.**

### If wall-clock allows, cells are extended, not replaced

Extra jobs appended to a cell carry their own `job_cost` and add points to that cell's curve,
which is why the allowance is expressed per job rather than per cell. Nothing already recorded
is discarded to do it.

### Seeds, corrected

v1 wrote `seed = SEED_BASE + 1000 * cell_index + job_index`, which assumed 4 jobs per cell.
With one job per cell the formula actually used is

```
seed = SEED_BASE + cell_index        # source: tools/make_cells.py:122, SEED_BASE = 20261105
```

giving 20261106 through 20261115, **distinct per cell** (S15). No two jobs share a seed, and no
two differently-seeded jobs are pooled onto one curve point.

### Budget after the revision

| Phase | Jobs | **inferred** GPU-hours |
|---|---|---|
| 0, setup (CPU only, `--partition=short`) | 7 | 7 x 0 = 0.00 |
| 1, attestation + multi-chain smoke | 4 (3 failed, 1 completed) | 0.3031 |
| failed sweep cells (argparse defect) | 6 | 55 s x 1 / 3600 = 0.0153 |
| 2, sweep as resubmitted | 10 | 10 x 2 x 1 = 20.00 |
| **committed total** | | 0.3031 + 0.0153 + 20.00 = **20.32** |
| **unspent against the ceiling** | | 100.00 - 20.32 = **79.68** |

The run will finish well under the ceiling. That is a fact about the wall-clock available, not
evidence that the methods are cheap, and the report states it that way.
