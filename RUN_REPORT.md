<!-- version: report-draft, 2026-11-05. The sweep has not finished; sections 6-8 are
     marked NOT YET PRODUCED and carry no numbers rather than provisional ones. -->
# Run report — arm B, Proteina-Complexa inference-time search

`RUN_ID`: **`complexa_armB_20261105`** — the one id for the whole run, exported by every job
that records, and what groups these jobs into one run.

Briefs: `AGENTS.md` **agents-v6**, sha256 `ccff61af…9e7280`; `PROJECT.md` **complexa-v6**,
sha256 `af3f6dc7…2a84992`. Both are committed verbatim in `briefs/`, so a later reader can read
the text rather than trust a fingerprint. No operator checksum was issued for either, so there is
nothing to reconcile against; the note in `run_manifest.json` says exactly what byte range was
hashed.

Every figure below is tagged **measured**, **inherited** or **inferred** (R7). Arithmetic is shown
for anything derived (R5). Status claims name a command, its exit code and the artifact it was
supposed to produce (R3).

---

## 1. What this run has and has not established

**It has not produced the comparison PROJECT.md asks for.** At the time of writing, the sweep has
produced designs from exactly one job, and that job was a smoke test of two samples. Sections 6–8
are empty on purpose.

What it has established, each from a command in this session:

1. The pipeline runs end to end on this cluster, on a three-chain target, and produces scored
   per-sequence results (job 11638953).
2. **The ipAE scale correction PROJECT.md warns about is real and decisive.** On the stored scale
   all 18 scored sequences pass the ipAE criterion; on the Å scale none does. The difference
   between a 100% success rate and a 0% one is a factor of 31.
3. Four dependency pins this repository does not record are required for it to import at all.
4. Three of my own defects, each of which cost GPU time: see section 5.

---

## 2. Environment

**measured.** Hardware attested per S2 from inside a GPU allocation, `nvidia-smi
--query-gpu=name,memory.total,compute_cap` → exit 0:

| | smoke job 11638953 | sweep cells |
|---|---|---|
| GPU present | yes | yes |
| vendor / model | NVIDIA H200 | NVIDIA A100-SXM4-40GB |
| VRAM | 143771 MiB | 40960 MiB |
| compute capability | 9.0 | 8.0 |
| CUDA runtime | 12.6 (torch 2.7.0+cu126) | 12.6 |

All five S2 fields are present for both. bf16 is available on both (cc ≥ 8.0), which is what S3's
rule about naming the GPU model exists to guarantee.

**The two card models are not interchangeable and the run does not pretend otherwise.** Why the
sweep moved from H200 to A100, and the fact that it did so by accident before it did so on
purpose, is section 5.3.

### Dependency pins this repository does not record

**measured**, each found by a failing import in a CPU-only job (`--partition=short`, 0
GPU-hours), each fixed and re-verified by the full import chain in job 11638848 → exit 0:

| package | installed by default | required | what broke |
|---|---|---|---|
| torchaudio | unpinned latest | **2.7.0+cu126** | `undefined symbol: torch_library_impl`; transformers imports torchaudio unconditionally |
| torchvision | unpinned latest | **0.22.0+cu126** | same ABI family as above |
| atomworks | 3.0.4 | **2.2.1** | 3.x dropped `AtomSelectionStack`, which `pdb_utils.py:51,554` calls |
| dm-haiku | unpinned latest | **0.0.16** | `jax.core.take_current_trace` does not exist in the pinned jax 0.4.29 |

`env/build_uv_env.sh` pins torch and jax and leaves their companions to the resolver. Every one of
these four is the newest release of an unpinned package failing against a pinned one. The full
resolved set, 235 packages, is `logs/pip_freeze.txt` (`uv pip freeze` → exit 0).

**All dependency debugging ran on `--partition=short` at 0 GPU-hours.** That was deliberate: four
rounds of it would otherwise have cost GPU time to discover import errors.

---

## 3. The ipAE scale, which is the one result this run can state with confidence

PROJECT.md warns that the repository stores interface pAE divided by 31. **measured**, from
`binder_analysis_utils.py:79` and `refolded_structure_utils.py:131` in the mirrored source at
commit `8fc6f5c8`: the divisor is 31.0, and it is applied before the value is written.

**measured**, job 11638953, 18 scored sequences from 2 designs on `38_TNFalpha`:

```
ipae_stored      n=18  min=0.8400   max=0.9500
ipae_angstrom    n=18  min=26.0400  max=29.4500
plddt_0to1       n=18  min=0.5900   max=0.7160
scrmsd_angstrom  n=18  min=0.4866   max=2.2308
```

**inferred.** The success rule is ipAE < 7 Å **and** pLDDT > 0.9 **and** scRMSD < 1.5 Å, all three
on a single sequence.

- Applied to `ipae_stored`, every one of 18 values is below 7 → the ipAE criterion passes 18/18.
- Applied to `ipae_angstrom` = stored × 31, the minimum is 26.04 → it passes 0/18.

26.04 = 0.84 × 31. A run that scored on the stored column would report a 100% ipAE pass rate from
data in which nothing comes within 19 Å of the threshold. **This is the single most important
thing in this report**, and it is a property of the data format, not of the method.

pLDDT independently fails: max 0.7160 against a threshold of 0.9. **Successes from this job: 0 of
18 sequences, 0 of 2 designs.** Two samples on a hard three-chain target is not evidence that the
method fails; it is the expected outcome at that sample count and it is reported as such.

### Per-sequence, not per-metric-best

**measured.** The repository's result CSV columns `{seq}_complex_i_pAE` and friends are *per-metric
bests across sequences*. Scoring from them would count a design whose best ipAE came from one
sequence and whose best pLDDT came from another. `tools/record_results.py` therefore reads
`sequence_type_stats.json`, where `complex_stats[i]` and `rmsd_stats[i]` are the same sequence, and
raises on any length mismatch rather than zipping unequal lists. Native and ProteinMPNN sequences
are recorded as separate categories (`seq_source` = `native` / `mpnn`), as PROJECT.md requires.

### Multi-chain, exercised first (S10)

**measured.** Sample directories `job_0_n_488_id_0_bon_orig0_r0` and `job_0_n_545_id_0_bon_orig1_r0`.
**inferred**: `38_TNFalpha` contributes 3 chains × 146 residues = 438, so the binders are
488 − 438 = **50** and 545 − 438 = **107** residues. The three chains survive generation rather
than being collapsed, which is the S10 failure case and it was run before any single-chain target.

---

## 4. Compute ledger

**measured**, every row from `sacct -j <id> -X -P -n -o JobID,State,ElapsedRaw,Start,End,
Partition,NodeList,ReqTRES` saved to `$RUN_ROOT/sacct/<id>.psv` before anything was derived from it
(S11), and charged through `emit.job_cost`, which computes GPU-hours itself (S7).

In **completion order**, not score order:

| job | what | state | elapsed | GPUs | **inferred** GPU-hours | cumulative |
|---|---|---|---|---|---|---|
| 11637680 | env build | FAILED | — | 0 | 0.000000 | 0.0000 |
| 11638131 | env repair | COMPLETED | — | 0 | 0.000000 | 0.0000 |
| 11638284 | smoke, a100 | CANCELLED pending | 0 s | 1 | 0.000000 | 0.0000 |
| 11638336 | smoke | FAILED | 67 s | 1 | 67 × 1 / 3600 = 0.018611 | 0.0186 |
| 11638423 | smoke | FAILED | 308 s | 1 | 308 × 1 / 3600 = 0.085556 | 0.1042 |
| 11638677, 11638744, 11638848 | CPU pins | — | — | 0 | 0.000000 | 0.1042 |
| 11638953 | **smoke, succeeded** | COMPLETED | 716 s | 1 | 716 × 1 / 3600 = 0.198889 | **0.3031** |
| 11639185/229/242/259/264/280 | argparse failures | FAILED | 55 s total | 1 each | 55 × 1 / 3600 = 0.015278 | 0.3184 |
| 11646330 | CUDA OOM | FAILED | 349 s | 1 | 349 × 1 / 3600 = 0.096944 | 0.4153 |
| 11646351 | cancelled after start | CANCELLED | 29 s | 1 | 29 × 1 / 3600 = 0.008056 | **0.4234** |
| 11646367/373/391/394/402/416 | cancelled while pending | CANCELLED | 0 s | 1 | 0.000000 | 0.4234 |
| 11646703 | charging | COMPLETED | — | 0 | 0.000000 | 0.4234 |

**inferred**, cumulative = **0.4234 of 100 GPU-hours = 0.42%**.

Every job is charged, including the ones that produced nothing and the six that were cancelled
before they ever held a GPU. A ledger listing only the jobs that ran would read as though nothing
had been abandoned.

**The ceiling is not the binding constraint on this run; wall-clock is.** The QOS allows 4
concurrent GPUs, so the originally planned 40 sweep jobs of 2 h is **inferred**
40 × 2 / 4 = 20.0 hours of wall-clock before any queueing, and every submitted cell has sat
`PENDING (Priority)` behind other users' work. `PLAN_budget.md` v2 section 8 records the revision
from 40 jobs / 80.0 GPU-hours to 10 jobs / 20.0 GPU-hours and what it costs.

---

## 5. My own defects

Three, all of which spent GPU time. Each is here because a reader checking this run's cost curve
is entitled to know which of it bought nothing.

### 5.1 A version probe that reported a conclusion from an empty list

Job 11638677 printed `FAIL no atomworks version on the index provides
AtomSelectionStack.from_contig`. **That conclusion was false.** `uv pip index versions` does not
exist in this uv build — exit 2, `unrecognized subcommand 'index'` — so the candidate list was
empty, the loop body never executed, and the script reported a negative result it had never
tested. I wrote a loop without checking the exit code of the command feeding it, which is the
exact failure S13 and R3 exist to prevent. `tools/find_working_version.py` now treats an empty
release list as a hard error, never a negative result.

### 5.2 `--steps` before the overrides: six jobs, zero work

Six sweep cells failed in 3–24 s each:

```
complexa design: error: argument --steps: invalid choice: '++run_name=…'
```

`cli_runner.py:964` declares `--steps` with `nargs="+"` and `:970` declares `overrides` as a
positional `nargs="*"`, so the flag placed first consumes the overrides as step names. **The smoke
test that passed had run all four stages in one call and never passed `--steps` at all.** I split
generate from evaluate afterwards — for a good reason, so an overrunning generation would skip
scoring deliberately rather than be killed mid-evaluation — and submitted six jobs on the strength
of a test that had not exercised the line I had just changed. **inferred** cost: 55 s × 1 GPU /
3600 = 0.015278 GPU-hours.

### 5.3 VRAM: committed to the largest unit of work without re-measuring (S4)

Job 11646330 reached generation and died:

```
torch.OutOfMemoryError: Tried to allocate 4.67 GiB.
GPU 0 has a total capacity of 39.49 GiB of which 1.27 GiB is free.
  concat_pair_feature_factory.py:442
```

The tensor is `[b, n_extended, n_extended, pair_dim]` — quadratic in residues, linear in batch —
and `38_TNFalpha` is 438 target residues before the binder. The batch was the repository default
`generation.dataloader.batch_size=16` (`binder_generate.yaml:19`; `search.max_batch_size` inherits
it at `:49`).

The only VRAM figure this run had was **12580 MiB, measured at `nsamples=2` on a 143771 MiB
H200**. I then committed to `nsamples=32` on a 40960 MiB A100 without re-measuring. S4 says
measure the smallest unit before committing to the largest; that step was skipped. **inferred**
cost: 349 × 1 / 3600 = 0.096944 GPU-hours, plus the 0.008056 of a seventh job cancelled after it
had started.

Fix: `batch_size=2` set explicitly on both pipeline calls — the batch the one successful
generation actually used, not a guess — which also bounds `max_batch_size` for beam search and FK
steering.

### 5.4 The GPU model the committed script asks for is not the one that ran

`cell_*.sbatch` carried `#SBATCH --gres=gpu:h200:1`, byte-identical on both sides
(md5 `651868c5…`). `scontrol show job` → exit 0 reported `TresPerNode=gres/gpu:a100:1` for all
eight. Cause, from the connector's source: `runners/hpc.py:143` always places `--gres` on the
**sbatch command line** from its config unless the submit passes one, and a command-line flag
overrides the in-script directive. Earlier jobs reached H200 because I passed `--gres` explicitly;
the resubmission loop did not.

This did not confound the method comparison — all eight were `a100:1`, so no method had faster
hardware than another — and S3 was still satisfied, since a model was named. **The defect is that
the committed artifact did not describe what ran.** The scripts are left as submitted and
corrected in `notes/gres_override_discovery.md`; editing them afterwards would misdescribe history
in the other direction. Submits now pass `--gres` explicitly so the two agree.

---

## 6. Results — NOT YET PRODUCED

The sweep has not completed. No per-method success counts, no success-per-GPU-hour figures and no
curve exist yet, and none are stated here. When they do, each will come from `emit` and the curve
from `emit.curve()`, which is the same function the operator's tools use.

## 7. Clustering — NOT YET PRODUCED

Foldseek ran inside the smoke job's analyze step, but with 0 successes there was nothing to
collapse. Clustering will be reported per (target, method, seq_source) category at two neighbouring
thresholds, with the count of merges that were real structural comparisons stated separately. **A
category with fewer than two successes will be reported as "clustering not exercised", never as a
cluster count of 1.**

## 8. Method comparison — NOT YET PRODUCED

All five methods are configured from the repository's own implementations and verified against the
source (`notes/method_verification.md`): FK steering's shipped `temperature: 0.1` is confirmed to
be β = 1/0.1 = 10 as PROJECT.md specifies, from `fk_steering.py:223-231`; beam search matches on
all four parameters; the reward is −i_pae with all other weights 0.0. MCTS constants are the
repository's own defaults, chosen so MCTS is neither advantaged nor handicapped by a value I
picked while the other four run at shipped settings.

Method 5 is **"Generate and Hallucinate, repo-internal variant"** throughout. It refines with the
model repository's own `refinement.algorithm=sequence_hallucination`, **not** with a hallucination
baseline's optimiser. Its curve is not comparable to one produced with a baseline refiner and must
not be presented as if it were.

---

## 9. Deviations from the briefs, declared

1. **Pushing with no `run_remote.txt`.** R8's default is commit locally, do not push. The operator
   instructed in-session to push to `https://github.com/Subhash-269/agent_research_exp_1.git`, and
   the workspace is pushed to that remote and no other. I did not create the repository, add a
   second remote, or change any repository's visibility.
2. **Environment setup used the `gpu-env-build` skill**, which asks to be declared.
3. **Repo defaults disabled to save GPU time**: `compute_monomer_metrics`, `compute_designability`,
   `compute_codesignability` and `compute_esm_metrics` are all `false`. None feeds the three
   success criteria. This makes evaluation cheaper and the numbers are not comparable to a run that
   left them on.
4. **MMseqs2 is absent** and its step fails inside `analyze`. PROJECT.md names Foldseek for
   collapsing near-duplicates; Foldseek is installed and ran. No sequence-identity clustering is
   reported.
5. **`PLAN_budget.md` v1 described a sweep that was never submitted.** v2 section 8 records the
   revision and, explicitly, what it gives up: one job per cell means one point per cell, so
   methods are comparable at their endpoints but **not along the curve**, which is what PROJECT.md
   asks for.

---

## 10. Unverified claims

Every statement in this run I could not check myself, one line each.

- **No cross-family comparison was made.** BindCraft, BoltzDesign1 and AlphaDesign were not
  installed, not run and not reported. They are deferred by PROJECT.md because their environments
  cannot be built inside this budget — **not** because they were tried and failed. Nothing in this
  report should be read as a comparison between Proteina-Complexa and a hallucination-based tool.
- Whether the repository under test is private. The operator named the remote; I did not inspect
  or change its visibility, and this session cannot see it.
- Whether a job allocated `gres/gpu=1` can actually use the second device. `nvidia-smi` printed two
  A100 rows inside an allocation whose `AllocTRES` says `gres/gpu=1`. Observed: two rows, and
  `gres/gpu=1`. Not observed: `CUDA_VISIBLE_DEVICES` inside the job, or whether work can be placed
  on the second device. Charging uses 1 GPU per job because that is what the scheduler allocated.
- The connector configuration's `gres` value. `connector_config.json` is restricted by AGENTS.md
  to running the CLI and was not read. The effective request was established from `scontrol`, not
  from the file.
- A100 throughput for this pipeline. Every per-sample timing figure in this run (75.2 s/sample
  generate, 104.8 s/sample evaluate, ~400 s fixed) was measured on an **H200** in job 11638953.
  `nsamples=16` for best-of-n is sized from those H200 numbers with headroom, not from an A100
  measurement, because no A100 generation has yet completed.
- Whether `nsamples` as set will fill the 2 h allowance on an A100. If generation overruns, the
  walltime guard stops before evaluation and reports structures generated but unscored; those
  contribute no successes.
- Operator checksums for the two briefs. None was issued, so the values in `run_manifest.json` are
  this session's own and there is nothing to reconcile them against.
- Whether 23 targets would behave like the 2 run. **inferred**: covering all 23 at the same matched
  allowance is 5 methods × 23 targets × 8.0 = 920.0 GPU-hours = 9.2× the ceiling. The other 21
  targets are **under-resourced — not attempted and not failed.**
