<!-- version: complexa-v6  draft 2026-11-05 -->
# Protein binder design: inference-time search comparison

## Context

This is a purely computational task. The
targets, the methods and the success criteria below are fixed and must be used exactly as given.

**Every number you report is one you calculated in this session.** A number taken from the web is not a result. If you encounter one, do not use it.

## Task

Determine which inference-time search method over a pretrained generative model for protein binder
design performs best under a matched compute budget. Carry every candidate through the full evaluation pipeline and report the number of unique
successes as a function of cumulative GPU-hours.

The run succeeds if you produce the three evaluation numbers (ipAE, complex pLDDT, and binder scRMSD) on real folded complexes for every
method-target pair you attempt, and a success-vs-compute curve. It does not require any particular
method to win, or any target to yield successes. A run that produces honest curves with zero
successes on a hard target has done what was asked; a run that produces no numbers has not, however
much infrastructure it built.

## Model

Use NVIDIA's Proteina-Complexa, a pretrained target-conditioned generative model for protein binder
design. Code: `github.com/NVIDIA-BioNeMo/Proteina-Complexa`. Checkpoints: Hugging Face,
`nvidia/NV-Proteina-Complexa-*-160M-v1`, Protein-Target variant.

## Methods

Each method spends additional compute at sampling time to improve design quality.

1. **Best-of-N sampling**: Given an increasing compute
budget, we grow the number of generated samples N and select all binders with ipAE < 7 Å.
2. **Beam search**: We maintain a set of N (beam width) denoising trajectories ("beams"). From each beam element i, we initiate L (branching factor)
new stochastic denoising trajectories that we run for K denoising steps to obtain a total of N×L new
candidate states. We now stochastically simulate all candidates i towards clean partially latent states, decode, fold the resulting sequences, and calculate
the candidates' Rewards. We can now select the top-N candidates to form the updated beam after K steps. In practice, we set N = 4, L = 4, and use 400 discretized denoising steps following the LaProteina
schedule, with K = 100 steps per beam search update. To further scale inference-time compute, the
entire beam search procedure is repeated multiple times until the wall-clock time limit is reached.
3. **Feynman–Kac Steering** is related to beam search, but instead of top-N
uses importance sampling to sample the tilted distribution. Specifically,
we sub-sample N new states from the N×L candidates with probability proportional to exp(β*Reward), where β is an inverse temperature scaling.
Here we use the same hyperparameters as our beam search (N = 4, L = 4,
K = 100), and set the inverse temperature to β = 10 to strongly favor high-reward trajectories.

4. **Monte Carlo Tree Search**: build a search tree over the generation process and use an
explore/exploit rule (UCB-style) to decide which branches to expand, scoring intermediate states
with the same Reward. Choose the exploration constant and the number of simulations yourself, and
document both.
5. **Generate and Hallucinate**: use the model to produce a starting candidate, then refine it
further with a separate optimisation procedure.
   **Deviation you must record.** The larger experiment defines this category as refining with one
   of the hallucination baselines' own optimisation procedure. Those tools are deferred below, so
   use the model repository's own refinement path instead (`refinement.algorithm`, whose default is
   `null`). Report it as **"Generate and Hallucinate, repo-internal variant"** everywhere it
   appears, state that it is not the baseline-tool version the full protocol calls for, and do not
   present its curve as comparable to one produced with a baseline refiner. If the repository's
   refinement path does not work, report that and run four methods.

Where your search procedure needs a scalar signal (Reward), use the negative of the ipAE score of the designed complex.

**Methods are matched on GPU-hours, not on sample count.** One beam search step over N×L candidates
does far more work than N independent draws. Compare at equal spend, and state the spend for each
method-target pair.

## Baselines — deferred, do not run

The hallucination-based tools BindCraft, BoltzDesign1 and AlphaDesign are out of scope for this run.
Do not install them, do not run them, and do not report numbers for them. They are deferred because
their environments cannot be built inside this budget, not because they were tried and failed — say
it that way. Your report's unverified-claims section states that no cross-family comparison was made.

## Targets

| Target Name | PDB ID | Target Chain & Residues | Hotspot Residues | Binder Length | Source |
|---|---|---|---|---|---|
|BBF-14|9HAG|A1-112|–|70-250|BindCraft|
|Betv1|AF2 prediction|A1-159|A24|70-185|BindCraft|
|CbAgo (PAZ)|AF2 prediction|A180-276|A268|70-160|BindCraft|
|CD45|AF2 prediction|d1 domain|A1|80-200|BindCraft|
|Claudin1|AF2 prediction|A1-188|A31, A46, A55, A152|80-175|BindCraft|
|CrSAS6|AF2 prediction|A1-145|–|90-200|BindCraft|
|Derf21|AF2 prediction|A1-118|A10|70-185|BindCraft|
|Derf7|AF2 prediction|A1-196|A115|70-185|BindCraft|
|HER2-AAV|1N8Z|C23-268|C165, C170|60-100|BindCraft|
|IFNAR2|2LAG|A1-103|A45, A73, A75, A77, A89, A91|60-175|BindCraft|
|PD-1|AF2 prediction|A1-115|A33, A95, A97, A102|80-150|BindCraft|
|SpCas9|AF2 prediction|A96-174, A306-446|A360|70-150|BindCraft|
|BHRF1|2WH6|A2-158|A65, A74, A77, A82, A85, A93|80-120|AlphaProteo|
|H1|5VLI|A1-50, A76-80, A107-111, A258-322, B1-68, B80-170|B21, B45, B52|40-120|AlphaProteo|
|IL7RA|3DI3|B17-209|B58, B80, B139|50-120|AlphaProteo|
|IL17A|4HSA|A17-29, A41-131, B19-127|A94, A116, B67|50-140|AlphaProteo|
|Insulin|4ZXB|E6-155|E64, E88, E96|40-120|AlphaProteo|
|PD-L1|5O45|A17-132|A56, A115, A123|50-120|AlphaProteo|
|SARS-CoV-2 RBD|6M0J|E333-526|E485, E489, E494, E500, E505|80-120|AlphaProteo|
|TNF-α|1TNF|A12-157, B12-157, C12-157|A113, C73|50-120|AlphaProteo|
|TrkA|1WWW|X282-382|X294, X296, X333|50-120|AlphaProteo|
|VEGFA|1BJ1|V14-107, W14-107|W81, W83, W91|50-140|AlphaProteo|

Use the target files the model repository ships, referenced from its own
`configs/targets/targets_dict.yaml`, rather than re-fetching from RCSB and slicing them yourself.

**S10. Multi-chain targets are not optional, and they are the failure case.** Several targets above
carry more than one chain. Analysis code that assumes a single chain silently inflates the
unique-success count to the raw file count — a wrong answer that reports cleanly. Run at least one
multi-chain target early, before committing budget to the rest, and report what the multi-chain path
required. Exercising it early turns a silent wrong answer into a loud failure.

## Evaluation pipeline

Identical for every method and every target. Run all of it.

For each generated candidate:

1. Produce 8 redesigned binder sequences via ProteinMPNN, and evaluate the model's own native
   sequence directly with no redesign, reporting the two as separate categories. Both are required.
2. Fold the complex using an AlphaFold2-based predictor with target templating and initial-guess
   initialisation enabled.
3. Score each folded result on interface predicted alignment error (ipAE), complex pLDDT (normalised to 0–1), and binder
   self-consistency RMSD (scRMSD). Do not reuse the generation reward as your final evaluation score. Re-fold and re-score every
   design through this full pipeline.
4. A design counts as a success only if, for a single sequence: ipAE < 7 Å, complex pLDDT > 0.9,
   and binder scRMSD < 1.5 Å.
   - **ipAE scale — verify before you score anything.** The repository stores interface pAE divided
     by 31. Convert to Å before applying `< 7`. Applying `< 7` to the stored value passes
     essentially every design, and the run then reports a near-100% success rate that looks like a
     result. Verify the factor against the repository rather than trusting this sentence, and report
     the observed min and max of each metric so a reader can see the threshold discriminates.
5. Collapse near-duplicate successful designs by structural similarity before counting, and
   report the number of distinct clusters as your "the number of unique successes" figure. Use Foldseek as the clustering method.
   - **State the similarity threshold you used**, and report the unique count at two neighbouring
     thresholds as well. The count is sensitive to this choice, and a single number without its
     threshold cannot be compared against anything.
   - Report how many merge decisions were made by actual structural comparison, versus forced by a
     shortcut such as a length or chain-count mismatch. A cluster count where no pair was ever
     structurally compared is not a clustering result. If a category has fewer than two successes,
     say clustering was not exercised rather than reporting a cluster count of 1 as a result.

## Compute

**Ceiling: 100 GPU-hours total of allocated GPU time**, including environment setup inside the
allocation and all failed attempts. Checkpoint and parameter downloads staged outside the allocation
do not count against it. Stop at the ceiling and report where you got to. Do not request an
extension inside the run. Cancelled jobs, failed smokes and retries are charged in full.

Report unique successes as a running function of cumulative GPU-hours in completion order, per
method and per target. Methods must be comparable at any point along the
curve, not only at the end. Each method on a given target should get the same GPU-hour allowance.

Before the first GPU job, write a plan allocating the ceiling across all
methods and targets. Update the plan as you run GPU jobs. A method or target that cannot be given a
fair share of the budget must be reported as under-resourced.
