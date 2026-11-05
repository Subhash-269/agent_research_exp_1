<!-- version: methodver-v1  2026-11-05 -->
# Checking each method's settings against PROJECT.md, in the source

PROJECT.md specifies hyperparameters for several of the five methods. Rather
than assume the repository's defaults match, each was read in
`third_party/Proteina-Complexa` at commit
`8fc6f5c85b38a2585ffaac555e708576dc4e8533`. Line references are to that tree.

Every figure here is **inherited** (R7) — read from a file, with the file named.
Nothing in this document was measured.

---

## The reward signal

PROJECT.md: "Where your search procedure needs a scalar signal (Reward), use
the negative of the ipAE score of the designed complex."

`configs/pipeline/binder/binder_generate.yaml`, `reward_model.af2folding.reward_weights`:

```
con: 0.0   i_pae: -1.0   plddt: 0.0   dgram_cce: 0.0   min_ipae: 0.0   ...
```

Every weight is zero except `i_pae = -1.0`. The composite reward is therefore
exactly the negative interface pAE. **Matches.**

One scale note: the reward is computed on the repository's *stored* interface
pAE, which is Angstroms ÷ 31. Since 31 is a positive constant, the negative
stored value and the negative Angstrom value induce the same ordering and the
same softmax up to a rescaling of the temperature — which matters for FK
steering and is picked up below.

## Beam search

PROJECT.md: N = 4 (beam width), L = 4 (branching), 400 denoising steps on the
LaProteina schedule, K = 100 steps per beam-search update.

`binder_generate.yaml`:

```
search.beam_search.n_branch: 4        -> L = 4
search.beam_search.beam_width: 4      -> N = 4
search.step_checkpoints: [0, 100, 200, 300, 400]
```

The checkpoints are 0, 100, 200, 300, 400, i.e. updates every K = 100 steps
across 400 steps. **Matches on all four.** Nothing is overridden.

## Feynman-Kac steering

PROJECT.md: same N, L, K as beam search, and "sub-sample N new states from the
N x L candidates with probability proportional to exp(beta * Reward)", with
beta = 10.

The repository exposes `fk_steering.temperature: 0.1`, not a beta. Whether
that is beta or 1/beta decides whether the shipped default is the specified
setting or is off by a factor of 100, so it was read rather than guessed.

`src/proteinfoundation/search/fk_steering.py:223-231`:

```python
# -- importance sampling: P(sample) ~ exp(reward / temperature) --
...
logits = rewards_for_selection / temperature
sampling_probs = torch.softmax(logits, dim=1)
```

So P ∝ exp(Reward / temperature). With `temperature = 0.1`:

```
beta = 1 / temperature = 1 / 0.1 = 10
```

**The shipped default already is beta = 10.** It is used unchanged, and no
override is passed.

Two qualifications a reader should have:

1. `fk_steering.py:242` samples with `torch.multinomial(..., replacement=True)`.
   PROJECT.md says "sub-sample N new states from the N x L candidates" without
   stating whether that is with or without replacement. Sampling with
   replacement is the standard Feynman-Kac/SMC resampling step, and it is what
   this run used. Recorded as a difference in reading, not a deviation I chose.

2. The reward fed to the softmax is on the stored ipAE scale (Angstroms ÷ 31),
   so beta = 10 here is beta = 10 *on that scale*. On an Angstrom scale the
   equivalent inverse temperature would be 10 / 31 = 0.32. PROJECT.md does not
   say which scale its beta is on. I used the repository's own default without
   rescaling, because rescaling would mean choosing a number PROJECT.md does
   not give me, and because beam search — the method FK steering is meant to be
   compared against — ranks on the same stored scale.

## Best-of-N

PROJECT.md: "grow the number of generated samples N and select all binders with
ipAE < 7 Angstrom."

`search.algorithm: best-of-n`, `search.best_of_n.replicas`. The selection at
`ipAE < 7 Å` is not done by the search; it is the evaluation threshold, applied
after the full refold pipeline, by `emit.score`'s `strict-v1` rule. PROJECT.md
step 3 is explicit that the generation reward must not be reused as the final
score, so the two are kept apart deliberately.

## Monte Carlo Tree Search

PROJECT.md leaves the exploration constant and the number of simulations to me,
and asks that both be documented.

```
MCTS_N_SIMULATIONS      = 20    # source: binder_generate.yaml mcts.n_simulations
MCTS_EXPLORATION_CONST  = 1.0   # source: binder_generate.yaml mcts.exploration_constant
MCTS_EXPLORATION_PROB   = 0.5   # source: binder_generate.yaml mcts.exploration_prob
```

These are the repository's own defaults, used unchanged. The reason for taking
them rather than picking my own: the other four methods all run at shipped
settings, so a hyperparameter I chose for MCTS alone would be the one thing
that differs between it and the rest, and any gap in the result would be
unattributable.

## Generate and Hallucinate, repo-internal variant

PROJECT.md requires this to be reported under that exact name, to state that it
is not the baseline-tool version the full protocol calls for, and not to present
its curve as comparable to one produced with a baseline refiner.

`binder_generate.yaml`:

```
refinement.algorithm: null        # options: null, sequence_hallucination
refinement.enable_greedy_optimization: true
refinement.n_greedy_iters: 15
refinement.loss_weights: {pae: 0.4, plddt: 0.1, i_pae: 0.1, con: 1.0,
                          i_con: 1.0, rg: 0.3, i_ptm: 0.05, helix_binder: -0.3}
```

The refinement path is `sequence_hallucination`, driven by ColabDesign AF2
losses. It is the repository's own refinement, not BindCraft's, BoltzDesign1's
or AlphaDesign's. PROJECT.md defers those three and tells me to use this path
instead and to label it. If it does not work, PROJECT.md says to report that
and run four methods.

## Deferred baselines

BindCraft, BoltzDesign1 and AlphaDesign are **not installed, not run and not
reported**. They are deferred because their environments cannot be built inside
this budget — not because they were tried and failed. No cross-family
comparison was made.
