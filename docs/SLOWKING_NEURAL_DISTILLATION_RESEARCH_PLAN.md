# Slowking neural distillation research plan

Status: research proposal only. A full replacement architecture is permitted
for investigation, but this document creates no runtime, selector, checkpoint,
training, serving, freeze, registration, or submission authority.

## Recommendation

Do not rebuild the old sparse guide-loss system as the primary Slowking brain.
Use all observed Slowking policies as an archetype-wide behavior prior, add
offline value learning so the student can improve rather than merely imitate, and use the
exact simulator for bounded belief-aware search at critical decisions. Distill
the improved search policy back into a fast option-conditioned actor.

The proposed system has four parts:

1. a causal public-state/history encoder;
2. a hierarchical option-conditioned policy and distributional critic;
3. exact-simulator policy improvement over top candidate actions; and
4. a fast distilled actor for ordinary play, with search reserved for
high-value branch points.

The first executable heuristic surrogate is now available. It is valuable as
a sparse teacher feature, confidence mask, and regression baseline—not as the
brain. Across 768 games it matches 410/430 confirmed covered decisions (95.3%)
and 226/228 on the `ShumpeiNomura` lineage, while abstaining on the overwhelming
majority of the 25,917 nontrivial single-selection prompts. This is the desired
shape: near-deterministic bot rules become explicit; strategy that needs prize,
resource, target-subset, or future-turn reasoning remains a learned/search
problem.

### Recommended winning architecture

1. Archetype-wide option-conditioned behavior cloning on all 768 games.
2. Sparse heuristic scores/rule IDs as offline auxiliary features, never serving logits.
3. Existing strategic, setup, combo-state, value, and distributional heads.
4. IQL-style conservative offline improvement.
5. Belief-aware exact-simulator search for payloads, targets, discards, pivots, and prize maps.
6. Distillation of improved search distributions into the fast actor.
7. RLM use for offline replay mining and counterexample discovery, not live control.

RLMs are useful in this project, but chiefly as offline analysts and curriculum
generators. A recursive language model is not the best live controller for a
structured, latency-sensitive game with an exact simulator.

## Data program

### Pin the teacher lineage

The expanded audit confirms 768 Slowking games across two named teams and three
exact lists: 311 `vibechu` games with the older list, 17 `ShumpeiNomura` games
with a one-card intermediate list, and 440 `ShumpeiNomura` games with the final
list. All 768 games are policy- and strategic-learning eligible. Deck contents
and legal actions determine which card-specific routes are available; exact
fingerprints remain optional conditioning and evaluation metadata rather than
an eligibility gate. Split by calendar day—never by individual frame—so
near-duplicate games and adjacent policy versions cannot leak across train and
validation.

Use three data strata:

- final-list `ShumpeiNomura` seats (440 games): action imitation, return targets,
  and final-list evaluation;
- intermediate-list `ShumpeiNomura` seats (17 games): the same learning targets
  plus explicit Spectrier-versus-second-Smoochum capability evidence;
- older `vibechu` seats (311 games): the same archetype policy, value, and causal
  head targets, with Boomerang Energy, Counter Gain, and Secret Box routes
  enabled only when those cards are present.

Preserve losses. Win-only cloning hides recovery failures and exaggerates
actions correlated with already-winning states. Use outcome/advantage weighting
only after a critic has been validated.

### Targets to derive

For every legal decision, retain the complete legal option set and train:

- chosen-option policy target;
- game return and distributional value;
- action-value/advantage for the chosen action;
- macro intent: setup, search, stack, pivot/gust, attach/recover, copied attack,
  targets, or end turn;
- top-card provenance and stack validity;
- next-attacker Energy readiness;
- two-Slowpoke continuity;
- copied source: Kyurem, Conkeldurr, Annihilape, or fallback;
- one-/two-turn Prize map and knockout timing;
- Bench liability and multi-Prize exposure;
- opponent hidden-remainder belief, derived only from submitted deck minus
  public evidence; and
- exact causal next-state targets from the replay transition.

Every unavailable label is masked. No hidden future state may enter the actor.

## Proposed model: Slowking Deliberative Policy v2

### Representation

Encode zones and Pokémon as structured card/instance tokens with ownership,
location, evolution ancestry, HP/damage, Energy, Tools, public effects, Prize
counts, turn flags, and action history. Use a causal Transformer or state-space
history encoder, plus cross-attention from each legal option to the encoded
state. Score legal options independently before combining them, so target and
resource choices are not collapsed into an option-blind state vector.

Add list-conditioning and policy-lineage embeddings. They let the network share
mechanics with older Slowking data without pretending that Counter Gain and
Boomerang lines exist in the `ShumpeiNomura` list.

### Hierarchical actor

Use a learned macro latent with these roles:

- develop Slowpoke chain;
- establish mobility/draw support;
- construct top deck;
- prepare next attacker;
- choose copied attack and targets;
- recover resources;
- disrupt/gust;
- close the game; and
- pass/end.

The macro is auxiliary context, not a hard rule gate. The final action remains
an option-conditioned score over the exact legal set. Train a direct flat
policy head in parallel as a safety baseline and for ablation.

### Critic and offline improvement

Start with behavioral cloning, then fit an in-distribution offline critic. An
IQL-style objective is a strong first choice because it can improve over the
behavior policy without evaluating arbitrary unseen actions. Use expectile
value regression, temporal-difference Q targets, and advantage-weighted policy
updates. Compare against plain behavior cloning and a return-conditioned
Decision Transformer; do not assume sequence modeling alone improves wins.

Use conservative uncertainty estimates. When critic ensembles disagree, fall
back toward the teacher prior rather than extrapolating.

Primary references:

- Implicit Q-Learning: <https://arxiv.org/abs/2110.06169>
- Decision Transformer: <https://arxiv.org/abs/2106.01345>

### Exact-simulator search

The project already has the game simulator, so do not learn a MuZero/Dreamer
dynamics model as the default. Exact rules are more valuable than an
approximate world model. Use the actor as a prior and the critic as a leaf
evaluator, then search only the top-K legal actions.

Because the game is partially observed, search over particles sampled from the
opponent belief state rather than one invented hidden hand. Each particle must
be consistent with the submitted deck and all public evidence. Use a
POMCP/information-set-MCTS-like controller or bounded determinization with
aggregation across particles.

Critical search nodes include:

- final top-stack choice;
- Kyurem/Conkeldurr/Annihilape selection;
- Trifrost’s three-target subset;
- Prime Catcher target plus pivot;
- Destined Fight and simultaneous-win states;
- large Ultra Ball/Night Stretcher discard choices;
- Wondrous Patch target;
- Bench commitment that opens a short Prize route; and
- any state where policy entropy or critic disagreement is high.

Distill the improved root visit distribution and search Q-values into the
actor. This converts expensive counterfactual reasoning into a fast policy over
time. MuZero and Dreamer remain useful references for search/value
distillation, but their learned dynamics are unnecessary while exact simulation
is available:

- MuZero: <https://arxiv.org/abs/1911.08265>
- DreamerV3: <https://arxiv.org/abs/2301.04104>

## Where RLMs fit

Recursive Language Models treat a long prompt as an external environment and
let a root model inspect it programmatically through recursive calls. That is a
good match for offline work over thousands of large replay transcripts:

- retrieve all instances of a tactical question;
- write and execute state filters;
- cluster decision motifs;
- compare wins and losses with matched public features;
- propose candidate rules and find counterexamples;
- produce human-readable guide revisions with exact episode citations; and
- generate a hard-case curriculum for the neural learner.

It is not a strong default live policy. RLM recursion solves context management,
not game-tree search; it is costly, nondeterministic, and awkward to constrain
to millisecond-scale legal actions. If tested live, use an RLM-style
orchestrator only at a tiny set of critical nodes, with tools that expose the
structured state, legal actions, critic values, and bounded simulator queries.
The language model may select analysis programs, but the simulator remains the
rules authority and the neural/search policy remains the action authority.

Primary RLM reference: <https://arxiv.org/abs/2512.24601>.

## Training stages

### Stage A — archetype-wide teacher clone

Train the option-conditioned actor on all 768 replay games. Supply the submitted
deck and exact legal option set so card-specific behavior is learned from state,
not used as a dataset filter. Report
episode-held-out action agreement overall and separately for setup, stacking,
attack source, target subset, recovery, and pivot decisions. Calibrate entropy;
high top-1 agreement on trivial prompts must not hide failure on decisive
branches.

Inject the frozen heuristic surrogate only as offline metadata:

- rule id, option score, margin, and abstention mask become auxiliary features
  or targets;
- a learned residual actor predicts every legal action, including uncovered
  prompts;
- teacher agreement is reported separately from actor agreement;
- the actor must operate with the heuristic channel zeroed as a mandatory
  ablation; and
- no heuristic score is added to serving logits.

This makes the reverse-engineered rules useful for sample efficiency without
placing a brittle rule system in the winning path.

### Stage B — causal heads and offline critic

Train next-state, Prize-map, Energy-readiness, belief, value, and advantage
heads. Apply IQL-style advantage weighting only after held-out value calibration
and rank correlation pass. Retain an unweighted behavior-cloning checkpoint as
the immutable baseline.

### Stage C — simulator policy improvement

Run bounded belief-aware search from replay states and fresh simulator states.
Create immutable search receipts containing state identity, belief particles,
candidate actions, visit counts, Q distributions, chosen action, and simulator
version. Distill only valid, causal root targets.

### Stage D — population self-play

Train against the archetype teacher clone, frozen project specialists, public top
agents, and strategically different archetypes. Keep one active learner and
freeze opponent checkpoints. Mix first/second seats exactly and keep replay
evaluation isolated from training.

### Stage E — planner distillation and runtime policy

Distill search into the actor until most ordinary decisions do not need online
search. Gate the search budget using calibrated policy entropy, critic ensemble
disagreement, and detected critical macros. The runtime must fail closed to the
fast actor if the planner exceeds its bounded budget.

## Required experiments

Run paired ablations from the same data split and initialization:

1. existing flat policy baseline;
2. archetype-wide behavior cloning over all 768 games;
3. behavior cloning plus hierarchical macro targets;
4. behavior cloning plus IQL critic/advantage weighting;
5. actor plus exact-simulator critical-node search;
6. search-distilled actor;
7. optional RLM-assisted curriculum; and
8. optional learned world model only if exact-simulator throughput is the
   measured bottleneck.

Measure:

- paired realized win rate with confidence intervals;
- first/second and matchup-stratified results;
- teacher agreement and search-improvement regret at critical nodes;
- value calibration and action-rank correlation;
- invalid-action and causal-information violations;
- latency p50/p95/p99 and search-budget exceedance;
- leave-one-head/macro ablations; and
- exploitability against held-out opponents and policy versions.

Promotion requires a fresh training-ineligible paired game gate. Replay action
agreement, guide-label accuracy, or offline return estimates cannot substitute
for wins.

## Stop conditions

Reject or redesign the system if:

- archetype-wide coverage is too sparse at decisive action classes;
- deck contents or legal actions fail to distinguish card-specific routes;
- the critic improves offline metrics but lowers paired wins;
- search gains disappear under hidden-state particles;
- an RLM path adds latency without measured critical-node improvement;
- learned dynamics underperform exact simulation on parity tests; or
- any actor input contains opponent-private or future information.
