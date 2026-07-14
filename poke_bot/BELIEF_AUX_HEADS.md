# Belief + Blackwell strategy aux heads

Neural heads supply **root-only** signals from info-set `state_vec`. They must
**never** be written into `features.build_board_tokens`. Leaf remote inference
(`batched_infer`) stays policy+value only.

## Scope table

| | **Scope A — Core / deck-agnostic** | **Scope B — Blackwell Hammer specialist** |
|---|---|---|
| **Where** | 3080 Ti / small `train_core_kernel`, shared trunk | GPU1 large profile, `hammer-pult`, `launch_blackwell` / RR |
| **Belief particle priors** | `opp_hand_head`, `opp_remainder_head`, wired `aux_head` | Same (shared) |
| **Own prizes** | Exact `PublicBeliefHistory.own_known_prizes` (not NN) | Same |
| **Lethal / can-take-prizes-now** | **Out of scope** | `lethal_threat_head` (IN SCOPE) |
| **Prize-map / KO-path / prize-race** | **Out of scope** | `prize_race_head` scaffold (IN SCOPE); full prize-map sequence **deferred** |
| **Matchup-specific techs** | No | Future room on specialist only |
| **Training weights** | Strategy weights **0**; belief weights on | RR defaults: lethal `0.15`, prize-race `0.10` |
| **Deploy gate** | Strategy search bias off | `POKEBOT_BLACKWELL_STRATEGY_HEADS` / hammer+`POKEBOT_GPU_PROFILE=blackwell` |

## Scope A heads (core + Blackwell)

- `aux_head` — archetype CE → soft-reweights `EmpiricalDeckPosterior` hypotheses
- `opp_hand_head` / `opp_remainder_head` — `Linear(d_model → CARD_VOCAB)` multilabel
  priors over legal hidden multisets

## Scope B heads (Blackwell Hammer only)

- `lethal_threat_head` — `Linear(d_model → 1)` logit for
  **P(own prize count decreases soon)** (can-take-prize / KO-path available)
- `prize_race_head` — `Linear(d_model → 2)` regresses normalized
  `[own_prizes/6, opp_prizes/6]` (prize-race / KO-path **scaffold**)

**Deploy choice (v1):** optional **root-only value bias** from the lethal logit
(`blackwell_heads.root_value_bias_from_lethal`). Not injected into bags; not
forced onto tiny core_kernel. Policy temperature bias deferred.

Modules are architecture-present for warm-start compatibility; core training
does **not** require non-zero strategy loss weights or strategy labels.

## Labels

### Scope A (belief)

Prefer self-play / sim JSON that dumps both seats’ private zones. Remask via
`replay_import._strip_opp_private` keeps GT in `aux_labels` (`opp_hand`,
`opp_deck_order`, `opp_prizes`) while clearing board-visible obs. Ladder seat
obs often lack private GT → multilabel BCE is **masked** (zero contrib).

Helpers: `train.belief_multihots_from_aux_labels`, `train.masked_belief_card_bce`.

### Scope B (strategy) — honest construction

Attached by `blackwell_heads.attach_blackwell_strategy_labels` during
`replay_import` / `dataset.convert_record`:

| Key | Construction | Caveat |
|---|---|---|
| `prize_race` | Public board prize counts → `[own/6, opp/6]` | Always available when prize lists/counts exist |
| `lethal_threat` | Post-hoc: within next H (=8) **own** decision frames, did `own` prize count **decrease**? | Outcome supervision along the played line — **not** a public damage calculator. Approximates “KO / prize-take was available soon.” |

Masked when absent. CLI weights:

- Scope A: `--aux-loss-weight`, `--opp-hand-loss-weight`, `--opp-remainder-loss-weight`
- Scope B: `--lethal-threat-loss-weight`, `--prize-race-loss-weight`
  (defaults **0** on bootstrap/core_kernel; **0.15 / 0.10** on `train_round_robin`)

## Warm-start (late head adds)

1. Distinct named heads — never grow `aux_head` into a kitchen sink  
2. `load_model_from_checkpoint` allowlists missing
   `opp_hand_head.*` / `opp_remainder_head.*` /
   `lethal_threat_head.*` / `prize_race_head.*`  
3. Uniform particle fallback while `model.warm_started_belief_heads` contains
   Scope A **card** heads  
4. Scope B search bias skips freshly warm-started strategy heads  
5. Checkpoint provenance: `aux_heads_present` + `warm_started_belief_heads`

## Gating helpers

- `blackwell_heads.blackwell_strategy_heads_enabled()` — search/deploy gate  
- `launch_blackwell.py` sets `POKEBOT_GPU_PROFILE=blackwell`,
  `POKEBOT_BLACKWELL_STRATEGY_HEADS=1`, `POKEBOT_PRIMARY_ARCHETYPE=hammer-pult`  
- `CoreTrainConfig.lethal_threat_loss_weight` / `prize_race_loss_weight` stay **0.0`

## Deferred within Blackwell scope

- Full **prize-map sequence** / multi-step KO-path soft targets over attack lines  
- Public damage-calc lethal labels (type chart / energy legal attacks)  
- Matchup-specific tech heads on the specialist  
- Policy-temperature bias from lethal (value bias shipped instead)
