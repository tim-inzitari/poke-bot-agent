# Pure RL Pipeline

Independent pure-RL line on branch `cursor/pure-rl-full-rebuild-2d48`
(PR vs `main` — not a Hope merge). Hope tip archived at
`archive/a-new-hope-pre-pure-rl` (+ tag).

1. **Deck-agnostic core** (Stage A) with AWR, search off, full-box hardware  
2. **Warm-start `hammer-pult`** specialist → held-out gate → greedy submit  

## Abhyuday / field guidance (design gospel)

Source: [Kaggle discussion 717697 — “Sharing my Reinforcement Learning journey”](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/717697)
(Abhyuday / pure-RL competitor comments).

**Cited resources**

| Resource | Role |
|---|---|
| [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) | Explicit RL primer he linked (“helped me understand the basics”) — actor–critic, advantages, on-policy freshness |
| [YouTube `eKC5PlYoboE`](https://www.youtube.com/watch?v=eKC5PlYoboE) | Cited in our runbook as foundations context; treat as RL / actor-critic mindset alongside Spinning Up (his post also points at TCG strategy videos for *human* meta study — we still keep **no rules knowledge as skill prior**) |

**Checklist (enforced)**

1. **Pure RL / pure self-play** — millions of variations; no hand-written rules knowledge as the skill prior (`PURE_RL_SELF_PLAY_FRAC` default **0.85**; self vs recent-self pool).
2. **Not AlphaZero-style** — no MCTS visit-target overnight path; `mcts_sims=0` in collect.
3. **Competition starter is terrible** — refuse starter paths; fresh small seed; `bootstrap_mix=0`.
4. **Efficient board representation + small policy** — `pure_rl_model_config()` ~**1.6M** params (`d_model=16`); fail-closed if `>3.5M`; prefer **&lt;2M**.
5. **High throughput** — aspirational ~7k SPS via volume: host CPU + GPU0/1 leaves + Elmo + bert whole-game farms.
6. **Refined curriculum** — self-play first; core multi-archetype decks then widen; official public bots for **gate** + light mix only.
7. **Representation richness** — audit obs for decisions (ongoing); do not starve the info set.
8. **Spinning Up mindset** — actor–critic AWR, `A = R − V` (stale V), fresh short replay window.
9. **Top-250-cards style focus early** — Stage A samples a small multi-archetype deck pool (not full 2k-card BC); widen later after gate.

## Spinning Up alignment

Guide: [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/spinningup.html).
We keep **AWR** (advantage-weighted regression on played actions) rather than a
half-baked PPO — same actor-critic / advantage spirit, simpler for high-SPS
imperfect-info collect.

| Spinning Up idea | Pure-RL knob / code |
|---|---|
| Actor–critic: π from advantages, V ≈ E[return\|s] | AWR on `selected_index`; value head → terminal W/L/D; **no** CE to behavior π |
| Advantage = return − baseline | `A = R − V(s)` with **stale/detached** V (optional whitening) |
| On-policy / fresh data | `PURE_RL_REPLAY_WINDOW_SHARDS` (default 2); `bootstrap_mix=0` enforced |
| Exploration vs exploitation | Collect: temperature sample; eval/submit: **greedy** |
| Prefer simple algorithms | Search OFF; not AlphaZero MCTS visit targets |
| Discount γ | **γ = 1** (undiscounted Monte Carlo terminal return) |

## Small model (mandatory)

`poke_bot.pure_rl.model_profile.pure_rl_model_config()` — default
`d_model=16`, 1/1/1 layers, `ff=32`, `ctx=32` → ~**1.6M** params.
Launch **fail-closes** if `sum(p.numel()) > 3.5M`. Do **not** warm-start from
Hope's d=256 primary or the competition starter.

## Fatal learning contract

- `TrainConfig.pure_rl_defaults()` → AWR on factorized `selected_index`  
- Stale (detached) value baseline; optional advantage whitening; weight clip  
- Soft `history_policy` CE targets **hard-fail**; `bootstrap_mix=0`  
- Aux / strategy head loss weights **0**; `POKEBOT_BLACKWELL_STRATEGY_HEADS=0`  
- Collect: `mcts_sims=0`, temperature sample (annealed); eval/submit: greedy  
- Fresh-data window; self-distill abort  
- **Self-play first**; public bots = held-out gate + light mix  

## Full hardware (from Stage A)

| Resource | Role |
|---|---|
| CPU workers | Max sim / in-flight games (self-play primary) |
| GPU1 Blackwell | Train + majority leaf replicas |
| GPU0 3080 Ti | Same-model leaf replicas |
| Elmo + bert | Additive whole-game farms (light public mix + capacity) |
| Overlap | Collect shard `t+1` while training shard `t` |

One active AWR trainee on the host. Remotes are **not** a second trainer.

Profile: `poke_bot.pure_rl.hardware.full_hardware_profile()`.  
Remote protocol: `poke_bot.remote_jobs.RemoteWorkerFarm` + `iter_additive_results`.  
Default endpoints: `192.168.1.143:8765,bert.local:8766`.

## Commands

```bash
# CI / wiring canary (no CABT, no remotes)
POKEBOT_PYTHON=/home/inzi/miniconda3/envs/poke-bot-agent/bin/python \
  bash scripts/canary_pure_rl.sh

# Overnight core (fresh small seed; self-play heavy; remotes on)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0
export POKEBOT_PYTHON=/home/inzi/miniconda3/envs/poke-bot-agent/bin/python

$POKEBOT_PYTHON -u scripts/launch_pure_rl.py \
  --mode core \
  --run-name pure_rl_core_overnight_<UTC> \
  --preflight-profile none \
  --log outputs/logs/pure_rl_core.log \
  --remote-worker-endpoints 192.168.1.143:8765,bert.local:8766 \
  -- \
  --base-checkpoint outputs/pure_rl/<run>/checkpoints/seed.pt \
  --iterations 1000 \
  --games-per-iter 256 \
  --heldout-games 200 \
  --gate-wr 0.70

# After CORE_GATE_PASSED: warm-start specialist
python -u scripts/warm_start_pure_rl_specialist.py \
  --core-checkpoint outputs/pure_rl/pure_rl_core_overnight/checkpoints/iter_XXXXX.pt \
  --run-name pure_rl_hammer --archetype hammer-pult
```

## Gates

- Held-out official four baselines, ≥200 seat-balanced games, WR ≥ 0.70  
- Exclude `baseline_failed` forfeits (`poke_bot.pure_rl.eval_public`)  
- Abort promote on self-distill / ~zero advantage (`poke_bot.pure_rl.aborts`)  

## Layout

```
outputs/pure_rl/<run-name>/
  shards/
  checkpoints/
  metrics/
  manifest.json
  CORE_GATE_PASSED | SPECIALIST_GATE_PASSED
```

## Post-core / specialist (future)

**Do not change Stage A core self-play overnight for these.** Sequence: finish
`CORE_GATE_PASSED` → then revisit for hammer-pult / specialist warm-start.

| Notebook | Notes (skim / deferred) |
|---|---|
| [beicicc — ptcg-public-experiment-snapshot-jul15](https://www.kaggle.com/code/beicicc/ptcg-public-experiment-snapshot-jul15) | Public experiment snapshot (Jul 15). Revisit after core for specialist deck ideas, representation tricks, and training recipes — **not** Stage A collect. |
| [makimakiai — ptcg-public-28-plus-sample-4-roster-update](https://www.kaggle.com/code/makimakiai/ptcg-public-28-plus-sample-4-roster-update) | Public 28+ + sample-4 roster update. Candidate later for specialist/meta roster mix and gate opponent sampling — **park only** until core gate. |
