# poke-bot-agent

Competition-grade PTCG agent for the Kaggle **pokemon-tcg-ai-battle** ladder.
The production-trusted path is a causal Transformer policy/value network over
each acting seat's deployment-visible observation/action history. Training and
incremental KV-cache serving share that contract. Single-guessed-world CABT
MCTS is retained only as an explicitly labeled oracle diagnostic; it cannot
generate trusted targets, pass promotion, or ship in a submission.

Primary archetype is **hard-set to `hammer-pult`** (Hammer Dragapult) via
`POKEBOT_PRIMARY_ARCHETYPE=hammer-pult` on every launch command. Pure
`dragapult` and tech variants are Phase 6 specialists. Design notes live under
`outputs/notes/`.

> Code default if the env var is unset: `deck_pool.primary_archetype()` returns
> `"dragapult"`. Always export `POKEBOT_PRIMARY_ARCHETYPE=hammer-pult` for the
> Phase 5 primary pipeline.

## Python environment

Always use the conda env (base `python` fails on the Blackwell `sm_120` GPU):

```bash
/home/inzi/miniconda3/envs/poke-bot-agent/bin/python   # torch 2.13.0+cu130
```

Do **not** use `/home/inzi/miniconda3/bin/python` (base, `2.5.1+cu124`).
Details: `outputs/notes/blackwell_torch.md`.

## GPU / device mapping

`poke_bot/config.py` sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` by default so torch
indices match `nvidia-smi`:

| Index | Card | Role |
|-------|------|------|
| `CUDA_VISIBLE_DEVICES=0` | RTX 3080 Ti (12 GB) | core-kernel / Phase 6 specialists |
| `CUDA_VISIBLE_DEVICES=1` | RTX PRO 5000 Blackwell (~48 GB) | primary hammer-pult bootstrap / RL train + default leaf-eval server |

Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID` before `CUDA_VISIBLE_DEVICES` (without
it, torch's "fastest-first" order inverts the indices and idx0 becomes
Blackwell). Full-hardware profile (`SIM_WORKERS=32`, per-GPU bf16
`BatchProfile`, ~100 GB RAM hot-set + swap overflow, `OomGuard`):
`outputs/notes/rl_loop_and_runtime.md`.

Primary model default (`config.MODEL`): `d_model=256`, layers 4/4/2, 8 heads,
`ff_dim=1024`, with causal realized-history context up to `max_context=320`.
Multi-card choices use a teacher-forced autoregressive decoder: each stage
chooses one remaining option or an explicit STOP after `minCount`. This gives
complete support to ordered legal selections without materializing factorial
action sets. Exhaustive enumeration remains oracle-search-only and fails
closed above its diagnostic cap.
Core-kernel `--gpu-profile 3080ti` uses a
leaner `d_model=192` arch for warm-start specialists (see
`outputs/notes/core_kernel.md`).

Belief aux heads (`opp_hand_head`, `opp_remainder_head`, wired `aux_head`) are
**root-only particle priors**, not board features; own prizes are exact belief
state. Blackwell Hammer **Scope B** also trains `lethal_threat_head` /
`prize_race_head` (core keeps those loss weights at 0). Details:
[`poke_bot/BELIEF_AUX_HEADS.md`](poke_bot/BELIEF_AUX_HEADS.md). Submission stays
greedy/policy until a separate belief-MCTS deploy pass.

## Setup

```bash
# 1. Competition data bundle (cg runtime + ptcg_engine + card data). Idempotent.
bash scripts/setup_competition_data.sh            # SKIP_EPISODES=1 to skip episode zips

# 2. Restore the baseline opponent field (gitignored payloads → 26 agents).
bash scripts/download_baselines.sh                # --force to re-fetch
```

## Test profiles

Testing is tiered so implementation edits do not accidentally launch native
full-game suites:

```bash
# Default development/pre-edit gate: deterministic CPU-only fixtures.
bash scripts/test_quick.sh

# Required before launch: quick, then concurrent isolated GPU0/GPU1 trusted
# 128-simulation two-iteration smokes, then one compatibility game/baseline.
bash scripts/test_canary.sh

# Manual/nightly/release only: exhaustive-roster native integration.
bash scripts/test_full.sh
```

The budgets are 30 seconds (`quick`), 20 minutes (`canary`), and 3 hours
(`full`); every command prints measured wall time and appends it to
`outputs/test-runs/suite_history.jsonl`. Pytest markers are `unit`,
`native`, `gpu`, `integration`, and `slow`. Plain `pytest` still behaves
normally and collects everything (the opt-in native test skips unless the
canary runner enables it). The invariant-to-test map is
[`tests/profile_manifest.json`](tests/profile_manifest.json).

Baseline compatibility results may be reused only when the manifest, every
baseline `main.py`/deck, and native simulator digest are unchanged. GPU
canaries are never cached, so checkpoint/config generation mismatches remain
visible. A canary validates launch correctness, target provenance, completed
simulation counts, device isolation, and clean worker shutdown—not win rate.
Promotion/evaluation sample sizes are separate and unchanged. `full` is not a
launch prerequisite unless a changed invariant has no canary coverage.

Both launch wrappers enforce preflight by default:

```bash
$PY scripts/launch_blackwell.py --run-name <lineage> -- <trainer args>
$PY scripts/launch_core_pipeline.py --run-name <lineage> -- <pipeline args>
```

Run the central canary once before starting both concurrent production
lineages, then pass `--preflight-profile quick` to each wrapper to avoid
contending with the newly started peer. `--preflight-profile none` is an
explicit emergency/operator override.

For concurrent production on this host, use 24 core workers / 3 GPU0 leaf
servers and let Blackwell auto-tune up to 40 workers with 6 GPU1 leaf servers.
The 32/4 core profile is reserved for GPU0-only canaries: under simultaneous
full Blackwell load it can exhaust an 8-second move deadline. Core collection
emits per-four-game heartbeats so the unattended stall gate remains causal
during long native waves.

## Pipeline

All commands use the conda python above. Pin the GPU per the table.

```bash
PY=/home/inzi/miniconda3/envs/poke-bot-agent/bin/python

# --- Phase 4: bootstrap data collection (hard-set primary = hammer-pult) ---
# Parallel streaming collector; expand-day pull toward the 2000-game target.
# Current JSONL ≈ 1308 games (target 2000).
$PY scripts/collect_archetype.py --force-primary hammer-pult --target 2000

# --- Phase 4: supervised bootstrap train on the Blackwell (idx1) ---
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
POKEBOT_PRIMARY_ARCHETYPE=hammer-pult \
$PY scripts/train_bootstrap.py --archetype hammer-pult --resume auto
#   flags: --jsonl --epochs --lr --games-per-batch --max-decisions-per-batch
#          --val-frac --patience --max-games --no-amp --no-cache --seed

# --- Phase 5: round-robin RL vs the full baseline field ---
# Sim workers stay CPU-only; OUR leaf eval defaults to GPU server(s) on the
# same device as the train step (--leaf-gpu auto → Blackwell), leaving the
# 3080 Ti free for core-kernel / Phase 6. Trusted play is policy-first;
# candidates are immutable and promoted only after direct draw-aware evaluation.
CUDA_DEVICE_ORDER=PCI_BUS_ID \
POKEBOT_PRIMARY_ARCHETYPE=hammer-pult \
$PY scripts/launch_blackwell.py -- \
    --archetype hammer-pult --resume auto \
    --agent-mode policy --leaf-eval gpu-server --leaf-gpu auto
#   launch_blackwell.py generates a unique run name for checkpoints/replays,
#   truncates outputs/logs/blackwell.log, and starts one fail-safe/trim monitor.
#   train_round_robin defaults (verify with -h): --iterations 10000,
#     --games-per-opp 16 / --games-per-opp-late 16 / --curriculum-switch-iter 0,
#     --min-games-per-opp 12 / --max-games-per-opp 24 (belief-MCTS budget band),
#     --workers 0 (auto → RL_GAMES_IN_FLIGHT=40, RAM-capped), --leaf-servers 2,
#     --train-epochs 1, --train-lr 5e-5, --bootstrap-mix 0.25,
#     --history-mix 1.0, --replay-fraction 0.50, --promotion-games 80,
#     --promotion-max-games 160, --promotion-min-pairs 40,
#     --promotion-threshold 0.5, --gate 0.55, --mcts-sims 128
#   NOTE: there is NO --gpu-profile on this script (that flag is train_core_kernel).
#   Experience: current + immutable prior-iteration replay + bootstrap anchor.
#   --agent-mode belief-mcts is the trusted search-target path; oracle-mcts is
#   diagnostic-only (cannot train/promote/deploy).

# --- Strict formal eval (independent balanced seats; draw-aware uncertainty) ---
$PY scripts/eval_vs_baselines.py \
    --checkpoint outputs/checkpoints/hammer-pult_round_robin.best.pt \
    --agent-mode policy --games-per-opp 100 --min-games-per-opp 100

# --- Deck-agnostic core kernel on the 3080 Ti (idx0), parallel to the primary ---
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
$PY scripts/train_core_kernel.py --device cuda --gpu-profile 3080ti --resume auto
#   flags: --device --gpu-profile {3080ti,blackwell,none} --run-name --epochs
#          --resume --smoke --probe --warm-start <archetype>
```

## Ops

### Fail-safe unattended monitor

`scripts/launch_blackwell.py` launches the uniquely named round-robin run and
one `scripts/unattended_monitor.py`. The monitor records process-group CPU/RSS,
system RAM, GPUs, disk/log growth, progress, and latest gate/metric lines. It
stops the run group on trusted-path corruption, digest/reload failures, repeated
OOM, or a configured progress stall. It never deletes run artifacts; a low or
noisy win rate alone is not fatal.

### One bounded Blackwell log

The only user-facing Blackwell console path is
`outputs/logs/blackwell.log`, independent of the run name:

```bash
tail -F outputs/logs/blackwell.log

# Blackwell GPU + recent training output, refreshed every two seconds
watch -d -n 2 'nvidia-smi -i 1; tail -n 80 outputs/logs/blackwell.log'
```

Each launch truncates that real file before opening it with `O_APPEND`. The
monitor trims it in place on the same inode above **256 MiB**, retaining the
newest **16 MiB** on whole-line boundaries where practical. It creates no
rotating archives and never removes historical logs. Override with
`--log-threshold-mb` / `--log-keep-mb`, or `POKEBOT_LOG_THRESHOLD_MB` /
`POKEBOT_LOG_KEEP_MB`; `POKEBOT_LOG_TRIM_INTERVAL` controls the check interval.

To bound an already-running legacy log without restarting training, point the
stable path at its current log inode and run one standalone owner:

```bash
$PY scripts/log_trimmer.py --file outputs/logs/blackwell.log \
    --threshold-mb 256 --keep-mb 16 --interval 30
$PY scripts/log_trimmer.py --file outputs/logs/blackwell.log --once
```

The next `launch_blackwell.py` invocation replaces an adoption symlink without
deleting or truncating its historical target, then writes directly to the real
stable file.

### Baseline prune (one-shot maintenance)

Formal round-robin/eval never turns a baseline crash into our win and never
silently shrinks the expected field; any unavailable/failing opponent
invalidates the result. Deleting confirmed hard-crashers from disk + manifest
is a separate maintenance pass:

```bash
$PY scripts/prune_broken_baselines.py            # scan + delete
$PY scripts/prune_broken_baselines.py --dry-run  # report only
```

Already run once: manifest **29 → 26** (see `baselines/README.md` /
`excluded_broken`).

## Competition submission

Kaggle expects a `.tar.gz` with `main.py` + `deck.csv` at the **top level**
(not a subfolder), plus `cg/libcg.so` and the trained checkpoint. Limit: 5
submissions/team/day; only the two most recent stay ranked (~24h settle).

```bash
# helper packs main.py + deck.csv + model.pt + cg/ + poke_bot/
bash scripts/build_submission.sh [checkpoint.pt] [out_dir]
# or manually:
tar -czf dist/submission.tar.gz -C /path/to/package .
kaggle competitions submit -c pokemon-tcg-ai-battle -f dist/submission.tar.gz -m "message"
```

Kaggle API credentials live at `~/.kaggle/kaggle.json` (outside the repo).

## Notes index

| Note | Topic |
|------|-------|
| `outputs/notes/archetype_pivot.md` | Plan contracts (primary, field, info-set, context) |
| `outputs/notes/primary_archetype_lock.md` | Why `hammer-pult` |
| `outputs/notes/rl_loop_and_runtime.md` | Phase 5 RL loop + hardware + failure policy |
| `outputs/notes/belief_aux_heads.md` | Belief aux heads (root-only priors) |
| `outputs/notes/core_kernel.md` | Deck-agnostic trunk + warm-start |
| `outputs/notes/gpu_batched_selfplay.md` | Batched leaf eval |
| `outputs/notes/phase6_priority.md` | Specialist order (after Phase 5) |
| `outputs/notes/build_resume.md` | On-disk progress / next build |
| `outputs/notes/blackwell_torch.md` | `sm_120` torch env |
| `outputs/notes/max_context.md` | `MAX_CONTEXT=320` evidence |
