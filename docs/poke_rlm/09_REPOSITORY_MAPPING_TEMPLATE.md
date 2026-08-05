# Repository Mapping — Filled Audit (Prompt 01)

> Filled from read-only inspection of `/workspace` on branch
> `cursor/experimentation-026a` @ `acc04af7bf9c896dee2e5fbef4268fd7dde6b677`.
> Production search/training behavior was not modified for this audit.
> Items that could not be measured in this cloud environment are marked
> **BLOCKED** with the reason.

## 1. Repository identity

| Item | Actual value | Evidence/command |
|---|---|---|
| Repository root | `/workspace` | `git rev-parse --show-toplevel` → `/workspace` |
| Default/current branch | `cursor/experimentation-026a` (audit HEAD); preferred merge base `codex/worksession-20260728`; remote `main` also present | `git branch --show-current`; cloud `base_branch` |
| Remote | `https://github.com/tim-inzitari/poke-bot-agent` | `git remote get-url origin` |
| Primary language/version | Python; `.python-version` = `3.11`; cloud runtime observed `Python 3.12.3` | `cat .python-version`; `python3 --version` |
| Dependency manager | `requirements.txt` (no `pyproject.toml` / `setup.py`) | `ls requirements.txt` |
| Dependencies listed | `torch`, `numpy`, `scipy`, `scikit-learn`, `tqdm`, `ipykernel`, `jupyter`, `kaggle-environments` | `cat requirements.txt` |
| Test runner | `pytest` via `pytest.ini` (`testpaths = tests`) | `pytest.ini`; command `pytest -m unit …` |
| Formatter/linter/type checker | **BLOCKED / not configured in-repo** — no `ruff.toml`, `mypy.ini`, `.flake8`, `.pre-commit-config.yaml`, or `pyproject.toml` | `ls` of those paths → missing |
| Kit AGENTS vs workspace AGENTS | Workspace root `AGENTS.md` is the Pokemon RL **controller contract**. PokeRLM kit instructions installed at `docs/poke_rlm/AGENTS_POKE_RLM.md` (not overwriting root `AGENTS.md`) | file presence |

## 2. Current agent entry points

| Responsibility | File path | Symbol/signature | Notes |
|---|---|---|---|
| Competition agent entry | `submission/main.py` | `def agent(obs_dict: dict) -> list[int]` (`:258`) | Deck from `deck.csv`; turn-order resolved before heavy imports |
| Turn/decision loop | `poke_bot/agent.py` | `PolicyAgent.__call__(self, obs_dict: dict) -> list[int]` (`:863+`) | Dispatches MCTS / RTP / greedy |
| Current policy inference | `poke_bot/agent.py` | `PolicyAgent.greedy_select` (`:654`); `_factorized_greedy_prepared` | Factorized legal-option stages; encode-once then `decode_options` |
| Experimental RTP path (this branch) | `poke_bot/agent.py` + `poke_bot/recursive_turn_planner/agent_bridge.py` | `PolicyAgent.rtp_select` (`:665`); `RTPAgentBridge.select` | **Default `use_recursive_turn_planner=True` on this experiment branch**; greedy fallback |
| MCTS/search entry | `poke_bot/agent.py` | `mcts_select` → oracle `MCTS.search` or `belief_mcts_select` → `BeliefMCTS.search` | Oracle single-world requires `oracle_mode`; trusted path is belief MCTS |
| Fallback policy | `submission/main.py`, `poke_bot/agent.py`, `poke_bot/submission_budget.py` | `trusted_search_or_greedy_select`; `_fail_closed_legal`; `SubmissionSearchBudget.plan` | Search failure → frozen greedy for same decision; illegal → legal random clamp |
| Kaggle packaging entry | `submission/main.py` | `agent` | Search disabled by default via `submission/search_config.json` `"enabled": false` |

Dispatch on this branch (`PolicyAgent.__call__`):

```text
select is None → deck
use_mcts → mcts_select
elif use_recursive_turn_planner + local model → rtp_select
else → greedy_select
```

## 3. CABT interface

| Contract | File path | Symbol/signature | Measured behavior |
|---|---|---|---|
| Build/start battle | `poke_bot/cg_env.py` | `battle_start(deck0, deck1) -> (obs_dict, StartData)` (`:105`) | Pass-through to `cg` Game API |
| Step / submit option indices | `poke_bot/cg_env.py` | `battle_select(select_list: list[int]) -> dict` (`:110`) | Engine consumes ordered option-index list |
| Finish battle | `poke_bot/cg_env.py` | `battle_finish()` (`:115`) | Frees battle memory |
| Clone/search begin | `poke_bot/cg_env.py` | `search_begin(obs_dict, search_inputs, manual_coin=False)` (`:240`) | Creates search world via `_cg_api().search_begin` |
| Search step | `poke_bot/cg_env.py` | `search_step(search_id, select)` (`:250`) | Advances search node by option indices |
| Search release | `poke_bot/cg_env.py` | `search_release(search_id)` (`:255`) | Wrapper present; few/no call sites beyond wrapper |
| Search end | `poke_bot/cg_env.py` | `search_end()` (`:260`) | Ends search session |
| Observe / typed obs | `poke_bot/cg_env.py` | `to_observation(obs_dict)` (`:96`) | `_cg_api().to_observation_class` |
| Legal sample (game) | `poke_bot/cg_env.py` | `random_legal_select(obs_dict, rng=None)` (`:156`) | Samples `k ∈ [minCount,maxCount]` distinct indices |
| Legal sample (search) | `poke_bot/cg_env.py` | `legal_select_from_searchstate(search_state, rng=None)` (`:265`) | Same pattern on SearchState |
| Hidden/public views | `poke_bot/features.py`, `poke_bot/belief.py` | `assert_info_set`; `assert_deployment_observation` | Opponent hand must be `None`; belief rejects privileged keys |
| Turn boundary | `poke_bot/features.py`, RTP bridge | `state.turn` encoded as `turn/10`; `turn_key_from_obs -> (seat, turn)` | RTP plan persistence keyed by `(yourIndex, turn)` |
| Seed control | **partial** | `PolicyAgent.rng`, `BeliefMCTS.rng`; no `cg_env` seed arg | Official libcg seeding opaque (`engine_rebuild/interfaces.py` notes); prefer `manual_coin` / recorded traces |

### Legal-option example (from unit fixture)

`tests/test_belief_mcts.py:37-44`:

```python
"select": {
    "type": 0,
    "context": 0,
    "minCount": 1,
    "maxCount": 1,
    "option": [{"type": 14}, {"type": 13}],
    "deck": None,
}
```

- Raw indices: `0`, `1` into `option[]`
- Semantic fields on options: at least `type` (engine OptionType); richer bindings encoded in `features.build_option_tokens`
- Continuation/action-group: complete actions are ordered permutations of indices for counts in `[minCount, maxCount]` (`features.enumerate_action_combos`, cap `MAX_ACTION_COMBOS=4096`)
- Factorized deployment path: `features.factorized_action_candidates(obs, prefix)` autoregressive stages with explicit STOP when `minCount` met

**Live CABT execution in this cloud pod:** **BLOCKED** — `cg` runtime not installed (`paths.cg_runtime_dir` raises `FileNotFoundError`).

## 4. Observation pipeline

| Item | File path/symbol | Shape/type | Visibility status |
|---|---|---|---|
| Public + own private board | `features.build_board_tokens` | `SparseVector` with `NUM_BOARD_TOKENS=24` words → model `[B,24,D]` | Acting-seat info set |
| Global scalars | `features.py` board builder | includes `state.turn/10`, first-player flag, seat one-hot | Public/acting |
| Own private state | hand / prizes / deck remainder in board bag | Sparse features | Acting seat only |
| Opponent hidden state | `features.assert_info_set` | `opp.hand is None` required | **Must be excluded** |
| Privileged keys rejected (belief) | `belief.assert_deployment_observation` | rejects opp `deck`, `deckOrder`, `prizeOrder`, `privateState`, etc. | Deployment-forbidden |
| History/event tokens | `PolicyAgent.board_history`, `previous_action_history`; model `forward_history_batch` / KV cache | list of board SparseVectors + optional previous-action SparseVectors; temporal `[B,T,D]` | Causal realized history only |
| Deck/archetype context | `PolicyAgent.deck`; matchup route via `ShadowMatchupAdapterRouter` / runtime tree | deck `list[int]` len 60; route int (`-1` unknown) | Deck known to actor; route from public evidence |
| Feature schema version | `features.FEATURE_SCHEMA_VERSION = 5`; model buffer `_feature_schema_version` | int32 | Checkpointed |
| Canonical observation hash | **BLOCKED / not found as a single deployment hash helper** | — | Public belief fingerprint exists in belief MCTS (`information-state fingerprint`); no universal obs SHA helper located for all paths |

### Training-only / privileged fields (must not enter deployment inputs)

From `belief.assert_deployment_observation` and `replay_import._strip_opp_private`:

- Opponent `hand` (may exist in privileged replay; stripped to aux labels)
- Opponent `deck` / `deckOrder` / `deck_order`
- Opponent `prizeOrder` / `prize_cards` / `hiddenPrize` / `truePrize`
- `privateState`
- Future/outcome labels used by strategic heads / teacher search (training targets only)

## 5. Current model

### Configuration parents

| Profile | File/symbol | Configuration | Parameter count |
|---|---|---|---:|
| Global `ModelConfig` defaults | `poke_bot/config.py` `ModelConfig` | `d_model=256`, spatial/temporal=4, option_decoder=2, `n_heads=8`, `ff_dim=1024`, `max_context=320`, history+KV | **BLOCKED live build** (needs `cg` for vocab); not the current Pure-RL production lean default |
| Pure-RL lean default | `poke_bot/pure_rl/model_profile.py` `pure_rl_model_config` | `d_model=96`, spatial=4, temporal=0, option=4, `ff_dim=384`, stateless, dense card2vec | Tests require `1e6 ≤ n ≤ 2e6` and `≤ 3.0M` target / `≤ 3.5M` fail (`test_pure_rl_awr.py`, `PURE_RL_PARAM_*`) |
| Pure-RL history variant | same file | temporal=1, history+KV, max_context=320 | Still `< 2e6` per `test_pure_rl_state_profile.py` |
| Alakazam ordinary refresh (receipt) | `state/final_format_alakazam_model_inventory_r79.json` | d=96, heads=8, spatial=4, temporal=1, option=4, ff=384, context=320 | **1,684,103** learned |
| Alakazam H10-I target (receipt) | same | d=96, spatial=7, temporal=3, option=7, ff=2496, context=320 | **10,352,606** learned |
| Slowking candidate inventory (receipt) | `state/slowking_candidate_parameter_inventory_v1.json` | includes fusion/setup/combo + V6 adapters | **1,910,963** total |

### Components

| Component | File/symbol | Configuration | Parameter count |
|---|---|---|---:|
| Embeddings | `TemporalCabtTransformer` EmbeddingBag or `FactorizedCard2Vec` | vocab from live `cg` (`card_vocab` documented ~1268 in features comments) | Included in receipts above; live recount **BLOCKED** without `cg` |
| State encoder | spatial + optional temporal transformer | see profiles | Included in receipts |
| Policy head | `policy_head = Linear(D,1)` over option hidden | logits `[B,max_N]` | Included |
| Value/evaluator | `value_head` → `tanh` → `[B]` | state-conditioned | Included |
| Action representation | `decode_options` / `option_hidden` | `option_hidden [B,max_N,D]`; logits `[B,max_N]` | — |
| Latent lookahead (optional) | `ActionConditionedLatentLookahead` | default width 512; enabled flags default **False** | Formula: **658,690** @ D=256/W=512; **412,130** @ D=96/W=512 |
| Decision fusion (optional) | `CausalDecisionFusion` | default disabled; width 16 | ~78k–199k depending on v1/v2/v3 + optional heads (code/tests) |
| Matchup adapters V5 | `matchup_adapters.py` | 18 experts, `96→8→96` | **29,520** |
| Matchup adapters V6 | `matchup_adapters_v6.py` | 64 slots, same MLP | **104,960** |
| Experimental RTP attachment (this branch) | `RecursiveTurnPlanner` | profiles `global_transformer` / `pure_rl` | **893,718** (d256/w512); **127,798** (d96/w192) measured via `sum(p.numel())` |
| Expanded strategic heads | `EXPANDED_HEAD_SPECS` in `model.py` | option: q/type/target/resource=1, utility=6; state: tactical 3×6, response 7, forecast 6, phase 5, outcome 3, remaining_turns 1; setup 9; combo 32 | Optional / H10 |

### Representative tensor shapes

| Tensor | Shape | Evidence |
|---|---|---|
| Board spatial memory | `[B, 24, D]` | `NUM_BOARD_TOKENS=24`; `encode_board`; `test_decode_options_shapes.py` |
| State vector | `[B, D]` or `[D]` | `temporal_encode` / `pool_cls` |
| Policy logits | `[B, max_N]` | padded → `-inf` |
| Option hidden | `[B, max_N, D]` | `decode_options(..., return_hidden=True)` |
| Value | `[B]` | `tanh(value_head)` |
| Latent lookahead next state | `[B, N, D]` | `test_marnie_latent_lookahead_r114.py` |
| Latent lookahead value / aid | `[B, N]` | same |

## 6. Training and data

| Responsibility | File/symbol | Current behavior |
|---|---|---|
| Protocol authority (human) | `docs/RL_TRAINING_PROTOCOL.md` | schema `poke_bot.rl_training_protocol/v2` |
| Protocol authority (numeric) | `config/rl_protocol.yaml` | schema `poke_bot.rl_protocol/v2` |
| Mutable specialist state | `state/specialists.yaml` | schema `poke_bot.specialist_state/v1` |
| Supervised trainer | `poke_bot/train.py` `train_bootstrap`; `scripts/train_bootstrap.py` | Bootstrap epochs from protocol |
| RL/self-play trainer | `scripts/train_pure_rl.py` `run_full_loop`; `poke_bot/pure_rl/multi_env_self_play.py` | Pure-RL high-volume path |
| Replay / shards | `poke_bot/pure_rl/shards.py` CompactDecision/Game/ShardWriter; replay cache tests | Device-resident / compact shards |
| Evaluation holdout | `poke_bot/pure_rl/eval_public.py`; `scripts/eval_vs_baselines.py`; holdout args in `train_pure_rl.py` | Formal gates + research holdout exclusion |
| Checkpoint registry | `poke_bot/checkpoint.py` `load_checkpoint` / `immutable_torch_save` / `assert_trusted_policy_checkpoint`; `pure_rl/model_registry.py` `freeze_model` | Digest-bound trust |
| Specialist state updates | `state/specialists.yaml` via validators/dashboard (`scripts/dashboard_snapshot.py`, handoff validators) | Trainers do not silently rewrite protocol YAML |
| Simulator/version provenance | receipts under `state/*` + collection receipts referenced in `rl_protocol.yaml` | Host paths on train box; not all present in this cloud checkout |

### Confirmed protocol numbers (`config/rl_protocol.yaml`)

| Item | Value | Location evidence |
|---|---:|---|
| Bootstrap supervised epochs | **25** | `bootstrap.supervised_epochs` / training-structure `exact_bootstrap_epochs` |
| Baseline-phase games / iter | **8192** | `new_training_games: 8192` |
| Baseline public / self-play | **7168 / 1024** | `public_mix_games` / `active_specialist_self_play_games` |
| High-volume final-submit games / iter | **16384** | `games_per_iteration: 16384` |
| High-volume self-play / public | **2048 / 14336** | `self_play_games_per_iteration` / `public_opponent_games_per_iteration` |
| RL + rehearsal cycle | 5 RL epochs then 5 rehearsal | `rl_epochs_per_cycle` / `expert_rehearsal_epochs_per_cycle` |
| Kit AGENTS claim of “exactly 3,000 isolated eval games” | **Not the current high-volume gate** | Current premium/control gates use **4250** premium + **1000** official control (`rl_protocol.yaml` ~789–794). Treat kit prose as design intent; YAML wins. |

## 7. Current search latency

### Code ceilings (authoritative in-repo)

| Constant | Value | Source |
|---|---:|---|
| `SearchConfig.sims_per_move` | 64 | `poke_bot/config.py` |
| Belief MCTS default `max_sims` / `min_trusted_sims` | 128 | `belief_mcts.py`, `agent.py` |
| Submission `maximum_sims` / `minimum_sims` | 50 / 50 | `submission/search_config.json`, `submission_budget.py` |
| Submission `maximum_calls` (per game process budget accounting) | 340 | same |
| Submission search enabled | **false** | `submission/search_config.json` |
| Complex-option threshold | 8 → sims ×1.5 | `SearchConfig.complex_option_threshold`, `mcts.planned_sims` |
| Move / game time budgets | move 8.0s; game 600.0s | `SearchConfig` |

### ~75 simulator-call claim — **REVISED**

| Finding | Evidence |
|---|---|
| **75 is not a production search constant** in `config.py`, `mcts.py`, `belief_mcts.py`, or `submission_budget.py` | `rg 75` on those files → no budget constant |
| 75 appears in **PokeRLM kit docs/schemas/RTP ablations** as a “legacy measured whole-turn ceiling” | `docs/poke_rlm/*`, `config/poke_rlm_planner.example.yaml`, `VERIFY_ABLATONS["legacy_mcts_75"]` |
| Actual submission hard caps when search enabled | **50 sims/decision**, up to **340 planned calls/game**, search currently **disabled** |
| Belief/local diagnostic search | commonly **128 sims/decision** (not 75/turn) |
| Whole-turn vs per-decision | Belief/oracle MCTS run **per atomic decision** (`PolicyAgent` call). A turn with many micro-decisions multiplies simulators; this is the structural problem PokeRLM targets |

**Conclusion:** Treat “~75 calls consume the whole turn” as an **external/design empirical claim from the kit**, not as a checksum-bound in-repo receipt. Do not implement against 75 as if it were `SearchConfig`. Prefer measuring on the training host; until then use code ceilings above + kit target 0–16 online verifies.

### Measured latency in this environment

| Metric | Simple turn | Median turn | Long/nonlinear turn | Max observed |
|---|---:|---:|---:|---:|
| Strategic decisions | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| Simulator calls | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| Model calls | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| State encodes | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| Total turn ms | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| p95 across sample | — | **BLOCKED** | — | — |
| Peak memory | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** |

**BLOCKED reason:** this cloud checkout has no `cg` runtime and no access to the Blackwell/train host fixtures required for end-to-end turn latency. Component benchmarks exist but were not executed here (would need engine/GPU):

- `scripts/bench_sim_throughput_model.py` (documented walls 730–3290 ms for synthetic 256×40 decision microbench in `docs/sim_gpu_multi_game_throughput.md`)
- `scripts/benchmark_belief_search.py`
- `scripts/benchmark_cuda_leaf.py`
- `scripts/benchmark_remote_model_gps.py`

No dedicated in-repo benchmark emits p50/p95 **whole-turn** `total_turn_ms` today.

## 8. Tests and safety coverage

| Requirement | Existing test path | Gap/action |
|---|---|---|
| Deterministic replay | `tests/test_replay_writer.py`; `tests/test_pure_rl_replay_cache.py` | Keep; add RTP plan-trace replay later |
| Legal option index | `tests/test_mcts_correctness.py` (ordered actions); `tests/test_recursive_turn_planner.py` legality; `tests/test_test_profiles.py` | Need engine-backed RTP legal-index integration test (native) |
| Hidden information | `features.assert_info_set`; `tests/test_public_matchup_router.py`; `tests/test_team_rockets_spidops_current_deck_guide.py` (invariance); belief deployment asserts | Add explicit RTP encode leakage test |
| Search call budget | `tests/test_submission_budget.py` | Does not encode whole-turn 75; document kit vs code |
| Model-call count | RTP `max_neural_passes` tested in `tests/test_recursive_turn_planner.py` | No production telemetry assertion for greedy encode-once |
| Timeout/fallback | `tests/test_submission_budget.py`; `tests/test_runtime_and_metrics.py`; `trusted_search_or_greedy_select` | RTP fallback covered lightly via bridge unit tests |
| Checkpoint compatibility | `tests/test_pure_rl_recovery_and_scheduling.py`; representation/context tests | Unchanged |
| Kaggle package smoke | `tests/test_go_first_contract.py` (submission main); `tests/test_submission_budget.py`; kaggle queue tests | Search still disabled in packaged config |
| Safe unit tests run this audit | `tests/test_recursive_turn_planner.py` + `tests/test_rtp_agent_bridge.py` + latent lookahead → **19 passed**; `test_submission_budget` one failure due missing `data/training_mixes/top_ladder_representatives.v1.json` in this checkout | Data fixture absent in cloud |

## 9. Proposed integration map

| PokeRLM component | Existing module to extend | New file only if necessary | Rationale |
|---|---|---|---|
| Deployment observation type | `features.assert_info_set` + `belief.assert_deployment_observation` + board SparseVector | Optional thin `poke_bot/poke_rlm/observation.py` TypedDict later | Do not fork featurization |
| Structured legal action | `features.factorized_action_candidates` / `enumerate_action_combos` / `Action = list[int]` | Optional dataclass wrapper | Engine index list remains authority |
| Parallel action decoder | `TemporalCabtTransformer.decode_options(..., return_hidden=True)` | — | Already batched; RTP bridge already prefers `option_hidden` |
| Complexity router | `SearchConfig.complex_option_threshold=8`; RTP `should_recurse` / trivial skip | — | Align thresholds; keep direct policy exit |
| Plan IR/compiler | experimental `poke_bot/recursive_turn_planner/types.py` | Promote/replace with kit `schemas/turn_plan.schema.json` types under `poke_bot/poke_rlm/plan_ir.py` | Current AST is skeleton vs kit IR |
| Recursive planner | `poke_bot/recursive_turn_planner/planner.py` + `agent_bridge.py` | Keep evolving; do not add second planner trunk | Already swapped into `PolicyAgent` on this branch behind flag |
| Latent dynamics | `ActionConditionedLatentLookahead` + `LookaheadBackedDynamics` | Expand structured delta heads later | Reuse lookahead; avoid parallel D |
| Validator/ledger | `TypedLegalityVerifier` (membership-only today) | Strengthen with resource ledger using engine select bounds | Exact legality stays outside net |
| Stateful executor/repair | `PlanExecutor` + bridge turn key `(seat,turn)` | — | Already persisted per turn |
| Trace/metrics | `RTPBridgeDiagnostics`; MCTS diagnostics dicts | Add turn-level sim/model/encode counters | Needed before claiming latency wins |

## 10. Audit conclusion

- **Safest first integration point:** keep using `PolicyAgent` non-MCTS path + `RTPAgentBridge` encode-once/`option_hidden` (already on this experiment branch), with **default-off when merging to production** until parity gates pass. Do not enable belief MCTS as primary planner.
- **Existing components reusable unchanged:** `cg_env` CABT wrappers, `features` legal enumeration + info-set asserts, `decode_options` batched scoring, greedy factorized fallback, submission budget/search-disabled packaging, protocol YAML authorities.
- **Components requiring migration adapters:** plan IR (skeleton → kit schema), legality verifier (membership → resource ledger), latent dynamics (scalar rollout → structured short-horizon deltas), checkpoint/inventory schema for RTP modules, whole-turn telemetry.
- **Highest-risk interface:** CABT legal option-index submission + hidden-info boundary. Any planner that invents indices or peeks opponent private fields fails closed.
- **Missing tests that block implementation:** engine-native RTP legal-index + leakage tests; whole-turn latency harness (p50/p95/max, sim/model/encode counts); checkpoint load of RTP-attached weights; production default-off flag verification on `main`/`worksession` (this branch currently defaults RTP **on**).
- **Exact Phase 1 file list (docs/interfaces/tests only — do not implement in this audit):**
  1. `docs/poke_rlm/09_REPOSITORY_MAPPING_TEMPLATE.md` (this file)
  2. `state/poke_rlm_redesign.yaml` (from example; progress record)
  3. `poke_bot/poke_rlm/__init__.py` (namespace; optional alias to recursive_turn_planner)
  4. `poke_bot/poke_rlm/plan_ir.py` (typed IR aligned to `schemas/turn_plan.schema.json`)
  5. `poke_bot/poke_rlm/telemetry.py` (turn counters: encodes, model calls, sim calls, ms)
  6. `tests/test_poke_rlm_plan_ir.py`
  7. `tests/test_poke_rlm_hidden_info.py`
  8. `scripts/bench_poke_rlm_turn_latency.py` (host-only; cg required)
  9. Keep `PolicyAgent.use_recursive_turn_planner` **False** on production branches until gates pass
- **Baseline artifact/checkpoint IDs (from receipts in checkout):**
  - Ordinary Alakazam refresh learned params **1,684,103** — `state/final_format_alakazam_model_inventory_r79.json` (`epoch_05.pt` sha256 `21cd3227…052206`)
  - H10-I validated canary learned params **10,352,606** — same inventory (`core9_hotstart_h10_nonpromotable_v2.pt` sha256 `1d2b6f04…010459`)
  - Slowking candidate total **1,910,963** — `state/slowking_candidate_parameter_inventory_v1.json`
  - Submission search config: enabled **false**, max sims **50**, max calls **340** — `submission/search_config.json`

### Smallest safe Phase 1 plan

1. Freeze this mapping as the integration authority for PokeRLM on the experiment branch.
2. Add telemetry + kit-aligned plan IR **without** changing CABT or training protocol numbers.
3. Keep greedy as deterministic fallback; ensure production defaults remain search-off and RTP-off until measured turn latency + illegal-action rate gates exist on a host with `cg`.
4. Measure the 75-call claim on the training host with `scripts/bench_poke_rlm_turn_latency.py` (to be added) before treating it as a hard budget.
5. Only then expand latent dynamics / teacher distillation (later prompts).
