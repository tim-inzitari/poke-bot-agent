# Engine rebuild / fork: multi-game + GPU-friendly throughput

Branch: `cursor/sim-gpu-multi-game-693f`  
Companion: [sim GPU / multi-game throughput](sim_gpu_multi_game_throughput.md)  
Authority: [Kaggle discussion 717141](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/717141) (`ptcg_engine.zip`)

**Status:** first-class engineering track — not deferred research fluff.  
Leaves + worker farms still matter for SPS *today*. This doc is the parallel path: **rebuild/fork the competition engine** so many games (and later selected hot loops) stop being process-isolated CPU singletons.

---

## 0. Source inventory (this workspace)

| Artifact | Expected path | Present here? |
|---|---|---|
| `ptcg_engine.zip` / C++ tree | `kaggle/input/pokemon-tcg-ai-battle/ptcg_engine/ptcgProgram 22/` (`paths.PTCG_ENGINE_DIR`) | **No** — competition bundle not downloaded |
| Python `cg/` bindings | `…/sample_submission/sample_submission/cg/{api,game,sim,utils}.py` + `libcg.so` | **No** (no Kaggle creds in this cloud env) |
| Public API docs | [matsuoinstitute.github.io/cabt](https://matsuoinstitute.github.io/cabt/) | **Yes** — used below |
| In-repo wrappers | `poke_bot/cg_env.py`, `worker_pool.py`, `mcts.py` | **Yes** |

Obtain source on a machine with Kaggle auth:

```bash
bash scripts/setup_competition_data.sh   # SKIP_EPISODES=1 OK
# then: unzip / inspect ptcg_engine; keep LICENSE + README with any fork
```

Until the zip is local, design below is grounded in **published Game/Search/Sim APIs**, our wrappers, and discussion 717141. Assumptions are marked **`[ASSUME]`**.

---

## 1. Why one-battle-per-process exists (code-level)

### Hard evidence from the public / shipped API surface

1. **Game API is a singleton session** — `cg.game.battle_start` / `battle_select` / `battle_finish` / `visualize_data` take **no battle handle**. Docs: “finish and clean up the **current** battle.” That is a process-global current-battle, not `Battle* env_id`.
2. **ctypes core is module-global** — `cg.sim` documents a process-wide `lib` plus unbound instance APIs: `GameInitialize`, `BattleStart`, `AgentStart`, `BattleFinish`, `GetBattleData`, `Select`, `VisualizeData`, `SearchBegin`/`SearchStep`/`SearchEnd`/`SearchRelease`, `AllCard`, `AllAttack`. No `env_id` in those names.
3. **Search is multi-tree, still session-scoped** — `search_begin` → `SearchState.searchId`; `search_step(search_id, …)`. Multiple trees OK *inside one search session*; `search_end()` tears the session down. `MultiTreeMCTS` already batches **leaves**, not battle steps.
4. **Our pool treats libcg as non-shareable** — `WorkerPool` spawn + recycle (`worker_recycle_games`) because “libcg is sequential CPU and holds per-process state that grows slowly.” That is operational confirmation of leaks / sticky native heaps, not just API style.
5. **Hardware knobs encode the constraint** — `HardwareConfig.sim_workers` = one battle per process; `parallel_games` only sizes multi-tree / reanalyse *around* that singleton.

### Likely C++ causes (`[ASSUME]` until zip audited)

| Suspect | Why it matches the API | Rebuild action |
|---|---|---|
| Static / file-scope `Battle` or `g_battle` | Explains no handle in Game API | `Battle` as value/arena object; API takes `Battle*` / `env_id` |
| Process-global RNG / coin stream | Coin flips + shuffle must be deterministic per seed | Per-env `Rng` (PCG/xoshiro); optional `manual_coin` already exists for search |
| Card / effect tables as writable globals | Card metadata load once; procs mutate shared caches | Read-only shared card DB; mutable state only on env |
| Non-reentrant effect / select stacks | Nested selects, looking zones | Stacks on `Battle`; no static scratch |
| Sticky allocators / unreclaimed search nodes | Explains recycle-after-N-games | Arena + explicit reset; pool recycle becomes optional |
| Bugs already reported in 717141 | `ToolCountProc` shadow; `Export.cpp` off-by-one | Fix in fork; parity suite catches regressions |

**Conclusion:** one-battle-per-process is not a training-policy choice — it is the **natural consequence of a non-reentrant C API**. Throughput scale-out today = **many OS processes**. The rebuild removes that bottleneck so **N envs share one process** (CPU first), then optionally GPU-accelerates *narrow* kernels.

---

## 2. Target architecture

```
                    ┌─────────────────────────────────────────┐
                    │  Shared immutable card / attack tables  │
                    └───────────────────┬─────────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
   Env[0] Battle                  Env[1] Battle                 Env[N-1]
   + Rng + select stack           + Rng + …                     + …
         │                              │                              │
         └──────────────┬───────────────┴──────────────┬───────────────┘
                        ▼                              ▼
              MultiEnv.step_batch(actions[N])     optional GPU kernels
              (CPU SoA / arena)                   (RNG batch, legality
                                                   bitsets, damage math)
                        │
                        ▼
              obs tensors / SparseVectors → existing leaf servers
```

### Layers

| Layer | Role | GPU? |
|---|---|---|
| **A. Instance Battle** | Reentrant `Battle` + per-env RNG; Python/C ABI with `env_id` | No |
| **B. MultiEnv / arena** | Contiguous N states, `reset`/`step`/`step_batch`, auto-reset dones | No (big win) |
| **C. SoA layout** | Pack hot fields (HP, energies, flags) for SIMD / cache | CPU SIMD first |
| **D. GPU kernels** | Only data-parallel kernels with no branching rules | **Partial** |
| **E. Leaf path** | Unchanged coalesce leaf servers | Already GPU |

### What is honest to put on GPU vs keep on CPU

| Keep on CPU (rules / control flow) | Candidate GPU / SIMD later |
|---|---|
| Card effect dispatch, targeting, nested selects | Batch coin / RNG draws |
| Zone mutations with legality side effects | Damage / weakness / resistance arithmetic |
| Search tree bookkeeping (`searchId` graphs) | Featurize SparseVectors for many envs |
| Full CABT ruling branches | Bitmask “is energy attach legal?” style checks **after** CPU proposes candidates |

A full “PTCG on CUDA/JAX” rewrite is **not** the near-term plan: parity risk kills training if transitions diverge. Prefer **fork → multi-env CPU → profile → GPU only hot loops**.

---

## 3. Concrete code-level change list (once zip is present)

### 3.1 Inventory pass (day 0–1)

1. List every `static` / anonymous-namespace global in `ptcgProgram 22`.
2. Map `BattleStart` / `Select` / `SearchBegin` to owning objects.
3. Find RNG + shuffle entry points; note seed control.
4. Confirm leak sites (search node pools, serialize buffers).
5. Record official bugs from 717141 and fix in fork.

### 3.2 Multi-instance / arena / SoA (CPU)

1. Introduce `struct BattleEnv { Battle battle; Rng rng; … }`.
2. Replace Game API with:
   - `env_create` / `env_reset(deck0, deck1, seed)`
   - `env_step(env_id, select_list) -> Obs`
   - `env_destroy` / `env_reset_inplace`
3. Keep legacy singleton wrappers calling `env_id=0` for drop-in `cg.game` compatibility during migration.
4. Arena: bump allocator per env or shared pool with generation tags; `search_release` returns nodes to freelist.
5. SoA optional phase: `Hp[N]`, `EnergyMask[N][…]` for the hottest loops identified by `perf`.

### 3.3 Batched `step` across N envs

```text
step_batch(actions: list[list[int] | None]) -> BatchObs
  for i in active:
      if actions[i] is None: continue  # waiting on policy
      obs[i] = env_step(i, actions[i])
      if done[i]: auto_reset(i) optional
```

Python spike: `poke_bot.engine_rebuild.interfaces.MultiEnv` (this PR).  
C++ later: single entry that loops in C without N× ctypes round-trips (critical — ctypes per step will dominate).

### 3.4 Optional GPU port

1. Profile pure-RL collect after MultiEnv CPU: if **policy/IPC** still dominates, stop (leaves already GPU).
2. If **sim_step** dominates, extract 1–2 kernels (RNG, damage) behind a CPU fallback.
3. Never block Stage A overnight on GPU rules.

---

## 4. Milestone ranking (effort / risk / gain)

| # | Milestone | Effort | Risk | Gain | Verdict |
|---|---|---|---|---|---|
| **M0** | Download zip; static/global inventory; golden capture harness vs `libcg` | S | Low | Unlocks everything | **Do first on training box** |
| **M1** | Instance `Battle*` + per-env RNG; legacy singleton shim | M | Med (subtle globals) | Many battles / process; cut process count | **Primary rebuild goal** |
| **M2** | Arena + leak-free search; drop aggressive recycle | M | Med | Stable long workers; better leaf occupancy | With M1 |
| **M3** | `step_batch` in C++ (no per-step ctypes) + Python `MultiEnv` | M | Med | Pure-RL SPS when sim-bound | **Next coding milestone after M0 inventory** |
| **M4** | SoA / SIMD on hottest fields | M–L | Med | Extra CPU throughput | After profiler says yes |
| **M5** | GPU kernels for RNG/damage/featurize only | L | High (parity) | Large only if sim-bound | Conditional |
| **M6** | Full vectorized rules on GPU/JAX | XL | Very high | Speculative TCGJax-class | Research only |

**Parity strategy (non-negotiable)**

1. **Golden games** — fixed decks + fixed seeds (or `manual_coin` paths); compare winner, length, and every `(select_type, option_ids, chosen)` transition.
2. **Transition hash** — `sha256` over canonical JSON of `(obs_select_fingerprint, action, next_select_fingerprint, logs_delta)` vs official `libcg`.
3. **Card suite** — one game per major mechanic family (tool attach, stadium, energy acceleration, confusion coin, prize take).
4. **Fail closed** — fork never trains overnight until golden suite green; ship digest beside `competition-libcg-sha256:…`.

Spike helpers: `poke_bot/engine_rebuild/parity.py`.

---

## 5. How this coexists with leaf servers

| Path | Role |
|---|---|
| Official `libcg` + `WorkerPool` | Production collect **now** |
| Coalesced GPU leaves | Policy throughput **now** (spike already wired) |
| Engine fork MultiEnv | Reduce process tax + sim latency **next** |
| Remote whole-game farms | Additive wall-clock games/hour without fork |

Rebuild does **not** replace leaves. It removes the “32–40 processes × one battle” ceiling so leaf batches stay full with fewer IPC speakers.

---

## 6. Recommended sequence (serious, not fluff)

1. **On box with data:** M0 inventory + golden harness against `libcg`.
2. **M1+M3 spike in C++:** multi-env CPU with `step_batch`; keep `cg.game` shim.
3. **Wire pure-RL collect** to `MultiEnv` behind a flag; A/B SPS vs process pool.
4. **Only then** consider M4/M5 from profiles.
5. Keep leaf coalesce tuning (`LEAF_SERVER_COALESCE_MS`) independent.

**Next concrete coding milestone (after source present):** implement M1 handle API + Python `MultiEnv` binding that passes the golden transition-hash suite for ≥10 fixed seeds.

Until then, this repo ships the **Python interface + FakeMultiEnv + parity harness** under `poke_bot/engine_rebuild/` so wiring and tests can proceed without `libcg`.

---

## 7. Out of scope

- Committing `ptcg_engine` source, `libcg.so`, or Kaggle secrets into git.
- Claiming official GPU `libcg`.
- Blocking overnight Stage A on the fork.
