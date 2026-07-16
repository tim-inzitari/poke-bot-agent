# Engine rebuild / fork: multi-game + GPU-friendly throughput

Branch: `cursor/sim-gpu-multi-game-693f`  
Companion: [sim GPU / multi-game throughput](sim_gpu_multi_game_throughput.md)  
Authority: [Kaggle discussion 717141](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/717141) (`ptcg_engine.zip`)

**Status:** first-class engineering track — not deferred research fluff.  
Leaves + worker farms still matter for SPS *today*. This doc is the parallel path: **rebuild/fork the competition engine** so many games (and later selected hot loops) stop being process-isolated CPU singletons.

---

## 0. Source inventory (this workspace) — **DOWNLOADED**

| Artifact | Path | Present? |
|---|---|---|
| C++ engine headers / `Export.cpp` | `kaggle/input/pokemon-tcg-ai-battle/ptcg_engine/ptcgProgram 22/` | **Yes** (45 files; gitignored) |
| Python `cg/` + `libcg.so` | `…/sample_submission/sample_submission/cg/` | **Yes** |
| Public API docs | [matsuoinstitute.github.io/cabt](https://matsuoinstitute.github.io/cabt/) | Yes |
| In-repo wrappers | `poke_bot/cg_env.py`, `worker_pool.py` | Yes |

Credentials used only via `~/.kaggle/kaggle.json` (chmod 600, not in git). Re-download:

```bash
# with ~/.kaggle/kaggle.json present
bash scripts/setup_competition_data.sh   # or the focused download used in this spike
```

---

## 1. M0 finding: singleton is **Python**, not native

### Hard evidence from the downloaded source + live `libcg.so`

1. **C ABI is handle-based** — `Export.cpp` / `Api.h`:
   - `BattleStart(int* cards) -> StartData { ApiData* battlePtr; … }`
   - `Select(ApiData* data, …)`, `GetBattleData(ApiData*)`, `BattleFinish(ApiData*)`
   - `ApiData` owns `Game game`, `State state`, `Search search`, buffers (`ApiData.h`).
2. **Python wrapper is the singleton** — `cg/game.py` stores `Battle.battle_ptr` on a class and `battle_select` always uses that one pointer. Docs that say “current battle” describe this helper, not the DLL.
3. **Live proof (this cloud host):** started **32 concurrent** `BattleStart` handles, stepped each with `Select`, finished all — `MULTI_HANDLE_OK` / `32_HANDLE_OK`.
4. **Shared process globals that remain** (OK for multi-battle if treated read-mostly after init):
   - `inline std::unordered_map<…> CardTable / SkillTable / AttackTable / NameTable` (`Card.h`)
   - `InitializeAll()` asserts empty then fills once (`All.h` / `GameInitialize`)
   - `static JsonBuilder AllCardJson / AllAttackJson` in `Export.cpp` (AllCard/AllAttack only)
5. **Per-env mutable state is already on the heap object** — `Game` has its own `std::mt19937 rng`; `BattleData` has `Game` + `State` (`Game.h`, `BattleData.h`).
6. **Our WorkerPool still uses one battle per process** because collect talks through `cg.game` + historical leak/recycle caution — that is now a **software choice**, not a native hard limit.

### What rebuild still buys (honest)

| Keep / do now | Still worth a fork |
|---|---|
| **`LibcgMultiEnv`** — many `ApiData*` in one process via ctypes (shipped) | C++ `step_batch` to avoid N× ctypes/JSON per ply |
| Wire pure-RL collect to MultiEnv + GPU leaves | SoA / SIMD on hot `State` fields after profiles |
| | Optional GPU kernels (RNG/damage/featurize) |
| | Fix 717141 bugs (`ToolCountProc`, `Export` off-by-one) in fork |

**Revised conclusion:** one-battle-per-process was largely a **Python API / ops** constraint. Near-term throughput: multi-handle `libcg` in-process. Medium-term rebuild: batch step + layout + selective GPU — not “invent battle handles from scratch.”

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

## 3. Concrete code-level change list (post-M0)

### 3.1 Inventory — **done**

1. Statics/globals listed: `CardTable*` family, `AllCardJson`/`AllAttackJson`, no process-global `Battle`.
2. `BattleStart`/`Select`/`SearchBegin` own state via `ApiData*`.
3. RNG is `Game.rng` (`std::mt19937`) per `ApiData`; public `BattleStart` still reseeds from `random_device` (seed control gap for parity — fork or new API entry).
4. Recycle/leak story still operationally relevant for long workers; measure with MultiEnv soak.
5. 717141 bugs remain fork/fix candidates.

### 3.2 Near-term (no fork): Python MultiEnv on official handles

1. **`LibcgMultiEnv`** (`poke_bot/engine_rebuild/libcg_multi_env.py`) — done spike.
2. Wire pure-RL / WorkerPool path behind a flag to use N handles per process (or fewer fatter processes).
3. Soak test for native leaks vs process recycle.

### 3.3 Fork: batched `step` + seed control + SoA

```text
step_batch(actions: list[list[int] | None]) -> BatchObs
  // C++ loops Select+serialize once; one ctypes call
```

Also expose `BattleStartSeeded(cards, seed)` for golden parity.  
SoA optional after `perf` on MultiEnv collect.

### 3.4 Optional GPU port

1. Profile after in-process MultiEnv + GPU leaves.
2. If **sim_step** dominates, extract 1–2 kernels (RNG, damage) with CPU fallback.
3. Never block Stage A overnight on GPU rules.

---

## 4. Milestone ranking (effort / risk / gain)

| # | Milestone | Effort | Risk | Gain | Verdict |
|---|---|---|---|---|---|
| **M0** | Download source; prove multi-handle; inventory globals | S | Low | Reframes the whole project | **Done** |
| **M1** | Ship/validate `LibcgMultiEnv` in pure-RL canary (fewer processes) | S–M | Low–Med (leaks) | Cut process tax immediately | **Do next** |
| **M2** | Soak + recycle policy for multi-handle workers | S | Med | Stable overnight | With M1 |
| **M3** | C++ `step_batch` + seeded start in engine fork | M | Med | Kill ctypes/JSON overhead | Primary rebuild goal |
| **M4** | SoA / SIMD on hottest fields | M–L | Med | Extra CPU throughput | After profiler |
| **M5** | GPU kernels for RNG/damage/featurize only | L | High | Conditional | After sim-bound proof |
| **M6** | Full vectorized rules on GPU/JAX | XL | Very high | Speculative | Research only |

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

1. ~~Download source + inventory~~ **Done** — multi-handle proven.
2. **M1 canary:** pure-RL collect via `LibcgMultiEnv` (or hybrid: few processes × many handles) + GPU leaves; A/B SPS vs today’s WorkerPool.
3. **M2 soak** for leaks; adjust recycle.
4. **M3 fork:** C++ `step_batch` + seeded `BattleStart`; golden transition hashes.
5. **M4/M5** only if sim_step still dominates profiles.

**Next coding milestone:** wire `LibcgMultiEnv` into pure-RL self-play behind a flag and bench games/hour vs process pool on the training box.

---

## 7. Out of scope

- Committing `ptcg_engine` source, `libcg.so`, or Kaggle secrets into git.
- Claiming official GPU `libcg`.
- Blocking overnight Stage A on the fork.
