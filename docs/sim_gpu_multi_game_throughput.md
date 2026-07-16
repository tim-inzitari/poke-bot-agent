# Simulator GPU / multi-game throughput exploration

Branch: `cursor/sim-gpu-multi-game-693f` (additive spike on pure-RL tip).  
Reference: [Kaggle discussion 717141](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/717141)  
Status: explore + spike only — **does not** change overnight launch scripts.

## 1. Discussion 717141 (reconstructed)

The live Kaggle page often crashes (CSS chunk load failure). Content was recovered via `r.jina.ai` markdown mirror (2026-07-16).

### What it actually is

**Title:** “Game Engine Source Code” — Addison Howard (Kaggle Staff), ~15 days before this note, 108 upvotes.

**Core announcement**

- Official engine source published on the competition Data page as `ptcg_engine.zip`.
- Intended use: local testing, verification, and **training**.
- Competition-specific engine (may diverge from retail PTCG rulings).
- Do not exploit bugs; report vulnerabilities in forums.
- License / README must stay with the source.
- Some comments are in Japanese.

**Host clarification (submission legality)**

- Code **derived / adapted / compiled** from `ptcg_engine.zip` **may** be included in `submission.tar.gz` for this competition.
- Still bound by Pokémon Elements / commercial / winner-license rules.

**Notable thread comments (not GPU recipes)**

| Author | Claim |
|---|---|
| SpeedSci | Source release is “very important for using RL” |
| pao | Asked whether compiled/adapted engine may ship in submissions → **yes** (Addison) |
| KawattaTaido | Bug report: `ToolCountProc` loop variable shadowing → wrong player / crash |
| Prema Ananda | Visualizer replay `selected` off-by-one in `Export.cpp` / `ApiSelect` |

### What the discussion does *not* say

It does **not** prescribe VectorVisor, WASM-on-GPU, CUDA graphs, or a batched multi-env API. Those are **downstream inferences**: releasing C++ source is the prerequisite for anyone to fork the engine into a multi-instance / GPU-oriented sim. Community GPU-batch ideas (e.g. VectorVisor-style many WASM VMs, or research engines like TCGJax / PokeJAX) are adjacent literature, not claims inside 717141.

---

## 2. Current architecture (this branch)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Host collect (pure-RL / round-robin)                                 │
│                                                                      │
│  WorkerPool (spawn, N≈32–40)     POKEBOT_WORKER_CPU_ONLY=1           │
│    └─ per worker: import cg / libcg.so  →  ONE battle at a time      │
│         battle_start / battle_select / search_*   【CPU-only】         │
│         featurize SparseVectors                   【CPU】             │
│         ──IPC──► leaf req queue                                       │
│                                                                      │
│  Leaf servers (dual GPU)                                             │
│    GPU1 Blackwell: ~6 replicas   GPU0 3080 Ti: ~3 replicas           │
│    coalesce_ms (default 4) + max_batch (default 1024)                │
│    bf16 TemporalCabtTransformer.forward  【GPU】                      │
│                                                                      │
│  Train process: AWR on CUDA:1 (Blackwell)                            │
│  Remotes (Elmo/bert): additive whole-game farms                      │
└──────────────────────────────────────────────────────────────────────┘
```

### CPU-bound today

| Piece | Notes |
|---|---|
| `libcg.so` / `cg.game` | **Sequential, one battle per process.** Documented in `worker_pool.py` + `config.HardwareConfig`. No GPU path in the shipped binary. |
| `search_begin` / `search_step` | Per-tree CPU (MCTS path). Pure-RL overnight uses `mcts_sims=0` → skipped. |
| Worker recycle | Bounds libcg leak (`worker_recycle_games`); disabled while leaf channel held (pool rebuilt per iteration). |
| Featurization | CPU SparseVectors before IPC. |

### GPU-bound today

| Piece | Notes |
|---|---|
| Leaf servers (`batched_infer.run_leaf_server`) | **Already** batch/coalesce network evals across many games/workers. |
| AWR train step | Blackwell; overlaps with next collect shard. |
| Official `libcg` | **Cannot** run on GPU as shipped. Only neural leaves (and train) use CUDA. |

### Pure-RL collect gap (found on this spike)

`scripts/train_pure_rl.py` **starts** dual-GPU leaf servers and passes `remote_channel` into `WorkerPool`, but `remote_self_play_job` historically **ignored** the channel and loaded the ~1.6M policy on **CPU in every worker**. Round-robin `_worker_play` already used `remote_leaf_backend_from_worker()`.

So for Stage A self-play (the primary SPS path), GPU leaves were largely idle while sim workers burned CPU on local forwards.

**Spike fix (additive):** `poke_bot/pure_rl/leaf_self_play.py` + wire-up in `remote_sim_jobs.remote_self_play_job`:

- same champion ckpt both seats → both use coalesced GPU leaf (`gpu-leaf-both`)
- recent-self pool (different opp ckpt) → our seat on leaf, opp CPU-local (`gpu-leaf-us-only`)
- no leaf channel → previous CPU-local behavior

Overnight launcher scripts are untouched.

---

## 3. Design options (gain vs invasiveness)

Ranked for **expected collect throughput** on the dedicated dual-GPU box, honesty first.

| Rank | Option | Expected gain | Invasiveness | Verdict |
|---|---|---|---|---|
| **1** | **Wire pure-RL self-play → existing coalesced leaf servers** (this spike) | **High for policy SPS** (model was on CPU×N workers). Leaves already batched. | **Low** — reuse RR path; no overnight script edits. | **Do now** |
| **2** | Status quo+: more workers / leaf replicas / coalesce tuning + remote farms | Medium — already mostly dialed (`full_hardware_profile`, resource_watcher). | Low | Ongoing ops |
| **3** | In-process multi-game async + larger coalesce occupancy | Low–medium if leaf util still low after (1). | Low–medium | Measure first |
| **4** | CUDA graphs / compiled leaf forward | Small–medium once batches stable & large. | Medium (leaf server only) | After telemetry |
| **5** | Fork `ptcg_engine` → multi-env CPU (SoA / arena allocator, many states per process) | Medium–high on sim-step bound workloads; helps MCTS search trees more than pure-RL policy rollouts. | **High** (maintain fork, parity tests) | Post-core |
| **6** | Vectorized / GPU game steps (JAX/CUDA rewrite of rules) | Potentially huge SPS (TCGJax-class claims elsewhere) but **parity + legality risk**; pure-RL still needs legal CABT semantics. | **Very high** | Research only |
| **7** | WASM + VectorVisor-style many VMs on GPU | Throughput-oriented for compute-bound interpreters; high latency; engine is native C++ not WASM today. | Very high | Unlikely fit |

### Honest answer: can official `libcg` run on GPU?

**No.** The competition binary is a native CPU library with process-global battle state. GPU is for **neural net leaves + training** only, unless we compile a **custom** engine from `ptcg_engine.zip` (allowed for local train / possibly submission per 717141) that deliberately targets multi-instance or GPU — that is a new product, not a flag on `libcg.so`.

---

## 4. Spike artifacts

| Path | Role |
|---|---|
| `docs/sim_gpu_multi_game_throughput.md` | This design note |
| `poke_bot/pure_rl/leaf_self_play.py` | Leaf wiring plan (pure logic) |
| `poke_bot/remote_sim_jobs.py` | Self-play uses leaf backend when channel active |
| `scripts/bench_sim_throughput_model.py` | Synthetic coalesce vs CPU-local microbench (no CUDA/libcg) |
| `tests/test_leaf_self_play_wiring.py` | Plan unit tests |

### Microbench (wave-based synthetic model)

Run: `python3 scripts/bench_sim_throughput_model.py`

**Evidence from this cloud agent host (no CUDA/libcg; analytical wave model,
32 workers, 256 games × 40 decisions):**

| Setup | Wall | Speedup vs CPU-local |
|---|---|---|
| cpu-local (1.6M proxy) | 1472 ms | 1.00× |
| gpu-leaf both, `coalesce_ms=4` | 3290 ms | **0.45×** (loses) |
| gpu-leaf both, `coalesce_ms=0` | 730 ms | **2.02×** |
| Hope-large proxy, `coalesce_ms=4` | — | **2.19×** |
| Hope-large proxy, `coalesce_ms=0` | — | **7.31×** |

**Implication:** wiring self-play to leaves is the right *interface* (one resident
GPU net, no N× CUDA contexts). For the tiny pure-RL policy, also set
`LEAF_SERVER_COALESCE_MS=0` (or ~1) when enabling the path; default 4 ms is
tuned for larger nets. Confirm with real leaf telemetry (`inference_ms`,
`batch_occupancy`).

---

## 5. Recommended redesign (practical)

1. **Ship / validate spike (1)** on a short pure-RL canary: confirm `leaf_self_play_mode=gpu-leaf-both` in job results and rising leaf `batch_occupancy`.
2. **Keep** many CPU `libcg` workers + coalesced dual-GPU leaves as the production backbone.
3. **Do not** block Stage A overnight on a GPU rules engine; revisit `ptcg_engine` multi-env only if after (1)+(2) the profiler shows **sim_step** (not policy) as the dominant bound.
4. Prefer additive remote whole-game farms for more wall-clock games/hour without forking the engine.

---

## 6. Out of scope / non-goals

- Editing overnight trainers / unattended monitors.
- Committing competition engine source or secrets.
- Claiming official GPU `libcg` support.
