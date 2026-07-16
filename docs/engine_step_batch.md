# Engine `step_batch` (M3 spike)

Branch track: `cursor/engine-step-batch-benchmark-1dae`  
Companion: [engine_rebuild_multi_game.md](engine_rebuild_multi_game.md)

## What shipped

| Artifact | Path | In git? |
|---|---|---|
| C++ StepBatch shim (dlopen stock `libcg.so`) | `poke_bot/engine_rebuild/native/step_batch.cpp` | Yes |
| ABI notes | `poke_bot/engine_rebuild/native/step_batch_abi.md` | Yes |
| Build script | `poke_bot/engine_rebuild/native/build_step_batch.sh` | Yes |
| Python loader | `poke_bot/engine_rebuild/libcg_step_batch.py` | Yes |
| MultiEnv prefers native | `poke_bot/engine_rebuild/libcg_multi_env.py` | Yes |
| Microbench | `scripts/bench_step_batch.py` | Yes |
| CG mirror fetch (no Kaggle) | `scripts/fetch_cg_runtime_mirror.sh` | Yes |
| Built `.so` | `poke_bot/engine_rebuild/native/build/libcg_step_batch.so` | **No** (`*.so` gitignored) |
| Stock `libcg.so` / `ptcg_engine` sources | `kaggle/input/...` | **No** (gitignored) |

## Build status (cloud VM)

1. **Full in-tree `Export.cpp` Linux rebuild:** **BLOCKED** — `ptcg_engine/ptcgProgram 22/` absent; no `~/.kaggle/kaggle.json` / `KAGGLE_API_TOKEN` on this VM. Competition C++ tree is not mirrored publicly.
2. **Additive `libcg_step_batch.so`:** **BUILDS** with `g++ -std=c++20 -O3 -fPIC -shared` (no engine sources). Forwards stock symbols via `dlopen`.
3. **`BattleStartSeeded`:** **NOT in shim** — stock `BattleStart` links `std::random_device`; seeded start needs an in-tree `Export.cpp` patch once sources are local.

## Cloud inventory

| Need | Present? |
|---|---|
| Stock `libcg.so` + `cg/*.py` | Yes (via `scripts/fetch_cg_runtime_mirror.sh`) |
| `ptcg_engine` headers / `Export.cpp` | **No** — run `scripts/setup_competition_data.sh` with Kaggle auth |
| `g++` C++20 | Yes |

## How to build + bench (cloud or host)

```bash
# 1) stock runtime (mirror OR Kaggle)
bash scripts/fetch_cg_runtime_mirror.sh
# preferred when creds exist:
# bash scripts/setup_competition_data.sh

# 2) shim
bash poke_bot/engine_rebuild/native/build_step_batch.sh

# 3) microbench N=8,32,64
python scripts/bench_step_batch.py --ns 8,32,64 --steps 40 --repeats 5 --games 16
# results → outputs/bench_step_batch.json
```

Force Python loop: `POKEBOT_STEP_BATCH=0`.

## Copy onto host / Elmo / bert (promotion boundary)

Do **not** restart overnight trainers mid-collection. At **promotion** (or iter boundary):

```bash
# on build machine
bash poke_bot/engine_rebuild/native/build_step_batch.sh
SO=poke_bot/engine_rebuild/native/build/libcg_step_batch.so

# host training box — stage next to stock cg
scp "$SO" HOST:/path/to/cg/libcg_step_batch.so
# ensure LIBCG_SO points at stock libcg.so (shim dlopens it)
# export LIBCG_STEP_BATCH_SO=/path/to/cg/libcg_step_batch.so  # optional explicit

# Elmo / bert workers — same file beside their cg/libcg.so, then recycle
# workers at promotion only (redeploy_throughput_next_iter / pool recreate).
```

`LibcgMultiEnv` auto-loads the shim when the `.so` is discoverable; no trainer flag required (`POKEBOT_STEP_BATCH=1` default).

## Remaining blockers

- **In-tree fork / `BattleStartSeeded` / SoA:** needs `ptcg_engine` download (Kaggle).
- **GPU full-game steps:** still blocked until multi-env CPU fork lands and profiles show sim-bound work; this shim only removes N× ctypes crossings (JSON still produced by stock `GetBattleData`).
- **Honest GPU sim:** not this deliverable.

## License

Competition-use-only. Keep competition LICENSE with any redistributed binary. Do not commit full engine sources or secrets.
