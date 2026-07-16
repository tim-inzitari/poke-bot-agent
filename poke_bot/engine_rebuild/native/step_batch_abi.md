# `libcg_step_batch` C ABI

Additive shared library. Keeps stock `BattleStart` / `Select` / `GetBattleData` /
`BattleFinish` available (forwarded via `dlopen` of stock `libcg.so`) and adds
batched step helpers.

## Environment

| Var | Meaning |
|---|---|
| `LIBCG_SO` | Absolute path to stock `libcg.so` (default: `libcg.so` on `LD_LIBRARY_PATH`) |

## New exports

### `int StepBatchReady(void)`
`1` if stock symbols resolved.

### `const char* StepBatchLastError(void)`
Empty string or dlopen/dlsym error.

### `int StepBatch(...)`

```c
int StepBatch(
    void** handles,              // [n] ApiData*
    int n,
    const int* action_flat,      // concatenated option indices
    const int* action_offsets,   // [n] start into action_flat
    const int* action_lens,      // [n] len; 0 => skip Select
    int fetch_obs_on_skip,       // non-zero => GetBattleData on skips
    int copy_json,               // 0 = borrowed ptrs (stock lifetime); 1 = malloc copy
    int* out_errors,             // [n]
    char** out_jsons,            // [n] UTF-8 JSON
    int* out_select_players      // [n] or NULL
);
```

When `copy_json==0`, pointers are borrowed from stock `GetBattleData` (valid until
the next Select/GetBattleData/StepBatch on that handle). When `copy_json==1`,
free with `StepBatchFreeJsons`.

Per-env `out_errors[i]`:

| Code | Meaning |
|---|---|
| `0` | Select ok (or skip+fetch path with no Select) |
| stock `Select` code | Non-zero from official Select |
| `-1` | Skipped (null handle or zero-length action without error) |
| `-2` | Bad args / stock not loaded |
| `-3` | `malloc` failed copying JSON |

Call return: `0` ABI ok; `1` null buffers / bad `n`; `2` stock lib missing.

### `void StepBatchFreeJsons(char** jsons, int n)`
Frees each `out_jsons[i]` and nulls the slots.

## Not in this shim

`BattleStartSeeded(int* cards, uint64_t seed)` — requires in-tree fork of
`Export.cpp` / `ApiBattleStart` (stock uses `std::random_device`). See
`docs/engine_step_batch.md`.

## Python

`poke_bot.engine_rebuild.libcg_step_batch` loads this `.so` when present and
exposes `step_batch_native(...)`. `LibcgMultiEnv` prefers it automatically.
