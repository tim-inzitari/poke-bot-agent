# rl-libs

Standalone generalized RL infrastructure extracted for reuse across Kaggle / LAN
training stacks. **Opaque payloads only** — no Pokémon-specific schema, no
production wiring into poke-bot selectors or trainers.

Version **0.1.1** (post code-review correctness fixes). See `docs/CODE_REVIEW.md`
and `PUBLISH.md` for the `lib/rl-libs` standalone branch.

## Packages

| Package | Lang | Role |
|---|---|---|
| `rl_io` | C++17 + pybind | Crash-safe ordered writer, SHA-256, mmap blob packs |
| `rl_runtime` | C++17 + pybind | POSIX SHM request/response rings + batch coalesce |
| `proc_pool` | C++17 + pybind | Recyclable process supervisor (length-framed stdio) |
| `rl_eval` | Python | Wilson / bootstrap metrics, promotion gates, aborts |
| `torch_ckpt` | Python | Atomic / immutable `torch.save` helpers + digests |
| `artifact_registry` | Python | Digest-bound registry + receipt-backed retention |

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/rl_io_bench
./build/rl_shm_bench
```

Python:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Design rules

- Callers own competition schemas (JSON manifests, record bytes, leaf tensors).
- Hot paths are C++; Torch / gates / policy stay Python.
- Do **not** point live production trainers at these packages until explicitly ordered.

## Sibling

See also `wave-dispatch` for LAN mid-wave job dispatch.
