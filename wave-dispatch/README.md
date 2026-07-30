# wave_dispatch

Standalone **C++** LAN job-dispatch library for multi-machine RL collect waves.

Use it from any Kaggle / training project: fan opaque episode jobs across local
workers + remote TCP workers, rebalance on wall-clock completions, gather
results. Domain code (env, model, job schema) stays in your repo.

**This tree is independent of poke-bot-agent production.** It does not import,
link, or modify that trainer.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Artifacts:

| Binary / lib | Role |
|---|---|
| `libwave_dispatch.a` | Core library |
| `wave_echo_worker` | Synthetic remote worker |
| `wave_echo_client` | Local+remote scheduled wave demo |
| `wave_dispatch_tests` | Unit / round-trip tests |

## Quick demo

```bash
./build/wave_echo_worker --port 8765 --workers 4 &
./build/wave_echo_client --endpoint 127.0.0.1:8765 --jobs 64 --local 2
```

## API (C++)

```cpp
#include "wave_dispatch/wave_dispatch.hpp"

using namespace wave_dispatch;

// Worker process
serve_forever(handler, ServerConfig{.port = 8765}, hello_fn);

// Trainer
JobClient client("gpu-box", 8765);
client.connect();
MidWaveScheduler sched(cfg);
run_scheduled_wave(jobs, local_fn, remotes, sched, collect_cfg, on_result);
```

Jobs/results are opaque `nlohmann::json` objects. Transport is length-prefixed
JSON (`!I` + UTF-8), protocol version 1 — same family as the poke-bot remote
worker wire format, without Pokemon types.

## Layout

```text
include/wave_dispatch/   public headers
src/                     frame, client, server, scheduler, collector
apps/                    echo worker + client
tests/                   frame / scheduler / TCP round-trip
docs/PROTOCOL.md         wire format
```

## Config knobs

`SchedulerConfig` / `CollectConfig` — typed C++ structs. No `PURE_RL_*` or
poke-bot env coupling.

## Status

v0.1.0 — C++ core + echo apps + tests. Python bindings optional later.
