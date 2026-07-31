# wave_dispatch

Standalone **C++** LAN job-dispatch library (with **Python bindings**) for
multi-machine RL collect waves — tuned for max dispatch speed.

Use it from any Kaggle / training project: fan opaque episode jobs across local
workers + remote TCP workers, rebalance on wall-clock completions, gather
results. Domain code (env, model, job schema) stays in your repo.

**This tree is independent of poke-bot-agent production.** It does not import,
link, or modify that trainer.

## Speed stack (v0.2)

| Layer | Tech |
|---|---|
| Socket reactor | Standalone **Asio** multi-threaded `io_context`, `TCP_NODELAY`, 4 MiB buffers, `SO_REUSEPORT` |
| JSON meta | **simdjson** parse |
| Large payloads | **`WDB1` binary frames** — opaque blob, no JSON re-encode |
| Job queues | **moodycamel ConcurrentQueue** + atomic claim credits |
| Build | `-O3`, PIC for Python wheels |

Local bench (loopback, 2000 jobs × 8 KiB blob): ~1.2 GB/s payload, >100k jobs/s dispatch.

## Python (callable from other projects)

```bash
# from this directory (needs a C++ toolchain + Python headers)
pip install -e .

# or after a cmake python build:
#   cmake -S . -B build-py -DWAVE_DISPATCH_BUILD_PYTHON=ON ...
#   PYTHONPATH=python pytest python/tests
```

```python
from wave_dispatch import (
    JobClient,
    ServerConfig,
    serve_forever,
    MidWaveScheduler,
    SchedulerConfig,
    CollectConfig,
    run_scheduled_wave,
)

# Worker
cfg = ServerConfig()
cfg.port = 8765
serve_forever(
    handler=lambda msg: {"type": "result", "ok": True, "result": msg["job"]},
    config=cfg,
    hello=lambda: {"workers": 4, "max_workers": 8, "default_workers": 4},
)

# Trainer
client = JobClient("127.0.0.1", 8765)
client.connect()
sched = MidWaveScheduler(SchedulerConfig())
ccfg = CollectConfig()
ccfg.local_workers = 2
ccfg.remote_chunk = 8
n = run_scheduled_wave(
    jobs=[{"id": i} for i in range(32)],
    local_submit=lambda job: {"ok": True, "echo": job},
    remote_clients=[client],
    scheduler=sched,
    config=ccfg,
    on_result=print,
)
```

## C++ build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

| Artifact | Role |
|---|---|
| `libwave_dispatch.a` | Core library |
| `_native*.so` | Python extension (`import wave_dispatch`) |
| `wave_echo_worker` | Synthetic remote worker |
| `wave_echo_client` | Local+remote scheduled wave demo |
| `wave_bench` | Throughput microbench (binary path) |
| `wave_dispatch_tests` | C++ unit / round-trip / binary tests |

### Quick C++ demo

```bash
./build/wave_echo_worker --port 8765 --workers 4 &
./build/wave_echo_client --endpoint 127.0.0.1:8765 --jobs 64 --local 2
```

## API (C++)

```cpp
#include "wave_dispatch/wave_dispatch.hpp"

using namespace wave_dispatch;

serve_forever(handler, ServerConfig{.port = 8765}, hello_fn);

JobClient client("gpu-box", 8765);
client.connect();
MidWaveScheduler sched(cfg);
run_scheduled_wave(jobs, local_fn, remotes, sched, collect_cfg, on_result);
```

Jobs/results are opaque JSON (nlohmann in C++, `dict` in Python). Transport is
length-prefixed JSON (`!I` + UTF-8), protocol version 1.

## Layout

```text
include/wave_dispatch/   public C++ headers
src/                     frame, client, server, scheduler, collector
python/wave_dispatch/    Python package
python/bindings/         pybind11 module
python/tests/            pytest smoke tests
apps/                    echo worker + client
tests/                   C++ frame / scheduler / TCP round-trip
docs/PROTOCOL.md         wire format
```

## Status

v0.2.0 — Asio + simdjson + binary frames + lock-free queues + Python bindings.
