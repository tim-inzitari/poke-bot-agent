# wave_dispatch

Standalone **C++** LAN job-dispatch library (with **Python bindings**) for
multi-machine RL collect waves — tuned for max dispatch speed.

Fan opaque episode jobs across local workers + remote TCP/UDS workers, rebalance
on wall-clock completions, gather results. Domain code (env, model, job schema)
stays in your competition repo.

## Speed stack (v0.3)

| Layer | Tech |
|---|---|
| Socket reactor | Standalone **Asio** (optional Linux **io_uring**), `TCP_NODELAY`, 4 MiB buffers, `SO_REUSEPORT` |
| Localhost | **Unix domain sockets** (`/tmp/wave_dispatch_<port>.sock`) |
| Connections | Persistent **`ConnectionPool`** (warm sockets across waves) |
| JSON meta | **simdjson** |
| Large payloads | **`WDB1` binary frames** + optional **LZ4** |
| Multi-job | Proto v2 **`jobs`/`results` batches** (one RTT for N jobs) |
| Queues | **moodycamel ConcurrentQueue** + atomic claims |
| Parent collect | Asio **async thread pool** (no connect-per-slot) |
| Buffers | Recycled **BufferPool** |

## Install (Python)

```bash
# system deps: cmake, C++17, liblz4-dev, (linux) liburing-dev
pip install -e .
```

```python
from wave_dispatch import (
    JobClient, WorkerFarm, ConnectionPool,
    ServerConfig, serve_forever,
    MidWaveScheduler, SchedulerConfig,
    CollectConfig, run_scheduled_wave,
)

# Add children by endpoint; add a new parent by calling run_scheduled_wave
farm = WorkerFarm(["gpu-a:8765", "gpu-b:8766", "127.0.0.1:8767"])
farm.connect()
pool = ConnectionPool()
for ep in ["gpu-a:8765", "gpu-b:8766", "127.0.0.1:8767"]:
    pool.ensure(ep, 4)

cfg = SchedulerConfig()
cfg.remote_defaults = {ep: 4 for ep in ["gpu-a:8765", "gpu-b:8766", "127.0.0.1:8767"]}
cfg.remote_maxima = {ep: 8 for ep in cfg.remote_defaults}
sched = MidWaveScheduler(cfg)

ccfg = CollectConfig()
ccfg.batch_size = 16
ccfg.compress_blobs = True
ccfg.use_connection_pool = True
ccfg.prefer_uds = True

n = run_scheduled_wave(
    jobs=[{"id": i} for i in range(256)],
    local_submit=lambda job: {"ok": True, "echo": job},
    remote_clients=list(farm.clients()),
    scheduler=sched,
    config=ccfg,
    pool=pool,
)
```

## Build (C++)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/wave_bench --jobs 4000 --blob 8192 --batch 16
```

## Layout

```text
include/wave_dispatch/   public headers
src/                     implementation
python/wave_dispatch/    Python package
apps/                    echo_worker, echo_client, wave_bench
tests/                   C++ tests
.github/workflows/ci.yml Linux/macOS build + wheels
```

## License

MIT — see [LICENSE](LICENSE).
