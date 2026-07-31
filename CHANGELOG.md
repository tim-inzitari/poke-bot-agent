# Changelog

## 0.3.1

- Code review (see `docs/CODE_REVIEW.md`): fail-closed collector, in-flight claims,
  no silent `ok:false` completions, no blind retry after write, idle socket close,
  strict batch unpack, protocol version 2.

## 0.3.0

- Persistent `ConnectionPool` (warm sockets across waves)
- Proto v2 multi-job frames (`jobs` / `results`) with optional LZ4 blob compression
- Buffer pool for frame encode/decode churn reduction
- Unix domain sockets for localhost (auto `/tmp/wave_dispatch_<port>.sock`)
- Async Asio parent collector (thread-pool posts, not connect-per-slot)
- Optional Linux `io_uring` Asio backend when liburing is present
- `wave_bench` reports pool hit stats
- CI workflow + packaging hygiene for standalone GitHub repo

## 0.2.0

- Asio multi-threaded reactor, simdjson, `WDB1` binary frames
- moodycamel ConcurrentQueue collector
- Python bindings (`pip install -e .`)

## 0.1.0

- Initial C++ protocol, client/server, mid-wave scheduler, echo apps
