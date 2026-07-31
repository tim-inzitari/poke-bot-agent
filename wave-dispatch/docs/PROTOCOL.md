# wave_dispatch wire protocol v1 (+ binary fast path)

TCP. One in-flight job per socket. Concurrency = open sockets.
Transport: **Asio** multi-threaded reactor, `TCP_NODELAY`, large SO buffers.

## Frame

```
uint32_be length
payload (length bytes)
```

Max frame: 256 MiB.

### JSON payload (control / small jobs) — v1 compatible

Payload is a UTF-8 JSON object (first byte `{`).

### Binary payload (`WDB1`) — fast path

```
'W' 'D' 'B' '1'
uint32_be meta_len
meta_json (meta_len bytes, small)
blob (remaining bytes — opaque trajectory / tensor bytes)
```

Use binary frames whenever result bodies are large. Meta stays tiny JSON;
blob is never re-encoded.

## Handshake

Client → Server:

```json
{"type":"hello","proto":1,"client":"wave-dispatch"}
```

Server → Client: `hello_ok` with `workers` / `max_workers` / capabilities.

## Data plane

JSON:

```json
{"type":"job","kind":"play","job":{...}}
```

```json
{"type":"result","ok":true,"result":{...}}
```

Binary: same meta fields inside `WDB1` meta; blob echoed/returned beside meta.

## Control

`ping`/`pong`, `health`, `bye`, plus opaque control frames (`reload`, …).

Idle timeouts on the server are **retried** (farm sockets survive wave gaps).
