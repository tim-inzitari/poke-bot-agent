# wave_dispatch wire protocol v1

TCP. One in-flight job per socket. Concurrency = number of open sockets.

## Frame

```
uint32_be length
utf-8 JSON object  (length bytes)
```

Max frame: 256 MiB.

## Handshake

Client → Server:

```json
{"type":"hello","proto":1,"client":"wave-dispatch"}
```

Server → Client:

```json
{
  "type":"hello_ok",
  "proto":1,
  "workers":4,
  "max_workers":8,
  "default_workers":4,
  "hostname":"...",
  "device":"cpu",
  "job_kinds":["play","echo"],
  "capabilities":["echo_v1"]
}
```

Extra hello fields are allowed (opaque to the library).

## Data plane

```json
{"type":"job","kind":"play","job":{ /* opaque */ }}
```

```json
{"type":"result","ok":true,"result":{ /* opaque */ }}
```

## Control

| type | purpose |
|---|---|
| `ping` / `pong` | liveness |
| `health` / `health_ok` | optional |
| `bye` | clean close |
| other | passed to handler (e.g. reload) |

Idle `recv` timeouts on the server are **retried** so farm sockets survive gaps between waves.
