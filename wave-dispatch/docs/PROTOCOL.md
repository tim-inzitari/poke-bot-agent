# wave_dispatch wire protocol

TCP and/or Unix domain sockets. One in-flight request per socket.
Concurrency = open sockets. Transport: **Asio** (optional Linux **io_uring**).

## Frame

```
uint32_be length
payload (length bytes)
```

Max frame: 256 MiB.

### JSON payload (control / small)

UTF-8 JSON object (starts with `{`).

### Binary payload (`WDB1`)

```
'W' 'D' 'B' '1'
uint32_be meta_len
meta_json
blob
```

### Batch (`type=jobs` / `type=results`)

Meta:

```json
{
  "type": "jobs",
  "kind": "play",
  "n": 3,
  "items": [{...}, {...}, {...}],
  "blob_lens": [0, 65, 65],
  "blob_codec": "mixed"
}
```

Blob: for each item with `blob_lens[i] > 0`:
`uint8 codec (0=none,1=lz4)` + payload bytes.

LZ4 payloads are prefixed with LE u32 original size.

## Handshake

`hello` → `hello_ok` with `workers`, `capabilities` including
`binary_v1`, `batch_v2`, `lz4_v1`.

## Localhost

Servers with `auto_uds=true` also bind `/tmp/wave_dispatch_<port>.sock`.
Clients/pools with `prefer_uds=true` use that path for `127.0.0.1` / `localhost`.
