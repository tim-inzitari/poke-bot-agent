# Senior engineer code review — wave_dispatch 0.3.0 → 0.3.1

Review date: 2026-07-31. Production poke-bot was not touched.

## Verdict (pre-fix)

**BLOCK** for production collect waves. Several failure modes could silently
shrink a wave, count remote errors as completed work, hang remote-only runs, or
duplicate jobs after ambiguous reconnects.

## Findings

| Sev | Area | Issue |
|---|---|---|
| P0 | collector | Jobs dequeued before durable result; partial failures not fatal |
| P0 | collector | Batch `ok:false` items counted as completed |
| P0 | ClaimLedger | Cap treated as lifetime eligibility → remote workers can spin forever |
| P1 | JobClient | Blind retry after write may duplicate non-idempotent jobs |
| P1 | Session | Idle timer re-armed forever; never closes socket |
| P1 | batch | Weak unpack validation (`n`, codecs, trailing bytes) |
| P1 | protocol | Docs say batch v2 but `kProtoVersion` stayed 1 |
| P2 | server | Handlers run on Asio I/O threads (latency under heavy jobs) |
| P2 | UDS | `/tmp` + `0666` defaults are loose |
| P2 | tests | Mostly happy-path |

## Fixes in 0.3.1

1. Collector: requeue failed jobs; require `completed == N` and zero errors; reject `ok:false`.
2. ClaimLedger: in-flight concurrency with `release_remote`.
3. Client: retry only if the request was not written.
4. Server idle timer closes the socket on expiry.
5. Strict batch unpack + protocol version 2.
6. Regression tests for partial failure / batch errors / idle close.

## Residual follow-ups

- Request IDs + server-side idempotency cache
- Bounded job executor off the reactor thread
- Hardened UDS directory/permissions
- Real async connect timeouts
