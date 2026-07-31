# Elmo remote-worker safety runbook

Elmo has two explicit lifecycles built from the same immutable image:

- `docker-compose.yml` (or `docker-compose.host.yml`) is the observed canary:
  four sim processes, a 100-job stop, and `restart: "no"`.
- `docker-compose.production.yml` is an override applied only after the canary.
  Routine whole-service job-count rotation is disabled because it drops every
  trainer socket. Per-child recycling and the independent RSS, free-RAM,
  capacity, and identity guards remain active.

Never use the production override by itself. Pick exactly one base file for the
host's GPU attachment and apply the production file second.

## Limits in both modes

- `40g` memory and `40g` memory+swap: about 40 GiB RAM and no container swap.
- `pids_limit: 256` and `init: true`: bound process fan-out and reap children.
- Docker JSON logs rotate at 10 MiB, retaining three files.
- Health uses the framed protocol and requires healthy, identity-matched leaves.
- Simulator children recycle every 16 games.
- The service reserves 16 TCP connections so the advertised 12-game load
  cannot consume the control/health path.
- A running pool gets 60 seconds to restore full ready capacity while a
  recycled child completes initialization. A stopped pool or any recorded
  initializer failure still exits immediately with watchdog code `70`.
- The worker drains immediately at 30 GiB uniquely charged cgroup-v2 memory
  (process-tree PSS fallback) or 24 GiB host available RAM. Summed descendant
  RSS remains diagnostic-only because spawned workers share pages and summing
  their RSS double-counts those pages. Watchdog exits are failures, not planned
  rotations.
- Production reloads atomically publish an exact path+SHA-256 record under
  `runtime-logs`; a later lifetime refuses to start if that record cannot be
  reproduced from the read-only `/workspace/checkpoint` mount.
- The image is pinned to `poke-bot-truenas-worker:safety-20260717.4`; its
  entrypoint, supervisor, and Python package are baked together.
- The prior `poke-bot-truenas-worker:safety-20260717` tag is retained unchanged
  for rollback; never rebuild or retag it.
- Non-smoke startup still requires exact token `20260717` (no newline) in
  `runtime-logs/REMOTE_WORKER_ARMED`. This runbook only verifies an already
  approved token; it does not create one.

## Production restart circuit

The remote worker retains reserved exit code `75` for compatibility with an
explicitly requested planned drain; routine job-count rotation is disabled.
The supervisor accepts that code only after at least 60 seconds, waits 10
seconds, and starts the next bounded lifetime. Each worker and all of its
manager/pool/leaf descendants run in one isolated process group. The supervisor
forwards TERM/INT/HUP to that group, allows up to 75 seconds for Python cleanup,
then sends KILL before Docker's 90-second stop deadline.

Every other exit consumes one of three failure slots in a rolling one-hour
window. Before each worker start, the supervisor writes
`runtime-logs/elmo-supervisor/active.attempt`. If the whole container or cgroup
is killed, the next Docker start sees the stale active marker and consumes a
slot. At three failures the supervisor writes an `open` state to
`circuit.state` and exits `0`. Production Compose uses `on-failure:3` only as a
final supervisor-crash guard, so that clean circuit-open exit cannot be
restarted by Docker.

The circuit does not silently reclose while the service is down. After the
one-hour window expires, or after an operator archives the failure state after
diagnosis, an explicit `docker compose up` is required.

## Checkpoint continuity across rotation

Production sets two variables that canary and Bert do not set:

- `POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE` points to
  `runtime-logs/elmo-supervisor/active-checkpoint.json`.
- `POKEBOT_REMOTE_CHECKPOINT_ROOT` confines every resumable checkpoint to the
  read-only `/workspace/checkpoint` bind mount.

After every fully acknowledged reload, the worker re-hashes the staged file and
atomically publishes its resolved path and digest before returning success to
the trainer. Startup treats this record as authoritative. Missing or corrupt
JSON, a missing file, a symlink/path escaping the checkpoint root, or a digest
mismatch returns a startup failure; it never falls back to `model.pt`.

The explicit `seed-active-checkpoint` preflight below is the only first-boot
bootstrap. It creates the record from `model.pt` only when no record exists. If
a valid record already exists, it validates and preserves it, so rerunning the
rollout cannot undo a later trainer reload. An existing invalid record fails
closed and must be investigated rather than overwritten.

## Build and static validation (does not start Elmo)

From `/mnt/Main/Elmo/poke-bot-agent`:

```bash
docker build \
  --file containers/truenas-worker/Dockerfile \
  --tag poke-bot-truenas-worker:safety-20260717.4 \
  .
docker image inspect poke-bot-truenas-worker:safety-20260717.4 \
  --format '{{ index .Config.Labels "org.opencontainers.image.version" }}'
# Confirm the pre-change image still exists as a distinct rollback artifact.
docker image inspect poke-bot-truenas-worker:safety-20260717 \
  --format 'rollback_id={{.Id}} version={{index .Config.Labels "org.opencontainers.image.version"}}'

cd containers/truenas-worker
BASE=docker-compose.yml
# Use docker-compose.host.yml instead only on a nvidia-container-toolkit host.
docker compose -f "$BASE" config --quiet
docker compose -f "$BASE" -f docker-compose.production.yml config --quiet
docker compose -f "$BASE" -f docker-compose.production.yml config | \
  grep -E 'restart:|mem_limit:|memswap_limit:|pids_limit:|MAX_SERVICE_JOBS|TREE_RSS|MIN_FREE_RAM|MAX_CONNECTIONS|CAPACITY_RECOVERY_GRACE|RECYCLE_GAMES|ACTIVE_CHECKPOINT|CHECKPOINT_ROOT'
test "$(cat runtime-logs/REMOTE_WORKER_ARMED)" = 20260717
```

Do not continue if either rendered config fails, if the image label differs,
or if the pre-approved arm file check fails.

## Observed canary

The canary remains fail-closed and never restarts itself:

```bash
cd /mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker
BASE=docker-compose.yml
docker compose -f "$BASE" up -d --force-recreate --wait worker
docker compose -f "$BASE" logs -f --tail=200 worker
docker inspect poke-bot-truenas-worker \
  --format 'restart={{.HostConfig.RestartPolicy.Name}} max={{.HostConfig.RestartPolicy.MaximumRetryCount}} memory={{.HostConfig.Memory}} swap={{.HostConfig.MemorySwap}} pids={{.HostConfig.PidsLimit}}'
```

Raise both `ELMO_SIM_WORKERS` and `ELMO_SIM_DEFAULT_WORKERS` together only
through the observed `4 -> 8 -> 12 -> 16 -> 20` sequence. The entrypoint
rejects anything above 20.

## Production rollout after the passed canary

Use the same base selected during validation:

```bash
cd /mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker
BASE=docker-compose.yml
docker compose -f "$BASE" -f docker-compose.production.yml config --quiet
docker compose -f "$BASE" -f docker-compose.production.yml \
  run --rm --no-deps worker seed-active-checkpoint
cat runtime-logs/elmo-supervisor/active-checkpoint.json
docker compose -f "$BASE" -f docker-compose.production.yml \
  up -d --force-recreate --wait worker
docker inspect poke-bot-truenas-worker \
  --format 'restart={{.HostConfig.RestartPolicy.Name}} max={{.HostConfig.RestartPolicy.MaximumRetryCount}} memory={{.HostConfig.Memory}} swap={{.HostConfig.MemorySwap}} pids={{.HostConfig.PidsLimit}}'
docker compose -f "$BASE" -f docker-compose.production.yml ps
docker compose -f "$BASE" -f docker-compose.production.yml logs --tail=200 worker
cat runtime-logs/elmo-supervisor/circuit.state
cat runtime-logs/elmo-supervisor/active.attempt
cat runtime-logs/elmo-supervisor/active-checkpoint.json
```

Expected production inspection is `restart=on-failure max=3`, 40 GiB equal
memory/swap byte caps, and `pids=256`. `circuit.state` should begin with
`closed`; `active.attempt` should normally begin with `active` while serving.

For a graceful operator stop:

```bash
docker compose -f "$BASE" -f docker-compose.production.yml stop -t 90 worker
```

To return to the no-restart canary configuration:

```bash
docker compose -f "$BASE" up -d --force-recreate --wait worker
```

## Image rollback

`docker-compose.rollback.yml` changes only the image reference back to the
preserved pre-grace tag. Apply it last so the selected GPU attachment,
production lifecycle, resource bounds, mounts, and durable state remain
unchanged. Diagnose and archive the `.2` logs before replacing the container.

For a production rollback:

```bash
docker image inspect poke-bot-truenas-worker:safety-20260717 >/dev/null
docker compose -f "$BASE" -f docker-compose.production.yml \
  -f docker-compose.rollback.yml config --quiet
docker compose -f "$BASE" -f docker-compose.production.yml \
  -f docker-compose.rollback.yml up -d --force-recreate --wait worker
```

For a no-restart canary rollback, omit `docker-compose.production.yml`. Never
point `safety-20260717` at the `.2` image; rollback depends on distinct image
IDs and immutable tags.

## Circuit-open recovery

First inspect the persisted reason and logs; do not repeatedly run `up`:

```bash
cat runtime-logs/elmo-supervisor/circuit.state
cat runtime-logs/elmo-supervisor/active.attempt
cat runtime-logs/elmo-supervisor/failures.epoch
docker compose -f "$BASE" -f docker-compose.production.yml logs --tail=500 worker
docker inspect poke-bot-truenas-worker --format '{{json .State}}'
```

After fixing the checkpoint, configuration, GPU, RSS, or host-RAM cause, the
rolling window can expire naturally. To resume immediately, archive (do not
erase) the evidence and explicitly recreate the service:

```bash
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e runtime-logs/elmo-supervisor/failures.epoch || \
  mv runtime-logs/elmo-supervisor/failures.epoch \
     "runtime-logs/elmo-supervisor/failures.epoch.$stamp"
test ! -e runtime-logs/elmo-supervisor/circuit.state || \
  mv runtime-logs/elmo-supervisor/circuit.state \
     "runtime-logs/elmo-supervisor/circuit.state.$stamp"
docker compose -f "$BASE" -f docker-compose.production.yml \
  up -d --force-recreate --wait worker
```
