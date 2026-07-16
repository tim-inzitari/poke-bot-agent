# Remote worker cutover — Elmo (TrueNAS) + bert

Boundary-safe path to redeploy remote whole-game farms with current MultiEnv
wiring and (optionally) a faster `libcg` fork **without** mid-collection kills
or simulator version storms.

**Sibling overnight relaunch:** do not interrupt live collect. Prep anytime;
cut over only at promotion/iter boundary.

## Inventory

### Elmo / TrueNAS (`192.168.1.143:8765`)

| Piece | Path / command |
|-------|----------------|
| Dockerfile | `containers/truenas-worker/Dockerfile` |
| Compose | `containers/truenas-worker/docker-compose.yml` |
| Entrypoint | `containers/truenas-worker/entrypoint.sh` |
| Ops notes | `containers/truenas-worker/OPS.md` |
| Stage/build/export | `scripts/deploy_truenas_worker.sh` |
| Local staging dirs (gitignored) | `.truenas_worker_build/`, `.truenas_worker_stage/` |
| SMB stage | `//truenas.local/main/poke-bot-agent/` |
| TrueNAS dataset | `/mnt/Main/main/poke-bot-agent/` |
| Runtime | **TrueNAS host Docker** (not nested Incus elmo Docker) |
| Checkpoint stage from trainer | SMB `…/containers/truenas-worker/checkpoint/` → `/workspace/checkpoint/` |

Exact prep commands (training box):

```bash
bash scripts/deploy_truenas_worker.sh --status
bash scripts/deploy_truenas_worker.sh --stage
bash scripts/deploy_truenas_worker.sh --build          # needs Docker on train box
bash scripts/deploy_truenas_worker.sh --export-image   # tar.gz → SMB
bash scripts/deploy_truenas_worker.sh --print-load     # copy/paste for TrueNAS
```

Optional fork bake (when parallel C++ `step_batch` `.so` is ready):

```bash
bash scripts/deploy_truenas_worker.sh --libcg-fork /path/to/cg --all
# runtime override instead of bake:
#   LIBCG_FORK_HOST=./libcg_fork CG_LIB_PATH=/workspace/libcg_fork docker compose up -d
```

### bert (`bert.local:8766`)

| Piece | Detail |
|-------|--------|
| Deploy style | **Native Mac Python** — SSH git sync, **not** Docker image |
| Script | `scripts/run_remote_worker.py --port 8766 --leaf-gpu mps` |
| Sync helper | `redeploy_remote_boundary.sh --sync-bert-code` (no restart) |
| Restart | `--cutover-remotes` or `redeploy_throughput_next_iter.sh` bert block |
| `CG_LIB_PATH` | competition `sample_submission/.../cg` (`libcg.dylib`) |
| Checkpoint stage | trainer rsync/SFTP into bert checkout (see `poke_bot/remote_jobs.py`) |

## Scripts

| Script | Role |
|--------|------|
| `scripts/deploy_truenas_worker.sh` | Elmo image stage/build/export (never restarts) |
| `scripts/redeploy_remote_boundary.sh` | **Boundary cutover orchestrator** (this doc) |
| `scripts/redeploy_throughput_next_iter.sh` | Throughput knobs + bert sync; does **not** rebuild Elmo image |
| `scripts/canary_remote_worker.py` | Hello/health; `--require-match-local` fail-closed |
| `scripts/canary_game_accuracy.py` | MultiEnv live rules canary |
| `scripts/run_remote_worker.py` | Worker process (reports `simulator_version` in hello) |

## Overnight cutover checklist

### Phase A — prep (safe while collect is running)

- [ ] On training box, checkout the deploy branch (default `cursor/pure-rl-full-rebuild-2d48` or the cutover branch).
- [ ] If fork `.so` ready: note path for `--libcg-fork`.
- [ ] Run prep (no kills):

```bash
bash scripts/redeploy_remote_boundary.sh \
  --pull --stage-elmo --build-elmo --export-elmo --sync-bert-code \
  ${LIBCG_FORK:+--libcg-fork "$LIBCG_FORK"}
```

- [ ] Confirm SMB has `containers/truenas-worker/poke-bot-truenas-worker.tar.gz`.
- [ ] Confirm bert `git rev-parse --short HEAD` matches (worker still on old process until Phase B).

### Phase B — wait for safe boundary

Prefer **after** log line:

```text
[pure_rl] iter=N games=… heldout_wr=… gate=…
```

That line is emitted **after** remote reload/pin + heldout for that iter.
Do **not** restart mid-`collect` wave.

```bash
# automatic poll of outputs/logs/pure_rl.log OR GO flag:
bash scripts/redeploy_remote_boundary.sh --wait-boundary --cutover-all

# or, when you see the iter line yourself:
mkdir -p outputs/state && touch outputs/state/remote_cutover_go
# (waiter will observe the flag)
```

### Phase C — cutover (remotes then host)

`--cutover-all` does:

1. TrueNAS: `docker load` (if tar present) + `compose down/up`
2. bert: git sync + restart `run_remote_worker.py :8766`
3. Fail-closed canaries (`--require-match-local` + game accuracy)
4. Host: stop `launch_pure_rl` / `train_pure_rl` / monitor / watcher; relaunch with MultiEnv defaults + both remotes

If TrueNAS SSH is unavailable, `--cutover-remotes` prints `--print-load` commands;
run those on the TrueNAS host, then re-run canaries + `--cutover-host --skip-wait`.

### Phase D — verify (fail closed)

```bash
python scripts/canary_remote_worker.py 192.168.1.143:8765 --require-match-local
python scripts/canary_remote_worker.py bert.local:8766 --require-match-local
# expect identical simulator_version strings
tail -f outputs/logs/pure_rl.log
# expect: GAME_ACCURACY_OK, multi_env=4, remote workers alive, no FAIL-CLOSED digest
```

Abort cutover if canary exit ≠ 0. Do not skip with `POKEBOT_SKIP_GAME_ACCURACY`
unless an operator explicitly overrides.

## Version-storm avoidance (hard rules)

1. **Same `simulator_version`** on host, Elmo, and bert before relaunching collect.
2. **No mid-collection remote restart** — wait for iter/promotion boundary.
3. **Prep ≠ cutover** — image build/export/code sync must not recreate containers.
4. **Digest pins** — Elmo checkpoints are digest-addressed under SMB `checkpoint/`;
   basename overwrites caused historical pin storms (fixed in trainer staging).
5. **Monitor patterns** — `unattended_monitor` fails closed on host
   `initial/reload/leaf response checkpoint digest mismatch` and `FAIL-CLOSED.*digest`;
   remote pin soft-drops are intentionally ignored.
6. **Fork cutover** — only enable fork `CG_LIB_PATH` on **all** participants in the
   same boundary window; mixed stock Jul-1 `libcg.so` + fork is a storm.

## What still needs host / Elmo access

This cloud agent **cannot**:

- Reach LAN (`192.168.1.143`, `bert.local`) or SSH to TrueNAS/bert
- Run Docker here (no daemon) or download competition `libcg` into the image
- Touch the live overnight trainer on the user's host

Operators on the training box must run Phase A–D. Cloud work ships the scripts,
compose/image plumbing, canary fail-closed checks, and this runbook.
