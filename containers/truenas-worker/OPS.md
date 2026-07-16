# TrueNAS `elmo` remote worker ops

## Role

**Deploy target:** TrueNAS SCALE host Docker listening on **`192.168.1.143:8765`**
(Incus instance **`elmo`** GPU passthrough remains available, but nested Docker
inside elmo is blocked by userns — production worker runs on the **TrueNAS host**).

| Fact | Value |
|------|-------|
| Worker LAN endpoint | **`192.168.1.143:8765`** |
| TrueNAS host | `192.168.1.143` / `truenas.local` (SSH via env/`ssh-agent`; **never** store passwords in repo) |
| elmo Incus LAN IP | `192.168.1.252` (GPU device host; not the TCP endpoint today) |
| GPU | NVIDIA GeForce RTX 3060 LHR (GA106), **12288 MiB** |
| CPU / RAM | 24 threads / ~125 GiB |

LAN offload: **CPU sims + MultiEnv** + **leaf inference on the 3060 LHR**.
Training box ships whole-game jobs over TCP `:8765`.

Sizing:

| Knob | Value | Why |
|------|-------|-----|
| `SIM_WORKERS` | **20** | 24 threads − leaf + Docker/OS headroom |
| `LEAF_SERVERS` | **2** | Keep GA106 12 GB fed without OOM |
| `LEAF_MAX_BATCH` | **192** | 12 GB headroom |
| `LEAF_QUEUE_DEPTH` | **96** | Match larger batches |
| `POKEBOT_MULTI_ENV` | **1 → 4** | Python `LibcgMultiEnv` handles / OS worker |
| `cpus` / `mem_limit` | **20 / 56g** | Cap below free RAM |

## Inventory — how the image is built / loaded

| Step | Where | Command |
|------|-------|---------|
| Stage context → SMB | training box | `bash scripts/deploy_truenas_worker.sh --stage` |
| Build image | training box (Docker) | `bash scripts/deploy_truenas_worker.sh --build` |
| Export tar → SMB | training box | `bash scripts/deploy_truenas_worker.sh --export-image` |
| Optional fork bake | training box | `… --libcg-fork /path/to/cg` (dir with `libcg.so` or `cg/`) |
| Load + compose | TrueNAS host | see `LOAD_ON_ELMO.sh.txt` / `--print-load` |
| Boundary cutover | training box | `bash scripts/redeploy_remote_boundary.sh …` |

Staging paths:

- SMB share: `//truenas.local/main/poke-bot-agent/`
- gvfs default: `/run/user/$UID/gvfs/smb-share:server=truenas.local,share=main/poke-bot-agent`
- On TrueNAS dataset: `/mnt/Main/main/poke-bot-agent/` (also linked under `/mnt/Main/Elmo/…`)
- Local build artifacts (gitignored): `.truenas_worker_build/`, `.truenas_worker_stage/`

Compose bind-mounts only `checkpoint/` + `runtime-logs/` (+ optional `libcg_fork/`).
Python code + `libcg` are **baked into the image** (except fork override mount).

## Discovery notes

- Host has NVIDIA driver **550.142** + Docker; elmo has `/dev/nvidia*` after
  physical GPU device `gpu0`.
- TrueNAS UI/API requires GPU devices with **`gpu_type: PHYSICAL`**.
- Apt `nvidia-utils` on Ubuntu plucky pulls mismatched 570 NVML — do **not**.
- Incus `nvidia.runtime=true` failed mount hooks; keep PHYSICAL PCI passthrough.
- **Current deploy:** worker on **TrueNAS host Docker** with GPU devices,
  `192.168.1.143:8765`.

## Stage / build / export (safe mid-collect)

On the training box — **does not restart workers or trainers**:

```bash
bash scripts/deploy_truenas_worker.sh --status
bash scripts/deploy_truenas_worker.sh --stage
bash scripts/deploy_truenas_worker.sh --build
bash scripts/deploy_truenas_worker.sh --export-image
# or: bash scripts/deploy_truenas_worker.sh --all

# When C++ fork .so is ready (optional):
bash scripts/deploy_truenas_worker.sh --libcg-fork /path/to/cg --all
```

## Load on TrueNAS (boundary only)

```bash
bash scripts/deploy_truenas_worker.sh --print-load
# Then on TrueNAS host (SSH), at promotion/iter boundary only:
cd /mnt/Main/main/poke-bot-agent/containers/truenas-worker
gunzip -fk poke-bot-truenas-worker.tar.gz || true
docker load -i poke-bot-truenas-worker.tar
mkdir -p checkpoint runtime-logs
docker compose down && docker compose up -d
```

Or from the training box with key auth:

```bash
export TRUENAS_SSH_USER=…   # never commit secrets
bash scripts/redeploy_remote_boundary.sh --wait-boundary --cutover-remotes
```

## Smoke (does not touch overnight trainers)

```bash
python scripts/canary_remote_worker.py 192.168.1.143:8765
# Fail-closed before cutover:
python scripts/canary_remote_worker.py 192.168.1.143:8765 --require-match-local
```

Expect: `hello_ok`, `gpu` contains `3060`, `workers=20`, `leaf_alive=true`,
and `simulator_version` matching the training box when `--require-match-local`.

## bert.local (native Mac whole-game worker — not Docker)

**Endpoint:** `bert.local:8766` / `192.168.1.157:8766`  
(Port **8765** on bert is Hermes `local-router`; poke-bot uses **8766**.)

| Fact | Value |
|------|-------|
| Host | Apple M4 Pro, 14 cores, ~64 GiB |
| Deploy style | **SSH git sync + native Python** (no NVIDIA Docker) |
| Checkout | `~/workspace/poke-bot-agent` |
| `CG_LIB_PATH` | competition `libcg.dylib` under sample_submission `cg/` |
| Leaf | local **MPS** |
| Workers | `10` sim + `1` leaf |

Code-only sync (safe mid-collect):

```bash
bash scripts/redeploy_remote_boundary.sh --sync-bert-code
```

Worker restart (boundary only) is handled by `--cutover-remotes` / the existing
`scripts/redeploy_throughput_next_iter.sh` bert block.

## Version-storm avoidance

1. Never restart Elmo/bert mid-collection wave.
2. Prep image/tar/code first; cut over only after `[pure_rl] iter=N …` or GO flag.
3. Host + Elmo + bert must share the same `simulator_version` (libcg digest).
4. `unattended_monitor` already fails closed on host digest mismatches; remote
   pin soft-drops are ignored — still canary with `--require-match-local`.
5. Prefer `scripts/redeploy_remote_boundary.sh --wait-boundary --cutover-all`.

Full checklist: [docs/REMOTE_WORKER_CUTOVER.md](../../docs/REMOTE_WORKER_CUTOVER.md).
