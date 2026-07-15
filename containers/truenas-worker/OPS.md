# TrueNAS `elmo` remote worker ops

## Role

**Deploy target:** TrueNAS SCALE Incus instance **`elmo`**
([UI](https://192.168.1.143/ui/instances/view/elmo)) — not host Apps, not bert.

| Fact | Value |
|------|-------|
| elmo LAN IP | **`192.168.1.252`** |
| TrueNAS host | `192.168.1.143` (SSH via env/`ssh-agent`; **never** store passwords in repo) |
| GPU in elmo | NVIDIA GeForce RTX 3060 LHR (GA106), **12288 MiB** |
| CPU / RAM in elmo | 24 threads / ~125 GiB |

LAN offload: **CPU belief-MCTS / sims** + **leaf inference on the 3060 LHR inside
elmo**. Training box ships whole-game jobs over TCP `:8765`.

Sizing:

| Knob | Value | Why |
|------|-------|-----|
| `SIM_WORKERS` | **20** | 24 threads − leaf + Docker/OS headroom |
| `LEAF_SERVERS` | **2** | Keep GA106 12 GB fed without OOM |
| `LEAF_MAX_BATCH` | **192** | 12 GB headroom (was 96 on mistaken 8 GB Ti) |
| `LEAF_QUEUE_DEPTH` | **96** | Match larger batches |
| `cpus` / `mem_limit` | **20 / 56g** | Cap below free RAM |

## Discovery notes

- Host has NVIDIA driver **550.142** + Docker; elmo has `/dev/nvidia*` after
  physical GPU device `gpu0`.
- TrueNAS UI/API requires GPU devices with **`gpu_type: PHYSICAL`** (not just
  Incus `gputype=physical`). Fix via:
  `midclt call virt.instance.device_update elmo '{"name":"gpu0","dev_type":"GPU","gpu_type":"PHYSICAL","pci":"0000:2d:00.0"}'`
- Apt `nvidia-utils` on Ubuntu plucky pulls mismatched 570 NVML — do **not**.
  Use curated host userspace at `/mnt/Main/Elmo/nvidia-host` (`LD_LIBRARY_PATH`)
  matching 550.142.
- Staging share: `/mnt/Main/main/poke-bot-agent` (SMB `//truenas.local/main/...`),
  also linked as `/mnt/Main/Elmo/poke-bot-agent` / mounted at `/opt/poke-bot-agent`.
- Incus `nvidia.runtime=true` failed mount hooks; keep PHYSICAL PCI passthrough.

## Stage / build / start

On the training box:

```bash
bash scripts/deploy_truenas_worker.sh --stage
bash scripts/deploy_truenas_worker.sh --build
bash scripts/deploy_truenas_worker.sh --export-image
```

On elmo (from TrueNAS host via `incus exec elmo`):

```bash
cd /mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker
# after --export-image lands poke-bot-truenas-worker.tar.gz here:
gunzip -fk poke-bot-truenas-worker.tar.gz || true
docker load -i poke-bot-truenas-worker.tar
mkdir -p checkpoint runtime-logs
# put champion .pt at checkpoint/model.pt
docker compose up -d
```

If nested Docker GPU fails inside elmo, fall back to host Docker on
`192.168.1.143:8765` with the same image (same physical 3060).

**Current deploy note:** Incus userns blocks nested Docker (`sysctl
net.ipv4.ip_unprivileged_port_start`). Worker runs on **TrueNAS host Docker**
with `--gpus all`, listening on `192.168.1.143:8765`. GPU passthrough into
elmo is left untouched.

## Smoke (does not touch overnight trainers)

```bash
/home/inzi/miniconda3/envs/poke-bot-agent/bin/python \
  scripts/canary_remote_worker.py 192.168.1.143:8765
```

Expect: `hello_ok`, `gpu` contains `3060`, `workers=20`, `leaf_alive=true`.

## bert.local (phase 2 — native Mac whole-game worker)

**Endpoint:** `bert.local:8766` / `192.168.1.157:8766`  
(Port **8765** on bert is already taken by Hermes `local-router` on
`127.0.0.1`; poke-bot uses **8766** bound to `0.0.0.0`.)

| Fact | Value |
|------|-------|
| Host | Apple M4 Pro, 14 cores, ~64 GiB |
| Deploy style | Native Python (no NVIDIA Docker) |
| Checkout | `~/workspace/poke-bot-agent` on bert |
| `CG_LIB_PATH` | `…/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg` (**competition `libcg.dylib` arm64**, not `cg-lib`) |
| Leaf | local **MPS** (`--leaf-gpu mps`), co-located with sims |
| Workers | `10` sim + `1` leaf (leave headroom vs 14 cores) |
| Protocol | Same whole-game TCP as elmo — **not** 1-leaf/RPC over ~45 ms RTT |

### Start / restart on bert

```bash
# on bert (SSH). Auth is env/agent — never store passwords in this repo.
cd ~/workspace/poke-bot-agent && source .venv/bin/activate
export CG_LIB_PATH="$HOME/workspace/poke-bot-agent/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg"
export POKEBOT_CHECKPOINT="$HOME/workspace/poke-bot-agent/outputs/checkpoints/model.pt"
nohup python scripts/run_remote_worker.py \
  --host 0.0.0.0 --port 8766 \
  --workers 10 --leaf-servers 1 --leaf-gpu mps \
  --leaf-max-batch 96 --leaf-queue-depth 48 \
  --checkpoint "$POKEBOT_CHECKPOINT" --cg-lib-path "$CG_LIB_PATH" \
  > logs/remote_worker.8766.log 2>&1 &
```

### Smoke (training box; does not touch overnight trainers)

```bash
/home/inzi/miniconda3/envs/poke-bot-agent/bin/python \
  scripts/canary_remote_worker.py bert.local:8766
```

Expect: `hello_ok`, `device=mps`, `workers=10`, `leaf_alive=true`.
