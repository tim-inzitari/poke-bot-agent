# Ubuntu Two-Week Runner

This branch adds an unattended Linux runner for a 7950X + 128GB RAM + CUDA GPU box. It is designed to run native CABT simulations on Ubuntu and train on CUDA, defaulting to the largest visible GPU, which should be the 48GB RTX 5000 Blackwell Pro.

## Quick Start

```bash
git checkout codex/ubuntu-long-runner
bash scripts/ubuntu_setup.sh
cp configs/ubuntu_two_week.env.example configs/ubuntu_two_week.env
tmux new -s poke-agent
bash scripts/ubuntu_long_run.sh configs/ubuntu_two_week.env
```

Detach from tmux with `Ctrl-b d`. Reattach with:

```bash
tmux attach -t poke-agent
```

Watch logs from another shell:

```bash
tail -f outputs/logs/latest_ubuntu_long_run.log
```

## What It Runs

- Uses `scripts/ubuntu_setup.sh` to create `.venv`, install CUDA PyTorch, install repo requirements, and fetch Kaggle CABT inputs when Kaggle credentials are present.
- Uses `scripts/ubuntu_long_run.sh` as the two-week appliance:
  - probes CUDA, disk, model config, deck, and CABT availability;
  - bootstraps from scratch if no checkpoint exists by generating weighted CABT rollout data;
  - trains `outputs/checkpoints/temporal_current.pt`;
  - repeatedly runs native self-play versus the accessible public baseline agents;
  - trains after each rollout batch with aggressive early stopping;
  - saves checkpoints, manifest state, logs, and optional submission tarballs after each iteration.
- Uses manifest resume in `outputs/checkpoints/ubuntu_long_run/manifest.json`; rerunning the script continues from the next iteration.

## Default Hardware Policy

- `POKE_AGENT_SELECT_LARGEST_CUDA=1` chooses the visible CUDA GPU with the most VRAM.
- On your box, that should choose the 48GB RTX 5000 for training/inference.
- To force a GPU, set either:

```bash
POKE_AGENT_CUDA_DEVICE=1
```

or expose only one GPU:

```bash
CUDA_VISIBLE_DEVICES=1
```

The 3080 Ti is left unused by default so the long run is simple and stable. Use it for a second experimental process only after the main run is healthy.

## Important Config

Edit `configs/ubuntu_two_week.env` before starting:

- `RUN_HOURS=336`: two weeks.
- `BOOTSTRAP_GAMES=20000`: initial random/weighted CABT data if no checkpoint exists.
- `SELF_PLAY_GAMES=250`: active games per iteration.
- `SELF_PLAY_EVAL_GAMES=50`: eval games after each collection phase.
- `BATCH_GAMES=8`: conservative 48GB CUDA batch for the current KAN model.
- `EARLY_STOP_PATIENCE=5` and `EARLY_STOP_MIN_DELTA=0.01`: aggressive train-after-self-play early stopping.
- `SELF_PLAY_BASELINES=public`: excludes the fragile Kokinn/Roman fallback approximations.
- `BUILD_SUBMISSION_EACH_ITER=1`: builds a local tarball snapshot, but does not submit to Kaggle.

No automatic Kaggle submissions are made. This preserves the competition submission limit.

## Outputs

- Rollouts: `outputs/rollouts/ubuntu_long_run.jsonl`
- Checkpoints: `outputs/checkpoints/ubuntu_long_run/iter_*.pt`
- Latest symlink: `outputs/checkpoints/ubuntu_long_run/latest.pt`
- Manifest: `outputs/checkpoints/ubuntu_long_run/manifest.json`
- Logs: `outputs/logs/ubuntu_long_run_*.log`
- Submission snapshots: `outputs/submissions/submission_iter_*.tar.gz`

## Recovery

If the box reboots or the session dies:

```bash
cd /path/to/poke-agent
tmux new -s poke-agent
bash scripts/ubuntu_long_run.sh configs/ubuntu_two_week.env
```

The runner reads the manifest, finds the latest checkpoint, and resumes at the next iteration. It also retries transient failures up to `MAX_CONSECUTIVE_FAILURES`.

## Kaggle Inputs

CABT is Linux-native and requires `kaggle/input/cg-lib/cg/libcg.so`. The setup script can download it if Kaggle auth is configured:

```bash
mkdir -p ~/.kaggle
# put kaggle.json there
chmod 600 ~/.kaggle/kaggle.json
scripts/download-kaggle-inputs.sh
```

## Sanity Checks

Run this anytime:

```bash
.venv/bin/python scripts/ubuntu_probe.py
```

You want:

- `cuda_available: true`
- selected device pointing at the 48GB GPU
- `simulator: available=True`
- a valid Dragapult deck with 60 cards
- either an existing checkpoint or bootstrap enabled
