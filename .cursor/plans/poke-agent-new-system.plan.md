---
name: Poke agent new-system refactor
overview: Isolate poke_agent for the dual-machine layout — 3080 Ti + 3060 train host, RTX PRO 5000 48GB LAN Ollama — with device roles, an inference backend, and a collect/train/eval self-play split. Kaggle submission stays offline Torch.
todos:
  - id: device-roles
    content: Add TRAIN_DEVICE / INFER_DEVICE resolution in device.py + config.py with backward-compatible defaults and unit tests
    status: pending
  - id: wire-train-infer
    content: Wire train path to TRAIN_DEVICE and self-play/policy path to INFER_DEVICE (stop silent CPU force when INFER_DEVICE is set)
    status: pending
  - id: inference-backend
    content: Extract LocalTorchBackend behind a small InferenceBackend interface; PolicyRuntime/beam_search use it; score-parity smoke
    status: pending
  - id: self-play-stages
    content: Add collect-only / train-only / eval-only CLI stages to run_self_play.py for dual-GPU role split
    status: pending
  - id: ollama-assist
    content: Add optional OLLAMA_BASE_URL client utility for LAN Qwen assist (tooling only — not CABT policy, not submission)
    status: pending
  - id: topology-docs
    content: Document Topology E (dual NVIDIA + LAN Ollama) in ARCHITECTURE.md/README with env examples
    status: pending
isProject: true
---

# Poke agent new-system refactor

Branch already cut: `cursor/poke-agent-new-system-4659` (do all work here).

## Target topology

| Machine | GPUs | Role |
|---|---|---|
| Train host | 3080 Ti (`cuda:0`) + 3060 (`cuda:1`) | CABT sim, self-play collect, transformer train |
| LAN box | RTX PRO 5000 Blackwell 48GB | Ollama / Qwen3.6 assist only |

**Hard rules**

- Kaggle `submission/` + `build_submission.sh` stay offline Torch — do not call Ollama from competition path
- JSONL remains the data contract
- Qwen is **not** the battle policy in this build
- Prefer role split (train vs infer GPU) over tensor-parallel across 3080 Ti + 3060

## Current pain (why this)

- [`poke_agent/device.py`](poke_agent/device.py) only returns bare `cuda` → dual GPUs fight on `cuda:0`
- [`poke_agent/self_play.py`](poke_agent/self_play.py) ~1275: `workers > 1` forces inference to CPU
- No remote/Ollama hooks; Elmo rsync is the only cross-machine pattern
- `PolicyRuntime` assumes in-process weights

## Implementation

### 1. Device roles

**Files:** `poke_agent/device.py`, `poke_agent/config.py`, `tests/test_device.py` (new)

- Add `resolve_train_device()` / `resolve_infer_device()` 
- Env: `TRAIN_DEVICE`, `INFER_DEVICE` (e.g. `cuda:0`, `cuda:1`, `cpu`, `mps`)
- Unset → keep today’s `torch_device()` behavior
- Map into `build_config` settings + `_ENV_MAP`
- Unit tests for resolution / fallbacks

### 2. Wire train + infer callers

**Files:** `poke_agent/main.py`, `poke_agent/multi_train.py`, `poke_agent/self_play.py`, `scripts/run_self_play.py`

- Train loops use train device
- Collect / `PolicyRuntime` use infer device
- Replace blind `workers > 1 → cpu` with: use `INFER_DEVICE` when set; only fall back to CPU when multiprocess + shared CUDA context is unsafe and no explicit infer device
- Print clear `train_device=` / `infer_device=` in logs

### 3. InferenceBackend

**Files:** new `poke_agent/inference/` (or `poke_agent/backends.py`), `poke_agent/policy_agent.py`, light touch `beam_search.py`

```text
InferenceBackend
  └─ LocalTorchBackend  # current PolicyRuntime path
```

- Beam search / self-play talk to the interface
- No behavior change for local checkpoints
- Smoke: same action scores for fixed checkpoint + state

### 4. Self-play stage CLI

**Files:** `scripts/run_self_play.py`, `poke_agent/self_play.py`

- Flags or subcommands: `--collect-only`, `--train-only`, `--eval-only`
- Same JSONL / checkpoint paths as today
- Enables 3080 Ti train while 3060 collects without one mega-process owning both

### 5. Ollama assist (optional)

**Files:** new `poke_agent/assist/ollama_client.py` (thin), env `OLLAMA_BASE_URL`

- Health check + simple chat helper for ops/dev
- **Not** wired into self-play policy or submission
- Fail soft if URL unset

### 6. Docs

**Files:** `docs/ARCHITECTURE.md`, `README.md`, keep/align `docs/NEW_SYSTEM_REFACTOR_PLAN.md`

- Add Topology E: dual NVIDIA train host + LAN Ollama
- Env example:

```bash
# train host
export TRAIN_DEVICE=cuda:0
export INFER_DEVICE=cuda:1
export OLLAMA_BASE_URL=http://blackwell-host:11434
```

## Out of scope

- Qwen choosing CABT moves
- DDP / tensor-parallel train
- Moving cg-lib to the Blackwell box
- Rewriting the transformer architecture

## Verify

```bash
pytest tests/test_device.py -q
pytest tests/ -q --ignore=tests/slow  # or project’s usual suite
python -c "from poke_agent.device import resolve_train_device, resolve_infer_device; print(resolve_train_device(), resolve_infer_device())"
# if cg-lib available:
# TRAIN_DEVICE=cuda:0 INFER_DEVICE=cuda:1 python scripts/run_self_play.py --help
scripts/build_submission.sh  # still works; no Ollama dependency
```

## Build order

Execute todos in listed order. Stop after `self-play-stages` if time-boxed; `ollama-assist` + `topology-docs` are fast follow on the same branch.
