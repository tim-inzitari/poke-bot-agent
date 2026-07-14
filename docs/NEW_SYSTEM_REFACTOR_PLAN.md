# New System Refactor Plan

Isolated branch: `cursor/poke-agent-new-system-4659`  
Base: `main` @ `pre-large refactor`

## Goal

Refactor `poke_agent` so the **new dual-machine topology** is a first-class layout,
without breaking Kaggle submission or the existing JSONL → train → checkpoint loop.

## Target topology

| Machine | GPUs | Role |
|---|---|---|
| **Train host** | 3080 Ti + 3060 | CABT sim, self-play collect, transformer train/eval |
| **Infer host (LAN)** | RTX PRO 5000 Blackwell 48GB | Ollama / Qwen3.6 (dev assist, tooling) — not competition runtime |

Constraints that do **not** change:

- Kaggle submission stays **offline Torch** (`submission/` + `build_submission.sh`).
- JSONL rollouts remain the cross-machine data contract.
- cg-lib still requires Linux for simulation.

## Why refactor (current pain)

1. `device.py` always picks `cuda:0` — no train vs infer GPU roles; dual local GPUs fight.
2. Self-play couples sim + local Torch inference + train in one process tree; `workers > 1` forces inference to CPU.
3. No remote inference / Ollama hooks — only Elmo-style rollout rsync exists today.
4. `PolicyRuntime` / beam search assume in-process model weights.
5. Docs/topology still Mac + Elmo centric; dual NVIDIA + LAN Ollama undocumented.
6. Config dual-homed (`config.py` + `model_catalog.py`); hard to extend for device roles.

## Non-goals (this branch)

- Rewriting the transformer architecture from scratch.
- Using Ollama/Qwen **inside** Kaggle evaluation.
- Full multi-node distributed training (DDP/NCCL cluster).
- Deleting notebooks or the legacy single-model path in phase 1.

## Phased plan

### Phase 0 — Branch isolation (done)

- [x] Cut `cursor/poke-agent-new-system-4659` from `main`
- [ ] Keep `main` / other experiment branches untouched
- [ ] All new-system work lands only here until merge criteria met

### Phase 1 — Device & role model (small, high leverage)

**Files:** `poke_agent/device.py`, `poke_agent/config.py`, tests

Introduce explicit roles instead of one opaque picker:

| Setting | Meaning | Example |
|---|---|---|
| `TRAIN_DEVICE` | Where gradients run | `cuda:0` (3080 Ti) |
| `INFER_DEVICE` | Local policy scoring | `cuda:1` (3060) or `cpu` |
| `CUDA_VISIBLE_DEVICES` | Host pin (docs + scripts) | `0,1` on train box |
| `OLLAMA_BASE_URL` | LAN assist endpoint | `http://blackwell:11434` |

Deliverables:

- Env overrides for train/infer device
- No behavior change when unset (keep today’s `torch_device()` default)
- Unit tests for resolution order
- Short note in README: “3080 Ti = train, 3060 = infer”

**Exit:** `scripts/train_agent.py` and a smoke self-play still work with defaults.

### Phase 2 — Inference backend interface

**Files:** `poke_agent/policy_agent.py`, new `poke_agent/inference/`, `beam_search.py` (consume interface only)

```text
InferenceBackend
  ├─ LocalTorchBackend   (current PolicyRuntime path)
  └─ (later) RemoteScorer  # optional; for split workers — not Ollama-as-policy
```

Important distinction:

- **Ollama/Qwen** = human/dev assist (code, deck notes, ops). Wire as optional client utility.
- **Policy scoring for CABT** = Torch checkpoint (local GPU). Do not pretend Qwen is the battle policy unless a later experiment explicitly adds that.

Deliverables:

- `PolicyRuntime` implements / wraps `LocalTorchBackend`
- Beam search / self-play call the interface, not raw module guts
- Optional thin `ollama_client.py` for tooling scripts only

**Exit:** Identical scores vs baseline checkpoint on a fixed smoke matchup.

### Phase 3 — Self-play split along existing seams

**Files:** `poke_agent/self_play.py`, `scripts/run_self_play.py`, pipeline shell

Split the loop into clear stages (already sequential — make them invocable):

1. **Collect** — needs cg-lib; uses `INFER_DEVICE` (3060) for policy; writes JSONL  
2. **Train** — uses `TRAIN_DEVICE` (3080 Ti); reads JSONL window  
3. **Eval** — configurable device; reports win rates  

Deliverables:

- CLI flags: `--collect-only`, `--train-only`, `--eval-only` (or subcommands)
- Multi-worker collect no longer silently forces CPU if `INFER_DEVICE` is set sanely
- Docs: recommended dual-GPU env on train host

**Exit:** Curriculum self-play one iteration on dual-GPU host without OOM fights.

### Phase 4 — Topology docs & ops scripts

**Files:** `docs/ARCHITECTURE.md`, `README.md`, new `scripts/` helpers as needed

Add **Topology E — Dual NVIDIA train host + LAN Ollama**:

- Train host env template
- Blackwell Ollama host checklist (model pull, keep-alive, firewall)
- What does / does not cross the network (JSONL and checkpoints yes; cg-lib no; Kaggle agent no)

**Exit:** New contributor can bring up B-layout from docs alone.

### Phase 5 — Catalog / package cleanup (only after 1–3 stable)

**Files:** `model_catalog.py`, `model_registry.py`, `agents.py`, maybe package layout

- Register device-aware agent entries without forking train path
- Trim stale config docs vs live `config.py`
- Optional: relocate assist clients under `poke_agent/assist/` so core train stays pure

**Exit:** Multi-model train + active agent resolve still green; no submission breakage.

## Suggested cut order (do not skip)

```text
Phase 1 device roles
    → Phase 2 InferenceBackend (local)
        → Phase 3 self-play stage CLI + dual-GPU
            → Phase 4 docs/ops
                → Phase 5 catalog cleanup
```

Leave `submission/` and `build_submission.sh` untouched until Phase 3 is green.

## Test bar per phase

| Phase | Must pass |
|---|---|
| 1 | Existing unit tests + new device resolution tests |
| 2 | Score parity smoke (fixed seed / fixed checkpoint) |
| 3 | `run_self_play.py` one iteration collect→train→eval |
| 4 | Doc-only / script dry-run |
| 5 | `train_models` + `resolve_agent` + submission build smoke |

## Merge criteria

- Defaults preserve current single-GPU behavior
- Dual-GPU train host documented and smoke-tested
- Ollama is optional assist, not required for train/self-play/submit
- No regression in `scripts/build_submission.sh` / validate path
- Plan phases 1–3 complete; 4–5 can follow in same branch or fast-follow PRs

## Out of scope reminders

| Idea | Verdict |
|---|---|
| Qwen plays CABT moves | Later experiment only; not this refactor’s core |
| Tensor-parallel train across 3080 Ti + 3060 | Skip; prefer role split (train vs infer) |
| Moving cg-lib to Blackwell box | No — keep sim on train/Linux host |
| Replacing JSONL contract | No |

## Immediate next step

Implement **Phase 1** on this branch: `TRAIN_DEVICE` / `INFER_DEVICE` with backward-compatible defaults and tests.
