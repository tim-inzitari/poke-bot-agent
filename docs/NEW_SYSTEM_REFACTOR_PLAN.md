# New System Refactor Plan

**Buildable plan:** [`.cursor/plans/poke-agent-new-system.plan.md`](../.cursor/plans/poke-agent-new-system.plan.md)

Branch: `cursor/poke-agent-new-system-4659`

## Status

Implemented on this branch:

1. `TRAIN_DEVICE` / `INFER_DEVICE` / `OLLAMA_BASE_URL` in config + `poke_agent/device.py`
2. Train / self-play callers wired to role devices
3. `poke_agent/inference` (`LocalTorchBackend` / `create_inference_backend`)
4. Self-play stages: `--collect-only` / `--train-only` / `--eval-only`
5. Optional `poke_agent/assist/ollama_client.py`
6. Topology E docs in `ARCHITECTURE.md` + `README.md`

## Quick env

```bash
export TRAIN_DEVICE=cuda:0
export INFER_DEVICE=cuda:1
export OLLAMA_BASE_URL=http://blackwell-host:11434
```
