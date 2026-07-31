# rl-libs extraction notes

## Source mapping (poke-bot → library)

| Library surface | Inspired by | What was generalized |
|---|---|---|
| `rl_io.OrderedWriter` | `poke_bot/replay_writer.py` | Ordered commit + fsync + atomic state; opaque records |
| `rl_io.BlobPack*` | `pure_rl/expert_*_pack.py` | Named blobs + manifest + mmap verify; no Torch/schema |
| `rl_runtime.ShmRing` | `batched_infer` remote queues | SHM ring + coalesce; model forward stays caller-owned |
| `proc_pool.Supervisor` | `worker_pool.py` | Spawn/recycle/monitor; worker argv is opaque |
| `rl_eval.*` | `eval_metrics.py`, `promotion.py`, `aborts.py` | Stats/gates without Pokémon fields |
| `torch_ckpt.*` | `checkpoint.py` atomic helpers | No matchup adapter contracts |
| `artifact_registry.*` | `artifact_retention.py` / registry patterns | Receipt retire + JSON registry |

## Production safety

This tree is additive. It must not mutate:

- selector / systemd trainer units
- healthy training processes
- Elmo/Bert production compose/launchd
- live `PURE_RL_MID_ITER_SCHEDULER` pins

Wire-up is a separate, explicitly ordered change.
