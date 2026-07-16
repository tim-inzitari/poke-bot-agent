# Throughput redeploy — next iter

Branch: `cursor/sim-gpu-multi-game-693f`

## What turns on by default

`scripts/launch_pure_rl.py` now setdefaults:

| Knob | Default |
|------|---------|
| `POKEBOT_MULTI_ENV` / `PURE_RL_MULTI_ENV` | `1` → **4** envs/OS process |
| `PURE_RL_LEAF_COALESCE_MS` | `0` (tiny policy) |
| Remote farms | `192.168.1.143:8765,bert.local:8766` |

Disable multi-env: `POKEBOT_MULTI_ENV=0`.

## One-shot on the training box

```bash
cd /home/inzi/poke-bot-agent   # or your checkout
bash scripts/redeploy_throughput_next_iter.sh
# at next iter / when ready to bounce:
bash scripts/redeploy_throughput_next_iter.sh --restart-now
```

That script: pulls the branch, canaries Elmo+bert, syncs/restarts bert `:8766`, prints (or runs) `launch_pure_rl` with the knobs.

## After restart — confirm in logs

Look for:

- `multi_env=4 leaf_coalesce_ms=0`
- `leaf_modes` / `leaf_self_play_mode=gpu-leaf-both` (or `gpu-leaf-us-only`)
- rising SPS / games per second vs prior iter

## Remotes

- **bert:** script SSH-syncs + restarts worker (needs key/agent auth).
- **Elmo:** host Docker image not rebuilt by this script; TCP farm still works. Rebuild compose only if you need new worker code on Elmo.
