# Throughput redeploy — next iter

Branch: `cursor/sim-gpu-multi-game-693f`

## What turns on by default

`scripts/launch_pure_rl.py` now setdefaults:

| Knob | Default |
|------|---------|
| `POKEBOT_MULTI_ENV` / `PURE_RL_MULTI_ENV` | `1` → **4** envs/OS process |
| `PURE_RL_LEAF_COALESCE_MS` | `0` (tiny policy) |
| `POKEBOT_LIVE_POOL` | `1` — **auto-rebalance** via `resource_watcher --emit-live-pool` |
| Remote farms | `192.168.1.143:8765,bert.local:8766` |

Watcher ratchets `workers` / `leaf_servers` from CPU/RAM/VRAM headroom and writes `outputs/state/live_pool_plan.json`. Pure-RL applies at the **next iteration boundary** (never mid-collect).

Disable multi-env: `POKEBOT_MULTI_ENV=0`.  
Disable live pool: `POKEBOT_LIVE_POOL=0` or `launch_pure_rl --no-resource-watcher`.

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

- `multi_env=4 leaf_coalesce_ms=0 live_pool=on`
- `PURE_RL_WATCHER pid=… emit_live_pool=1`
- `leaf_modes` / `leaf_self_play_mode=gpu-leaf-*`
- later: `[pure_rl] live_pool_plan seq=… apply workers=…->…`
- rising SPS / games per second vs prior iter

Also: `tail -f outputs/logs/resource_watcher.log` and `cat outputs/state/live_pool_plan.json`.

## Remotes

- **bert:** script SSH-syncs + restarts worker (needs key/agent auth).
- **Elmo:** host Docker image not rebuilt by this script; TCP farm still works. Rebuild compose only if you need new worker code on Elmo.
