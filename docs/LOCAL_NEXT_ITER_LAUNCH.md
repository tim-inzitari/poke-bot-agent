# LOCAL AGENT HANDOFF — launch throughput at next iter

**Status:** ready / super-confirmed  
**Branch:** `cursor/pure-rl-full-rebuild-2d48` (tip includes merge `d2e6e06`+)  
**Do this on the training box (LAN to Elmo/bert). Do NOT kill mid-collect.**

## Instruction for local Cursor agent

At the **next pure-RL iteration boundary** (after current collect finishes; prefer promotion/iter boundary, not mid-wave):

1. `cd` to the poke-bot-agent checkout on this machine.
2. Pull the pure-RL mainline:
   ```bash
   git fetch origin cursor/pure-rl-full-rebuild-2d48
   git checkout cursor/pure-rl-full-rebuild-2d48
   git pull --ff-only origin cursor/pure-rl-full-rebuild-2d48
   ```
3. Redeploy + restart with throughput defaults (multi-env, leaf coalesce 0, live-pool watcher, remotes, **fail-closed game accuracy**):
   ```bash
   bash scripts/redeploy_throughput_next_iter.sh --restart-now
   ```
4. Confirm in logs:
   - `GAME_ACCURACY_OK`
   - `multi_env=4 leaf_coalesce_ms=0 live_pool=on`
   - `PURE_RL_WATCHER … emit_live_pool=1`
   - `leaf_modes` / `gpu-leaf-*`
5. Report SPS / games-per-sec vs prior iter once the new iter is rolling.

## Constraints

- No password commits.
- Do not attach experimental flags to unrelated overnight Hope/RR PIDs.
- If accuracy canary fails, **stop** (do not skip unless operator sets `POKEBOT_SKIP_GAME_ACCURACY=1`).
- Bert sync is attempted by the redeploy script; Elmo host Docker image is unchanged (TCP farm still used).

## Defaults that will be on

| Knob | Value |
|------|--------|
| Multi-env | 4 battles / OS process |
| Leaf coalesce | 0 ms |
| Live pool | on (resource_watcher emits plan; apply next iter) |
| Remotes | `192.168.1.143:8765,bert.local:8766` |
