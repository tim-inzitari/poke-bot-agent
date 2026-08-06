# Prompt to paste into Cursor

Take over this repository and continue the active Pokemon RL goal autonomously and conservatively:

`/Users/tsinzitari/Documents/poke-agent-codex`

Before doing anything, read `AGENTS.md`, `GOAL.md` revision **170**, and `CURSOR_HANDOFF.md` completely. Then read every canonical source named by `GOAL.md` that is relevant to the action you are about to take. Treat `GOAL.md` as authoritative; `ops/current_goal_requirements.json` is only a compatibility projection.

Preserve the worktree. Do not run `git reset`, `git clean`, `git checkout --`, broad formatters, or mass rewrites. Do not overwrite concurrent or unrelated edits. Do not rewrite `GOAL.md` unless the owner explicitly changes the design.

Safety is non-negotiable: never terminate, signal, disconnect, replace, or classify an SSH, Codex, Cursor, terminal, editor, or other interactive session as stale. Never use `kill`, `pkill`, `killall`, shell signals, or process-tree termination. The LAN identity in `AGENTS.md` is owner-controlled and hard-allowed. Control training/dashboard workloads only through their declared service manager, and never restart healthy training to make metadata agree.

## Active revision-170 contract

1. **Crustle is abandoned for now.** Preserve every Crustle artifact (including sealed iter5 collection receipt). Do not delete, recollect, or restart Crustle RL/RTP while abandon receipts/masks remain. Receipt: `outputs/state/crustle-owner-abandon-r170.json`.

2. **Activate Slop Box H10 + RTP** (`teal-mask-ogerpon-ex`), separately versioned from historical Teal/Slop Box. Identity: `state/slop_box_h10_rtp_prestage_identity_r170.json`. Bootstrap unit: `pokebot-final-format-slop-box-h10-rtp-bootstrap.service`.

3. **Fail-closed until** `outputs/bootstrap/slop-box-h10-rtp/expert_trajectory_shard.jsonl` exists; then start the bootstrap oneshot via systemd. Expert bootstrap must also emit the initial RTP cut (`rtp_shadow_planner.pt`). Guide stays RL-training-only; RTP stays neural-only.

4. Warm-start from checksum-bound Marnie H10 freeze `sha256:f20efb20…` (recommended primary) and optional Alakazam teacher `sha256:02c014ad…`. Never rewrite parents; require migration/parity before bootstrap.

5. Keep Bert Alakazam rejoin poller held via `launchctl bootout` until a canonical CPU/MPS selector receipt exists.

6. Refresh heartbeats (`state/goalmd_loop_heartbeat_slop_box_r170.json` / train `outputs/state/…`) as you progress. Runtime truth: `GOAL.md` + receipts + managed-service state.
