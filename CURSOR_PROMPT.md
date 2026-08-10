# Prompt to paste into a NEW Codex / Cursor account

Take over this repository for a **code fix** of a production failure:

`/Users/tsinzitari/Documents/poke-agent-codex`

## Read first (mandatory)

1. **`CODEX_ITER0_CONTRACT_LOOP.md`** — full knowledge pack (symptoms, quarantine tallies, ruled-out bugs, deployed changes, fix targets).
2. `AGENTS.md`
3. `GOAL.md` revision **175** + `state/alakazam-rtp-owner-hard-swap-r175.json`
4. Top of `CURSOR_HANDOFF.md`

Treat `GOAL.md` as authoritative. `ops/current_goal_requirements.json` is only a compatibility projection.

## Mission

**Break the Alakazam RTP r175 iter0 exact-collection loop.** The unit keeps failing:

`exact collection contract failed: self_play=1024/1024 public_mix=<short>/7172 retained=<short>/8196`

then quarantines and RESUMEs iter0 forever. Latest known fail: **attempt_0040** `public_mix=5673/7172` retained `6697/8196`. Earlier **attempt_0038** missing **816** were all `strong_public_practice` (10 non-roster18 gate specialists). Pins/self_play/diverse were OK. Refill bars complete but retention does not.

Owner wants a **code** fix so retention/promote/multi-play actually fills those cells. Do **not** weaken the exact collection contract. Do **not** set `PURE_RL_PUBLIC_MIX_LOCAL_ONLY=1` (owner rejected remotes=0). Keep RTP on; remotes engaged; multi packing preferred when safe — contract correctness beats GPS.

## Safety

Never kill/signal interactive sessions. No process-tree termination. Control workloads only via `systemctl --user` (train), `launchctl` (Bert), `sudo docker` (Elmo). Preserve worktree (no destructive git). Do not rewrite `GOAL.md` unless owner changes design.

## Hosts

SSH aliases + key `~/.ssh/id_ed25519_poke_lan` IdentitiesOnly yes: `train` 192.168.1.151 inzi, `bert` 192.168.1.158 tsinzitari, `elmo` 192.168.1.143 admin. Live PYTHONPATH `/home/inzi/poke-bot-agent`; unit `pokebot-final-format-alakazam-rtp-r175-rl.service`.

## Already fixed (do not re-break)

- Illegal ordered action prefix fallback on `ActionSpaceTooLarge` (not the current loop).
- Guide text/registry rebound without refeature.

## Deployed and suspect

Play+self_play remote multi pack @4 (`run_play_multi` for jobs with `spec`). attempt_0040 shortfall **grew** after play multi — inspect promote + multi result schema vs single `remote_play_job`.

## Done when

iter0 commits with retained **8196** and training advances past collection (not another `attempt_00NN`).
