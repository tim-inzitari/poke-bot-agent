# Codex knowledge — Alakazam RTP r175 iter0 exact-collection loop

**Audience:** new Codex / Cursor account with no prior chat memory.  
**Date:** 2026-08-07 ~17:25 EDT  
**Owner verdict:** production has completely failed — unit keeps looping exact-collection failures on **iter 0**. This handoff is for a **code fix attempt**, not metadata reconciliation.

Repo (Mac worktree): `/Users/tsinzitari/Documents/poke-agent-codex`  
Live train tree (PYTHONPATH): `/home/inzi/poke-bot-agent`  
Deploy twin (launcher): `/home/inzi/poke-bot-agent-deployments/final-format-alakazam-fusion-v3-r104/`

Read order: this file → `AGENTS.md` → `GOAL.md` rev **175** → `state/alakazam-rtp-owner-hard-swap-r175.json` → `CURSOR_HANDOFF.md` top block.

Do **not** rewrite `GOAL.md` unless the owner changes the design. Preserve the worktree (no `git reset` / `clean` / `checkout --`).

---

## 1. Mission

Break the **iter0 exact collection contract loop** so the run can commit:

| Slice | Required |
|---|---|
| self_play | **1024** |
| public_mix | **7172** |
| total retained | **8196** |
| pin floors | ≥1024 Grimmsnarl `specialist-marnie-final-format-h10-f20efb20f5c3`, ≥512 Crustle `specialist-crustle-final-format-h10-7efd8d4113e7` |

Then continue r175 RL with RTP on. Owner constraints that still apply:

- Remotes **must stay engaged** on public_mix: `PURE_RL_PUBLIC_MIX_LOCAL_ONLY=0` (owner rejected remotes=0).
- RTP on in sims (`POKEBOT_USE_RECURSIVE_TURN_PLANNER=1`, ckpt `alakazam-r175.live/rtp_shadow_planner.pt`).
- Combo head OFF; guide ON; Slop Box CE hold `ExecStart=/bin/false`.
- Service control only: `systemctl --user` (train), `launchctl` (Bert), `sudo docker` (Elmo). **No kill/pkill/signals/process-tree.**
- Never restart healthy training just to reconcile metadata. Restart is OK when fixing a broken collect path / remotes.

---

## 2. Hosts / SSH

Key: `~/.ssh/id_ed25519_poke_lan`, IdentitiesOnly yes.

| Alias | Host | User | Role |
|---|---|---|---|
| `train` | 192.168.1.151 | inzi | RL unit + PYTHONPATH tree |
| `bert` | 192.168.1.158 | tsinzitari | remote worker `:8766` (hop via train if Mac auth flaps) |
| `elmo` | 192.168.1.143 | admin | docker `poke-bot-truenas-worker` `:8765` |

Bert LaunchAgent: `com.pokebot.remote-worker-8766-h10-r80`  
Bert cwd: `/Users/tsinzitari/workspace/poke-bot-agent-h10-r79-stage`  
Elmo overlay: `/mnt/Main/Elmo/poke-bot/rtp-r175-overlay/`

Unit: `pokebot-final-format-alakazam-rtp-r175-rl.service`  
Progress: `outputs/final_format_alakazam_rtp_r175/logs/rl.progress.status`  
Run: `outputs/pure_rl/final_format_alakazam_rtp_r175_i_v6_8k/`  
Quarantine: `…/quarantine/iter_00000/attempt_NNNN/`

---

## 3. What “failed” looks like

After public_mix (+ refill rounds), trainer raises:

```text
RuntimeError: exact collection contract failed:
  self_play=1024/1024 public_mix=<N>/7172 retained=<N+1024>/8196
```

Shard is quarantined; systemd restarts; RESUME `next_iteration=0`; loop forever.

**Important:** progress bars hitting 7172/7172 or refill 100% **do not mean retention succeeded**. Bars count finished jobs; contract counts retained source games with records.

Code site (deploy twin + live tree):

- `scripts/train_pure_rl.py` ~10696–10785 — `_PUBLIC_MIX_TARGETED_REPLACEMENT_ROUNDS` refill then exact contract assert.
- Retention keys: `retained_public_indices`, `with_record`, promote via `_promote_replacement_spool`.

---

## 4. Measured shortfalls (immutable quarantine evidence)

### attempt_0038 (pre play-multi on public_mix)
- Contract: `public_mix=6356/7172` retained `7380/8196` → **missing 816**
- self_play 1024 OK; diverse_public **2586** OK; pins OK
- **All 816** from `strong_public_practice` — 10 non-roster18 gate specialists, ~77–84 each:

| miss | exp | got | opponent_id |
|---:|---:|---:|---|
| 84 | 270 | 186 | specialist-archaludon-ex-gate-iter5-251298117902 |
| 83 | 270 | 187 | specialist-dragapult-dusknoir-gate-iter15-b6996ed641b1 |
| 82 | 270 | 188 | specialist-rockets-mewtwo-gate-iter5-fc2f9a525a86 |
| 82 | 270 | 188 | specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98 |
| 82 | 270 | 188 | specialist-garchomp-gate-iter5-61fbb254944f |
| 82 | 270 | 188 | specialist-dudunsparce-gate-iter15-a1e944fcb4c4 |
| 82 | 269 | 187 | specialist-teal-mask-ogerpon-ex-gate-iter14-5c74cfb63626 |
| 82 | 269 | 187 | specialist-hammer-pult-gate-iter15-c256a0ababee |
| 80 | 269 | 189 | specialist-team-rockets-spidops-gate-iter5-4ab63dc94d5a |
| 77 | 269 | 192 | specialist-thwackey-gate-iter5-0435f335fde6 |

Roster18 / public NN rows in the strong plan hit their floors. Pins Grimmsnarl/Crustle met.

### attempt_0040 (after enabling play multi-pack @4)
- Contract: `public_mix=5673/7172` retained `6697/8196` → **missing 1499** (worse)
- Same failure class (public_mix retention), larger hole
- Quarantine: `…/attempt_0040/` (shard ~2.8GB)

Earlier related: attempt_0037 ~6182/7172 (~990 missing strong_public). Pattern is stable: **strong_public gate specialists under-retain; refill does not close the hole.**

### How to re-analyze a quarantine shard
```bash
# on train
python3 - <<'PY'
# stream quarantine/.../shards/iter_00000.jsonl
# count by target_provenance.opponent_training_group
# for strong_public_practice compare opponent_id counts vs
# collection_plans/iter_00000.json per_opponent
PY
```
Fields live under `target_provenance` (`opponent_training_group`, `opponent_id`, `self_play`, `target_source`).

---

## 5. What is NOT the bug (already ruled out)

1. **Illegal ordered action / bounds=[4,4]**  
   Root cause was RTP `ActionSpaceTooLarge` treating incomplete prefixes `[0]`/`[1]` as complete actions when select needs exactly 4 of 7.  
   **Fixed** (re-raise → factorized greedy; validate with `factorized_teacher_forcing_stages`; `rtp_max_action_combos=1024`).  
   Last illegals ~13:58 EDT. attempt_0038/0040 segments had **0** illegal / `our_failed`.  
   **Do not retrain RTP for this.** Do not reintroduce prefix fallback.  
   Files: `poke_bot/recursive_turn_planner/agent_bridge.py`, `poke_bot/agent.py`, `poke_bot/poke_rlm/agent_hooks.py`.

2. **Pin floors** — already met when contract fails.

3. **Guide refeature** — pilot guide text/yaml/registry rebound (`guide_contract_sha256=f2ce4dfc…`) with **`refeature: false`**. Receipt: `outputs/state/alakazam-rtp-r175-pilot-guide-update.json`.

4. **LOCAL_ONLY=1 as the product solution** — temporarily made retention pass historically; **owner forbade remotes=0**. Keep `PURE_RL_PUBLIC_MIX_LOCAL_ONLY=0`.

---

## 6. Changes already deployed this session (relevant to the loop)

### A. Illegal-action fix (keep)
- RTP / poke_rlm bridges re-raise `ActionSpaceTooLarge`.
- Agent validates ordered actions before submit.

### B. Remote multi packing (live; may interact with retention)
Owner wanted multi on remotes for GPS.

- Env: `POKEBOT_REMOTE_SELF_PLAY_MULTI_GAMES=4`
- Train `--multi-env-per-worker 8`, MemoryHigh=120G / Max=123G
- `poke_bot/remote_jobs.py` `client_self_play_multi_pack`: packs **both** `kind=self_play` **and** `kind=play` when worker advertises `self_play_multi`
- `poke_bot/remote_sim_jobs.py` `remote_self_play_multi_job`: if child jobs have `spec` → `run_play_multi`, else `run_self_play_multi`
- `poke_bot/pure_rl/multi_env_self_play.py` **`run_play_multi`**: LibcgMultiEnv, our `PolicyAgent` vs package opponent from `spec` (`load_baseline_agent`)

Live proof after deploy:  
`scheduled_dispatch kind=play … self_play_multi_pack={'192.168.1.143:8765': 4, '192.168.1.158:8766': 4}`

**Hypothesis to test in code:** play multi packing / promote path drops or mis-keys strong_public jobs (attempt_0040 worse than 0038). Fix must preserve GPS gains if possible, but **contract correctness > GPS**.

### C. Public mix local frac
- Drop-in: `~/.config/systemd/user/pokebot-final-format-alakazam-rtp-r175-rl.service.d/50-public-mix-local-only-r175.conf`
- `PURE_RL_PUBLIC_MIX_LOCAL_ONLY=0`
- `PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC=0.70`

---

## 7. Suspected code defect areas (priority for fix)

1. **`_promote_replacement_spool` / targeted refill** (`train_pure_rl.py`)  
   Refill bars complete (multiple rounds: 1761→1521→1285→1050→…) but `retained_public_indices` still short. Promote may reject valid spares (historical Crustle r167: opp_archetype / specialist-id mismatch). Search for promote contract mismatches on strong_public `opponent_id` / `spec.id` / `opp_archetype`.

2. **`run_play_multi` result shape vs play retention**  
   Retention may require fields that single-game `_worker_play` sets (`record_json`, provenance, job_index, training_eligible). Multi path must be byte-compatible with what `_collect_wave` counts as retained. Compare a retained single-play row vs a `run_play_multi` row.

3. **Remote pack failure → silent hole**  
   If a 4-game pack fails after partial success, confirm retries re-queue **all** missing `job_index`s. Check whether pack-level failure burns 4 strong_public cells without refill credit.

4. **strong_public practice record receipt** (same function, ~10786+)  
   Separate fail path: `strong-public practice record receipt failed`. Confirm whether 0040 hit exact contract only or also practice receipt issues.

5. **Do not “fix” by weakening the exact contract** unless owner orders it. Goal language forbids weakening exact collection.

---

## 8. RTP mental model (owner clarification)

- Engine step = **one** legal action for current `select`.
- RTP plans ahead on our turn and searches for a **best sequence**, then emits the next primitive each step (`continue_plan` / replan).
- `bounds=[4,4]` with `n_options=7` is **one** multi-index select (pick exactly 4 of 7), not “4 RTP steps.” Incomplete `[0]` was illegal for that single step.

---

## 9. Safety / ops rules

- Never terminate interactive SSH/Codex/Cursor sessions; LAN identity in `AGENTS.md` is hard-allowed.
- No process-tree kills; only declared service managers.
- Prefer promotion/iter boundary restarts; mid-collection restart only for fail-closed / version storm / this contract loop fix.
- Bert restore: prefer `launchctl bootout` + `bootstrap` (avoid hung `kickstart -k` / `lsof`). Mac→Bert may hit “Too many authentication failures”; hop via `train`.

---

## 10. Immediate verification commands

```bash
ssh train 'systemctl --user show pokebot-final-format-alakazam-rtp-r175-rl.service -p ActiveState -p MainPID -p NRestarts -p ActiveEnterTimestamp'
ssh train 'cat /home/inzi/poke-bot-agent/outputs/final_format_alakazam_rtp_r175/logs/rl.progress.status'
ssh train 'ls /home/inzi/poke-bot-agent/outputs/pure_rl/final_format_alakazam_rtp_r175_i_v6_8k/quarantine/iter_00000 | tail'
ssh train 'grep "exact collection contract failed" /home/inzi/poke-bot-agent/outputs/final_format_alakazam_rtp_r175/logs/rl.log | tail'
ssh train 'grep "self_play_multi_pack" /home/inzi/poke-bot-agent/outputs/final_format_alakazam_rtp_r175/logs/rl.log | tail -5'
```

Success signal: append-only commit of iter0 (`commits/iter_00000.json` or ledger `next_iteration>=1`) with retained 8196, then train/fit advances — not another `attempt_00NN` quarantine.

---

## 11. Suggested code-attempt plan

1. Diff retained-row schema: single `remote_play_job` vs `run_play_multi` child result; fix any missing fields that block retention/promote.
2. Instrument or unit-test `_promote_replacement_spool` against strong_public missing indices using attempt_0040 jobs/plan.
3. If multi pack is drop-causing: fail-closed per-game inside pack, or temporarily pack only `self_play` until play multi is proven — **only if** needed for contract; owner still wants multi when safe.
4. Deploy fixed files to: Mac worktree, `train:/home/inzi/poke-bot-agent/`, deploy twin, Bert stage, Elmo overlay; restart workers via launchctl/docker; restart RL unit once.
5. Watch one full iter0 through commit; re-run strong_public per-opponent tally on the committed shard.

---

## 12. Key receipts on train

- `outputs/state/alakazam-rtp-r175-illegal-action-fix.json`
- `outputs/state/alakazam-rtp-r175-public-mix-local-only-recovery.json` (historical LOCAL_ONLY=1)
- `outputs/state/alakazam-rtp-r175-pilot-guide-update.json` (`refeature: false`)
- `outputs/state/alakazam-rtp-r175-codex-handoff.json`
- `state/alakazam-rtp-owner-hard-swap-r175.json`
- Quarantines: `attempt_0038`, `attempt_0040` (primary)

Also see `CURSOR_PROMPT.md` (paste prompt) and top of `CURSOR_HANDOFF.md`.
