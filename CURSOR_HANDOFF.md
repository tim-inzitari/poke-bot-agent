# Cursor takeover handoff

## CRITICAL — Codex code-fix NOW (2026-08-07 ~17:25 EDT)

**Owner:** production completely failed — iter0 exact-collection loop. New Codex account: read **`CODEX_ITER0_CONTRACT_LOOP.md` first**, then this block.

**Mission:** Code-fix retention/promote/(play) multi so iter0 commits **8196** retained with remotes ON and RTP ON. Do not weaken the exact contract. Do not force `LOCAL_ONLY=1`.

### Failure truth
- Unit `pokebot-final-format-alakazam-rtp-r175-rl.service` on `train` loops: collect → refill bars → `exact collection contract failed` → quarantine → RESUME iter0.
- **attempt_0038:** `public_mix=6356/7172` (missing **816**), all `strong_public_practice`, 10 non-roster18 gate specialists ~80 each. Pins/self_play/diverse OK. **0** illegal in that attempt.
- **attempt_0040:** `public_mix=5673/7172` retained `6697/8196` (missing **1499**) — **worse after play multi-pack @4**.
- Progress bar 100% ≠ retained contract.

### Live env (must keep)
- RTP ON; `POKEBOT_REMOTE_SELF_PLAY_MULTI_GAMES=4`; train `--multi-env-per-worker 8`; MemoryHigh=120G/Max=123G
- `PURE_RL_PUBLIC_MIX_LOCAL_ONLY=0`, `MIN_LOCAL_FRAC=0.70`
- PYTHONPATH `/home/inzi/poke-bot-agent`; run `outputs/pure_rl/final_format_alakazam_rtp_r175_i_v6_8k/`
- Bert `:8766` LaunchAgent; Elmo docker + `rtp-r175-overlay`

### Code already shipped this session
- Illegal-action fix (keep).
- `client_self_play_multi_pack` packs `self_play` **and** `play`; `run_play_multi` for jobs with `spec`.
- Suspect: promote spool and/or `run_play_multi` result schema vs `_worker_play` retention.

### Do / don't
- DO fix promote/retention/multi-play so strong_public cells fill under remotes.
- DO deploy to train tree + twin + Bert stage + Elmo overlay; service-manager restarts only.
- DON'T retrain RTP for bounds=[4,4]; DON'T LOCAL_ONLY=1; DON'T weaken exact contract; DON'T rewrite GOAL.md; DON'T kill interactive sessions.

### Paste prompt
See `CURSOR_PROMPT.md`. Full dossier: `CODEX_ITER0_CONTRACT_LOOP.md`.


---

## CRITICAL — LAN `self_play_multi` packing (Bert+Elmo) (2026-08-07 ~12:04 EDT)

- **Owner override** of GOAL r112/r124 single-game-per-socket for LAN remotes: trainer now submits `self_play_multi` packs at **4 games/socket** to **both** Elmo `:8765` and Bert `:8766`.
- **Live proof**: `self_play_multi_pack={'192.168.1.143:8765': 4, '192.168.1.158:8766': 4}`; train `multi_env=8`; collect advanced to **~870/1024 @ ~8 game/s** with remotes=52; `MemoryHigh=120G` / `MemoryMax=123G`; Slop Box hold still `ExecStart=/bin/false`.
- **Why packing was dead before**: remotes already had `POKEBOT_MULTI_ENV_PER_WORKER=4`, but dispatch stayed single-game while remotes were up. **RAM was never the gate — dispatch was.**
- **Fixes deployed**:
  - Train `poke_bot/remote_jobs.py` (PYTHONPATH `/home/inzi/poke-bot-agent`): SMB-less Elmo resident mapping via `checkpoint_digest_verify_v1` when gvfs SMB is missing/wedged; bounded gvfs dir probes.
  - Elmo overlay + image: `remote_sim_jobs.py` strips `collect_privileged_belief` on official libcg for multi; `multi_env_self_play.py` soft-skips hidden-snapshot RuntimeError. Overlay path: `/mnt/Main/Elmo/.../rtp-r175-overlay/poke_bot/pure_rl/multi_env_self_play.py`.
  - Bert stage trees patched the same; LaunchAgent `com.pokebot.remote-worker-8766-h10-r80` (avoid `kickstart -k` churn mid-listen).
- **Receipt**: `outputs/state/alakazam-rtp-r175-remote-self-play-multi-packing.json`.
- **Residual**: occasional remote pack retries on `illegal ordered action length/content` (policy runtime); not packing/capability. Do not restart healthy RL to reconcile metadata.

## CRITICAL — Bert/Elmo RTP remote sync (2026-08-07 ~11:00 EDT)

- **Deployed** train rtp-cpu-local + trusted RTP stamps to Bert `:8766` and Elmo `:8765` without intending to restart healthy train; remote flap still forced a later systemd restart of `pokebot-final-format-alakazam-rtp-r175-rl.service` (now MainPID **3817272**, NRestarts=0, collecting).
- **Artifacts synced**: `poke_bot/agent.py`, `leaf_self_play.py`, `recursive_turn_planner/**`, `poke_rlm/**` (required import), `train_round_robin.py`, RTP ckpt `sha256:dde7b813…` (`alakazam-r175.live/rtp_shadow_planner.pt`).
- **Env on remotes**: `POKEBOT_USE_RECURSIVE_TURN_PLANNER=1`, `POKEBOT_RTP_CHECKPOINT=…/rtp_shadow_planner.pt`, specialist/sizing; **`POKEBOT_MULTI_ENV_PER_WORKER=4` preserved**.
- **Bert**: LaunchAgent + supervised script; RTP env confirmed on live process. Probe `rl-alakazam-self:probe-rtp2-0-918766` → `target_source=recursive_turn_planner` `trusted=true` `leaf_self_play_mode=rtp-cpu-local`; `job_kinds` includes `self_play_multi`.
- **Elmo**: compose overlay `/mnt/Main/Elmo/.../rtp-r175-overlay` + production env/volumes (incl. durable `run_remote_worker.py` multi mount). Probe `…-918765` → same RTP stamps; healthy; `self_play_multi` restored after image recreate.
- **Proof receipt**: `outputs/state/alakazam-rtp-r175-remote-sync.json`.
- **Train**: remotes additive capacity 52; live `collect` advancing (self_play then public_mix). Slop Box bootstrap/RL hold still `ExecStart=/bin/false`.


## CRITICAL — RTP now used in r175 live sims (2026-08-07 ~10:39 EDT)

- **Wrong before**: env had `POKEBOT_USE_RECURSIVE_TURN_PLANNER=1` + live sidecar, but GPU-leaf collect built `PolicyAgent(model=None, leaf_backend=…)` so RTP never inited → 100% `history_policy`. Also `_build_selfplay_record` rejected non-`history_policy`.
- **Canonical path** (`experiments/recursive_turn_planner/SWAP_IN.md`): RTP needs a **local** parent model. GPU-leaf-only cannot supply `option_hidden`/`state_vec` without leaf-protocol redesign → fail-closed to **`rtp-cpu-local`** when RTP is armed (leaves may stay up unused for those jobs).
- **Fix on train** (`/home/inzi/poke-bot-agent`, backups `*.bak-rtp-r175-*`):
  - `poke_bot/pure_rl/leaf_self_play.py` — `rtp_requires_local_model()` → plan mode `rtp-cpu-local`
  - `poke_bot/agent.py` — stamp `recursive_turn_planner` `trusted=True` + factorized one-hot stages
  - `scripts/train_round_robin.py` — trust RTP source; `_worker_play` loads local model when RTP armed
- **Did not fight** MemoryHigh=120G / Max=123G or `multi_env=8`; one clean restart onto MainPID **3765383**.
- **Proof (live shard `…/shards/iter_00000.jsonl`)**: majority `target_source=recursive_turn_planner` with `trusted=true` and nonzero `decisions` (e.g. ~442 RTP / ~233 history_policy mid-collect; RTP frac ~0.65). Remainder `history_policy` ≈ Bert/Elmo without RTP env/code sync (still trusted/trainable).
- **Receipt**: `outputs/state/alakazam-rtp-r175-collect-enablement.json`
- **RL**: active, `collect:self_play` iter0 advancing. **Slop Box hold** unchanged (`ExecStart=/bin/false`).

## CRITICAL — r175 RL OOM/flap recovery (2026-08-07 ~10:38 EDT)

- **Unit**: `pokebot-final-format-alakazam-rtp-r175-rl.service` on train.
- **Failure chain**: earlier systemd `oom-kill` (~112GiB peak) → later start flaps from (1) guide_contract digest mismatch (concurrent guide refresh), (2) clean-boundary reject of `collection.multi_env_per_worker` 4→8, (3) Bert `:8766` / Elmo `:8765` down after concurrent `run_remote_worker.py` syntax break (`remote_self_play_multi_job` import) → `BetweenIterSyncError` / `required remote farm did not connect`.
- **Fixes**: MemoryHigh=**120G** / MemoryMax=**123G** (drop-in `30-memory-high-r175-gps.conf`; Max>High); registry `multi_env=8`; allowlist `collection.multi_env_per_worker` + migration_0002; repaired Bert stage + Elmo container `run_remote_worker.py` syntax; relaunched Bert LaunchAgent + Elmo docker worker (both `MULTI_ENV_PER_WORKER=4`).
- **Live**: active MainPID **3765383**, NRestarts=0; remotes elmo=36/bert=16; phase **`collect:self_play` iter0**. RTP collect enablement applied on this same PID (see section above). Slop Box hold unchanged.

## Elmo worker memory 96G (2026-08-07 ~10:36 EDT)

- `poke-bot-truenas-worker` compose `mem_limit`/`memswap_limit` **64G→96G** via declared path `deployments/persistent-workers-20260720-v1/containers/truenas-worker` (`docker-compose.host.yml` + `docker-compose.production.yml`); recreated; **healthy**; cgroup/docker stats **96GiB** max.
- Host RAM ~125Gi. **No persistent ARC cap change** (prior mistaken 16G `zfs_arc_max` via `pokebot-zfs-arc-cap.service` **reverted** to 64G/`68719476736`). No manual ARC drop/ping — TrueNAS reclaim under pressure.
- Slop Box hold unchanged. Train Alakazam RL untouched.


## Owner try-run — train multi_env=8 + remotes multi_env≈4 (2026-08-07 ~10:38 EDT)

- **Train** `pokebot-final-format-alakazam-rtp-r175-rl.service` **active**: `--multi-env-per-worker 8`; `MemoryHigh=120G` / `MemoryMax=123G` (host ~124Gi — no headroom for 132–140G Max). Cleared stale `user.control` Memory* overrides.
- Needed for restart: guide digest rebind (`guide_contract_sha256`→`83a734f1…`) + allowlist `collection.multi_env_per_worker` in deployment `scripts/train_pure_rl.py` (migration `migration_0002.json`).
- **Live**: remotes capacity 52; `self_play_pool … multi_env_batches=0` (trainer still single-game LAN dispatch while remotes up); collect `self_play` iter0 started; NRestarts=0; no OOM.
- **Bert** `:8766`: LaunchAgent `POKEBOT_MULTI_ENV_PER_WORKER=4` (+ aliases); `self_play_multi` kind wired; launchctl running; env confirmed in process.
- **Elmo** `:8765`: compose `POKEBOT_MULTI_ENV_PER_WORKER=4`; clean `run_remote_worker.py` docker-cp’d after bad patch syntax; container healthy/listening; env confirmed.
- **Packing reality**: remote env/capability ready at ≈4, but trainer does not yet submit `self_play_multi` batches — packing inactive until dispatch is wired. Slop Box hold unchanged.

## Alakazam pilot guide refresh (2026-08-07 ~09:46 EDT)

- Owner replaced `docs/deck_guides/alakazam-final-refresh-expert-brief.txt` with **ALAKAZAM FINAL-REFRESH PILOT GUIDE** (matchup-specific Xerosic/Hammer/Shaymin/Battle Cage notes; Nighttime Mine guidance removed).
- Checksum binds bumped: `config/deck_guides/alakazam-final-refresh.yaml` expert_writeup sha `035a02ab…` / 1795 words; `state/final_format_alakazam_guide_ready_r79.json` contract+writeup digests.
- Guide still binds measurement digest `sha256:25878108…` (4× Alakazam modal list). Live r175 pilot CSV `alakazam-owner-rtp-pilot-r175` is a **different** 60-card list (3× Alakazam / 4× Dudunsparce 1264) — **not** silently rebased; Kaggle milestones keep forcing pilot.
- RL untouched; Slop Box hold unchanged; combo head remains OFF.

## CRITICAL — dash stale progress + 3080/legacy/GPS (2026-08-07 ~09:53 EDT)

### Dash progress before → after (Bert `:8780`)
- **Before**: age ~1123 s; `collect:public_mix` **886/7172** @ 3.28 s/game; sched `legacy_or_starting`; GPU0 `OUT OF FLEET`.
- **After (matched live public_mix)**: age ~22 s; **3115/7172** @ 6.81 game/s; sched `mid_iter`; GPU0 `PRODUCTION · 12 policy leaf replicas` (DELTA 0 vs train status).
- **After (current, honest mid-recollect)**: age ~26 s; `collect:self_play` **0/1024**; sched `mid_iter`; GPU0/1 PRODUCTION ~99%/83%.

### Stale `/api/status` root cause + fix
- Bert `pokebot-dashboard/v1/server.py` SSH snapshot **`timeout=15`**, but full train snapshot takes **~20 s** under load → refreshes timed out → cache froze.
- **Fix (dash-only)**: timeout **15→45**; `launchctl kickstart` `com.pokebot.training-dashboard`. Local `dashboard/lan/server.py` mirrored.

### 3080 out-of-fleet
- **Runtime**: never out — leaves GPU0=12 / GPU1=30. Overnight `hardware.py` preferred_index copy did not eject 3080.
- **Dash mislabel**: `_is_curriculum_service_unit` rejected `…-alakazam-rtp-r175-rl.service` (no `*-h10*`) → overlay `active=true` with `leaf_gpu0=0` → OUT OF FLEET.
- **Fix**: recognize `-rtp-r175-rl` + overlay refreshes `curriculum_worker_state`.

### Scheduler "legacy"
- Exact string: `scheduler_queue_state` → `mode: "legacy_or_starting"` when `endpoint_owned_queues` absent.
- **Runtime truth**: additive **`mid_iter`** (never legacy). Fixed to emit `mode=mid_iter`.

### GPS
- **Top cause**: cgroup MemoryHigh reclaim thrashing (~110 GiB > High ~96 GiB) → `cpu_hot`, GPU idle, `multi_env=1` despite launch `--multi-env-per-worker 4`.
- Raising High to **112G (=Max)** briefly restored GPS (~4–10 game/s) then **oom-killed** at 112 GiB peak (NRestarts=1). Interrupted public_mix quarantined as `attempt_0025`; no durable shard → fail-closed **recollect iter0** (`multi_env=4` again).
- **Corrected limits (no extra restart)**: `MemoryHigh=108G` / `MemoryMax=120G` via `30-memory-high-r175-gps.conf`; cleared transient `user.control/…/50-MemoryHigh.conf`.

### Artifacts / safety
- Train dash patches: workspace + `final-format-marnie-postupload-r136` `dashboard_snapshot.py`.
- Receipt: `outputs/state/alakazam-rtp-r175-fleet-scheduler-gps-recovery.json` (v2).
- RL active MainPID `3673358`, collecting self_play iter0. Slop Box CE hold unchanged.

## CRITICAL — Alakazam r175 Crustle ≥512 public-mix floor + Bert/Elmo sync (2026-08-07 ~09:03 EDT)

- **Owner hard assert**: every Alakazam r175 loop must schedule ≥512 games vs our Crustle specialist `specialist-crustle-final-format-h10-7efd8d4113e7` (checkpoint `sha256:7efd8d4113e7…`, content `sha256:359e3b4f…`), in addition to ≥1024 Grimmsnarl `specialist-marnie-final-format-h10-f20efb20f5c3` / `sha256:f20efb20f5c3…` and 1024 mirrors → 8196 fill.
- **Bert gap (real)**: stage worker `poke-bot-agent-h10-r79-stage` had the Crustle package on disk but **missing from `baselines/manifest.json`** → `baseline id is absent from local manifest` on `192.168.1.158:8766` during public_mix. Synced package+manifest (stage + workspace); kickstarted `com.pokebot.remote-worker-8766-h10-r80`.
- **Elmo**: synced Crustle (+ Marnie H10) into main/baseline-sync/Elmo manifests; refreshed `poke-bot-truenas-worker` so the container remounts (was also failing crustle-absent until restart).
- **Floor wiring (canonical knobs)**:
  - Sidecar: `outputs/final_format_alakazam_rtp_r175/runtime/owner_public_mix_pin_floors_r175.json` (`poke_bot.owner_public_mix_pin_floors/v1`)
  - Systemd drop-in: `~/.config/systemd/user/pokebot-final-format-alakazam-rtp-r175-rl.service.d/20-owner-public-mix-pin-floors.conf` → `POKEBOT_OWNER_PUBLIC_MIX_PIN_FLOORS=…`
  - Registry pins: `specialists.alakazam.owner_crustle_pin` floor 512 + `owner_grimmsnarl_pin` floor 1024 in `specialist_runtime_registry_h10_r175.json`; hard-swap JSON updated.
  - Enforcement: deployment `train_pure_rl.py` reassigns `diverse_public` jobs to meet floors (practice roster untouched).
- **Live iter0 evidence**: `collection_plans/iter_00000.owner_public_mix_pin_floors.json` — Crustle **512/512** (converted 383), Grimmsnarl **1024/1024** (converted 895). Unit active MainPID `3609393`, NRestarts=0, phase `collect:self_play` after rebuild.
- **Receipts**: `outputs/state/alakazam-rtp-r175-crustle-baseline-fleet-sync.json`, `outputs/state/alakazam-rtp-r175-crustle-public-mix-floor-arm.json`.
- **Slop Box CE hold**: unchanged.

## CRITICAL FIX — Alakazam r175 self-play retention flap (2026-08-07 ~08:47 EDT)
 (2026-08-07 ~08:47 EDT)

- **Symptom**: `pokebot-final-format-alakazam-rtp-r175-rl.service` flapping (~every 20m, NRestarts→21) with  
  `RuntimeError: exact self-play retention failed after bounded replacements: retained=876/1024`  
  during `collect:self_play_refill` iter=0 (never completed iter0).
- **Root cause**: Unit `PYTHONPATH=/home/inzi/poke-bot-agent` loaded a stale  
  `poke_bot/pure_rl/hardware.py` whose `pick_leaf_server_index()` lacked `preferred_index`,  
  while `batched_infer.py` already passed it. PolicyAgent fail-closed  
  `TypeError: pick_leaf_server_index() got an unexpected keyword argument 'preferred_index'`  
  (~44k log hits) → most self-play games produced no retainable record. Exact-1024 target  
  and 4 bounded replacement rounds could not close the 148-game gap.
- **Fix applied (train)**: copied preferred_index-capable `hardware.py` from deployment twin  
  `final-format-alakazam-fusion-v3-r104` onto `/home/inzi/poke-bot-agent/poke_bot/pure_rl/hardware.py`  
  (backup beside it); `systemctl --user restart` of the flapping RL unit. Did **not** lower 1024.  
  Slop Box CE hold untouched.
- **Verified**: after restart, `preferred_index` / FAIL-CLOSED errors = 0; shard has **1024/1024**  
  unique self-play `collection_job_index` values; unit advanced to `collect:public_mix`  
  (MainPID `3561037`, NRestarts=0). Receipt:  
  `outputs/state/alakazam-rtp-r175-self-play-retention-fix.json`.
- **Soft follow-up (non-blocking for this flap)**: Bert public-mix retries warn  
  `baseline id is absent from local manifest: specialist-crustle-final-format-h10-7efd8d4113e7`  
  — watch if public_mix retention later stresses; not the self-play root cause.

## Owner order — Alakazam Kaggle copy 2 (2026-08-07 ~12:37Z)

- **Action**: second Kaggle copy of bootstrap model `7480d81c54b1` as Alakazam (`first_if_allowed`, owner pilot deck `alakazam-owner-rtp-pilot-r175`; not 55188658).
- **Model**: `outputs/pure_rl/_protected/models/final-format-alakazam-rtp-r175-expert-bootstrap-v1/model.pt` · sha256 `7480d81c54b1b98955108401fc04c82e93b6afe626a70c1b52fd467cc0cb704b`
- **Bundle**: `outputs/submissions/final-format-alakazam-rtp-r175-expert-bootstrap/copy-2/submission.tar.gz` (reused copy-1 build `sha256:335d91c0f3f2…`)
- **Copy**: `2/2` · label `alakazam training milestone iter 0 copy 2/2 first 7480d81c54b1`
- **Submission id**: `55324802` (watcher uploaded; COMPLETE; publicScore 600.0). Prior copy 1 = `55315274`.
- **Receipt**: `outputs/state/alakazam-rtp-r175-kaggle-copy2-enqueued.json`
- **RL**: left alone — `pokebot-final-format-alakazam-rtp-r175-rl.service` remained `active`/`running` throughout (no restart).

## Dash-only Alakazam r175 rebind — 2026-08-07 ~08:38 EDT

- **Before `/api/status`**: `training.display_name=Slop Box`, `specialist_id=teal-mask-ogerpon-ex`, `phase=stopped:rehearsal:chao_hard_gate_remeasure`, run `final_format_slop_box_chao_hard_ce`.
- **After**: `training.display_name=Alakazam`, `specialist_id=alakazam`, `phase=collect:self_play_refill`, run `final_format_alakazam_rtp_r175_i_v6_8k`, iter `0`, `commit_count=0` (honest; no durable commits yet).
- **Canonical typed source**: `state/alakazam-rtp-owner-hard-swap-r175.json` (armed, rev 175).
- **Selector / progress files changed (train)**:
  - `~/.config/pokebot/specialist_runtime.env` → `POKEBOT_ACTIVE_SPECIALIST=alakazam` (was crustle)
  - `state/specialists.yaml` → `current.active_specialist=alakazam` + r175 `active_run`; specialists row `alakazam.active=true` / `teal-mask-ogerpon-ex.active=false` (restored after a truncated rewrite; full roster preserved)
  - `scripts/dashboard_snapshot.py` and deployment twin under `final-format-marnie-postupload-r136` → r175 progress projection preferred; Chao-hard suppressed while hard-swap armed
  - Receipt: `outputs/state/alakazam-rtp-r175-dashboard-rebind-r175.json`
- **RL untouched**: `pokebot-final-format-alakazam-rtp-r175-rl.service` MainPID stayed `3520668` active/collecting; Slop Box CE hold `ExecStart=/bin/false` preserved; bind-keeper already stopped.


## Every-5 refresh+Kaggle cadence (r175) — 2026-08-07 ~01:10 EDT

- **Armed**: **partial → now yes** (gap found and sidecar staged; RL not restarted)
- **Expert refresh already live**: `pokebot-final-format-alakazam-rtp-r175-rl.service` MainPID `2428302` / trainer `2428453` launch argv includes `--expert-rehearsal-every 5` and `--expert-rehearsal-epochs 5` (fires at start of iters 5/10/15/…; registry `isolated_refresh_contract.expert_rehearsal_every=5`).
- **Kaggle gap confirmed**: legacy `pokebot-final-format-alakazam-milestone-submissions.timer` still targets **r79** run-dir `final_format_alakazam_r79_h10_i_v6_8k` (r97 `5n+4` cadence) — **not** r175. Orchestrator only wrote a one-shot bootstrap `queued_request` (no recurring loop).
- **Fix applied (sidecar, no RL interrupt)**:
  - Script: `scripts/stage_alakazam_rtp_r175_milestone_submissions.py` (on train `/home/inzi/poke-bot-agent/...`)
  - Timer: `pokebot-final-format-alakazam-rtp-r175-milestone-submissions.timer` (enabled/active)
  - Service: `pokebot-final-format-alakazam-rtp-r175-milestone-submissions.service` (oneshot OK; `staged: []` until commits exist)
  - Eligible: durable commits **5,10,15,…,300** **and** `history[].expert_rehearsal` present (refresh-then-Kaggle fail-closed)
  - Deck: owner pilot `alakazam-owner-rtp-pilot-r175` / `first_if_allowed`; queue via existing `pokebot-kaggle-submission-queue.service`
  - Arming receipt: `outputs/state/alakazam-rtp-r175-every5-refresh-kaggle-armed.json`
- **Note**: live r175 measurement-deck digest is still `specialist_representatives.v1` (`25878108…`); milestone sidecar **forces** pilot (`660c1274…`) per owner r175 (same as bootstrap submit `55315274`).
- **RL proof at arm**: unit active, MainPID unchanged `2428302`, collect `self_play_refill iter=0` progressing.

## Kaggle enqueue status (r175 bootstrap) — 2026-08-07 05:06 UTC

- **Status**: submitted (canonical queue + watcher upload)
- **Model**: `final-format-alakazam-rtp-r175-expert-bootstrap-v1/model.pt`
- **Model sha256**: `sha256:7480d81c54b1b98955108401fc04c82e93b6afe626a70c1b52fd467cc0cb704b`
- **Deck lineage**: Alakazam owner RTP pilot `alakazam-owner-rtp-pilot-r175` (`decks/archetype-samples/alakazam-owner-rtp-pilot-r175.csv`, sha256 `1705f0f4db0c…`) — **not** Slop Box Cox/Chao `55188658`
- **Turn order**: `first_if_allowed`
- **Canonical queue**: `outputs/state/kaggle-submission-queue.json`
- **Label**: `alakazam training milestone iter 0 copy 1/1 first 7480d81c54b1`
- **Bundle**: `outputs/submissions/final-format-alakazam-rtp-r175-expert-bootstrap/copy-1/submission.tar.gz` (`sha256:335d91c0f3f2…`)
- **Kaggle submission id**: `55315274` (PENDING at enqueue time)
- **Receipts**: `outputs/state/alakazam-rtp-r175-kaggle-queue-request.json` (promoted from `queued_request`), `outputs/state/alakazam-rtp-r175-kaggle-enqueued.json`
- **Watcher**: `pokebot-kaggle-submission-queue.service` (active; automatic one-shot auth consumed)
- **RL**: left alone — `pokebot-final-format-alakazam-rtp-r175-rl.service` remained active/collecting during enqueue


Updated: 2026-08-07  
Repository: `/Users/tsinzitari/Documents/poke-agent-codex`  
Current owner contract: `GOAL.md`, revision **175** (authoritative)  
Live training mode: **Alakazam RTP hard-swap overnight loop**

## Morning status (2026-08-07 ~00:58 EDT / 04:58 UTC)

Babysit outcome: **RL/self-play is RUNNING** (iter 0 collecting).

### Phase reached
1. **CE complete** — 25/25 epochs; best `epoch_15.pt` (`sha256:7480d81c54b1…`); ready receipt written.
2. **Kaggle** — bootstrap `first_if_allowed` **submitted** as id `55315274` (PENDING). Deck lineage = owner pilot `alakazam-owner-rtp-pilot-r175` (not 55188658). Queue: `outputs/state/kaggle-submission-queue.json`; receipts: `alakazam-rtp-r175-kaggle-queue-request.json` + `alakazam-rtp-r175-kaggle-enqueued.json`. Watcher `pokebot-kaggle-submission-queue.service` uploaded.
3. **RL up** — `pokebot-final-format-alakazam-rtp-r175-rl.service` **active/running**.

### PIDs / units
- Orchestrator: finished (failed earlier at RL start boundary; CE+Kaggle receipts already done).
- RL unit MainPID **2428302** (`launch_pure_rl.py`)
- Trainer PID **2428453** (`train_pure_rl.py`); monitor **2428455**; watcher **2428454**
- CE bootstrap (completed): family `final-format-alakazam-rtp-r175-expert-bootstrap-v1`

### Owner self-play binds (verified in live collect + registry)
- `games_per_iteration=8196`, `self_play=1024`, `public_mix=7172`
- Grimmsnarl pin `f20efb20f5c3…` / floor **1024**/set (`specialist-marnie-final-format-h10-f20efb20f5c3`)
- `iteration_ceiling=300`; expert rehearsal every 5; combo loss **0.0** (guide 0.05)
- First log evidence:  
  `collect iter=0 self_play=1024 … public_mix=7172` then  
  `collect:self_play iter=0: 79%|…| 810/1024`

### Recoveries applied overnight (train)
- CE freeze previously failed on inherited combo fusion; later finalize wrote ready+family model.
- Missing RTP live sidecar cut+published: `outputs/rtp_fleet/alakazam-r175.live/rtp_shadow_planner.pt` (receipt `outputs/state/alakazam-rtp-r175-bootstrap-cut.json`).
- RL profile drop-in:  
  `~/.config/systemd/user/pokebot-final-format-alakazam-rtp-r175-rl.service.d/10-r175-profile-compat.conf`  
  (`POKEBOT_COMBO_STATE_HEAD_ENABLED=1` for architecture match; learning still off via loss weight 0; matchup registry path; history context).
- Seed-namespace offsets confirmed/patched for iter_max 300 (`FORMAL=31M`, `RESEARCH=62M`).
- Slop Box CE hold unchanged (`ExecStart=/bin/false`).

### Soft follow-ups (non-blocking)
- ~~Promote Kaggle `queued_request` → authorized queue entry / capture submission id.~~ **Done** — submission id `55315274`.
- Confirm Grimmsnarl floor realized in public-mix set composition once iter0 shard commits.

## Revision 175 owner boundary (active)

- Hard-swap off Jul24–Aug5 Slop Box CE recovery (stays held/`ExecStart=/bin/false`).
- Do **not** resurrect failed `pokebot-final-format-alakazam-r79-h10` (failed since 2026-08-02).
- Active loop:
  1. Expert refresh on last 5 Alakazam days **2026-08-01..2026-08-05**
  2. CE rebootstrap from Alakazam checkpoint `iter_00020.pt` / family `final-format-alakazam-r79-h10-refresh-v1`
  3. Kaggle `first_if_allowed`
  4. Self-play/public-mix RL with RTP: **1024 mirrors**, fill to **8196**, **≥1024 Grimmsnarl/set**
  5. Every 5 iterations: expert refresh → Kaggle → continue
- Iteration ceiling: **300**
- Pilot deck list id: `alakazam-owner-rtp-pilot-r175`  
  path: `decks/archetype-samples/alakazam-owner-rtp-pilot-r175.csv`
- Grimmsnarl pin (unique):  
  `sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381`  
  package: `specialist-marnie-final-format-h10-f20efb20f5c3`  
  checkpoint: `outputs/pure_rl/final_format_marnie_r104_h10_i_v6_8k/checkpoints/iter_00007.pt`
- Heads: all non-combo heads learning; guide ON (0.05 directional); **combo OFF**
- Typed canonical source: `state/alakazam-rtp-owner-hard-swap-r175.json`
- Units:
  - `pokebot-final-format-alakazam-rtp-r175-orchestrator.service`
  - `pokebot-final-format-alakazam-rtp-r175-rl.service`
- Loop state: `outputs/state/alakazam-rtp-owner-hard-swap-loop-r175.json`
- Orchestrator log: `outputs/final_format_alakazam_rtp_r175/logs/overnight-orchestrator.log`

## Spelling / count notes

- Owner `Xerosic's Mechinations` → data `Xerosic's Machinations` (id 1197)
- Owner `Basic Psychic Energy` → data `Basic {P} Energy` (id 5)
- Curly apostrophes in Boss's Orders / Lana's Aid
- Owner wrote 2× Dudunsparce but Pokémon(19)/60-card math requires 3×; bound **3× Dudunsparce**

## Safety / ops

1. Read `AGENTS.md` and `GOAL.md` completely before acting.
2. Preserve worktree; no destructive git.
3. systemd --user / launchctl only; never process-tree kill interactive sessions.
4. Do not restart healthy training merely to reconcile metadata.
5. Crustle remains abandoned (r170); do not delete Crustle artifacts.
