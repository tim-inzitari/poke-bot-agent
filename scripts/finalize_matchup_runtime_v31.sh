#!/usr/bin/env bash
# Complete the iteration-26 matchup-runtime transaction and resume iteration 27.
set -euo pipefail

ROOT=/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v31-matchup-runtime
PY=/home/inzi/miniconda3/envs/poke-bot-agent/bin/python
RUN=/home/inzi/poke-bot-agent/outputs/pure_rl/pure_rl_alakazam_temporal1_8k_teacher_v16_20260721
PARENT="$RUN/checkpoints/iter_00026.pt"
AUTH=/home/inzi/poke-bot-agent/outputs/state/alakazam-matchup-adapter-iter26-v31-authorization.json
FIT=/home/inzi/poke-bot-agent/outputs/matchup_adapters/alakazam-iter26-all22-v31/final.pt
MERGED="$RUN/checkpoints/iter_00026_matchup_v31.pt"
TREE_SOURCE=/home/inzi/poke-bot-agent/outputs/state/public-matchup-tree-calibrated-v31.json
TREE_RUNTIME=/home/inzi/poke-bot-agent/outputs/state/public-matchup-tree-runtime-v31.json
BOUNDARY_RECEIPT=/home/inzi/poke-bot-agent/outputs/state/alakazam-matchup-runtime-iter26-v31.json
REMOTE_MARKER=/home/inzi/poke-bot-agent/outputs/state/matchup-runtime-activation-v31.json
PRODUCTION_READY=/home/inzi/poke-bot-agent/outputs/state/matchup-runtime-v31-production-ready.json
STATUS=/home/inzi/poke-bot-agent/outputs/state/matchup-runtime-v31-finalizer.json
LOG=/home/inzi/poke-bot-agent/outputs/logs/matchup-runtime-v31-finalizer.log
PRODUCTION=pokebot-pure-rl-alakazam.service
FIT_UNITS=(
  pokebot-matchup-adapter-v31b.service
  pokebot-matchup-adapter-v31.service
  pokebot-matchup-adapter-v31-recovery.service
)
FLEET_UNITS=(
  pokebot-adapter-fleet-blackwell.service
  pokebot-adapter-fleet-3080.service
  pokebot-adapter-fleet-finalizer.service
)
DROPIN_SOURCE="$ROOT/.staging/zzzzzzzzzzzzzzzzzz-v31-matchup-runtime.conf"
DROPIN_TARGET=/home/inzi/.config/systemd/user/pokebot-pure-rl-alakazam.service.d/zzzzzzzzzzzzzzzzzz-v31-matchup-runtime.conf
HANDOFF_PATCH=/home/inzi/poke-bot-agent/outputs/state/matchup-runtime-v31-handoff-patch

exec >>"$LOG" 2>&1

status() {
  local phase="$1" detail="${2:-}"
  PHASE="$phase" DETAIL="$detail" STATUS="$STATUS" "$PY" - <<'PY'
import json, os, tempfile, time
from pathlib import Path
p=Path(os.environ["STATUS"])
p.parent.mkdir(parents=True, exist_ok=True)
payload={"schema":"poke_bot.matchup_runtime_finalizer/v1","phase":os.environ["PHASE"],"detail":os.environ["DETAIL"],"updated_at":time.time()}
fd,tmp=tempfile.mkstemp(prefix="."+p.name+".",dir=p.parent)
with os.fdopen(fd,"w") as f:
    json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,p)
PY
  printf '[%s] phase=%s %s\n' "$(date -Is)" "$phase" "$detail"
}

cleanup_error() {
  code=$?
  status failed "exit=$code line=${BASH_LINENO[0]:-unknown}"
  exit "$code"
}
trap cleanup_error ERR

# On a later login/reboot both enabled units may be started by default.target.
# A previously completed handoff is a successful no-op; any other active
# production state remains a hard fail-closed violation.
if [[ "$(systemctl --user is-active "$PRODUCTION" 2>/dev/null || true)" == active ]]; then
  completed_phase="$($PY - "$STATUS" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    print(str(json.loads(path.read_text()).get("phase") or ""))
except Exception:
    print("")
PY
)"
  if [[ -s "$PRODUCTION_READY" && -s "$BOUNDARY_RECEIPT" ]]; then
    "$PY" "$ROOT/scripts/assert_matchup_runtime_production_ready.py" \
      --receipt "$PRODUCTION_READY"
    if [[ "$completed_phase" != complete ]]; then
      status complete "recovered=production_active production_ready=verified"
    fi
    printf '[%s] already complete; production is active\n' "$(date -Is)"
    trap - ERR
    exit 0
  fi
  status failed "production_active_before_completed_matchup_handoff"
  exit 21
fi
cd "$ROOT"
status waiting_for_exact_adapter_fit "target=25 epochs parent=iter_00026"
while [[ ! -s "$FIT" ]]; do
  fit_state=inactive
  for fit_unit in "${FIT_UNITS[@]}"; do
    candidate_state="$(systemctl --user show "$fit_unit" -p ActiveState --value 2>/dev/null || true)"
    if [[ "$candidate_state" == active || "$candidate_state" == activating ]]; then
      fit_state="$candidate_state:$fit_unit"
      break
    fi
  done
  if [[ "$fit_state" != active:* && "$fit_state" != activating:* ]]; then
    fleet_state=inactive
    for fleet_unit in "${FLEET_UNITS[@]}"; do
      candidate_state="$(systemctl --user show "$fleet_unit" -p ActiveState --value 2>/dev/null || true)"
      if [[ "$candidate_state" == active || "$candidate_state" == activating ]]; then
        fleet_state="$candidate_state:$fleet_unit"
        break
      fi
    done
    if [[ "$fleet_state" == active:* || "$fleet_state" == activating:* ]]; then
      status waiting_for_exact_adapter_fleet "state=$fleet_state target=25 epochs"
      sleep 20
      continue
    fi
    status adapter_fit_not_running "fit=${fit_state:-unknown} fleet=${fleet_state:-unknown}"
    exit 20
  fi
  sleep 20
done

status waiting_for_exact_fit_commit
fit_exit_deadline=$(( $(date +%s) + 900 ))
while :; do
  fit_running=0
  for fit_unit in "${FIT_UNITS[@]}" "${FLEET_UNITS[@]}"; do
    candidate_state="$(systemctl --user show "$fit_unit" -p ActiveState --value 2>/dev/null || true)"
    if [[ "$candidate_state" == active || "$candidate_state" == activating ]]; then
      fit_running=1
      break
    fi
  done
  [[ "$fit_running" -eq 0 ]] && break
  if [[ "$(date +%s)" -ge "$fit_exit_deadline" ]]; then
    status failed "final.pt exists but fitter did not commit/exit within 900s"
    exit 22
  fi
  sleep 2
done

status validating_exact_fit_commit
FIT="$FIT" AUTH="$AUTH" PARENT="$PARENT" "$PY" - <<'PY'
from pathlib import Path
from poke_bot import checkpoint
from poke_bot.matchup_adapter_activation import (
    validate_adapter_training_authorization,
)
import os

fit_path = Path(os.environ["FIT"]).resolve()
auth_path = Path(os.environ["AUTH"]).resolve()
parent_path = Path(os.environ["PARENT"]).resolve()
validate_adapter_training_authorization(auth_path, parent_checkpoint=parent_path)
saved = checkpoint.load_checkpoint(fit_path, map_location="cpu")
extra = dict(saved.get("extra") or {})
state = dict(extra.get("streaming_matchup_adapter_state") or {})
config = dict(extra.get("streaming_matchup_adapter_train_config") or {})
if not (
    int(saved.get("epoch", -1)) == 25
    and int(state.get("epoch", -1)) == 25
    and state.get("complete") is True
    and int(state.get("train_sequences_consumed", -1)) == 0
    and int(config.get("epochs", -1)) == 25
    and config.get("exact_epochs") is True
    and extra.get("matchup_adapter_fit_complete") is True
    and extra.get("matchup_adapters_runtime_enabled") is False
    and Path(str(extra.get("matchup_adapter_activation_receipt") or "")).resolve()
        == auth_path
    and str(extra.get("matchup_adapter_parent_checkpoint_digest") or "")
        == checkpoint.checkpoint_digest(parent_path)
):
    raise SystemExit("final adapter checkpoint is not the exact committed 25-epoch fit")
print("exact_fit_commit_ok", checkpoint.checkpoint_digest(fit_path))
PY

status installing_post_fit_authorization_compatibility
(
  cd "$HANDOFF_PATCH"
  sha256sum --check MANIFEST.sha256
)
# These two files participate in the fitter's immutable implementation
# identity, so they are deliberately installed only after final.pt exists.
# The completed checkpoint therefore records the exact implementation that
# produced it, while the handoff/production source can accept the later
# clean-boundary rehearsal authorization it was trained under.
install -m 0644 "$HANDOFF_PATCH/matchup_adapter_activation.py" \
  "$ROOT/poke_bot/matchup_adapter_activation.py"
install -m 0644 "$HANDOFF_PATCH/train.py" "$ROOT/poke_bot/train.py"
install -m 0644 "$HANDOFF_PATCH/archetypes.py" "$ROOT/poke_bot/archetypes.py"
install -m 0755 "$HANDOFF_PATCH/apply_matchup_runtime_at_boundary.py" \
  "$ROOT/scripts/apply_matchup_runtime_at_boundary.py"
install -m 0644 "$HANDOFF_PATCH/test_matchup_adapter_activation.py" \
  "$ROOT/tests/test_matchup_adapter_activation.py"
install -m 0644 "$HANDOFF_PATCH/test_matchup_adapters.py" \
  "$ROOT/tests/test_matchup_adapters.py"
install -m 0644 "$HANDOFF_PATCH/test_ladder_archetype_registry.py" \
  "$ROOT/tests/test_ladder_archetype_registry.py"
install -m 0644 "$HANDOFF_PATCH/test_apply_matchup_runtime_boundary.py" \
  "$ROOT/tests/test_apply_matchup_runtime_boundary.py"
"$PY" -m py_compile \
  poke_bot/matchup_adapter_activation.py \
  poke_bot/train.py \
  poke_bot/archetypes.py \
  scripts/apply_matchup_runtime_at_boundary.py

status validating_adapter_fit
"$PY" -m pytest -q \
  tests/test_matchup_adapter_streaming_trainer.py \
  tests/test_matchup_adapter_activation.py \
  tests/test_matchup_adapters.py \
  tests/test_ladder_archetype_registry.py \
  tests/test_apply_matchup_runtime_boundary.py \
  tests/test_public_matchup_router.py \
  tests/test_remote_leaf_routing.py \
  tests/test_matchup_runtime_reboot_gate.py \
  tests/test_pure_rl_lineage_retention.py \
  tests/test_strong_public_gate.py \
  tests/test_pure_rl_recovery_and_scheduling.py

if [[ ! -s "$MERGED" ]]; then
  status merging_adapter_fit "output=$MERGED"
  "$PY" scripts/merge_dormant_matchup_adapters.py \
    --parent-checkpoint "$PARENT" \
    --adapter-checkpoint "$FIT" \
    --activation-receipt "$AUTH" \
    --output "$MERGED"
fi

if [[ ! -s "$TREE_RUNTIME" ]]; then
  status activating_precision_gated_tree
  "$PY" scripts/activate_public_matchup_tree.py \
    --source "$TREE_SOURCE" \
    --checkpoint "$MERGED" \
    --output "$TREE_RUNTIME" \
    --min-precision 0.93 \
    --min-support 5000 \
    --consecutive-required 2
fi

if [[ ! -s "$REMOTE_MARKER" ]]; then
  status building_remote_activation_marker
  "$PY" scripts/build_remote_matchup_runtime_marker.py \
    --checkpoint "$MERGED" \
    --tree "$TREE_RUNTIME" \
    --output "$REMOTE_MARKER"
fi

status validating_clean_boundary
"$PY" scripts/apply_matchup_runtime_at_boundary.py \
  --run-dir "$RUN" \
  --merged-checkpoint "$MERGED" \
  --parent-checkpoint "$PARENT" \
  --activation-authorization "$AUTH" \
  --runtime-tree "$TREE_RUNTIME" \
  --receipt "$BOUNDARY_RECEIPT" \
  --expected-last-iteration 26 \
  --validate-only

status redeploying_remote_code_and_checkpoint
REMOTE_BOOTSTRAP_CHECKPOINT="$MERGED" \
REMOTE_PREFLIGHT_PROFILE=none \
  scripts/redeploy_remote_self_play.sh "$LOG.remote-redeploy"

status installing_remote_runtime_bundles
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
elmo_tree_tmp="/tmp/public-matchup-tree-runtime-v31-$stamp.json"
elmo_marker_tmp="/tmp/matchup-runtime-activation-$stamp.json"
scp -q "$TREE_RUNTIME" "elmo:$elmo_tree_tmp"
scp -q "$REMOTE_MARKER" "elmo:$elmo_marker_tmp"
ssh elmo "set -e; sudo -n docker stop poke-bot-truenas-worker >/dev/null; d=/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/checkpoint; sudo -n cp -p \"\$d/public-matchup-tree-runtime-v31.json\" \"\$d/public-matchup-tree-runtime-v31.json.before-$stamp\" 2>/dev/null || true; sudo -n cp -p \"\$d/matchup-runtime-activation.json\" \"\$d/matchup-runtime-activation.json.before-$stamp\" 2>/dev/null || true; sudo -n install -m 0444 \"$elmo_tree_tmp\" \"\$d/public-matchup-tree-runtime-v31.json\"; sudo -n install -m 0444 \"$elmo_marker_tmp\" \"\$d/matchup-runtime-activation.json\"; rm -f \"$elmo_tree_tmp\" \"$elmo_marker_tmp\"; sudo -n docker start poke-bot-truenas-worker >/dev/null"

bert_tree_tmp="/tmp/public-matchup-tree-runtime-v31-$stamp.json"
bert_marker_tmp="/tmp/matchup-runtime-activation-$stamp.json"
scp -q "$TREE_RUNTIME" "bert.local:$bert_tree_tmp"
scp -q "$REMOTE_MARKER" "bert.local:$bert_marker_tmp"
ssh bert.local "set -e; domain=gui/\$(id -u); target=\$domain/com.pokebot.remote-worker-8766; launchctl print \"\$target\" >/dev/null; d=/Users/tsinzitari/workspace/poke-bot-agent/outputs/checkpoints; cp -p \"\$d/public-matchup-tree-runtime-v31.json\" \"\$d/public-matchup-tree-runtime-v31.json.before-$stamp\" 2>/dev/null || true; cp -p \"\$d/matchup-runtime-activation.json\" \"\$d/matchup-runtime-activation.json.before-$stamp\" 2>/dev/null || true; install -m 0444 \"$bert_tree_tmp\" \"\$d/public-matchup-tree-runtime-v31.json\"; install -m 0444 \"$bert_marker_tmp\" \"\$d/matchup-runtime-activation.json\"; rm -f \"$bert_tree_tmp\" \"$bert_marker_tmp\"; nohup launchctl kickstart -k \"\$target\" >/dev/null 2>&1 </dev/null &"

status verifying_remote_runtime
EXPECTED_TREE="$($PY -c 'import hashlib,sys; print("sha256:"+hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$TREE_RUNTIME")" \
EXPECTED_CHECKPOINT="$($PY -c 'from poke_bot.checkpoint import checkpoint_digest; import sys; print(checkpoint_digest(sys.argv[1]))' "$MERGED")" \
EXPECTED_MARKER="$($PY -c 'import hashlib,sys; print("sha256:"+hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$REMOTE_MARKER")" \
EXPECTED_ROUTES="$($PY -c 'import json,sys; print(json.dumps(sorted(json.load(open(sys.argv[1]))["accepted_archetype_ids"]),separators=(",",":")))' "$REMOTE_MARKER")" \
"$PY" - <<'PY'
import json,os,time
from poke_bot.remote_jobs import RemoteJobClient,parse_endpoint
expected_tree=os.environ["EXPECTED_TREE"]
expected_checkpoint=os.environ["EXPECTED_CHECKPOINT"]
expected_marker=os.environ["EXPECTED_MARKER"]
expected_routes=json.loads(os.environ["EXPECTED_ROUTES"])
for endpoint in ("192.168.1.143:8765","bert.local:8766"):
    deadline=time.monotonic()+300
    last=None
    while time.monotonic()<deadline:
        try:
            c=RemoteJobClient(*parse_endpoint(endpoint),timeout_s=10,connect_timeout_s=5,control_timeout_s=10)
            info=c.connect(); health=c.health(); c.close()
            runtime=health.get("matchup_runtime") or {}
            if (health.get("ok") is True and info.checkpoint_digest==expected_checkpoint and
                runtime.get("checkpoint_digest")==expected_checkpoint and
                runtime.get("marker_digest")==expected_marker and
                runtime.get("tree_digest")==expected_tree and
                sorted(runtime.get("accepted_archetype_ids") or [])==expected_routes and
                runtime.get("continuous_reevaluation") is True and
                runtime.get("one_route_per_decision") is True and
                runtime.get("unknown_route_exact_bypass") is True and
                int(runtime.get("consecutive_required") or 0)==2):
                print(endpoint,"runtime_ok",len(runtime["accepted_archetype_ids"])); break
            last=(info.checkpoint_digest,health)
        except Exception as exc:
            last=repr(exc)
        time.sleep(2)
    else:
        raise SystemExit(f"remote runtime verification failed {endpoint}: {last}")
PY

status publishing_boundary_learner
"$PY" scripts/apply_matchup_runtime_at_boundary.py \
  --run-dir "$RUN" \
  --merged-checkpoint "$MERGED" \
  --parent-checkpoint "$PARENT" \
  --activation-authorization "$AUTH" \
  --runtime-tree "$TREE_RUNTIME" \
  --receipt "$BOUNDARY_RECEIPT" \
  --expected-last-iteration 26

status installing_production_v31
test -s "$DROPIN_SOURCE"
install -m 0644 "$DROPIN_SOURCE" "$DROPIN_TARGET"
systemctl --user daemon-reload
systemd-analyze --user verify "$PRODUCTION"

status publishing_production_ready
PRODUCTION_READY="$PRODUCTION_READY" BOUNDARY_RECEIPT="$BOUNDARY_RECEIPT" \
MERGED="$MERGED" TREE_RUNTIME="$TREE_RUNTIME" REMOTE_MARKER="$REMOTE_MARKER" \
DROPIN_TARGET="$DROPIN_TARGET" "$PY" - <<'PY'
import hashlib, json, os, tempfile
from pathlib import Path

def digest(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + value

paths = {
    "boundary_receipt": Path(os.environ["BOUNDARY_RECEIPT"]).resolve(),
    "merged_checkpoint": Path(os.environ["MERGED"]).resolve(),
    "runtime_tree": Path(os.environ["TREE_RUNTIME"]).resolve(),
    "remote_marker": Path(os.environ["REMOTE_MARKER"]).resolve(),
    "production_dropin": Path(os.environ["DROPIN_TARGET"]).resolve(),
}
if any(not path.is_file() or path.stat().st_size <= 0 for path in paths.values()):
    raise SystemExit("production-ready inputs are missing")
marker = json.loads(paths["remote_marker"].read_text(encoding="utf-8"))
tree = json.loads(paths["runtime_tree"].read_text(encoding="utf-8"))
runtime = dict(tree.get("runtime_contract") or {})
accepted = sorted(str(value) for value in marker.get("accepted_archetype_ids") or ())
if not (
    marker.get("runtime_enabled") is True
    and marker.get("continuous_reevaluation") is True
    and marker.get("one_route_per_decision") is True
    and accepted
    and accepted == sorted(runtime.get("accepted_archetype_ids") or ())
    and runtime.get("unknown_route_exact_bypass") is True
    and runtime.get("one_route_per_decision") is True
):
    raise SystemExit("production-ready runtime contract is invalid")
payload = {
    "schema": "poke_bot.matchup_runtime_production_ready/v1",
    "runtime_enabled": True,
    "iteration": 27,
    "accepted_archetype_ids": accepted,
    "artifacts": {
        name: {"path": str(path), "digest": digest(path)}
        for name, path in sorted(paths.items())
    },
}
ready = Path(os.environ["PRODUCTION_READY"]).resolve()
ready.parent.mkdir(parents=True, exist_ok=True)
if ready.exists():
    if json.loads(ready.read_text(encoding="utf-8")) != payload:
        raise SystemExit("existing production-ready receipt conflicts")
else:
    fd, temporary = tempfile.mkstemp(prefix="." + ready.name + ".", dir=ready.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, ready)
    finally:
        Path(temporary).unlink(missing_ok=True)
print("production_ready_ok", digest(ready))
PY
systemctl --user reset-failed "$PRODUCTION" || true
systemctl --user start "$PRODUCTION"

status verifying_iteration_27_start
EXPECTED_CHECKPOINT="$($PY -c 'from poke_bot.checkpoint import checkpoint_digest; import sys; print(checkpoint_digest(sys.argv[1]))' "$MERGED")" \
"$PY" - <<'PY'
import json,os,subprocess,time
from pathlib import Path
runtime=Path("/home/inzi/poke-bot-agent/outputs/pure_rl/pure_rl_alakazam_temporal1_8k_teacher_v16_20260721/iteration_runtime.json")
expected=os.environ["EXPECTED_CHECKPOINT"]
deadline=time.monotonic()+600
while time.monotonic()<deadline:
    active=subprocess.run(["systemctl","--user","is-active","pokebot-pure-rl-alakazam.service"],text=True,stdout=subprocess.PIPE).stdout.strip()
    if active=="failed": raise SystemExit("production failed during iteration-27 startup")
    try: row=json.loads(runtime.read_text())
    except Exception: row={}
    if (int(row.get("iteration",-1))==27 and row.get("phase")=="collect" and row.get("checkpoint_digest")==expected):
        print("iteration27_runtime_ok",row); break
    time.sleep(2)
else: raise SystemExit("iteration 27 did not publish the activated learner")
PY

status complete "iteration=27 runtime=enabled"
trap - ERR
