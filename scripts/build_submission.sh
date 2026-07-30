#!/usr/bin/env bash
# Pack a Kaggle submission tarball: main.py, deck.csv, model.pt, competition cg/.
# Usage:
#   bash scripts/build_submission.sh [checkpoint.pt] [out_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${1:-}"
OUT_DIR="${2:-$ROOT/outputs/submission}"
STAGE="$OUT_DIR/stage"
TARBALL="$OUT_DIR/submission.tar.gz"
ARCH="${POKEBOT_PRIMARY_ARCHETYPE:-dragapult}"
DECK_SRC="${POKEBOT_SUBMISSION_DECK:-$ROOT/submission/deck.csv}"
MATCHUP_TREE_SRC="${POKEBOT_SUBMISSION_MATCHUP_TREE:-}"
MATCHUP_ROSTER_SRC="$ROOT/state/matchup_adapter_roster.json"
TURN_ORDER_PREFERENCE="${POKEBOT_SUBMISSION_TURN_ORDER:-first_if_allowed}"
SEARCH_CONFIG_SRC="$ROOT/submission/search_config.json"
BELIEF_PRIOR_BUILDER="$ROOT/scripts/build_submission_belief_posterior.py"
BELIEF_PRIOR_SOURCES=(
  "$ROOT/data/training_mixes/top_ladder_representatives.v1.json"
  "$ROOT/data/training_mixes/specialist_representatives.v1.json"
)

if [[ -z "$CKPT" ]]; then
  for cand in \
    "$ROOT/outputs/checkpoints/${ARCH}_round_robin.best.pt" \
    "$ROOT/outputs/checkpoints/${ARCH}_bootstrap.best.pt" \
    "$ROOT/outputs/checkpoints/${ARCH}_bootstrap.latest.pt"; do
    if [[ -f "$cand" ]]; then
      CKPT="$cand"
      break
    fi
  done
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "ERROR: no checkpoint (pass path as \$1)" >&2
  exit 1
fi
if [[ ! -f "$DECK_SRC" ]]; then
  echo "ERROR: submission deck does not exist: $DECK_SRC" >&2
  exit 1
fi
if [[ "$TURN_ORDER_PREFERENCE" != "first_if_allowed" && \
      "$TURN_ORDER_PREFERENCE" != "second_if_allowed" ]]; then
  echo "ERROR: invalid POKEBOT_SUBMISSION_TURN_ORDER=$TURN_ORDER_PREFERENCE" >&2
  exit 1
fi
if [[ ! -f "$SEARCH_CONFIG_SRC" || ! -f "$BELIEF_PRIOR_BUILDER" ]]; then
  echo "ERROR: default belief-MCTS submission assets are missing" >&2
  exit 1
fi
for source in "${BELIEF_PRIOR_SOURCES[@]}"; do
  if [[ ! -f "$source" ]]; then
    echo "ERROR: public belief-prior source does not exist: $source" >&2
    exit 1
  fi
done

# Deployment is policy-first/history-only. Privileged single-world search must
# never be packaged accidentally, even if enabled in the caller's environment.
if [[ "${POKEBOT_SEARCH_MODE:-policy}" != "policy" || \
      "${POKEBOT_ALLOW_ORACLE_DECK:-0}" == "1" ]]; then
  echo "ERROR: refusing to package oracle/privileged search configuration" >&2
  exit 1
fi
PYTHON="${POKEBOT_PYTHON:-/home/inzi/miniconda3/envs/poke-bot-agent/bin/python}"
"$PYTHON" - "$CKPT" <<'PY'
import sys
from poke_bot import checkpoint as checkpoint_mod
from poke_bot.checkpoint import assert_trusted_policy_checkpoint
from poke_bot.dormant_adapter_compat import validate_zero_dormant_checkpoint
assert_trusted_policy_checkpoint(sys.argv[1])
payload = checkpoint_mod.load_checkpoint(sys.argv[1], map_location="cpu")
dormant = dict((payload.get("extra") or {}).get("dormant_matchup_adapter_bank") or {})
if dormant:
    validate_zero_dormant_checkpoint(
        sys.argv[1],
        allow_trained=(
            dormant.get("schema")
            == "poke_bot.trained_dormant_matchup_adapter/v1"
        ),
    )
print("OK: trusted history-policy checkpoint")
PY

CG_SRC=""
for cand in \
  "$ROOT/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg" \
  "$ROOT/kaggle/input/cg-lib/cg"; do
  if [[ -d "$cand" ]]; then
    CG_SRC="$cand"
    break
  fi
done
if [[ -z "$CG_SRC" ]]; then
  echo "ERROR: competition cg/ not found" >&2
  exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE"

echo "== build_submission"
echo "   ckpt=$CKPT"
echo "   deck=$DECK_SRC"
echo "   turn_order=$TURN_ORDER_PREFERENCE"
echo "   cg=$CG_SRC"
echo "   out=$TARBALL"

cp "$ROOT/submission/main.py" "$STAGE/main.py"
printf '{"schema":"poke_bot.submission_turn_order_profile/v1","turn_order_preference":"%s"}\n' \
  "$TURN_ORDER_PREFERENCE" >"$STAGE/turn_order_profile.json"
cp "$DECK_SRC" "$STAGE/deck.csv"
cp "$CKPT" "$STAGE/model.pt"
cp "$SEARCH_CONFIG_SRC" "$STAGE/search_config.json"
"$PYTHON" "$BELIEF_PRIOR_BUILDER" \
  --output "$STAGE/belief_decks.json" \
  --source "${BELIEF_PRIOR_SOURCES[0]}" \
  --source "${BELIEF_PRIOR_SOURCES[1]}"
cp -a "$CG_SRC" "$STAGE/cg"
if [[ -n "$MATCHUP_TREE_SRC" ]]; then
  if [[ ! -f "$MATCHUP_TREE_SRC" ]]; then
    echo "ERROR: submission matchup tree does not exist: $MATCHUP_TREE_SRC" >&2
    exit 1
  fi
  if [[ ! -f "$MATCHUP_ROSTER_SRC" ]]; then
    echo "ERROR: submission matchup roster does not exist: $MATCHUP_ROSTER_SRC" >&2
    exit 1
  fi
  "$PYTHON" - "$MATCHUP_TREE_SRC" <<'PY'
import sys
from poke_bot.public_matchup_router import PublicMatchupDecisionTree

tree = PublicMatchupDecisionTree.from_path(
    sys.argv[1], require_runtime_enabled=True
)
if not tree.runtime_accepted_archetype_ids:
    raise SystemExit("ERROR: submission matchup tree has no accepted routes")
print(
    "OK: activated public matchup tree",
    len(tree.runtime_accepted_archetype_ids),
    tree.digest,
)
PY
  cp "$MATCHUP_TREE_SRC" "$STAGE/matchup_tree.json"
  mkdir -p "$STAGE/state"
  cp "$MATCHUP_ROSTER_SRC" "$STAGE/state/matchup_adapter_roster.json"
fi

# Vendor poke_bot package needed at runtime (model/mcts/features/…).
mkdir -p "$STAGE/poke_bot"
rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$ROOT/poke_bot/" "$STAGE/poke_bot/"

# The dormant-adapter loader contract checksum-binds the dynamic simulation
# record loader alongside the package modules. Submission inference does not
# execute this training script, but the exact source file must be present so a
# bank-bearing checkpoint proves it carries the validated loader overlay.
mkdir -p "$STAGE/scripts"
cp "$ROOT/scripts/train_round_robin.py" "$STAGE/scripts/train_round_robin.py"

"$PYTHON" - "$STAGE" <<'PY'
import json
from pathlib import Path
import sys

stage = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(stage))
from poke_bot.submission_budget import SubmissionSearchBudget

config = json.loads((stage / "search_config.json").read_text())
budget = SubmissionSearchBudget.from_config(config, started_at=0.0)
prior = json.loads((stage / "belief_decks.json").read_text())
decks = prior.get("deck_lists") or []
if (
    config.get("enabled") is not False
    or config.get("algorithm") != "public_history_root_sampled_belief_mcts"
    or config.get("leaf_evaluator")
    != "trained_checkpoint_policy_value_head"
    or config.get("leaf_evaluator_checkpoint") != "submission_model_pt"
    or config.get("require_trained_state_evaluator") is not True
    or config.get("search_failure_behavior")
    != "greedy_current_decision_then_retry"
    or config.get("game_wide_greedy_only_for_time_budget") is not True
    or config.get("fallback") != "frozen_model_greedy_policy"
    or config.get("oracle_inputs_allowed") is not False
    or budget.hard_cap_s != 600.0
    or prior.get("schema") != "poke_bot.submission_belief_decks/v1"
    or prior.get("anonymous") is not True
    or prior.get("contains_opponent_identity") is not False
    or prior.get("deck_count") != len(decks)
    or len(decks) < 8
    or any(len(deck) != 60 for deck in decks)
):
    raise SystemExit("ERROR: packaged belief-MCTS contract is invalid")
print(
    "OK: default frozen policy-only deployment",
    f"hard_cap={budget.hard_cap_s:.0f}s",
    f"internal_deadline={budget.internal_deadline_s:.0f}s",
    f"final_greedy_reserve={budget.final_greedy_reserve_s:.0f}s",
    f"decks={len(decks)}",
)
PY

# A bank-bearing checkpoint must ship the exact loader implementation that was
# validated before fleet rollout. Ordinary legacy checkpoints remain unchanged.
"$PYTHON" - "$STAGE" <<'PY'
from pathlib import Path
import sys

stage = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(stage))
from poke_bot import checkpoint

payload = checkpoint.load_checkpoint(stage / "model.pt", map_location="cpu")
dormant = dict(dict(payload.get("extra") or {}).get("dormant_matchup_adapter_bank") or {})
if dormant:
    from poke_bot.dormant_adapter_compat import (
        loader_source_contract,
        validate_loader_root,
        validate_zero_dormant_checkpoint,
    )

    validate_zero_dormant_checkpoint(
        stage / "model.pt",
        allow_trained=(
            dormant.get("schema")
            == "poke_bot.trained_dormant_matchup_adapter/v1"
        ),
    )
    validate_loader_root(
        stage,
        role="submission",
        source_contract=loader_source_contract(stage),
        checkpoint_path=stage / "model.pt",
    )
    print("OK: frozen adapter-bank checkpoint + exact submission loader")
PY

# Isolated-smoke helper: prove no __file__ at import by compiling main.
"$PYTHON" - <<'PY' "$STAGE"
import ast, pathlib, sys
stage = pathlib.Path(sys.argv[1])
src = (stage / "main.py").read_text()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id == "__file__":
        # Allow only inside functions that are not module-level executed at import
        # — still forbid any __file__ per plan fail-closed contract.
        raise SystemExit("ERROR: submission/main.py must not reference __file__")
print("OK: no __file__ references in main.py")
PY

tar -czf "$TARBALL" -C "$STAGE" .
echo ">> wrote $TARBALL ($(du -h "$TARBALL" | awk '{print $1}'))"

# Isolated tarball smoke: extract to tmp, then load main.py exactly like Kaggle:
# by filename while the agent directory is absent from sys.path.  Do not set
# PYTHONPATH here; doing so previously masked a broken vendored-cg import.
SMOKE_DIR="$OUT_DIR/smoke_extract"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"
tar -xzf "$TARBALL" -C "$SMOKE_DIR"
(
  env -u PYTHONPATH -u CG_LIB_PATH \
    POKEBOT_SUBMISSION_SEARCH_DISABLE=1 \
    "$PYTHON" -I - "$SMOKE_DIR" <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

stage = Path(sys.argv[1]).resolve()
os.chdir(stage)
assert str(stage) not in sys.path, (stage, sys.path)
spec = importlib.util.spec_from_file_location("kaggle_submission_main", stage / "main.py")
assert spec is not None and spec.loader is not None
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
assert str(stage) not in sys.path, "main.py must remain lazily loaded before first act"
turn_order = json.loads((stage / "turn_order_profile.json").read_text())[
    "turn_order_preference"
]

for prompt, first_index, second_index in (
    ({"select": {"context": 41, "minCount": 1, "maxCount": 1,
                  "option": [{"type": 1}, {"type": 2}]}}, 0, 1),
    ({"select": {"context": "IS_FIRST", "minCount": 1, "maxCount": 1,
                  "option": [{"type": "No"}, {"type": "Yes"}]}}, 1, 0),
):
    expected = [first_index if turn_order == "first_if_allowed" else second_index]
    assert agent_mod.agent(prompt) == expected, (prompt, expected)
assert agent_mod._MODEL is None, "turn-order choice must not load the model"
print("OK: package honors", turn_order, "before model initialization")

# Exercise the exact first Kaggle call.  This must add the agent directory,
# import vendored cg, load the exact neural checkpoint, and return 60 cards.
deck = agent_mod.agent({"logs": [], "current": None, "select": None})
assert len(deck) == 60, len(deck)
assert str(stage) == sys.path[0], sys.path
from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start
print("OK: Kaggle-style isolated first act + vendored cg + neural load", len(deck))

obs, _ = battle_start(deck, deck)
steps = 0
go_first_seen = False
rng = random.Random(0)
while obs is not None and steps < 80:
    cur = obs.get("current") or {}
    if cur.get("result", -1) != -1:
        break
    sel = obs.get("select")
    if sel is None:
        break
    n = len(sel.get("option") or [])
    lo = int(sel.get("minCount", 0) or 0)
    hi = min(int(sel.get("maxCount", 0) or 0), n)
    if hi <= 0:
        break
    choice = agent_mod.agent(obs)
    assert all(0 <= int(index) < n for index in choice), (n, choice)
    assert lo <= len(choice) <= hi, (lo, len(choice), hi)
    context = sel.get("context")
    if context == 41 or str(context).replace("_", "").lower() == "isfirst":
        go_first_seen = True
        desired = "yes" if turn_order == "first_if_allowed" else "no"
        desired_integer = 1 if desired == "yes" else 2
        selected = [
            i for i, option in enumerate(sel.get("option") or [])
            if option.get("type") == desired_integer
            or str(option.get("type")).lower() == desired
        ]
        assert len(selected) == 1 and choice == selected, (
            "must honor packaged turn order",
            turn_order,
            choice,
            selected,
        )
    obs = battle_select(choice)
    steps += 1
battle_finish()
assert steps > 0
assert go_first_seen, "isolated engine battle never exercised IsFirst"
print(f"OK: Kaggle-style neural battle steps={steps}")
PY
)

# Default-policy isolated tarball smoke. This exercises the exact packaged
# config without an environment override and proves future submissions do not
# construct or invoke MCTS.
(
  env -u PYTHONPATH -u CG_LIB_PATH \
    "$PYTHON" -I - "$SMOKE_DIR" <<'PY'
import importlib.util
import os
from pathlib import Path
import random
import sys

stage = Path(sys.argv[1]).resolve()
os.chdir(stage)
spec = importlib.util.spec_from_file_location(
    "kaggle_submission_search_smoke", stage / "main.py"
)
assert spec is not None and spec.loader is not None
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
deck = agent_mod.agent({"logs": [], "current": None, "select": None})
from cg.game import battle_finish, battle_select, battle_start

obs, _ = battle_start(deck, deck)
steps = 0
agent_calls = 0
try:
    while obs is not None and steps < 80:
        current = obs.get("current") or {}
        if current.get("result", -1) != -1:
            break
        selection = obs.get("select")
        if selection is None:
            break
        choice = agent_mod.agent(obs)
        agent_calls += 1
        obs = battle_select(choice)
        steps += 1
finally:
    battle_finish()

budget = agent_mod._SEARCH_BUDGET
policy = agent_mod._POLICY
assert budget is not None
assert budget.enabled is False
assert budget.searches_used == 0
assert budget.disabled_reason is None, budget.disabled_reason
assert budget.consecutive_search_failures == 0
assert policy is not None and policy.use_mcts is False
assert policy.belief_mcts is False
assert policy.last_result is None
# These counters were added after the first frozen V5 specialists.  Their
# absence in an older, checksum-pinned policy runtime is equivalent to the
# untouched zero/None state and must not force replacement of that runtime.
assert getattr(policy, "last_search_fallback_reason", None) is None
assert int(getattr(policy, "fail_closed_count", 0)) == 0
assert agent_calls > 0
assert budget.final_greedy_reserve_s == 20.0
print(
    "OK: packaged policy-only default",
    f"calls={agent_calls}",
    "mcts_calls=0",
    f"final_reserve={budget.final_greedy_reserve_s:.0f}s",
)
PY
)

echo ">> isolated smoke dir: $SMOKE_DIR"
"$PYTHON" - "$TARBALL" <<'PY'
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

bundle = Path(sys.argv[1]).resolve()
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
receipt = Path(str(bundle) + ".go-first-verified.json")
turn_order = json.loads(
    (bundle.parent / "stage" / "turn_order_profile.json").read_text()
)["turn_order_preference"]
payload = {
    "schema": "poke_bot.submission_turn_order_attestation/v1",
    "file_sha256": "sha256:" + digest,
    "turn_order_preference": turn_order,
    "go_first_if_offered": turn_order == "first_if_allowed",
    "go_second_if_offered": turn_order == "second_if_allowed",
    "belief_mcts_default": False,
    "belief_mcts_leaf_evaluator": "trained_checkpoint_policy_value_head",
    "belief_mcts_leaf_evaluator_checkpoint": "submission_model_pt",
    "belief_mcts_hard_cap_s": 600.0,
    "belief_mcts_internal_deadline_s": 540.0,
    "belief_mcts_final_greedy_reserve_s": 20.0,
    "search_config_sha256": "sha256:" + hashlib.sha256(
        (bundle.parent / "stage" / "search_config.json").read_bytes()
    ).hexdigest(),
    "belief_decks_sha256": "sha256:" + hashlib.sha256(
        (bundle.parent / "stage" / "belief_decks.json").read_bytes()
    ).hexdigest(),
    "verified_cases": [
        "integer_enum",
        "string_enum_reversed_options",
        "live_engine_prompt",
    ],
    "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, receipt)
print("OK: wrote digest-bound go-first attestation", receipt)
PY
echo "DONE"
