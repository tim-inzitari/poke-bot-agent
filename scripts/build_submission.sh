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
MATCHUP_ROSTER_SRC="${POKEBOT_SUBMISSION_MATCHUP_ROSTER:-$ROOT/state/matchup_adapter_roster.json}"
TURN_ORDER_PREFERENCE="${POKEBOT_SUBMISSION_TURN_ORDER:-first_if_allowed}"
RTP_MODE="${POKEBOT_SUBMISSION_RTP_MODE:-default_off}"
DIRECT_NO_SEARCH_ASSETS="${POKEBOT_SUBMISSION_DIRECT_NO_SEARCH_ASSETS:-0}"
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
case "$RTP_MODE" in
  default_off|disabled|enabled|off|direct|recursive)
    ;;
  *)
    echo "ERROR: invalid POKEBOT_SUBMISSION_RTP_MODE=$RTP_MODE" >&2
    exit 1
    ;;
esac
if [[ "$DIRECT_NO_SEARCH_ASSETS" != "0" && "$DIRECT_NO_SEARCH_ASSETS" != "1" ]]; then
  echo "ERROR: POKEBOT_SUBMISSION_DIRECT_NO_SEARCH_ASSETS must be 0 or 1" >&2
  exit 1
fi
if [[ "$DIRECT_NO_SEARCH_ASSETS" == "1" && "$RTP_MODE" != "off" ]]; then
  echo "ERROR: search-asset-free packaging requires explicit RTP mode off" >&2
  exit 1
fi
if [[ "$DIRECT_NO_SEARCH_ASSETS" == "0" ]]; then
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
fi

# Deployment is policy-first/history-only. Privileged single-world search must
# never be packaged accidentally, even if enabled in the caller's environment.
if [[ "${POKEBOT_SEARCH_MODE:-policy}" != "policy" || \
      "${POKEBOT_ALLOW_ORACLE_DECK:-0}" == "1" ]]; then
  echo "ERROR: refusing to package oracle/privileged search configuration" >&2
  exit 1
fi
PYTHON="${POKEBOT_PYTHON:-/home/inzi/miniconda3/envs/poke-bot-agent/bin/python}"
"$PYTHON" - "$CKPT" "$MATCHUP_TREE_SRC" <<'PY'
import json
import sys
from pathlib import Path

from poke_bot import checkpoint as checkpoint_mod
from poke_bot.checkpoint import assert_trusted_policy_checkpoint
from poke_bot.dormant_adapter_compat import validate_zero_dormant_checkpoint
from poke_bot.matchup_adapter_routes import (
    require_runtime_route_binding,
    resolve_matchup_adapter_route_contract,
)
from poke_bot.public_matchup_router import PublicMatchupDecisionTree


TRAINED_DORMANT_SCHEMA = "poke_bot.trained_dormant_matchup_adapter/v1"
checkpoint_path = Path(sys.argv[1]).resolve()
tree_arg = sys.argv[2].strip()
assert_trusted_policy_checkpoint(checkpoint_path)
payload = checkpoint_mod.load_checkpoint(checkpoint_path, map_location="cpu")
extra = dict(payload.get("extra") or {})
dormant = dict(extra.get("dormant_matchup_adapter_bank") or {})
trained_adapter_bank = (
    dormant.get("schema") == TRAINED_DORMANT_SCHEMA
    and dormant.get("zero_output") is False
)
if dormant:
    validate_zero_dormant_checkpoint(
        checkpoint_path,
        allow_trained=(dormant.get("schema") == TRAINED_DORMANT_SCHEMA),
    )
if trained_adapter_bank:
    if not tree_arg:
        raise SystemExit(
            "ERROR: trained matchup adapter checkpoint requires "
            "POKEBOT_SUBMISSION_MATCHUP_TREE"
        )
    tree_path = Path(tree_arg).expanduser().resolve()
    if not tree_path.is_file():
        raise SystemExit(
            "ERROR: submission matchup tree does not exist: " + str(tree_path)
        )
    try:
        tree = PublicMatchupDecisionTree.from_path(
            tree_path, require_runtime_enabled=True
        )
        adapter_config = dict(extra.get("matchup_adapter_config") or {})
        route_contract = resolve_matchup_adapter_route_contract(adapter_config)
        runtime = dict(
            json.loads(tree_path.read_text()).get("runtime_contract") or {}
        )
        require_runtime_route_binding(runtime, route_contract)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "ERROR: trained matchup adapter runtime binding is invalid: " + str(exc)
        ) from exc
    checkpoint_archetype = str(payload.get("archetype_id") or "").strip().casefold()
    if (
        not checkpoint_archetype
        or checkpoint_archetype not in tree.runtime_accepted_archetype_ids
        or tuple(tree.targets) != route_contract.target_ids
        or tuple(tree.route_physical_slots) != route_contract.physical_slots
        or tree.adapter_format != route_contract.adapter_format
        or tree.slot_registry_digest != route_contract.slot_registry_digest
    ):
        raise SystemExit(
            "ERROR: trained matchup adapter tree is not bound to the exact "
            "checkpoint route contract"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise SystemExit(
            "ERROR: trained matchup adapter checkpoint lacks model_state_dict"
        )

    def output_nonzero(name):
        try:
            return int(name.detach().count_nonzero().item()) > 0
        except (AttributeError, RuntimeError, TypeError):
            return None

    # Runtime-enabled only means the router is armed.  A submitted tree must
    # also never select a route whose adapter output projection is still exact
    # zero; inspecting down-projection weights alone would not establish this.
    invalid_output_routes = []
    for archetype_id in sorted(tree.runtime_accepted_archetype_ids):
        slot = route_contract.physical_slot_by_target[archetype_id]
        prefix = f"matchup_adapter_bank.experts.{slot}.up."
        weight_nonzero = output_nonzero(state.get(prefix + "weight"))
        bias_nonzero = output_nonzero(state.get(prefix + "bias"))
        if weight_nonzero is None or bias_nonzero is None:
            invalid_output_routes.append(f"{archetype_id}@{slot}:missing-output")
        elif not (weight_nonzero or bias_nonzero):
            invalid_output_routes.append(f"{archetype_id}@{slot}:zero-output")
    if invalid_output_routes:
        raise SystemExit(
            "ERROR: runtime tree accepts adapter route(s) without a verified "
            "nonzero output projection: " + ", ".join(invalid_output_routes)
        )
    print(
        "OK: trained matchup adapter runtime binding",
        checkpoint_archetype,
        tree.digest,
    )
print("OK: trusted history-policy checkpoint")
PY

CG_SRC="${POKEBOT_SUBMISSION_CG_ROOT:-}"
if [[ -z "$CG_SRC" ]]; then
  for cand in \
    "$ROOT/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg" \
    "$ROOT/kaggle/input/cg-lib/cg"; do
    if [[ -d "$cand" ]]; then
      CG_SRC="$cand"
      break
    fi
  done
fi
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
echo "   rtp_mode=$RTP_MODE"
echo "   direct_no_search_assets=$DIRECT_NO_SEARCH_ASSETS"
echo "   cg=$CG_SRC"
echo "   out=$TARBALL"

cp "$ROOT/submission/main.py" "$STAGE/main.py"
printf '{"schema":"poke_bot.submission_turn_order_profile/v1","turn_order_preference":"%s"}\n' \
  "$TURN_ORDER_PREFERENCE" >"$STAGE/turn_order_profile.json"
cp "$DECK_SRC" "$STAGE/deck.csv"
cp "$CKPT" "$STAGE/model.pt"
if [[ "$DIRECT_NO_SEARCH_ASSETS" == "0" ]]; then
  cp "$SEARCH_CONFIG_SRC" "$STAGE/search_config.json"
  "$PYTHON" "$BELIEF_PRIOR_BUILDER" \
    --output "$STAGE/belief_decks.json" \
    --source "${BELIEF_PRIOR_SOURCES[0]}" \
    --source "${BELIEF_PRIOR_SOURCES[1]}"
fi
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

# New RTP candidates use an explicit three-arm profile.  ``disabled`` and
# ``enabled`` remain byte-compatible profile modes for immutable historical
# packages; only ``off``, ``direct``, and ``recursive`` are eligible for r197.
"$PYTHON" - "$STAGE" "$RTP_MODE" "${POKEBOT_SUBMISSION_RTP_CHECKPOINT:-}" \
  "${POKEBOT_SUBMISSION_RTP_PARENT_CHECKPOINT_SHA256:-}" \
  "${POKEBOT_SUBMISSION_RTP_PROMOTION_RECEIPT:-}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from poke_bot.rtp_evaluation_promotion import (
    RTPPromotionEvidenceError,
    read_r198_immutable_json_object,
    validate_r198_evaluation_receipt,
)


stage = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
sidecar_source = Path(sys.argv[3]).expanduser() if sys.argv[3] else None
expected_parent = sys.argv[4]
promotion_source = Path(sys.argv[5]).expanduser() if sys.argv[5] else None


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            value.update(block)
    return "sha256:" + value.hexdigest()


def canonical_digest(value: object) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def cards_digest(path: Path) -> str:
    cards = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cards.append(int(line.split(",", 1)[0]))
        except ValueError as exc:
            raise SystemExit(f"ERROR: packaged deck has a non-card row: {line!r}") from exc
    if len(cards) != 60:
        raise SystemExit(f"ERROR: packaged deck does not have 60 cards: {len(cards)}")
    return canonical_digest(cards)


def write_profile(value: dict) -> None:
    (stage / "runtime_profile.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


model = stage / "model.pt"
model_digest = digest(model)
base = {"schema": "poke_bot.submission_runtime_profile/v1"}
if mode == "default_off":
    pass
elif mode == "disabled":
    # Preserve immutable r195 no-RTP package semantics exactly.
    write_profile(
        {
            **base,
            "recursive_turn_planner": "disabled",
            "display": "NO RTP",
            "rtp_sidecar_packaged": False,
        }
    )
elif mode == "enabled":
    # Preserve immutable r195 RTP package semantics exactly.
    if sidecar_source is None or not sidecar_source.is_file():
        raise SystemExit("ERROR: enabled RTP submission lacks POKEBOT_SUBMISSION_RTP_CHECKPOINT")
    packaged = stage / "rtp_shadow_planner.pt"
    shutil.copy2(sidecar_source, packaged)
    write_profile(
        {
            **base,
            "recursive_turn_planner": "enabled",
            "display": "RTP",
            "rtp_sidecar_packaged": True,
            "rtp_checkpoint_sha256": digest(packaged),
            "specialist_id": "alakazam",
        }
    )
elif mode == "off":
    write_profile(
        {
            **base,
            "rtp_mode": "off",
            "recursive_turn_planner": "disabled",
            "display": "NO RTP",
            "rtp_sidecar_packaged": False,
            "model_checkpoint_sha256": model_digest,
        }
    )
elif mode == "direct":
    if sidecar_source is None or not sidecar_source.is_file():
        raise SystemExit("ERROR: direct RTP submission lacks POKEBOT_SUBMISSION_RTP_CHECKPOINT")
    if not expected_parent:
        raise SystemExit("ERROR: direct RTP submission lacks POKEBOT_SUBMISSION_RTP_PARENT_CHECKPOINT_SHA256")
    if model_digest != expected_parent:
        raise SystemExit("ERROR: direct RTP expected parent does not match packaged model")
    tree = stage / "matchup_tree.json"
    if not tree.is_file():
        raise SystemExit("ERROR: direct RTP submission requires a packaged matchup tree")
    packaged_sidecar = stage / "rtp_shadow_planner.pt"
    shutil.copy2(sidecar_source, packaged_sidecar)
    try:
        import torch
        sidecar_payload = torch.load(
            packaged_sidecar, map_location="cpu", weights_only=True
        )
    except Exception as exc:  # noqa: BLE001 - a malformed sidecar is fatal.
        raise SystemExit("ERROR: direct RTP sidecar cannot be read") from exc
    if not isinstance(sidecar_payload, dict) or not isinstance(
        sidecar_payload.get("config"), dict
    ):
        raise SystemExit("ERROR: direct RTP sidecar lacks its config")
    sidecar_config = dict(sidecar_payload["config"])
    write_profile(
        {
            **base,
            "rtp_mode": "direct",
            "recursive_turn_planner": "enabled",
            "display": "DIRECT RTP",
            "rtp_sidecar_packaged": True,
            "rtp_direct_bridge_only": True,
            "rtp_sizing_profile": "pure_rl_r197",
            "specialist_id": "alakazam",
            "model_checkpoint_sha256": model_digest,
            "parent_checkpoint_sha256": expected_parent,
            "rtp_checkpoint_sha256": digest(packaged_sidecar),
            "rtp_config_sha256": canonical_digest(sidecar_config),
            "max_neural_passes": sidecar_config.get("max_neural_passes"),
            "max_action_combos": 1024,
            "required_neural_passes": {"normal": 6, "forced_replan": 5},
            "deck_file_sha256": digest(stage / "deck.csv"),
            "deck_cards_sha256": cards_digest(stage / "deck.csv"),
            "matchup_tree_sha256": digest(tree),
        }
    )
elif mode == "recursive":
    if sidecar_source is None or not sidecar_source.is_file():
        raise SystemExit("ERROR: recursive RTP submission lacks POKEBOT_SUBMISSION_RTP_CHECKPOINT")
    if not expected_parent:
        raise SystemExit("ERROR: recursive RTP submission lacks POKEBOT_SUBMISSION_RTP_PARENT_CHECKPOINT_SHA256")
    if model_digest != expected_parent:
        raise SystemExit("ERROR: recursive RTP expected parent does not match packaged model")
    if promotion_source is None or not promotion_source.is_file():
        raise SystemExit("ERROR: recursive RTP submission lacks POKEBOT_SUBMISSION_RTP_PROMOTION_RECEIPT")
    tree = stage / "matchup_tree.json"
    if not tree.is_file():
        raise SystemExit("ERROR: recursive RTP submission requires a packaged matchup tree")
    try:
        promotion_identity, promotion = read_r198_immutable_json_object(
            promotion_source,
            label="RTP promotion receipt",
        )
    except RTPPromotionEvidenceError as exc:
        raise SystemExit("ERROR: RTP promotion receipt is not sealed immutable evidence: " + str(exc)) from exc
    evaluation_source = Path(
        str(promotion.get("evaluation_receipt_path") or "")
    ).expanduser()
    if not evaluation_source.is_file():
        raise SystemExit("ERROR: RTP promotion receipt evaluation receipt is unavailable")
    expected_evaluation_digest = str(promotion.get("evaluation_receipt_sha256") or "")
    if promotion.get("deck_file_sha256") != digest(stage / "deck.csv"):
        raise SystemExit("ERROR: RTP promotion receipt deck-file digest changed")
    try:
        validate_r198_evaluation_receipt(
            evaluation_source,
            expected_sha256=expected_evaluation_digest,
            require_local_evidence=True,
            expected_parent_checkpoint_sha256=expected_parent,
            expected_sidecar_sha256=digest(sidecar_source),
            expected_sidecar_config_sha256=str(promotion.get("sidecar_config_sha256") or ""),
            expected_deck_file_sha256=digest(stage / "deck.csv"),
            expected_deck_cards_sha256=cards_digest(stage / "deck.csv"),
            expected_matchup_tree_sha256=digest(tree),
        )
    except RTPPromotionEvidenceError as exc:
        raise SystemExit(
            "ERROR: RTP promotion evaluation receipt is non-promotable: " + str(exc)
        ) from exc
    packaged_sidecar = stage / "rtp_shadow_planner.pt"
    packaged_promotion = stage / "rtp_promotion_receipt.json"
    packaged_evaluation = stage / "rtp_evaluation_receipt.json"
    shutil.copy2(sidecar_source, packaged_sidecar)
    shutil.copy2(promotion_source, packaged_promotion)
    shutil.copy2(evaluation_source, packaged_evaluation)
    # A source receipt is verified before copying, but copy2 intentionally
    # retains source metadata.  The submitted package is its own sealed
    # evidence boundary: make both bytes exact 0444 and re-open them through
    # the same no-symlink immutable reader before profiling the package.
    for receipt_path in (packaged_promotion, packaged_evaluation):
        os.chmod(receipt_path, 0o444)
    try:
        packaged_promotion_identity, packaged_promotion_payload = (
            read_r198_immutable_json_object(
                packaged_promotion,
                label="packaged RTP promotion receipt",
                expected_sha256=promotion_identity["sha256"],
            )
        )
        packaged_evaluation_identity, _ = read_r198_immutable_json_object(
            packaged_evaluation,
            label="packaged RTP evaluation receipt",
            expected_sha256=expected_evaluation_digest,
        )
    except RTPPromotionEvidenceError as exc:
        raise SystemExit("ERROR: packaged RTP evidence is not sealed immutable evidence: " + str(exc)) from exc
    if (
        packaged_promotion_payload.get("evaluation_receipt_sha256")
        != packaged_evaluation_identity["sha256"]
        or packaged_promotion_identity["sha256"] != promotion_identity["sha256"]
    ):
        raise SystemExit("ERROR: packaged RTP promotion/evaluation receipt binding changed")
    try:
        import torch
        sidecar_payload = torch.load(
            packaged_sidecar, map_location="cpu", weights_only=True
        )
    except Exception as exc:  # noqa: BLE001 - a malformed sidecar is fatal.
        raise SystemExit("ERROR: recursive RTP sidecar cannot be read") from exc
    if not isinstance(sidecar_payload, dict) or not isinstance(
        sidecar_payload.get("config"), dict
    ):
        raise SystemExit("ERROR: recursive RTP sidecar lacks its config")
    sidecar_config = dict(sidecar_payload["config"])
    write_profile(
        {
            **base,
            "rtp_mode": "recursive",
            "recursive_turn_planner": "enabled",
            "display": "RTP",
            "rtp_sidecar_packaged": True,
            "rtp_sizing_profile": "pure_rl_r197",
            "specialist_id": "alakazam",
            "model_checkpoint_sha256": model_digest,
            "parent_checkpoint_sha256": expected_parent,
            "rtp_checkpoint_sha256": digest(packaged_sidecar),
            "rtp_config_sha256": canonical_digest(sidecar_config),
            "max_neural_passes": sidecar_config.get("max_neural_passes"),
            "max_action_combos": 1024,
            "required_neural_passes": {"normal": 6, "forced_replan": 5},
            "deck_file_sha256": digest(stage / "deck.csv"),
            "deck_cards_sha256": cards_digest(stage / "deck.csv"),
            "matchup_tree_sha256": digest(tree),
            "rtp_promotion_receipt_file": packaged_promotion.name,
            "rtp_promotion_receipt_sha256": digest(packaged_promotion),
            "rtp_evaluation_receipt_file": packaged_evaluation.name,
            "rtp_evaluation_receipt_sha256": digest(packaged_evaluation),
        }
    )
else:
    raise SystemExit(f"ERROR: unknown RTP mode {mode!r}")
PY

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

# Build-time validation intentionally invokes the same package-local helpers
# that Kaggle invokes on its first non-turn-order action.  It catches manually
# assembled archives and makes r197 receipt/sidecar binding fail before upload.
if [[ "$RTP_MODE" == "off" || "$RTP_MODE" == "direct" || "$RTP_MODE" == "recursive" ]]; then
  "$PYTHON" - "$STAGE" "$RTP_MODE" <<'PY'
import importlib.util
import os
from pathlib import Path
import sys

stage = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
os.chdir(stage)
sys.path.insert(0, str(stage))
spec = importlib.util.spec_from_file_location("submission_profile_check", stage / "main.py")
if spec is None or spec.loader is None:
    raise SystemExit("ERROR: packaged submission entrypoint cannot be imported")
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
profile = agent_mod._apply_runtime_profile()
if profile.get("rtp_mode") != mode:
    raise SystemExit("ERROR: packaged RTP mode changed before smoke")
if mode in {"direct", "recursive"}:
    model_digest = agent_mod._assert_profile_model_identity(profile)
    if mode == "direct":
        agent_mod._assert_direct_rtp_binding(profile, model_digest=model_digest)
    else:
        agent_mod._assert_recursive_rtp_binding(profile, model_digest=model_digest)
print("OK: packaged", mode, "RTP profile binding")
PY
fi

if [[ "$DIRECT_NO_SEARCH_ASSETS" == "0" ]]; then
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
else
  if [[ -e "$STAGE/search_config.json" || -e "$STAGE/belief_decks.json" ]]; then
    echo "ERROR: direct-policy package unexpectedly contains search assets" >&2
    exit 1
  fi
fi

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
    -u POKEBOT_MATCHUP_ADAPTER_RUNTIME \
    -u POKEBOT_PUBLIC_MATCHUP_TREE_PATH \
    POKEBOT_USE_RECURSIVE_TURN_PLANNER=1 \
    POKEBOT_RTP_CHECKPOINT=/definitely/not/a/packaged/rtp-sidecar.pt \
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
from poke_bot import checkpoint as checkpoint_mod
checkpoint_payload = checkpoint_mod.load_checkpoint(
    stage / "model.pt", map_location="cpu"
)
runtime_profile_path = stage / "runtime_profile.json"
if runtime_profile_path.is_file():
    runtime_profile = json.loads(runtime_profile_path.read_text())
    rtp_mode = runtime_profile.get("rtp_mode")
    if rtp_mode == "off" or (
        rtp_mode is None and runtime_profile["recursive_turn_planner"] == "disabled"
    ):
        assert runtime_profile["display"] == "NO RTP"
        assert runtime_profile["rtp_sidecar_packaged"] is False
        assert agent_mod._POLICY.use_recursive_turn_planner is False
        assert agent_mod._POLICY._rtp_bridge is None
        assert os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] == "0"
        assert "POKEBOT_RTP_CHECKPOINT" not in os.environ
        assert not (stage / "rtp_shadow_planner.pt").exists()
        print("OK: submitted runtime is NO RTP despite hostile inherited RTP env")
    elif rtp_mode == "direct":
        assert runtime_profile["display"] == "DIRECT RTP"
        assert runtime_profile["rtp_sidecar_packaged"] is True
        assert agent_mod._POLICY.use_recursive_turn_planner is True
        assert agent_mod._POLICY._rtp_bridge is not None
        assert getattr(agent_mod._POLICY._rtp_bridge, "_submission_direct_only", False)
        assert os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] == "1"
        sidecar = stage / "rtp_shadow_planner.pt"
        assert sidecar.is_file()
        assert os.environ["POKEBOT_RTP_CHECKPOINT"] == str(sidecar.resolve())
        assert (
            os.environ["POKEBOT_RTP_PARENT_CHECKPOINT_SHA256"]
            == runtime_profile["parent_checkpoint_sha256"]
        )
        assert "POKEBOT_RTP_ALLOW_UNTRAINED" not in os.environ
        config = agent_mod._assert_direct_rtp_binding(
            runtime_profile,
            model_digest=agent_mod._assert_profile_model_identity(runtime_profile),
        )
        agent_mod._assert_live_recursive_config(agent_mod._POLICY, config)
        print("OK: submitted runtime is direct bridge only")
    elif rtp_mode == "recursive":
        assert runtime_profile["display"] == "RTP"
        assert runtime_profile["rtp_sidecar_packaged"] is True
        assert agent_mod._POLICY.use_recursive_turn_planner is True
        assert agent_mod._POLICY._rtp_bridge is not None
        assert os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] == "1"
        sidecar = stage / "rtp_shadow_planner.pt"
        assert sidecar.is_file()
        assert os.environ["POKEBOT_RTP_CHECKPOINT"] == str(sidecar.resolve())
        assert os.environ["POKEBOT_RTP_SERVING_QUALIFIED"] == "1"
        assert (
            os.environ["POKEBOT_RTP_PARENT_CHECKPOINT_SHA256"]
            == runtime_profile["parent_checkpoint_sha256"]
        )
        assert os.environ["POKEBOT_RTP_PROMOTION_RECEIPT"] == str(
            (stage / "rtp_promotion_receipt.json").resolve()
        )
        assert (
            os.environ["POKEBOT_RTP_PROMOTION_RECEIPT_SHA256"]
            == runtime_profile["rtp_promotion_receipt_sha256"]
        )
        config = agent_mod._assert_recursive_rtp_binding(
            runtime_profile,
            model_digest=agent_mod._assert_profile_model_identity(runtime_profile),
        )
        agent_mod._assert_live_recursive_config(agent_mod._POLICY, config)
        assert int(config["max_neural_passes"]) == 256
        assert int(agent_mod._POLICY._rtp_bridge.max_action_combos) == 1024
        print("OK: submitted runtime is receipt-bound recursive RTP")
    elif runtime_profile["recursive_turn_planner"] == "enabled":
        assert runtime_profile["display"] == "RTP"
        assert runtime_profile["rtp_sidecar_packaged"] is True
        assert agent_mod._POLICY.use_recursive_turn_planner is True
        assert agent_mod._POLICY._rtp_bridge is not None
        assert os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] == "1"
        sidecar = stage / "rtp_shadow_planner.pt"
        assert sidecar.is_file()
        assert os.environ["POKEBOT_RTP_CHECKPOINT"] == str(sidecar.resolve())
        print("OK: submitted runtime is RTP with checksum-bound packaged sidecar")
    else:
        raise AssertionError(runtime_profile)
if agent_mod._checkpoint_has_trained_matchup_adapter_bank(checkpoint_payload):
    import hashlib

    tree_path = stage / "matchup_tree.json"
    assert tree_path.is_file(), "trained adapter package must ship matchup_tree.json"
    model = agent_mod._MODEL
    policy = agent_mod._POLICY
    assert model is not None and policy is not None
    bank = model.matchup_adapter_bank
    router = policy._matchup_adapter_shadow_router
    expected_tree_digest = "sha256:" + hashlib.sha256(
        tree_path.read_bytes()
    ).hexdigest()
    assert checkpoint_payload["model_config"]["matchup_adapters_enabled"] is False
    assert policy.matchup_adapter_runtime is True
    assert bank.enabled is True
    assert not any(parameter.requires_grad for parameter in bank.parameters())
    assert router.tree.runtime_enabled is True
    assert router.tree.digest == expected_tree_digest
    print("OK: trained adapter package enabled + frozen + exact tree")
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
    -u POKEBOT_MATCHUP_ADAPTER_RUNTIME \
    -u POKEBOT_PUBLIC_MATCHUP_TREE_PATH \
    POKEBOT_USE_RECURSIVE_TURN_PLANNER=1 \
    POKEBOT_RTP_CHECKPOINT=/definitely/not/a/packaged/rtp-sidecar.pt \
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
if budget is not None:
    assert budget.enabled is False
    assert budget.searches_used == 0
    assert budget.disabled_reason is None, budget.disabled_reason
    assert budget.consecutive_search_failures == 0
else:
    # Direct-policy archives intentionally omit both search assets.  In that
    # stricter package shape there is no search budget object to exercise.
    assert not (stage / "search_config.json").exists()
    assert not (stage / "belief_decks.json").exists()
assert policy is not None and policy.use_mcts is False
assert policy.belief_mcts is False
assert policy.last_result is None
# These counters were added after the first frozen V5 specialists.  Their
# absence in an older, checksum-pinned policy runtime is equivalent to the
# untouched zero/None state and must not force replacement of that runtime.
assert getattr(policy, "last_search_fallback_reason", None) is None
assert int(getattr(policy, "fail_closed_count", 0)) == 0
assert agent_calls > 0
if budget is not None:
    assert budget.final_greedy_reserve_s == 20.0
print(
    "OK: packaged policy-only default",
    f"calls={agent_calls}",
    "mcts_calls=0",
    (
        f"final_reserve={budget.final_greedy_reserve_s:.0f}s"
        if budget is not None
        else "search_budget=absent"
    ),
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
runtime_profile_path = bundle.parent / "stage" / "runtime_profile.json"
runtime_profile = (
    json.loads(runtime_profile_path.read_text())
    if runtime_profile_path.is_file()
    else {}
)
search_config_path = bundle.parent / "stage" / "search_config.json"
belief_decks_path = bundle.parent / "stage" / "belief_decks.json"
payload = {
    "schema": "poke_bot.submission_turn_order_attestation/v1",
    "file_sha256": "sha256:" + digest,
    "turn_order_preference": turn_order,
    "recursive_turn_planner": runtime_profile.get(
        "recursive_turn_planner", "default_off"
    ),
    "rtp_mode": runtime_profile.get("rtp_mode", "default_off"),
    "submission_message_required_literal": runtime_profile.get("display"),
    "model_checkpoint_sha256": runtime_profile.get("model_checkpoint_sha256"),
    "rtp_checkpoint_sha256": runtime_profile.get("rtp_checkpoint_sha256"),
    "rtp_config_sha256": runtime_profile.get("rtp_config_sha256"),
    "max_neural_passes": runtime_profile.get("max_neural_passes"),
    "max_action_combos": runtime_profile.get("max_action_combos"),
    "required_neural_passes": runtime_profile.get("required_neural_passes"),
    "rtp_promotion_receipt_sha256": runtime_profile.get(
        "rtp_promotion_receipt_sha256"
    ),
    "go_first_if_offered": turn_order == "first_if_allowed",
    "go_second_if_offered": turn_order == "second_if_allowed",
    "belief_mcts_default": False,
    "belief_mcts_leaf_evaluator": "trained_checkpoint_policy_value_head",
    "belief_mcts_leaf_evaluator_checkpoint": "submission_model_pt",
    "belief_mcts_hard_cap_s": 600.0,
    "belief_mcts_internal_deadline_s": 540.0,
    "belief_mcts_final_greedy_reserve_s": 20.0,
    "search_assets_packaged": search_config_path.is_file(),
    "search_config_sha256": (
        "sha256:" + hashlib.sha256(search_config_path.read_bytes()).hexdigest()
        if search_config_path.is_file()
        else None
    ),
    "belief_decks_sha256": (
        "sha256:" + hashlib.sha256(belief_decks_path.read_bytes()).hexdigest()
        if belief_decks_path.is_file()
        else None
    ),
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
