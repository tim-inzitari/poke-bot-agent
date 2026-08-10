#!/usr/bin/env python3
"""Stage/activate owner Alakazam RTP hard-swap loop (GOAL revision 175).

Cycle:
  expert refresh (last 5 days) -> CE rebootstrap from checkpoint ->
  Kaggle submit -> self-play loop (1024 mirrors, fill 8196, >=1024 Grimmsnarl
  pinned to f20efb20f5c3) -> every 5 iters expert refresh + Kaggle again.

Fail closed on missing deck/checkpoint/corpus/Grimmsnarl pin.
Combo head remains off. All other heads + guide stay live/nonzero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetypes import ALAKAZAM_FINAL_REFRESH_REPRESENTATIVE  # noqa: E402

SCHEMA = "poke_bot.alakazam_rtp_owner_hard_swap_r175/v1"
SPECIALIST_ID = "alakazam"
OWNER_REV = 175
GAMES_PER_ITER = 8196
SELF_PLAY_GAMES = 1024
SELF_PLAY_FRAC = SELF_PLAY_GAMES / GAMES_PER_ITER  # exact 1024 via round()
GRIMMSNARL_FLOOR = 1024
ITER_CEILING = 300
GUIDE_LOSS_WEIGHT = 0.05
SETUP_BOARD_LOSS_WEIGHT = 0.025
COMBO_LOSS_WEIGHT = 0.0
EXPERT_REHEARSAL_EVERY = 5
EXPERT_REHEARSAL_EPOCHS = 5
BOOTSTRAP_EPOCHS = 25

GRIMMSNARL_SHA256 = (
    "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
)
GRIMMSNARL_PREFIX = "f20efb20f5c3"
GRIMMSNARL_CHECKPOINT = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "final_format_marnie_r104_h10_i_v6_8k/checkpoints/iter_00007.pt"
)
GRIMMSNARL_PACKAGE_ID = "specialist-marnie-final-format-h10-f20efb20f5c3"
GRIMMSNARL_FAMILY = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/_protected/models/"
    "marnie-iter9-training-freeze-r163"
)

PARENT_CHECKPOINT = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "final_format_alakazam_r79_h10_i_v6_8k/checkpoints/iter_00020.pt"
)
PARENT_FAMILY = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/_protected/models/"
    "final-format-alakazam-r79-h10-refresh-v1"
)

DECK_CSV = Path(
    "/home/inzi/poke-bot-agent/decks/archetype-samples/"
    "alakazam-owner-rtp-pilot-r175.csv"
)
GUIDE = Path(
    "/home/inzi/poke-bot-agent/config/deck_guides/alakazam-final-refresh.yaml"
)
PROTOCOL = Path("/home/inzi/poke-bot-agent/config/rl_protocol.yaml")

WINDOW_START = "2026-08-01"
WINDOW_END = "2026-08-05"
EXPERT_OUT = Path(
    "/home/inzi/poke-bot-agent/data/bootstrap/"
    "expert-alakazam-last5-2026-08-01-2026-08-05-r175/alakazam"
)
EXPERT_STATUS = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "alakazam-last5-expert-materialize-r175.json"
)
ARCHIVE_CANDIDATES = (
    Path("/tmp/truenas_main/poke-bot-agent/archive/episode-days"),
    Path("/mnt/Main/main/poke-bot-agent/archive/episode-days"),
    Path("/home/inzi/poke-bot-agent/archive/episode-days"),
)

RUNTIME_ROOT = Path(
    "/home/inzi/poke-bot-agent/outputs/final_format_alakazam_rtp_r175"
)
REGISTRY_PATH = RUNTIME_ROOT / "runtime" / "specialist_runtime_registry_h10_r175.json"
SELECTOR_PATH = RUNTIME_ROOT / "runtime" / "specialist_runtime_h10_r175.env"
READY_PATH = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "final-format-alakazam-rtp-r175-bootstrap-ready.json"
)
CONTRACT_PATH = Path(
    "/home/inzi/poke-bot-agent/state/alakazam-rtp-owner-hard-swap-r175.json"
)
STATE_PATH = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "alakazam-rtp-owner-hard-swap-loop-r175.json"
)
BOOTSTRAP_RUN = Path(
    "/home/inzi/poke-bot-agent/outputs/bootstrap/"
    "final_format_alakazam_rtp_r175"
)
CPU_PACK = Path(
    "/home/inzi/poke-bot-agent/outputs/bootstrap/cpu-packs/"
    "final_format_alakazam_rtp_r175"
)
FAMILY_OUT = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/_protected/models/"
    "final-format-alakazam-rtp-r175-expert-bootstrap-v1"
)
RUN_NAME = "final_format_alakazam_rtp_r175_i_v6_8k"
PARENT_REGISTRY = Path(
    "/home/inzi/poke-bot-agent/outputs/final_format_alakazam_r79/runtime/"
    "specialist_runtime_registry_h10_r105_fusion_v3_directional_learner1536_iter20_exact.json"
)

LIVE_HEADS = (
    "policy",
    "value",
    "archetype",
    "opponent_hand",
    "opponent_remainder",
    "lethal_threat",
    "prize_race",
    "action_q",
    "action_type",
    "action_target",
    "action_resource",
    "action_utility",
    "tactical_outcome",
    "opponent_response",
    "resource_forecast",
    "game_phase",
    "outcome_distribution",
    "remaining_turns",
    "setup_board_outcome",
    "guide_strategic_directional_v2",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _set_arg(args_list: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    skip = False
    for item in args_list:
        if skip:
            skip = False
            continue
        if item == flag:
            skip = True
            continue
        out.append(item)
    out.extend([flag, value])
    return out


def resolve_archive_dir() -> Path:
    for candidate in ARCHIVE_CANDIDATES:
        probe = candidate / f"pokemon-tcg-ai-battle-episodes-{WINDOW_START}.zip"
        if probe.is_file():
            return candidate
    raise RuntimeError(
        "FAIL_CLOSED: last-5-day Alakazam archive days missing under "
        + ", ".join(str(path) for path in ARCHIVE_CANDIDATES)
    )


def verify_grimmsnarl_pin() -> dict[str, Any]:
    matches = []
    for path in (GRIMMSNARL_CHECKPOINT, GRIMMSNARL_FAMILY / "model.pt"):
        if not path.is_file():
            continue
        digest = sha256_file(path)
        if GRIMMSNARL_PREFIX in digest:
            matches.append({"path": str(path), "sha256": digest})
    unique = {row["sha256"] for row in matches}
    if len(unique) != 1 or GRIMMSNARL_SHA256 not in unique:
        raise RuntimeError(
            "FAIL_CLOSED: Grimmsnarl f20efb20f5c3 pin ambiguous or missing: "
            f"{matches}"
        )
    if matches[0]["sha256"] != GRIMMSNARL_SHA256:
        raise RuntimeError("FAIL_CLOSED: Grimmsnarl digest mismatch")
    return {
        "package_id": GRIMMSNARL_PACKAGE_ID,
        "checkpoint": str(GRIMMSNARL_CHECKPOINT),
        "checkpoint_sha256": GRIMMSNARL_SHA256,
        "family": str(GRIMMSNARL_FAMILY),
        "floor_games_per_set": GRIMMSNARL_FLOOR,
    }


def verify_deck() -> dict[str, Any]:
    if not DECK_CSV.is_file():
        raise RuntimeError(f"FAIL_CLOSED missing deck csv: {DECK_CSV}")
    cards = [
        int(line.strip())
        for line in DECK_CSV.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cards) != 60:
        raise RuntimeError(f"FAIL_CLOSED deck size {len(cards)} != 60")
    bound = list(ALAKAZAM_FINAL_REFRESH_REPRESENTATIVE)
    if sorted(cards) != sorted(bound):
        raise RuntimeError(
            "FAIL_CLOSED: archetypes.ALAKAZAM_FINAL_REFRESH_REPRESENTATIVE "
            "does not match owner pilot CSV; update archetypes.py first"
        )
    return {
        "path": str(DECK_CSV),
        "sha256": sha256_file(DECK_CSV),
        "card_count": 60,
        "list_id": "alakazam-owner-rtp-pilot-r175",
        "dudunsparce_count_note": (
            "Owner prose said 2x Dudunsparce but Pokemon(19)/60-card math "
            "requires 3x; bound 3x Dudunsparce"
        ),
        "spelling_mismatches": [
            "Boss's Orders -> Boss\u2019s Orders (curly apostrophe)",
            "Xerosic's Mechinations -> Xerosic\u2019s Machinations",
            "Lana's Aid -> Lana\u2019s Aid (curly apostrophe)",
            "Basic Psychic Energy -> Basic {P} Energy",
        ],
    }


def verify_parent() -> dict[str, Any]:
    if not PARENT_CHECKPOINT.is_file():
        raise RuntimeError(f"FAIL_CLOSED missing parent checkpoint: {PARENT_CHECKPOINT}")
    digest = sha256_file(PARENT_CHECKPOINT)
    if not (PARENT_FAMILY / "model.pt").is_file():
        raise RuntimeError(f"FAIL_CLOSED missing parent family: {PARENT_FAMILY}")
    return {
        "checkpoint": str(PARENT_CHECKPOINT),
        "checkpoint_sha256": digest,
        "family": str(PARENT_FAMILY),
        "family_model_sha256": sha256_file(PARENT_FAMILY / "model.pt"),
    }


def write_contract(
    *,
    deck: dict[str, Any],
    parent: dict[str, Any],
    grimmsnarl: dict[str, Any],
    archive_dir: Path,
) -> dict[str, Any]:
    contract = {
        "schema": SCHEMA,
        "owner_decision_revision": OWNER_REV,
        "status": "armed",
        "recorded_at_utc": utc_now(),
        "specialist_id": SPECIALIST_ID,
        "loop": {
            "order": [
                "expert_refresh_last_5_days",
                "ce_rebootstrap_from_checkpoint",
                "kaggle_submit_first_if_allowed",
                "self_play_public_mix_rl",
            ],
            "every_n_iterations": {
                "n": EXPERT_REHEARSAL_EVERY,
                "actions": ["expert_refresh", "kaggle_submit"],
            },
        },
        "expert_window": {
            "start": WINDOW_START,
            "end": WINDOW_END,
            "archive_dir": str(archive_dir),
            "out_dir": str(EXPERT_OUT),
            "status": str(EXPERT_STATUS),
        },
        "bootstrap": {
            "parent": parent,
            "epochs": BOOTSTRAP_EPOCHS,
            "family_out": str(FAMILY_OUT),
            "ready": str(READY_PATH),
            "run_dir": str(BOOTSTRAP_RUN),
            "guide_loss_weight": GUIDE_LOSS_WEIGHT,
            "setup_board_outcome_loss_weight": SETUP_BOARD_LOSS_WEIGHT,
            "combo_state_loss_weight": COMBO_LOSS_WEIGHT,
            "combo_state_head_enabled": True,
            "combo_state_route_enabled": False,
            "expanded_heads_enabled": True,
            "decision_fusion_enabled": True,
            "guide_training_mode": "strategic_directional_v2",
            "live_heads": list(LIVE_HEADS),
            "disabled_heads": ["combo_state"],
        },
        "self_play": {
            "games_per_iteration": GAMES_PER_ITER,
            "self_play_mirrors": SELF_PLAY_GAMES,
            "self_play_fraction": SELF_PLAY_FRAC,
            "public_mix_fill_games": GAMES_PER_ITER - SELF_PLAY_GAMES,
            "grimmsnarl_floor_per_set": GRIMMSNARL_FLOOR,
            "grimmsnarl": grimmsnarl,
            "iteration_ceiling": ITER_CEILING,
            "expert_rehearsal_every": EXPERT_REHEARSAL_EVERY,
            "expert_rehearsal_epochs": EXPERT_REHEARSAL_EPOCHS,
            "run_name": RUN_NAME,
            "rtp_enabled": True,
            "rtp_checkpoint": (
                "/home/inzi/poke-bot-agent/outputs/rtp_fleet/"
                "alakazam-r175.live/rtp_shadow_planner.pt"
            ),
        },
        "deck": deck,
        "units": {
            "orchestrator": "pokebot-final-format-alakazam-rtp-r175-orchestrator.service",
            "bootstrap": "pokebot-final-format-alakazam-rtp-r175-bootstrap.service",
            "rl": "pokebot-final-format-alakazam-rtp-r175-rl.service",
            "held_off": [
                "pokebot-final-format-slop-box-h10-rtp-bootstrap.service",
                "pokebot-final-format-alakazam-r79-h10.service",
            ],
        },
        "registry": str(REGISTRY_PATH),
        "selector": str(SELECTOR_PATH),
    }
    atomic_json(CONTRACT_PATH, contract)
    # Mirror under outputs/state for remote controllers that only watch outputs/.
    atomic_json(
        Path("/home/inzi/poke-bot-agent/outputs/state/alakazam-rtp-owner-hard-swap-r175.json"),
        contract,
    )
    return contract


def materialize_expert(*, archive_dir: Path, python: Path) -> None:
    EXPERT_OUT.parent.mkdir(parents=True, exist_ok=True)
    EXPERT_STATUS.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python),
        "-u",
        str(ROOT / "scripts/materialize_authoritative_guide_window_parallel.py"),
        "--start",
        WINDOW_START,
        "--end",
        WINDOW_END,
        "--archive-dir",
        str(archive_dir),
        "--out-dir",
        str(EXPERT_OUT.parent),  # parent contains per-archetype subdir writes
        "--status",
        str(EXPERT_STATUS),
        "--required-archetype",
        SPECIALIST_ID,
        "--current-deck-guide",
        SPECIALIST_ID,
        "--day-parallelism",
        "2",
        "--workers-per-day",
        "3",
        "--max-in-flight-per-day",
        "6",
        "--max-context",
        "320",
        "--memory-floor-gib",
        "16",
        "--min-records",
        "1",
    ]
    print("materialize:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def expert_corpus_ready() -> Path:
    # Prefer protected pointer if materializer emitted one.
    candidates = [
        EXPERT_OUT / "PROTECTED_EXPERT_CORPUS.json",
        EXPERT_OUT.parent / "PROTECTED_EXPERT_CORPUS.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    # Fall back to building a pinned pointer from a feature manifest if present.
    manifests = sorted(EXPERT_OUT.glob("*.features.json"))
    if not manifests:
        manifests = sorted(EXPERT_OUT.parent.glob("alakazam/**/*.features.json"))
    if not manifests:
        raise RuntimeError(
            f"FAIL_CLOSED: no Alakazam expert corpus under {EXPERT_OUT}"
        )
    # Use existing r109 packager pattern if available; otherwise require pointer.
    raise RuntimeError(
        "FAIL_CLOSED: materialization completed without PROTECTED_EXPERT_CORPUS.json; "
        f"found manifests={[str(path) for path in manifests[:5]]}"
    )


def run_bootstrap(*, python: Path, expert: Path) -> None:
    guide_ready = EXPERT_OUT / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    if not guide_ready.is_file():
        # Some materializers place ready beside parent.
        alt = EXPERT_OUT.parent / "alakazam" / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
        if alt.is_file():
            guide_ready = alt
    curriculum_dir = Path(
        "/home/inzi/poke-bot-agent/state/final_format_alakazam_curriculum_r79"
    )
    role = curriculum_dir / "alakazam-strategic-head-roles-r56.json"
    spec = curriculum_dir / "alakazam-strategic-curriculum-r56.json"
    validation = curriculum_dir / "alakazam-strategic-curriculum-validation-r56.json"
    for path in (role, spec, validation, GUIDE, expert, PARENT_FAMILY):
        if not path.exists():
            raise RuntimeError(f"FAIL_CLOSED bootstrap input missing: {path}")

    cmd = [
        str(python),
        "-u",
        str(ROOT / "scripts/run_specialist_expert_bootstrap.py"),
        "--archetype",
        SPECIALIST_ID,
        "--expert-corpus",
        str(expert),
        "--core-family",
        str(PARENT_FAMILY),
        "--allow-h10-specialist-parent",
        "--registry-root",
        str(FAMILY_OUT.parent),
        "--family",
        FAMILY_OUT.name,
        "--ready",
        str(READY_PATH),
        "--run-name",
        "final_format_alakazam_rtp_r175_bootstrap",
        "--run-dir",
        str(BOOTSTRAP_RUN),
        "--cpu-pack-root",
        str(CPU_PACK),
        "--epochs",
        str(BOOTSTRAP_EPOCHS),
        "--expanded-heads",
        "--decision-fusion",
        "--rl-protocol",
        str(PROTOCOL),
        "--current-deck-guide-contract",
        str(GUIDE),
        "--current-deck-guide-version",
        "powerful-hand-v1",
        "--strategic-curriculum-spec",
        str(spec),
        "--strategic-head-role-map",
        str(role),
        "--strategic-validation-receipt",
        str(validation),
        # Keep H10's combo module/loadable tensors; r175 disables only its
        # dedicated action/guide route through the environment below.
    ]
    if guide_ready.is_file():
        cmd.extend(["--current-deck-guide-corpus-ready", str(guide_ready)])
    print("bootstrap:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["POKEBOT_COMBO_STATE_HEAD_ENABLED"] = "1"
    env["POKEBOT_COMBO_STATE_ROUTE_ENABLED"] = "0"
    env["POKEBOT_COMBO_STATE_ROUTE_SPECIALIST"] = "alakazam"
    env["POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED"] = "1"
    env["POKEBOT_EXPANDED_HEADS_ENABLED"] = "1"
    env["POKEBOT_DECISION_FUSION_ENABLED"] = "1"
    env["POKEBOT_DECISION_FUSION_RUNTIME_ENABLED"] = "1"
    env["POKEBOT_H10_CAPACITY_ENABLED"] = "1"
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def stage_registry(*, expert: Path) -> dict[str, Any]:
    if not READY_PATH.is_file():
        raise RuntimeError(f"FAIL_CLOSED missing bootstrap ready: {READY_PATH}")
    if not PARENT_REGISTRY.is_file():
        raise RuntimeError(f"FAIL_CLOSED missing parent registry: {PARENT_REGISTRY}")
    source = read_json(PARENT_REGISTRY)
    source["owner_decision_revision"] = OWNER_REV
    source["minimum_terminal_iteration"] = 5
    source["iteration_ceiling"] = ITER_CEILING
    source["isolated_refresh_contract"] = {
        "schema": "poke_bot.alakazam_rtp_isolated_runtime_r175/v1",
        "specialist_id": SPECIALIST_ID,
        "games_per_iteration": GAMES_PER_ITER,
        "self_play_games": SELF_PLAY_GAMES,
        "self_play_fraction": SELF_PLAY_FRAC,
        "public_mix_games": GAMES_PER_ITER - SELF_PLAY_GAMES,
        "grimmsnarl_floor_per_set": GRIMMSNARL_FLOOR,
        "grimmsnarl_checkpoint_sha256": GRIMMSNARL_SHA256,
        "grimmsnarl_package_id": GRIMMSNARL_PACKAGE_ID,
        "learner_epochs_per_iteration": 1,
        "expert_rehearsal_every": EXPERT_REHEARSAL_EVERY,
        "expert_rehearsal_epochs": EXPERT_REHEARSAL_EPOCHS,
        "iteration_ceiling": ITER_CEILING,
        "combo_state_loss_weight": COMBO_LOSS_WEIGHT,
        "combo_state_head_enabled": True,
        "combo_state_route_enabled": False,
        "guide_loss_weight": GUIDE_LOSS_WEIGHT,
        "setup_board_outcome_loss_weight": SETUP_BOARD_LOSS_WEIGHT,
        "live_heads": list(LIVE_HEADS),
        "disabled_heads": ["combo_state"],
    }

    common = list(source.get("common_trainer_args") or [])
    common = _set_arg(common, "--games-per-iter", str(GAMES_PER_ITER))
    common = _set_arg(common, "--expert-rehearsal-every", str(EXPERT_REHEARSAL_EVERY))
    common = _set_arg(common, "--expert-rehearsal-epochs", str(EXPERT_REHEARSAL_EPOCHS))
    common = _set_arg(common, "--archetype-aux-loss-weight", "0.05")
    common = _set_arg(common, "--opp-hand-loss-weight", "0.05")
    common = _set_arg(common, "--opp-remainder-loss-weight", "0.05")
    common = _set_arg(common, "--lethal-threat-loss-weight", "0.025")
    common = _set_arg(common, "--prize-race-loss-weight", "0.025")
    source["common_trainer_args"] = common

    specialists = dict(source.get("specialists") or {})
    row = dict(specialists.get(SPECIALIST_ID) or {})
    row["run_name"] = RUN_NAME
    row["iteration_ceiling"] = ITER_CEILING
    row["minimum_terminal_iteration"] = 5
    row["guide_loss_weight"] = GUIDE_LOSS_WEIGHT
    row["setup_board_outcome_loss_weight"] = SETUP_BOARD_LOSS_WEIGHT
    row["combo_state_loss_weight"] = COMBO_LOSS_WEIGHT
    row["combo_state_head_enabled"] = True
    row["combo_state_route_enabled"] = False
    row["guide_training_mode"] = "strategic_directional_v2"
    row["guide_id"] = SPECIALIST_ID
    row["guide_contract"] = str(GUIDE)
    row["guide_contract_sha256"] = sha256_file(GUIDE)
    row["expert_manifest"] = str(expert)
    row["expert_manifest_sha256"] = sha256_file(expert)
    row["initial_checkpoint"] = str(FAMILY_OUT / "model.pt")
    if (FAMILY_OUT / "model.pt").is_file():
        row["initial_checkpoint_sha256"] = sha256_file(FAMILY_OUT / "model.pt")
    row["log"] = str(RUNTIME_ROOT / "logs" / "rl.log")
    row["owner_grimmsnarl_pin"] = {
        "package_id": GRIMMSNARL_PACKAGE_ID,
        "checkpoint_sha256": GRIMMSNARL_SHA256,
        "floor_games_per_set": GRIMMSNARL_FLOOR,
    }
    # Ensure combo is not required.
    targets = [
        str(value)
        for value in row.get("expert_required_target_coverage") or ()
        if str(value) != "combo_state_rows"
    ]
    row["expert_required_target_coverage"] = targets
    specialists[SPECIALIST_ID] = row
    source["specialists"] = specialists

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(REGISTRY_PATH, source)
    atomic_text(
        SELECTOR_PATH,
        "POKEBOT_ACTIVE_SPECIALIST=alakazam\n"
        f"PURE_RL_SELF_PLAY_FRAC={SELF_PLAY_FRAC:.16f}\n"
        "POKEBOT_COMBO_STATE_HEAD_ENABLED=1\n"
        "POKEBOT_COMBO_STATE_ROUTE_ENABLED=0\n"
        "POKEBOT_COMBO_STATE_ROUTE_SPECIALIST=alakazam\n"
        "POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED=1\n"
        "POKEBOT_EXPANDED_HEADS_ENABLED=1\n"
        "POKEBOT_DECISION_FUSION_ENABLED=1\n"
        "POKEBOT_DECISION_FUSION_RUNTIME_ENABLED=1\n"
        "POKEBOT_H10_CAPACITY_ENABLED=1\n"
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER=1\n"
        "POKEBOT_RTP_CHECKPOINT=/home/inzi/poke-bot-agent/outputs/rtp_fleet/alakazam-r175.live/rtp_shadow_planner.pt\n"
        "POKEBOT_RTP_SPECIALIST_ID=alakazam\n"
        "POKEBOT_RTP_SIZING_PROFILE=pure_rl\n",
    )
    return source


def queue_kaggle_milestone() -> dict[str, Any]:
    """Queue one nonblocking first_if_allowed milestone from bootstrap family."""
    model = FAMILY_OUT / "model.pt"
    if not model.is_file():
        raise RuntimeError("FAIL_CLOSED: bootstrap model missing before Kaggle queue")
    receipt = {
        "schema": "poke_bot.alakazam_rtp_r175_kaggle_queue/v1",
        "queued_at_utc": utc_now(),
        "specialist_id": SPECIALIST_ID,
        "checkpoint": str(model),
        "checkpoint_sha256": sha256_file(model),
        "deck": str(DECK_CSV),
        "deck_sha256": sha256_file(DECK_CSV),
        "turn_order_preference": "first_if_allowed",
        "status": "queued_request",
        "note": (
            "Orchestrator stages identity for the managed kaggle queue; "
            "pokebot-kaggle-submission-queue.service consumes authorized copies."
        ),
    }
    out = Path(
        "/home/inzi/poke-bot-agent/outputs/state/"
        "alakazam-rtp-r175-kaggle-queue-request.json"
    )
    atomic_json(out, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "materialize",
            "bootstrap",
            "register",
            "queue-kaggle",
            "full",
        ),
        default="full",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"),
    )
    parser.add_argument(
        "--skip-materialize-if-ready",
        action="store_true",
        help="Reuse an already-built PROTECTED_EXPERT_CORPUS when present.",
    )
    args = parser.parse_args()

    deck = verify_deck()
    parent = verify_parent()
    grimmsnarl = verify_grimmsnarl_pin()
    archive_dir = resolve_archive_dir()
    contract = write_contract(
        deck=deck, parent=parent, grimmsnarl=grimmsnarl, archive_dir=archive_dir
    )
    atomic_json(
        STATE_PATH,
        {
            "schema": "poke_bot.alakazam_rtp_owner_hard_swap_loop_state_r175/v1",
            "updated_at_utc": utc_now(),
            "phase": "preflight_ok",
            "contract": str(CONTRACT_PATH),
            "contract_sha256": sha256_file(CONTRACT_PATH),
        },
    )
    print(json.dumps({"preflight": "ok", "contract": str(CONTRACT_PATH)}, indent=2))
    if args.phase == "preflight":
        return 0

    expert: Path | None = None
    if args.phase in {"materialize", "full"}:
        if args.skip_materialize_if_ready:
            try:
                expert = expert_corpus_ready()
            except RuntimeError:
                expert = None
        if expert is None:
            materialize_expert(archive_dir=archive_dir, python=args.python)
            expert = expert_corpus_ready()
        atomic_json(
            STATE_PATH,
            {
                "schema": "poke_bot.alakazam_rtp_owner_hard_swap_loop_state_r175/v1",
                "updated_at_utc": utc_now(),
                "phase": "expert_materialized",
                "expert": str(expert),
                "expert_sha256": sha256_file(expert),
            },
        )
    if args.phase == "materialize":
        return 0

    if expert is None:
        expert = expert_corpus_ready()

    if args.phase in {"bootstrap", "full"}:
        run_bootstrap(python=args.python, expert=expert)
        atomic_json(
            STATE_PATH,
            {
                "schema": "poke_bot.alakazam_rtp_owner_hard_swap_loop_state_r175/v1",
                "updated_at_utc": utc_now(),
                "phase": "bootstrap_ready",
                "ready": str(READY_PATH),
                "family": str(FAMILY_OUT),
            },
        )
    if args.phase == "bootstrap":
        return 0

    if args.phase in {"register", "full"}:
        stage_registry(expert=expert)
        atomic_json(
            STATE_PATH,
            {
                "schema": "poke_bot.alakazam_rtp_owner_hard_swap_loop_state_r175/v1",
                "updated_at_utc": utc_now(),
                "phase": "registry_staged",
                "registry": str(REGISTRY_PATH),
                "selector": str(SELECTOR_PATH),
            },
        )
    if args.phase == "register":
        return 0

    if args.phase in {"queue-kaggle", "full"}:
        receipt = queue_kaggle_milestone()
        atomic_json(
            STATE_PATH,
            {
                "schema": "poke_bot.alakazam_rtp_owner_hard_swap_loop_state_r175/v1",
                "updated_at_utc": utc_now(),
                "phase": "kaggle_queued",
                "kaggle": receipt,
                "next": "start_rl_unit",
            },
        )

    print(
        json.dumps(
            {
                "status": "staged",
                "contract": str(CONTRACT_PATH),
                "registry": str(REGISTRY_PATH),
                "grimmsnarl": grimmsnarl,
                "self_play": contract["self_play"],
                "heads_live": LIVE_HEADS,
                "combo_head": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
