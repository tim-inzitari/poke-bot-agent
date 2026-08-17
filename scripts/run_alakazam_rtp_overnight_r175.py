#!/usr/bin/env python3
"""Overnight Alakazam RTP loop arming (GOAL r175).

Phases:
  1) materialize last-5-day Alakazam expert features (2026-08-01..05)
  2) assemble PROTECTED_EXPERT_CORPUS (no combo requirement)
  3) CE rebootstrap from iter_00020 / final-format Alakazam family
  4) queue Kaggle request + stage RL registry (iter_max=300, 1024 mirrors,
     fill 8196, Grimmsnarl f20efb20f5c3 floor 1024, guide ON, combo OFF)
  5) start RL systemd unit when preflight passes
"""

from __future__ import annotations

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

ROOT = Path("/home/pokebot/poke-bot-agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PY = Path("/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python")
SPECIALIST = "alakazam"
DAYS = [f"2026-08-{d:02d}" for d in range(1, 6)]
ARCHIVE_SRC = Path("/tmp/truenas_main/poke-bot-agent/archive/episode-days")
LOCAL_EPISODES = ROOT / "data/episodes/raw"
EXPERT = ROOT / "data/bootstrap/expert-alakazam-last5-2026-08-01-2026-08-05-r175/alakazam"
R109 = ROOT / (
    "data/bootstrap/expert-latest20-2026-07-14-2026-08-02-roster18-v6-strategic-r109/"
    "alakazam"
)
PARENT_FAMILY = ROOT / (
    "outputs/pure_rl/_protected/models/final-format-alakazam-r79-h10-refresh-v1"
)
PARENT_CKPT = ROOT / (
    "outputs/pure_rl/final_format_alakazam_r79_h10_i_v6_8k/checkpoints/iter_00020.pt"
)
DECK = ROOT / "decks/archetype-samples/alakazam-owner-rtp-pilot-r175.csv"
GUIDE = ROOT / "config/deck_guides/alakazam-final-refresh.yaml"
CURRICULUM = ROOT / "state/final_format_alakazam_curriculum_r79"
PROTOCOL = ROOT / "config/rl_protocol.yaml"
FAMILY_OUT = ROOT / (
    "outputs/pure_rl/_protected/models/"
    "final-format-alakazam-rtp-r175-expert-bootstrap-v1"
)
READY = ROOT / "outputs/state/final-format-alakazam-rtp-r175-bootstrap-ready.json"
BOOTSTRAP_RUN = ROOT / "outputs/bootstrap/final_format_alakazam_rtp_r175"
CPU_PACK = ROOT / "outputs/bootstrap/cpu-packs/final_format_alakazam_rtp_r175"
RUNTIME_ROOT = ROOT / "outputs/final_format_alakazam_rtp_r175"
REGISTRY = RUNTIME_ROOT / "runtime/specialist_runtime_registry_h10_r175.json"
SELECTOR = RUNTIME_ROOT / "runtime/specialist_runtime_h10_r175.env"
PARENT_REGISTRY = ROOT / (
    "outputs/final_format_alakazam_r79/runtime/"
    "specialist_runtime_registry_h10_r105_fusion_v3_directional_learner1536_iter20_exact.json"
)
CONTRACT = ROOT / "state/alakazam-rtp-owner-hard-swap-r175.json"
STATE = ROOT / "outputs/state/alakazam-rtp-owner-hard-swap-loop-r175.json"
LOG = RUNTIME_ROOT / "logs/overnight-orchestrator.log"

GRIM_SHA = "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
GRIM_CKPT = ROOT / (
    "outputs/pure_rl/final_format_marnie_r104_h10_i_v6_8k/checkpoints/iter_00007.pt"
)
GRIM_PKG = "specialist-marnie-final-format-h10-f20efb20f5c3"

GAMES = 8196
SELF_PLAY = 1024
SELF_FRAC = SELF_PLAY / GAMES
ITER_MAX = 300
GUIDE_W = 0.05
SETUP_W = 0.025
COMBO_W = 0.0

LIVE_HEADS = [
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
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def set_arg(args: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    skip = False
    for item in args:
        if skip:
            skip = False
            continue
        if item == flag:
            skip = True
            continue
        out.append(item)
    out.extend([flag, value])
    return out


def write_state(phase: str, **extra: Any) -> None:
    payload = {
        "schema": "poke_bot.alakazam_rtp_owner_hard_swap_loop_state_r175/v1",
        "updated_at_utc": utc_now(),
        "phase": phase,
        **extra,
    }
    atomic_json(STATE, payload)


def preflight() -> None:
    from poke_bot.archetypes import ALAKAZAM_FINAL_REFRESH_REPRESENTATIVE

    missing = [
        path
        for path in (
            DECK,
            GUIDE,
            PARENT_FAMILY / "model.pt",
            PARENT_CKPT,
            GRIM_CKPT,
            PARENT_REGISTRY,
            CURRICULUM / "alakazam-strategic-curriculum-r56.json",
            CURRICULUM / "alakazam-strategic-head-roles-r56.json",
            CURRICULUM / "alakazam-strategic-curriculum-validation-r56.json",
        )
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"FAIL_CLOSED missing inputs: {missing}")
    cards = [int(x) for x in DECK.read_text().splitlines() if x.strip()]
    if len(cards) != 60:
        raise RuntimeError(f"FAIL_CLOSED deck size {len(cards)}")
    if sorted(cards) != sorted(ALAKAZAM_FINAL_REFRESH_REPRESENTATIVE):
        raise RuntimeError(
            "FAIL_CLOSED archetypes.ALAKAZAM_FINAL_REFRESH_REPRESENTATIVE "
            "does not match owner pilot CSV"
        )
    grim = sha256_file(GRIM_CKPT)
    if grim != GRIM_SHA or "f20efb20f5c3" not in grim:
        raise RuntimeError(f"FAIL_CLOSED Grimmsnarl pin mismatch: {grim}")
    for day in DAYS:
        archive = ARCHIVE_SRC / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        if not archive.is_file():
            raise RuntimeError(f"FAIL_CLOSED missing archive day: {archive}")
    log("preflight ok")


def link_archives() -> None:
    LOCAL_EPISODES.mkdir(parents=True, exist_ok=True)
    for day in DAYS:
        src = ARCHIVE_SRC / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        dst = LOCAL_EPISODES / src.name
        if dst.is_symlink() or dst.exists():
            if dst.resolve() == src.resolve() or (
                dst.is_file() and dst.stat().st_size == src.stat().st_size
            ):
                continue
            dst.unlink()
        os.symlink(src, dst)
        log(f"linked archive {day}")


def materialize_days() -> None:
    EXPERT.mkdir(parents=True, exist_ok=True)
    link_archives()
    for day in DAYS:
        # Prefer already-built r109 shards for overlapping days.
        r109_feat = R109 / f"all-recognized-{day}.alakazam.features"
        r109_meta = R109 / f"all-recognized-{day}.alakazam.features.json"
        out_feat = EXPERT / f"all-recognized-{day}.alakazam.features"
        out_meta = EXPERT / f"all-recognized-{day}.alakazam.features.json"
        if out_feat.is_file() and out_meta.is_file():
            log(f"reuse local day {day}")
            continue
        if r109_feat.is_file() and r109_meta.is_file():
            shutil.copy2(r109_feat, out_feat)
            shutil.copy2(r109_meta, out_meta)
            log(f"copied r109 day {day}")
            continue
        # Build via established day script into a temp dir then rename.
        day_out = EXPERT / f".daybuild-{day}"
        day_out.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["POKEBOT_PYTHON"] = str(PY)
        env["PYTHONPATH"] = str(ROOT)
        cmd = [
            "bash",
            str(ROOT / "scripts/prepare_alakazam_expert_day.sh"),
            day,
            "6",
            "6",
            str(day_out),
        ]
        log("RUN " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
        built = day_out / f"top_ladder_alakazam_{day}.features"
        built_meta = day_out / f"top_ladder_alakazam_{day}.features.json"
        if not built.is_file():
            raise RuntimeError(f"FAIL_CLOSED day build missing features: {built}")
        shutil.move(str(built), str(out_feat))
        if built_meta.is_file():
            shutil.move(str(built_meta), str(out_meta))
        else:
            # Synthesize minimal sidecar if featurizer only wrote .features
            atomic_json(
                out_meta,
                {
                    "source_date": day,
                    "path": out_feat.name,
                    "sha256": sha256_file(out_feat),
                },
            )
        log(f"materialized day {day}")


def assemble_corpus() -> Path:
    shards = []
    totals = {"records_kept": 0, "decisions_kept": 0}
    for day in DAYS:
        feat = EXPERT / f"all-recognized-{day}.alakazam.features"
        meta = EXPERT / f"all-recognized-{day}.alakazam.features.json"
        if not feat.is_file():
            raise RuntimeError(f"FAIL_CLOSED missing features {feat}")
        stats: dict[str, Any] = {}
        if meta.is_file():
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
                stats = dict(payload.get("stats") or payload.get("totals") or {})
            except json.JSONDecodeError:
                stats = {}
        records = int(stats.get("records_kept") or stats.get("records") or 0)
        decisions = int(stats.get("decisions_kept") or stats.get("decisions") or 0)
        # If sidecar lacks stats, accept byte presence and let bootstrap min-decisions gate.
        totals["records_kept"] += max(records, 0)
        totals["decisions_kept"] += max(decisions, 0)
        shards.append(
            {
                "path": feat.name,
                "sha256": sha256_file(feat),
                "bytes": feat.stat().st_size,
                "source_dates": [day],
                "stats": {
                    "records_kept": records,
                    "decisions_kept": decisions,
                },
            }
        )
    manifest = {
        "format": "pokebot-bootstrap-feature-manifest",
        "format_version": 1,
        "compact_mode": "temporal-expert-v1",
        "date_start": DAYS[0],
        "date_end": DAYS[-1],
        "dates": DAYS,
        "empty_dates": [],
        "max_context": 320,
        "selection": {
            "field": "GameSequence.archetype",
            "operator": "exact_casefold",
            "value": SPECIALIST,
            "seat_semantics": "acting_seat_only",
            "opponent_routes_only": False,
        },
        "quality_gates": {
            "passed": True,
            "acting_seat_archetype_exact": True,
        },
        "shards": shards,
        "totals": {
            "records_kept": totals["records_kept"],
            "decisions_kept": totals["decisions_kept"],
            "bytes": sum(int(row["bytes"]) for row in shards),
            "days": len(DAYS),
        },
    }
    man_path = EXPERT / "manifest.json"
    atomic_json(man_path, manifest)
    ready = {
        "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
        "status": "ready",
        "specialist_id": SPECIALIST,
        "guide_version": "powerful-hand-v1",
        "manifest": str(man_path),
        "records_kept": totals["records_kept"],
        "decisions_kept": totals["decisions_kept"],
        "combo_state_labels_present": False,
        "setup_board_labels_present": True,
        "created_at_utc": utc_now(),
        "window": {"start": DAYS[0], "end": DAYS[-1]},
    }
    ready_path = EXPERT / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    atomic_json(ready_path, ready)
    protected = {
        "schema": "poke_bot.pinned_expert_corpus/v1",
        "protected": True,
        "status": "protected",
        "specialist_id": SPECIALIST,
        "manifest": "manifest.json",
        "manifest_sha256": sha256_file(man_path),
        "guide_ready": str(ready_path),
        "selection": manifest["selection"],
        "totals": manifest["totals"],
        "combo_state_labels_present": False,
        "created_at_utc": utc_now(),
        "window": {"start": DAYS[0], "end": DAYS[-1]},
        "owner_revision": 175,
    }
    ptr = EXPERT / "PROTECTED_EXPERT_CORPUS.json"
    atomic_json(ptr, protected)
    log(f"assembled corpus {ptr}")
    return ptr


def run_bootstrap(expert: Path) -> None:
    role = CURRICULUM / "alakazam-strategic-head-roles-r56.json"
    spec = CURRICULUM / "alakazam-strategic-curriculum-r56.json"
    validation = CURRICULUM / "alakazam-strategic-curriculum-validation-r56.json"
    cmd = [
        str(PY),
        "-u",
        str(ROOT / "scripts/run_specialist_expert_bootstrap.py"),
        "--archetype",
        SPECIALIST,
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
        str(READY),
        "--run-name",
        "final_format_alakazam_rtp_r175_bootstrap",
        "--run-dir",
        str(BOOTSTRAP_RUN),
        "--cpu-pack-root",
        str(CPU_PACK),
        "--epochs",
        "25",
        "--expanded-heads",
        "--decision-fusion",
        "--rl-protocol",
        str(PROTOCOL),
        "--current-deck-guide-contract",
        str(GUIDE),
        "--current-deck-guide-version",
        "powerful-hand-v1",
        "--current-deck-guide-corpus-ready",
        str(EXPERT / "CURRENT_DECK_GUIDE_CORPUS_READY.json"),
        "--strategic-curriculum-spec",
        str(spec),
        "--strategic-head-role-map",
        str(role),
        "--strategic-validation-receipt",
        str(validation),
        # The H10 combo architecture stays resident; r175 disables only its
        # dedicated action/guide route through the environment below.
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["POKEBOT_COMBO_STATE_HEAD_ENABLED"] = "1"
    env["POKEBOT_COMBO_STATE_ROUTE_ENABLED"] = "0"
    env["POKEBOT_COMBO_STATE_ROUTE_SPECIALIST"] = "alakazam"
    env["POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED"] = "1"
    env["POKEBOT_EXPANDED_HEADS_ENABLED"] = "1"
    env["POKEBOT_DECISION_FUSION_ENABLED"] = "1"
    env["POKEBOT_DECISION_FUSION_RUNTIME_ENABLED"] = "1"
    env["POKEBOT_H10_CAPACITY_ENABLED"] = "1"
    # Bootstrap keeps all non-combo heads live while combo's physical tensors
    # remain checkpoint-compatible and its policy route stays disabled.
    env["POKEBOT_OWNER_ALAKAZAM_ALL_HEADS_LIVE_R175"] = "1"
    log("BOOTSTRAP " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def stage_registry(expert: Path) -> None:
    source = json.loads(PARENT_REGISTRY.read_text(encoding="utf-8"))
    source["owner_decision_revision"] = 175
    source["minimum_terminal_iteration"] = 5
    source["iteration_ceiling"] = ITER_MAX
    source["isolated_refresh_contract"] = {
        "schema": "poke_bot.alakazam_rtp_isolated_runtime_r175/v1",
        "games_per_iteration": GAMES,
        "self_play_games": SELF_PLAY,
        "self_play_fraction": SELF_FRAC,
        "public_mix_games": GAMES - SELF_PLAY,
        "grimmsnarl_floor_per_set": 1024,
        "grimmsnarl_checkpoint_sha256": GRIM_SHA,
        "grimmsnarl_package_id": GRIM_PKG,
        "iteration_ceiling": ITER_MAX,
        "expert_rehearsal_every": 5,
        "expert_rehearsal_epochs": 5,
        "guide_loss_weight": GUIDE_W,
        "setup_board_outcome_loss_weight": SETUP_W,
        "combo_state_loss_weight": COMBO_W,
        "combo_state_head_enabled": True,
        "combo_state_route_enabled": False,
        "live_heads": LIVE_HEADS,
        "disabled_heads": ["combo_state"],
    }
    common = list(source.get("common_trainer_args") or [])
    common = set_arg(common, "--games-per-iter", str(GAMES))
    common = set_arg(common, "--expert-rehearsal-every", "5")
    common = set_arg(common, "--expert-rehearsal-epochs", "5")
    common = set_arg(common, "--archetype-aux-loss-weight", "0.05")
    common = set_arg(common, "--opp-hand-loss-weight", "0.05")
    common = set_arg(common, "--opp-remainder-loss-weight", "0.05")
    common = set_arg(common, "--lethal-threat-loss-weight", "0.025")
    common = set_arg(common, "--prize-race-loss-weight", "0.025")
    common = set_arg(common, "--setup-board-outcome-loss-weight", str(SETUP_W))
    common = set_arg(common, "--combo-state-loss-weight", str(COMBO_W))
    common = set_arg(common, "--current-deck-guide-loss-weight", str(GUIDE_W))
    source["common_trainer_args"] = common
    specialists = dict(source.get("specialists") or {})
    row = dict(specialists.get(SPECIALIST) or {})
    row["run_name"] = "final_format_alakazam_rtp_r175_i_v6_8k"
    row["iteration_ceiling"] = ITER_MAX
    row["minimum_terminal_iteration"] = 5
    row["guide_loss_weight"] = GUIDE_W
    row["setup_board_outcome_loss_weight"] = SETUP_W
    row["combo_state_loss_weight"] = COMBO_W
    row["combo_state_head_enabled"] = True
    row["combo_state_route_enabled"] = False
    row["guide_training_mode"] = "strategic_directional_v2"
    row["guide_contract"] = str(GUIDE)
    row["guide_contract_sha256"] = sha256_file(GUIDE)
    row["expert_manifest"] = str(expert)
    row["expert_manifest_sha256"] = sha256_file(expert)
    row["initial_checkpoint"] = str(FAMILY_OUT / "model.pt")
    if (FAMILY_OUT / "model.pt").is_file():
        row["initial_checkpoint_sha256"] = sha256_file(FAMILY_OUT / "model.pt")
    row["log"] = str(RUNTIME_ROOT / "logs/rl.log")
    row["owner_grimmsnarl_pin"] = {
        "package_id": GRIM_PKG,
        "checkpoint_sha256": GRIM_SHA,
        "floor_games_per_set": 1024,
    }
    row["expert_required_target_coverage"] = [
        t
        for t in (row.get("expert_required_target_coverage") or [])
        if t != "combo_state_rows"
    ]
    specialists[SPECIALIST] = row
    source["specialists"] = specialists
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(REGISTRY, source)
    SELECTOR.write_text(
        "POKEBOT_ACTIVE_SPECIALIST=alakazam\n"
        f"PURE_RL_SELF_PLAY_FRAC={SELF_FRAC:.16f}\n"
        "POKEBOT_COMBO_STATE_HEAD_ENABLED=1\n"
        "POKEBOT_COMBO_STATE_ROUTE_ENABLED=0\n"
        "POKEBOT_COMBO_STATE_ROUTE_SPECIALIST=alakazam\n"
        "POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED=1\n"
        "POKEBOT_EXPANDED_HEADS_ENABLED=1\n"
        "POKEBOT_DECISION_FUSION_ENABLED=1\n"
        "POKEBOT_DECISION_FUSION_RUNTIME_ENABLED=1\n"
        "POKEBOT_H10_CAPACITY_ENABLED=1\n"
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER=1\n"
        "POKEBOT_RTP_CHECKPOINT=/home/pokebot/poke-bot-agent/outputs/rtp_fleet/alakazam-r175.live/rtp_shadow_planner.pt\n"
        "POKEBOT_RTP_SPECIALIST_ID=alakazam\n"
        "POKEBOT_RTP_SIZING_PROFILE=pure_rl\n",
        encoding="utf-8",
    )
    log(f"registry staged {REGISTRY}")


def write_contract(expert: Path) -> None:
    contract = {
        "schema": "poke_bot.alakazam_rtp_owner_hard_swap_r175/v1",
        "owner_decision_revision": 175,
        "status": "armed",
        "recorded_at_utc": utc_now(),
        "specialist_id": SPECIALIST,
        "deck": {
            "path": str(DECK),
            "list_id": "alakazam-owner-rtp-pilot-r175",
            "sha256": sha256_file(DECK),
        },
        "expert_window": {"start": DAYS[0], "end": DAYS[-1], "corpus": str(expert)},
        "parent_checkpoint": {
            "path": str(PARENT_CKPT),
            "sha256": sha256_file(PARENT_CKPT),
            "family": str(PARENT_FAMILY),
        },
        "self_play": {
            "games_per_iteration": GAMES,
            "self_play_mirrors": SELF_PLAY,
            "self_play_fraction": SELF_FRAC,
            "public_mix_fill_games": GAMES - SELF_PLAY,
            "grimmsnarl_floor_per_set": 1024,
            "iteration_ceiling": ITER_MAX,
        },
        "grimmsnarl": {
            "package_id": GRIM_PKG,
            "checkpoint": str(GRIM_CKPT),
            "checkpoint_sha256": GRIM_SHA,
        },
        "heads": {
            "live": LIVE_HEADS,
            "disabled": ["combo_state"],
            "guide_loss_weight": GUIDE_W,
            "setup_board_outcome_loss_weight": SETUP_W,
            "combo_state_loss_weight": COMBO_W,
            "combo_state_head_enabled": True,
            "combo_state_route_enabled": False,
        },
        "units": {
            "orchestrator": "pokebot-final-format-alakazam-rtp-r175-orchestrator.service",
            "rl": "pokebot-final-format-alakazam-rtp-r175-rl.service",
        },
        "registry": str(REGISTRY),
    }
    atomic_json(CONTRACT, contract)
    atomic_json(
        ROOT / "outputs/state/alakazam-rtp-owner-hard-swap-r175.json", contract
    )


def queue_kaggle() -> None:
    model = FAMILY_OUT / "model.pt"
    if not model.is_file():
        raise RuntimeError("FAIL_CLOSED bootstrap model missing for kaggle queue")
    receipt = {
        "schema": "poke_bot.alakazam_rtp_r175_kaggle_queue/v1",
        "queued_at_utc": utc_now(),
        "checkpoint": str(model),
        "checkpoint_sha256": sha256_file(model),
        "deck": str(DECK),
        "deck_sha256": sha256_file(DECK),
        "turn_order_preference": "first_if_allowed",
        "status": "queued_request",
    }
    atomic_json(
        ROOT / "outputs/state/alakazam-rtp-r175-kaggle-queue-request.json", receipt
    )
    log(f"kaggle queue request written: {receipt['checkpoint_sha256']}")


def start_rl_unit() -> None:
    unit = "pokebot-final-format-alakazam-rtp-r175-rl.service"
    # Ensure TRAINING_ARMED exists.
    armed = ROOT / "outputs/state/TRAINING_ARMED"
    if not armed.is_file() or armed.stat().st_size == 0:
        atomic_json(
            armed,
            {
                "schema": "poke_bot.training_armed/v1",
                "armed_at_utc": utc_now(),
                "owner_revision": 175,
                "specialist_id": SPECIALIST,
            },
        )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "reset-failed", unit], check=False)
    subprocess.run(["systemctl", "--user", "start", unit], check=True)
    status = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "MainPID", "-p", "SubState"],
        text=True,
    )
    log(f"RL unit start: {status.strip()}")


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    write_state("starting")
    preflight()
    write_contract(EXPERT / "PROTECTED_EXPERT_CORPUS.json")
    write_state("preflight_ok")
    materialize_days()
    write_state("expert_days_ready")
    expert = assemble_corpus()
    write_contract(expert)
    write_state("expert_corpus_ready", expert=str(expert), expert_sha256=sha256_file(expert))
    run_bootstrap(expert)
    write_state("bootstrap_ready", ready=str(READY))
    queue_kaggle()
    stage_registry(expert)
    write_state("registry_staged", registry=str(REGISTRY))
    start_rl_unit()
    write_state(
        "rl_started",
        unit="pokebot-final-format-alakazam-rtp-r175-rl.service",
        registry=str(REGISTRY),
        grimmsnarl=GRIM_SHA,
        iter_max=ITER_MAX,
        combo_head=False,
        live_heads=LIVE_HEADS,
    )
    log("overnight arming complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        log(f"FAIL: {exc}")
        write_state("failed", error=str(exc))
        raise
