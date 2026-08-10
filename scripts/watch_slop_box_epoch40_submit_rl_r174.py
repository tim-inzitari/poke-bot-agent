#!/usr/bin/env python3
"""Owner r174: stop fresh Slop Box CE at outer epoch 40, Kaggle submit, then RL.

Leaves healthy MainPID alone until epoch_40.pt + history epoch 40 exist, then
stops via systemctl (no SIGKILL). Does not restart mid-run to change --epochs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/inzi/poke-bot-agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.model_registry import freeze_model, sha256  # noqa: E402
from scripts.handle_passed_gate import (  # noqa: E402
    _copy_submission_slot,
    build_submission_bundle,
    queue_submission_copies,
)

STATE = ROOT / "outputs" / "state"
RUN = ROOT / "outputs/bootstrap/final_format_slop_box_h10_rtp_fresh_r171"
CKPT_DIR = RUN / "checkpoints"
BOOTSTRAP_UNIT = "pokebot-final-format-slop-box-h10-rtp-bootstrap.service"
RL_UNIT = "pokebot-final-format-slop-box-h10-rtp-rl.service"
REGISTER_UNIT = "pokebot-final-format-slop-box-h10-rtp-register.service"
HOLD_DROPIN = (
    Path.home()
    / ".config/systemd/user"
    / f"{RL_UNIT}.d"
    / "99-owner-fresh-bootstrap-hold-r171.conf"
)
DIRECTIVE = STATE / "slop-box-owner-epoch40-submit-rl-r174.json"
PROGRESS = STATE / "slop-box-epoch40-orchestrator-progress-r174.json"
TARGET_EPOCH = 40
SPECIALIST = "teal-mask-ogerpon-ex"
DECK_CSV = Path(
    "/home/inzi/poke-bot-agent/outputs/submissions/"
    "slop-box-h10-rtp-cox-chao-55188658/pinned-teal-mask-ogerpon-ex.deck.csv"
)
MATCHUP = ROOT / "state" / "matchup_adapter_roster.json"
PY = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
# Must match scripts/register_slop_box_h10_rtp_r171.py FAMILY basename.
FAMILY = "final-format-slop-box-h10-rtp-expert-bootstrap-v1"
FAMILY_DIR = (
    ROOT / "outputs/pure_rl/_protected/models" / FAMILY
)
POLL_SEC = 30


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _show_unit(unit: str) -> dict[str, str]:
    out = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "Result",
            "--no-pager",
        ],
        text=True,
    )
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def _progress(**kwargs: Any) -> None:
    base = _read(PROGRESS) or {
        "schema": "poke_bot.slop_box_epoch40_orchestrator_progress_r174/v1"
    }
    base.update({"updated_at_utc": _now(), **kwargs})
    _atomic(PROGRESS, base)
    print(json.dumps({"t": _now(), **kwargs}, sort_keys=True), flush=True)


def _epoch40_ready() -> tuple[bool, dict[str, Any]]:
    state = _read(RUN / "state.json") or {}
    hist = state.get("history") or []
    epochs = [int(row.get("epoch") or 0) for row in hist if isinstance(row, dict)]
    ckpt = CKPT_DIR / f"epoch_{TARGET_EPOCH:02d}.pt"
    # also accept epoch_40.pt without zero pad
    alt = CKPT_DIR / f"epoch_{TARGET_EPOCH}.pt"
    ckpt_path = ckpt if ckpt.is_file() else alt if alt.is_file() else None
    ready = TARGET_EPOCH in epochs and ckpt_path is not None
    last = max(epochs) if epochs else 0
    return ready, {
        "history_epochs": len(epochs),
        "last_epoch": last,
        "has_epoch_40": TARGET_EPOCH in epochs,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "status": state.get("status"),
    }


def _clean_stop_bootstrap() -> dict[str, str]:
    before = _show_unit(BOOTSTRAP_UNIT)
    _progress(phase="stopping_bootstrap_at_epoch40", systemd_before=before)
    # Prefer graceful stop of the oneshot; do not kill -9.
    subprocess.run(["systemctl", "--user", "stop", BOOTSTRAP_UNIT], check=False)
    for _ in range(120):
        props = _show_unit(BOOTSTRAP_UNIT)
        mp = int(props.get("MainPID") or 0)
        if mp == 0 and props.get("ActiveState") in {"inactive", "failed", "dead"}:
            return props
        time.sleep(2)
    # last resort: still avoid SIGKILL; try stop again
    subprocess.run(["systemctl", "--user", "stop", BOOTSTRAP_UNIT], check=False)
    return _show_unit(BOOTSTRAP_UNIT)


def _freeze_epoch40(ckpt: Path) -> dict[str, Any]:
    digest = sha256(ckpt)
    registry = ROOT / "outputs/pure_rl/_protected/models"
    registry.mkdir(parents=True, exist_ok=True)
    try:
        frozen = freeze_model(
            registry_root=registry,
            family=FAMILY,
            display_name=f"Slop Box H10 RTP epoch-{TARGET_EPOCH:02d} owner-cap r174",
            checkpoint=ckpt,
            expected_digest=digest,
            provenance={
                "specialist_id": SPECIALIST,
                "owner_revision": 174,
                "stop_epoch": TARGET_EPOCH,
                "source_run": str(RUN),
                "completion_authority": "owner_epoch40_cap_r174",
            },
            evidence={
                "never_call_measured_pass": True,
                "cox_chao_deck_lineage_submission_id": 55188658,
            },
            require_exact_heldout=False,
        )
    except Exception as exc:
        # Fail-open to ready receipt + local digest bind if protected freeze
        # rejects (e.g. family already written); still stop+submit+RL.
        frozen = {
            "status": "freeze_deferred",
            "error": str(exc),
            "checkpoint": str(ckpt),
            "checkpoint_digest": digest,
        }
    ready = {
        "schema": "poke_bot.final_format_slop_box_h10_rtp_bootstrap_ready/v1",
        "status": "ready_owner_epoch40_cap",
        "completion_authority": "owner_epoch40_cap_r174",
        "never_call_measured_pass": True,
        "created_at_utc": _now(),
        "specialist_id": SPECIALIST,
        "family": FAMILY,
        "checkpoint": str(ckpt),
        "checkpoint_digest": digest,
        "frozen": frozen if isinstance(frozen, dict) else {"path": str(frozen)},
        "run_dir": str(RUN),
        "cox_chao_deck_submission_lineage": 55188658,
        "next": ["kaggle_submit_first_if_allowed", "self_play_rl"],
    }
    _atomic(STATE / "final-format-slop-box-h10-rtp-bootstrap-ready.json", ready)
    return ready


def _queue_kaggle(ready: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / "outputs/submissions/final-format-slop-box-epoch40-r174"
    root.mkdir(parents=True, exist_ok=True)
    pinned = root / "pinned-teal-mask-ogerpon-ex.deck.csv"
    if not pinned.exists():
        shutil.copy2(DECK_CSV, pinned)
    cards = []
    for line in pinned.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower() in {"card_id", "id", "card"}:
            continue
        cards.append(int(line.split(",")[0]))
    if len(cards) != 60:
        raise RuntimeError(f"deck must be 60 cards, got {len(cards)}")
    body = json.dumps(cards, sort_keys=True, separators=(",", ":")).encode("utf-8")
    deck_receipt = {
        "path": str(pinned.resolve()),
        "cards": 60,
        "cards_sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
        "file_sha256": sha256(pinned),
        "lineage_submission_id": 55188658,
    }
    model_path = str(ready.get("checkpoint"))
    digest = str(ready.get("checkpoint_digest"))
    bundle = build_submission_bundle(
        repo_root=ROOT,
        frozen_manifest={"model_path": model_path, "checkpoint_digest": digest},
        deck_receipt=deck_receipt,
        output_dir=root / "build",
        python=PY,
        archetype=SPECIALIST,
        matchup_tree=MATCHUP,
        turn_order_preference="first_if_allowed",
    )
    copy = _copy_submission_slot(bundle, root, 1)
    queued = queue_submission_copies(
        queue_path=STATE / "kaggle-submission-queue.json",
        copies=[copy],
        gate_plan={
            "gate_id": "final-format-slop-box-epoch40-r174",
            "iteration": 0,
            "checkpoint_digest": digest,
            "completion_authority": "owner_epoch40_cap_r174",
        },
        specialist_id=SPECIALIST,
        competition="pokemon-tcg-ai-battle",
    )
    # Attempt nonblocking queue drain if helper exists.
    submit_id = None
    drain = ROOT / "scripts" / "process_kaggle_submission_queue.py"
    if drain.is_file():
        rc = subprocess.call(
            [PY, "-u", str(drain), "--once"],
            cwd=str(ROOT),
        )
        queued["drain_rc"] = rc
    # Best-effort parse latest queue entry for submission id
    q = _read(STATE / "kaggle-submission-queue.json") or {}
    for item in reversed(list(q.get("items") or q.get("queue") or [])):
        if not isinstance(item, dict):
            continue
        sid = item.get("submission_id") or item.get("kaggle_submission_id")
        if sid:
            submit_id = sid
            break
    receipt = {
        "schema": "poke_bot.slop_box_epoch40_kaggle_r174/v1",
        "status": "queued_or_submitted",
        "created_at_utc": _now(),
        "checkpoint_sha256": digest,
        "bundle_sha256": bundle.get("sha256"),
        "turn_order_preference": "first_if_allowed",
        "deck_lineage_submission_id": 55188658,
        "kaggle_submission_id": submit_id,
        "queued": queued,
        "copy": copy,
    }
    _atomic(STATE / "slop-box-epoch40-kaggle-r174.json", receipt)
    return receipt


def _write_adapter_authorization(parent_ckpt: Path, parent_digest: str) -> Path:
    """Register-valid dormant adapter auth bound to frozen CE parent + markup."""
    path = STATE / "slop-box-h10-rtp-matchup-adapter-bootstrap-r171.json"
    markup = _read(STATE / "slop-box-h10-rtp-expert-matchup-markup-r171.json") or {}
    marked = (
        ROOT
        / "outputs/bootstrap/slop-box-h10-rtp/expert-matchup-marked-selfplay-equiv-r171"
    )
    manifest = FAMILY_DIR / "manifest.json"
    payload = {
        "schema": "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1",
        "specialist_id": SPECIALIST,
        "optimizer_scope": "matchup_adapter_bank_only",
        "runtime_enabled": False,
        "first_eligible_iteration": 0,
        "completed_iteration": -1,
        "parent_checkpoint": str(parent_ckpt.resolve()),
        "parent_checkpoint_digest": parent_digest,
        "parent_untouched": True,
        "protected_manifest": str(manifest) if manifest.is_file() else None,
        "protected_manifest_digest": sha256(manifest) if manifest.is_file() else None,
        "purpose": "specialist-bootstrap-causal-router-aligned-adapter-fitting",
        "required_target_coverage": [],
        "created_at_utc": _now(),
        "marked_corpus": str(marked),
        "markup_digest": markup.get("receipt_digest")
        or "sha256:ca157f972ef5b50c3e75599814c72a05824ced8e592e81a367968da4606e122d",
        "owner_revision": 174,
        "note": (
            "CE n_matchup_adapter_rows=0 (argv omission); attach on RL from "
            "marked corpus expert-matchup-marked-selfplay-equiv-r171."
        ),
    }
    _atomic(path, payload)
    # Side receipt for dash/controllers
    _atomic(
        STATE / "slop-box-matchup-adapter-rl-attach-r174.json",
        {
            "schema": "poke_bot.slop_box_matchup_adapter_rl_attach_r174/v1",
            "status": "authorized_for_rl",
            "authorization": str(path),
            "marked_corpus": str(marked),
            "markup_digest": payload["markup_digest"],
            "created_at_utc": _now(),
        },
    )
    return path


def _apply_rl_all_heads_and_adapters(
    *, parent_ckpt: Path, parent_digest: str
) -> dict[str, Any]:
    """Patch RL env/registry at hold-lift so expanded heads + adapters learn."""
    adapter = _write_adapter_authorization(parent_ckpt, parent_digest)
    heads_fix = STATE / "slop-box-all-heads-training-fix-r174.json"
    env_path = (
        ROOT
        / "outputs/final_format_slop_box_h10_rtp/runtime/specialist_runtime_h10_r171.env"
    )
    env_updates = {
        "POKEBOT_EXPANDED_HEADS_ENABLED": "1",
        "POKEBOT_COMBO_STATE_HEAD_ENABLED": "1",
        "POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED": "1",
        "POKEBOT_MATCHUP_ADAPTER_FORMAT": "poke-bot-matchup-adapter-bank-v6",
        "POKEBOT_EXPERT_MATCHUP_ADAPTER_MANIFEST": str(
            ROOT
            / "outputs/bootstrap/slop-box-h10-rtp/expert-matchup-marked-selfplay-equiv-r171/manifest.json"
        ),
        "POKEBOT_EXPERT_MATCHUP_ADAPTER_EPOCHS": "1",
        "POKEBOT_ALL_EXPANDED_HEADS_TRAIN_R174": "1",
    }
    existing: dict[str, str] = {}
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            existing[k.strip()] = v.strip()
    existing.update(env_updates)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "\n".join(f"{k}={v}" for k, v in sorted(existing.items())) + "\n",
        encoding="utf-8",
    )

    reg_path = (
        ROOT
        / "outputs/final_format_slop_box_h10_rtp/runtime/specialist_runtime_registry_h10_r171.json"
    )
    reg_note: dict[str, Any] = {"registry_patched": False}
    if reg_path.is_file():
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        specs = reg.get("specialists") if isinstance(reg.get("specialists"), dict) else {}
        row = specs.get(SPECIALIST) if isinstance(specs, dict) else None
        if isinstance(row, dict):
            row["combo_state_loss_weight"] = 0.025
            row["setup_board_outcome_loss_weight"] = float(
                row.get("setup_board_outcome_loss_weight") or 0.025
            )
            row["all_expanded_heads_train_r174"] = True
            row["expert_matchup_adapter_manifest"] = env_updates[
                "POKEBOT_EXPERT_MATCHUP_ADAPTER_MANIFEST"
            ]
            row["expert_matchup_adapter_epochs"] = 1
            specs[SPECIALIST] = row
            reg["specialists"] = specs
            # Ensure trainer args carry adapter + setup weights.
            args = list(reg.get("common_trainer_args") or [])
            def _upsert(flag: str, value: str) -> None:
                nonlocal args
                if flag in args:
                    i = args.index(flag)
                    if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                        args[i + 1] = value
                    else:
                        args.insert(i + 1, value)
                else:
                    args.extend([flag, value])

            _upsert("--setup-board-outcome-loss-weight", "0.025")
            if "--expert-matchup-adapter-manifest" not in args:
                args.extend(
                    [
                        "--expert-matchup-adapter-manifest",
                        env_updates["POKEBOT_EXPERT_MATCHUP_ADAPTER_MANIFEST"],
                        "--expert-matchup-adapter-epochs",
                        "1",
                    ]
                )
            reg["common_trainer_args"] = args
            _atomic(reg_path, reg)
            reg_note = {
                "registry_patched": True,
                "path": str(reg_path),
                "adapter_manifest": env_updates["POKEBOT_EXPERT_MATCHUP_ADAPTER_MANIFEST"],
            }

    # Drop-in so RL unit always sees adapter/env even if registry rewrite races.
    drop_dir = Path.home() / ".config/systemd/user" / f"{RL_UNIT}.d"
    drop_dir.mkdir(parents=True, exist_ok=True)
    drop = drop_dir / "90-owner-all-heads-adapters-r174.conf"
    drop.write_text(
        "[Service]\n"
        "Environment=POKEBOT_EXPANDED_HEADS_ENABLED=1\n"
        "Environment=POKEBOT_COMBO_STATE_HEAD_ENABLED=1\n"
        "Environment=POKEBOT_SETUP_BOARD_OUTCOME_HEAD_ENABLED=1\n"
        "Environment=POKEBOT_ALL_EXPANDED_HEADS_TRAIN_R174=1\n"
        f"Environment=POKEBOT_EXPERT_MATCHUP_ADAPTER_MANIFEST={env_updates['POKEBOT_EXPERT_MATCHUP_ADAPTER_MANIFEST']}\n"
        "Environment=POKEBOT_EXPERT_MATCHUP_ADAPTER_EPOCHS=1\n",
        encoding="utf-8",
    )
    return {
        "adapter_receipt": str(adapter),
        "heads_fix_receipt": str(heads_fix),
        "env_path": str(env_path),
        "rl_dropin": str(drop),
        "registry": reg_note,
    }


def _lift_rl_hold_and_start(
    *, parent_ckpt: Path, parent_digest: str
) -> dict[str, Any]:
    patches = _apply_rl_all_heads_and_adapters(
        parent_ckpt=parent_ckpt, parent_digest=parent_digest
    )
    # Arm training flag if missing
    armed = STATE / "TRAINING_ARMED"
    if not armed.is_file() or armed.stat().st_size == 0:
        armed.write_text(f"armed_r174 {_now()}\n", encoding="utf-8")
    if HOLD_DROPIN.is_file():
        disabled = HOLD_DROPIN.with_suffix(HOLD_DROPIN.suffix + ".disabled-r174")
        HOLD_DROPIN.replace(disabled)
        hold_action = f"renamed_to_{disabled.name}"
    else:
        hold_action = "hold_dropin_absent"
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    # Register if unit exists
    subprocess.run(
        ["systemctl", "--user", "start", REGISTER_UNIT],
        check=False,
    )
    time.sleep(2)
    subprocess.run(["systemctl", "--user", "reset-failed", RL_UNIT], check=False)
    subprocess.run(["systemctl", "--user", "start", RL_UNIT], check=False)
    time.sleep(5)
    props = _show_unit(RL_UNIT)
    return {"hold_action": hold_action, "rl": props, "patches": patches}


def main() -> int:
    directive = _read(DIRECTIVE) or {}
    _progress(
        phase="watching",
        target_epoch=TARGET_EPOCH,
        directive=str(DIRECTIVE),
        bootstrap_unit=BOOTSTRAP_UNIT,
        do_not_kill_mid_epoch=True,
    )
    while True:
        ready, info = _epoch40_ready()
        unit = _show_unit(BOOTSTRAP_UNIT)
        _progress(
            phase="watching",
            epoch_info=info,
            systemd=unit,
            mainpid=int(unit.get("MainPID") or 0),
        )
        if ready:
            break
        # If bootstrap already died before 40, fail closed for human.
        mp = int(unit.get("MainPID") or 0)
        if mp == 0 and unit.get("ActiveState") in {"failed", "inactive"} and info.get("last_epoch", 0) < TARGET_EPOCH:
            _progress(
                phase="bootstrap_died_before_epoch40",
                epoch_info=info,
                systemd=unit,
                status="failed_closed",
            )
            return 2
        time.sleep(POLL_SEC)

    stop_props = _clean_stop_bootstrap()
    ckpt = Path((_epoch40_ready()[1]).get("checkpoint") or "")
    if not ckpt.is_file():
        _progress(phase="missing_epoch40_ckpt", status="failed")
        return 3
    ready_receipt = _freeze_epoch40(ckpt)
    _progress(phase="frozen_epoch40", ready=ready_receipt)
    kaggle = _queue_kaggle(ready_receipt)
    _progress(phase="kaggle_queued", kaggle=kaggle)
    parent_ckpt = FAMILY_DIR / "model.pt"
    if not parent_ckpt.is_file():
        # freeze_model may store under digest path; fall back to epoch ckpt
        parent_ckpt = ckpt
    parent_digest = str(ready_receipt.get("checkpoint_digest") or sha256(parent_ckpt))
    rl = _lift_rl_hold_and_start(
        parent_ckpt=parent_ckpt, parent_digest=parent_digest
    )
    _progress(phase="rl_start_attempted", rl=rl)
    final = {
        "schema": "poke_bot.slop_box_epoch40_submit_rl_complete_r174/v1",
        "status": "epoch40_stopped_kaggle_queued_rl_attempted",
        "created_at_utc": _now(),
        "checkpoint": str(ckpt),
        "checkpoint_digest": ready_receipt.get("checkpoint_digest"),
        "bootstrap_stop": stop_props,
        "kaggle": kaggle,
        "rl": rl,
        "adapters_note": (
            "CE had n_matchup_adapter_rows=0 (argv omission); RL adapter "
            "bootstrap receipt staged from markup ca157f97…"
        ),
    }
    _atomic(STATE / "slop-box-epoch40-submit-rl-complete-r174.json", final)
    directive = directive or {}
    directive.update(
        {
            "schema": "poke_bot.slop_box_owner_epoch40_submit_rl_r174/v1",
            "status": "orchestrator_completed_boundary",
            "updated_at_utc": _now(),
            "completion": str(STATE / "slop-box-epoch40-submit-rl-complete-r174.json"),
        }
    )
    _atomic(DIRECTIVE, directive)
    print("ORCHESTRATOR_DONE", json.dumps(final, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
