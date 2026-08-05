#!/usr/bin/env python3
"""Validate the dormant Marnie→Crustle→population managed handoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


CRUSTLE_UNITS = (
    "pokebot-final-format-crustle-r113-h10-bootstrap.service",
    "pokebot-final-format-crustle-r113-h10-register.service",
    "pokebot-final-format-crustle-r113-h10-rl.service",
    "pokebot-final-format-crustle-r113-h10-gate-handler.service",
    "pokebot-final-format-crustle-r113-completion.service",
)
MINIMUM_CRUSTLE_RECORDS = 16_639


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def systemctl(*args: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def show(unit: str, *properties: str) -> dict[str, str]:
    raw = systemctl("show", unit, *(item for prop in properties for item in ("-p", prop)))
    return dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)


def require(text: str, *needles: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise RuntimeError(f"managed handoff invariant missing: {missing}")


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def validate_corpus_import(path: Path) -> dict[str, Any]:
    import_path = path.resolve()
    if not import_path.is_file():
        raise RuntimeError(f"Crustle v2 corpus import receipt missing: {import_path}")
    receipt = json.loads(import_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "poke_bot.prestaged_specialist_corpus_import/v1",
        "status": "ready",
        "specialist_id": "crustle",
        "guide_version": "crustle-north-star-v2",
        "active_training_modified": False,
    }
    for key, value in expected.items():
        require_equal(receipt.get(key), value, f"corpus import {key}")
    records = int(receipt.get("records", 0))
    if records < MINIMUM_CRUSTLE_RECORDS:
        raise RuntimeError(
            f"Crustle v2 corpus has {records} records; "
            f"requires at least {MINIMUM_CRUSTLE_RECORDS}"
        )

    destination = Path(str(receipt["destination"])).resolve()
    artifact_names = {
        "guide_ready": "CURRENT_DECK_GUIDE_CORPUS_READY.json",
        "protected_expert": "PROTECTED_EXPERT_CORPUS.json",
        "manifest": "manifest.json",
        "finalization": str(receipt["finalization_receipt_name"]),
    }
    artifacts = {name: destination / filename for name, filename in artifact_names.items()}
    for artifact in artifacts.values():
        if not artifact.is_file():
            raise RuntimeError(f"Crustle v2 corpus artifact missing: {artifact}")

    expected_digests = {
        "guide_ready": receipt["ready_receipt_sha256"],
        "protected_expert": receipt["protected_pointer_sha256"],
        "manifest": receipt["manifest_sha256"],
        "finalization": receipt["finalization_receipt_sha256"],
    }
    actual_digests = {name: sha256(artifact) for name, artifact in artifacts.items()}
    for name, expected_digest in expected_digests.items():
        require_equal(actual_digests[name], expected_digest, f"corpus artifact {name} sha256")

    guide_ready = json.loads(artifacts["guide_ready"].read_text(encoding="utf-8"))
    finalization = json.loads(artifacts["finalization"].read_text(encoding="utf-8"))
    for document_name, document, schema, status in (
        (
            "guide ready",
            guide_ready,
            "poke_bot.current_deck_guide_corpus_ready/v1",
            "ready",
        ),
        (
            "finalization",
            finalization,
            "poke_bot.crustle_guide_corpus_validation/v1",
            "ready_checksum_validated",
        ),
    ):
        require_equal(document.get("schema"), schema, f"{document_name} schema")
        require_equal(document.get("status"), status, f"{document_name} status")
        require_equal(document.get("specialist_id"), "crustle", f"{document_name} specialist")
        require_equal(
            document.get("guide_version"),
            "crustle-north-star-v2",
            f"{document_name} guide version",
        )
        for count_name in ("records", "decisions", "guide_rows"):
            require_equal(
                int(document.get(count_name, -1)),
                int(receipt[count_name]),
                f"{document_name} {count_name}",
            )

    return {
        "status": "ready_checksum_validated_imported",
        "import_receipt": str(import_path),
        "import_receipt_sha256": sha256(import_path),
        "destination": str(destination),
        "guide_version": receipt["guide_version"],
        "records": records,
        "decisions": int(receipt["decisions"]),
        "guide_rows": int(receipt["guide_rows"]),
        "artifact_sha256": actual_digests,
        "active_training_modified": False,
    }


def write_once(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable receipt already differs: {path}")
        return
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-root", type=Path, required=True)
    parser.add_argument("--effective-marnie-completion", type=Path, required=True)
    parser.add_argument("--corpus-import", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.receipt.resolve()
    prior_receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.is_file()
        else None
    )

    unit_root = args.unit_root.resolve()
    paths = {name: unit_root / name for name in CRUSTLE_UNITS}
    paths["marnie_completion"] = args.effective_marnie_completion.resolve()
    paths["capacity"] = unit_root / "pokebot-post-refresh-capacity-boundary-r104.service"
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"managed handoff unit missing: {path}")

    marnie_completion = paths["marnie_completion"].read_text(encoding="utf-8")
    bootstrap = paths[CRUSTLE_UNITS[0]].read_text(encoding="utf-8")
    register = paths[CRUSTLE_UNITS[1]].read_text(encoding="utf-8")
    trainer = paths[CRUSTLE_UNITS[2]].read_text(encoding="utf-8")
    completion = paths[CRUSTLE_UNITS[4]].read_text(encoding="utf-8")
    capacity = paths["capacity"].read_text(encoding="utf-8")
    require(
        marnie_completion,
        "--next-service pokebot-final-format-crustle-r113-h10-bootstrap.service",
        "--next-unit /home/inzi/.config/systemd/user/pokebot-final-format-crustle-r113-h10-bootstrap.service",
    )
    if "--next-service pokebot-post-refresh-capacity-boundary-r104.service" in marnie_completion:
        raise RuntimeError("Marnie completion still bypasses Crustle")
    require(
        bootstrap,
        "ConditionPathExists=/home/inzi/poke-bot-agent/outputs/state/final-format-marnie-r104-h10-completion-v1.json",
        "crustle-v2/CURRENT_DECK_GUIDE_CORPUS_READY.json",
        "OnSuccess=pokebot-final-format-crustle-r113-h10-register.service",
    )
    require(register, "OnSuccess=pokebot-final-format-crustle-r113-h10-rl.service")
    require(
        trainer,
        "PURE_RL_SIM_WORKERS=96",
        "PURE_RL_GAMES_IN_FLIGHT=96",
        "POKEBOT_LIVE_POOL_MAX_WORKERS=96",
        "OnSuccess=pokebot-final-format-crustle-r113-h10-gate-handler.service",
    )
    require(completion, "--next-service pokebot-post-refresh-capacity-boundary-r104.service")
    require(capacity, "--crustle-completion /home/inzi/poke-bot-agent/outputs/state/final-format-crustle-r113-h10-completion-v1.json")

    service_state: dict[str, dict[str, str]] = {}
    for unit in CRUSTLE_UNITS:
        values = show(unit, "LoadState", "ActiveState", "SubState")
        expected = {"LoadState": "loaded", "ActiveState": "inactive", "SubState": "dead"}
        if values != expected:
            raise RuntimeError(f"Crustle unit has premature authority: {unit} {values}")
        service_state[unit] = {
            "load_state": values["LoadState"],
            "active_state": values["ActiveState"],
            "sub_state": values["SubState"],
        }
    marnie_state = show(
        "pokebot-final-format-marnie-r104-h10-rl.service",
        "ActiveState",
        "SubState",
        "MainPID",
    )
    if (
        marnie_state.get("ActiveState") != "active"
        or marnie_state.get("SubState") != "running"
        or int(marnie_state.get("MainPID", "0")) <= 0
    ):
        raise RuntimeError(f"active Marnie training changed during handoff staging: {marnie_state}")

    corpus = validate_corpus_import(args.corpus_import) if args.corpus_import else None

    receipt = {
        "schema": (
            "poke_bot.post_marnie_crustle_handoff_stage/v2"
            if corpus
            else "poke_bot.post_marnie_crustle_handoff_stage/v1"
        ),
        "status": (
            "staged_inactive_corpus_ready_waiting_for_marnie_iteration_20"
            if corpus
            else "staged_inactive_waiting_for_marnie_iteration_20_and_v2_corpus"
        ),
        "owner_decision_revision": 113,
        "created_at_utc": (
            prior_receipt["created_at_utc"]
            if isinstance(prior_receipt, dict) and prior_receipt.get("created_at_utc")
            else datetime.now(timezone.utc).isoformat()
        ),
        "marnie_service": {
            "active_state": marnie_state["ActiveState"],
            "sub_state": marnie_state["SubState"],
        },
        "marnie_completion_next_service": CRUSTLE_UNITS[0],
        "crustle_service_state": service_state,
        "unit_sha256": {name: sha256(path) for name, path in paths.items()},
        "crustle_v2_corpus": corpus,
        "then_current_h10_registry_rebind_required_at_handoff": True,
        "crustle_training_or_selector_authority": False,
        "population_release_requires_crustle_completion": True,
    }
    write_once(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
