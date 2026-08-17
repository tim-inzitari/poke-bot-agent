#!/usr/bin/env python3
"""Train and freeze the Alakazam-only expert warm start on Blackwell."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint
from poke_bot.pure_rl.expert_rehearsal import resolve_expert_manifest
from poke_bot.pure_rl.model_registry import freeze_model, sha256, verify_frozen_model


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def resolve_filtered_manifest(
    path: Path, *, min_decisions: int
) -> tuple[Path, dict[str, Any]]:
    """Resolve a direct or checksum-pinned Alakazam feature manifest."""
    identity = resolve_expert_manifest(path, min_decisions=int(min_decisions))
    manifest_path = Path(identity.path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = dict(manifest.get("selection") or {})
    quality = dict(manifest.get("quality_gates") or {})
    totals = dict(manifest.get("totals") or {})
    if (
        manifest.get("format") != "pokebot-bootstrap-feature-manifest"
        or selection.get("field") != "GameSequence.archetype"
        or str(selection.get("value") or "").lower() != "alakazam"
        or selection.get("seat_semantics") != "acting_seat_only"
        or quality.get("passed") is not True
        or quality.get("acting_seat_archetype_exact") is not True
        or int(totals.get("decisions_kept", 0)) < int(min_decisions)
        or int(totals.get("records_kept", 0)) <= 0
    ):
        raise ValueError("Alakazam filtered feature manifest failed its contract")
    source = Path(str(manifest.get("source_manifest") or ""))
    source_matches = source.is_file() and sha256(source) == manifest.get(
        "source_manifest_sha256"
    )
    if not source_matches:
        # The source all-decks manifest is useful provenance, but the sealed
        # filtered corpus is the actual training input. Disk retention may
        # legitimately remove that much larger upstream corpus. Accept this
        # only through the explicit protected pointer whose digest pins the
        # filtered manifest bytes; direct-manifest callers still fail closed.
        pointer_path = Path(path).expanduser().resolve()
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            raw_manifest = Path(str(pointer.get("manifest") or "")).expanduser()
            pinned_manifest = (
                raw_manifest.resolve()
                if raw_manifest.is_absolute()
                else (pointer_path.parent / raw_manifest).resolve()
            )
            protected_fallback = (
                pointer.get("schema") == "poke_bot.pinned_expert_corpus/v1"
                and pointer.get("protected") is True
                and pinned_manifest == manifest_path
                and sha256(manifest_path) == pointer.get("manifest_sha256")
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            protected_fallback = False
        if not protected_fallback:
            raise ValueError(
                "filtered manifest source identity is unavailable or changed"
            )
    return manifest_path, manifest


def validate_filtered_manifest(path: Path, *, min_decisions: int) -> dict[str, Any]:
    """Validate and return metadata for a pinned acting-seat corpus."""
    _manifest_path, manifest = resolve_filtered_manifest(
        path, min_decisions=int(min_decisions)
    )
    return manifest


def bootstrap_command(
    *,
    python: Path,
    manifest: Path,
    run_name: str,
    init_checkpoint: Path,
    resume: bool,
    epochs: int,
    patience: int,
    min_decisions: int,
    device_resident: bool = True,
) -> list[str]:
    command = [
        str(python),
        "-u",
        str(ROOT / "scripts/train_bootstrap.py"),
        "--feature-manifest",
        str(manifest),
        "--archetype",
        "alakazam",
        "--run-name",
        str(run_name),
        "--model-profile",
        "pure-rl",
        "--epochs",
        str(int(epochs)),
        "--lr",
        "5e-5",
        "--games-per-batch",
        "512",
        "--max-decisions-per-batch",
        "12288",
        "--val-frac",
        "0.10",
        "--split-by-episode",
        "--patience",
        str(int(patience)),
        "--aux-loss-weight",
        "0",
        "--opp-hand-loss-weight",
        "0",
        "--opp-remainder-loss-weight",
        "0",
        "--lethal-threat-loss-weight",
        "0",
        "--prize-race-loss-weight",
        "0",
        "--min-usable-record-frac",
        "0.999",
        "--min-decisions",
        str(int(min_decisions)),
        "--seed",
        "20260720",
    ]
    if device_resident:
        command.append("--device-resident")
    if resume:
        command.extend(("--resume", "auto"))
    else:
        command.extend(("--resume", "0", "--init-checkpoint", str(init_checkpoint)))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-manifest", type=Path, required=True)
    parser.add_argument("--core-family", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-decisions", type=int, default=100_000)
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument(
        "--device-resident",
        dest="device_resident",
        action="store_true",
        help="Pack the corpus onto the selected GPU (the default).",
    )
    storage.add_argument(
        "--streaming",
        dest="device_resident",
        action="store_false",
        help=(
            "Stream batches from host memory. The transition service uses "
            "this on the 12 GiB 3080 Ti so Blackwell core training continues."
        ),
    )
    parser.set_defaults(device_resident=True)
    args = parser.parse_args()

    filtered_input = args.filtered_manifest.expanduser().resolve()
    filtered_path, filtered = resolve_filtered_manifest(
        filtered_input, min_decisions=int(args.min_decisions)
    )
    core = verify_frozen_model(args.core_family)
    core_path = Path(str(core["model_path"])).resolve()
    filtered_digest = sha256(filtered_path)

    bootstrap_family = args.registry_root.expanduser().resolve() / "alakazam_expert_bootstrap"
    if args.ready.is_file() and bootstrap_family.is_dir():
        ready = json.loads(args.ready.read_text(encoding="utf-8"))
        frozen = verify_frozen_model(bootstrap_family)
        if (
            ready.get("status") == "ready"
            and ready.get("checkpoint_digest") == frozen.get("checkpoint_digest")
            and ready.get("core_checkpoint_digest") == core.get("checkpoint_digest")
            and ready.get("filtered_manifest_sha256") == filtered_digest
        ):
            print(json.dumps(ready, indent=2), flush=True)
            return 0
        raise RuntimeError("existing Alakazam bootstrap readiness identity changed")

    latest = checkpoint.latest_path(args.run_name)
    command = bootstrap_command(
        python=args.python,
        manifest=filtered_path,
        run_name=args.run_name,
        init_checkpoint=core_path,
        resume=latest.is_file(),
        epochs=int(args.epochs),
        patience=int(args.patience),
        min_decisions=int(args.min_decisions),
        device_resident=bool(args.device_resident),
    )
    print(
        "[alakazam-bootstrap] exact acting-seat corpus "
        f"games={filtered['totals']['records_kept']} "
        f"decisions={filtered['totals']['decisions_kept']} "
        f"epochs<={int(args.epochs)} patience={int(args.patience)}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        return int(completed.returncode)
    result_path = ROOT / "outputs/train" / f"{args.run_name}_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    best = Path(str(result.get("best_path") or "")).expanduser().resolve()
    if not best.is_file():
        raise RuntimeError("Alakazam bootstrap did not publish a best checkpoint")
    best_digest = checkpoint.checkpoint_digest(best)
    frozen = freeze_model(
        registry_root=args.registry_root,
        family="alakazam_expert_bootstrap",
        display_name="Alakazam Expert Bootstrap",
        checkpoint=best,
        expected_digest=best_digest,
        provenance={
            "initialized_from_family": "deck_agnostic_core",
            "initialized_from_digest": core["checkpoint_digest"],
            "filtered_manifest": str(filtered_path),
            "filtered_manifest_sha256": filtered_digest,
            "acting_seat_archetype": "alakazam",
            "epochs_max": int(args.epochs),
            "early_stop_patience": int(args.patience),
            "device_resident": bool(args.device_resident),
            "training_result": result,
        },
        evidence={
            "kind": "supervised_expert_validation",
            "best_metric": result.get("best_metric"),
            "epochs_completed": len(result.get("history") or []),
        },
    )
    ready = {
        "schema": "poke_bot.alakazam_expert_bootstrap_ready/v1",
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "checkpoint": frozen["model_path"],
        "checkpoint_digest": frozen["checkpoint_digest"],
        "core_checkpoint_digest": core["checkpoint_digest"],
        "filtered_manifest": str(filtered_path),
        "filtered_manifest_sha256": filtered_digest,
        "records": int(filtered["totals"]["records_kept"]),
        "decisions": int(filtered["totals"]["decisions_kept"]),
        "epochs_max": int(args.epochs),
        "early_stop_patience": int(args.patience),
        "device_resident": bool(args.device_resident),
        "best_metric": result.get("best_metric"),
    }
    atomic_json(args.ready, ready)
    print(json.dumps(ready, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
