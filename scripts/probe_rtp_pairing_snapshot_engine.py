#!/usr/bin/env python3
"""Seal and prove the private v2 true-RNG pairing snapshot ABI.

This evaluation-only tool is intentionally separate from production training
and submission packaging.  It captures one exact post-BattleStart snapshot,
seals the physical bytes under a private root, restores the bytes in fresh OS
processes, compares a delayed duplicate transcript, exercises a divergent
policy from another fresh restore, and only then emits the immutable probe and
capability records consumed by r198 staging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poke_bot.engine_rebuild.rtp_pairing_snapshot import (  # noqa: E402
    PairingArtifactSet,
    RTPPairingSnapshotError,
    RtpPairingSnapshotEngine,
    SNAPSHOT_BOUNDARY_TAG,
    SNAPSHOT_CAPTURE_BOUNDARY,
    SNAPSHOT_SEAL_SCHEMA,
    canonical_digest,
    emit_true_rng_pairing_capability,
    emit_true_rng_pairing_probe,
    file_digest,
    frozen_file_identity,
    snapshot_abi_sha256,
    verify_pairing_case_binding,
)


def _lexical_absolute(path: str | Path) -> Path:
    raw = os.path.expanduser(os.fspath(path))
    if not os.path.isabs(raw):
        raw = os.path.join(os.getcwd(), raw)
    return Path(raw)


def _reject_symlinks(path: str | Path, *, label: str) -> Path:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        if component in ("", "."):
            continue
        if component == "..":
            current = current.parent
            continue
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RTPPairingSnapshotError(
                f"cannot inspect {label}: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RTPPairingSnapshotError(f"{label} traverses a symlink: {current}")
    return absolute


def _private_output_root(path: str | Path) -> Path:
    lexical = _reject_symlinks(path, label="private probe output root")
    if ".private" not in lexical.parts:
        raise RTPPairingSnapshotError(
            "private probe output root must contain a literal .private component"
        )
    lexical.mkdir(parents=True, exist_ok=True)
    _reject_symlinks(lexical, label="private probe output root")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_dir() or ".private" not in resolved.parts:
        raise RTPPairingSnapshotError("private probe output root is unsafe")
    return resolved


def _write_immutable_bytes(path: Path, material: bytes) -> Path:
    _reject_symlinks(path.parent, label="snapshot output parent")
    expected = "sha256:" + hashlib.sha256(material).hexdigest()
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RTPPairingSnapshotError(f"snapshot output is unsafe: {path}")
        if file_digest(path) != expected or stat.S_IMODE(metadata.st_mode) != 0o444:
            raise RTPPairingSnapshotError(
                f"sealed snapshot already exists with different content: {path}"
            )
        return path
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(material)
            stream.flush()
            os.fsync(stream.fileno())
        _reject_symlinks(path, label="sealed snapshot")
        os.chmod(path, 0o444)
    except Exception:
        raise
    return path


def _read_deck(path: str | Path) -> tuple[list[int], dict[str, Any]]:
    identity = frozen_file_identity(path)
    raw = Path(identity["path"]).read_text(encoding="utf-8")
    try:
        cards = [int(line.strip()) for line in raw.splitlines() if line.strip()]
    except ValueError as exc:
        raise RTPPairingSnapshotError(f"deck is not one numeric card id per line: {path}") from exc
    if len(cards) != 60:
        raise RTPPairingSnapshotError(f"deck must contain exactly 60 card ids: {path}")
    return cards, identity


def _verify_case_binding(
    path: str | Path,
    *,
    seed: int,
    deck0_identity: Mapping[str, Any],
    deck1_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = frozen_file_identity(path)
    identity, binding = verify_pairing_case_binding(
        artifact, expected_debug_seed=seed
    )
    declared = binding["ordered_deck_identities"]
    for index, (expected, actual) in enumerate(
        ((declared[0], deck0_identity), (declared[1], deck1_identity))
    ):
        if expected.get("sha256") != actual["sha256"]:
            raise RTPPairingSnapshotError(
                f"case binding deck {index} does not match the ordered deck file"
            )
        if expected.get("bytes") is not None and expected.get("bytes") != actual["bytes"]:
            raise RTPPairingSnapshotError(
                f"case binding deck {index} byte count does not match"
            )
    return identity, binding


def _artifacts(args: argparse.Namespace) -> PairingArtifactSet:
    return PairingArtifactSet.from_paths(
        engine_path=args.library_path,
        source_manifest_path=args.source_manifest_path,
        patch_path=args.patch_path,
        build_receipt_path=args.build_receipt_path,
    )


def _legal_action(observation: Mapping[str, Any], policy: str) -> list[int]:
    selection = observation.get("select")
    if not isinstance(selection, Mapping):
        raise RTPPairingSnapshotError("probe reached an observation without a selection")
    options = selection.get("option")
    if not isinstance(options, list):
        raise RTPPairingSnapshotError("probe selection does not expose legal options")
    minimum = selection.get("minCount")
    maximum = selection.get("maxCount")
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise RTPPairingSnapshotError("probe selection has an invalid minimum")
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise RTPPairingSnapshotError("probe selection has an invalid maximum")
    if minimum < 0 or minimum > maximum or minimum > len(options):
        raise RTPPairingSnapshotError("probe selection has impossible bounds")
    if policy == "first":
        return list(range(minimum))
    if policy == "last":
        return list(range(len(options) - minimum, len(options)))
    raise RTPPairingSnapshotError(f"unknown probe policy: {policy}")


def _child_result(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = _artifacts(args)
    engine = RtpPairingSnapshotEngine(args.library_path)
    engine.require_bound_artifacts(artifacts)
    with engine.restore_sealed_snapshot_manifest(args.snapshot_seal_path) as battle:
        initial = battle.observation()
        initial_sha256 = canonical_digest(initial)
        actions: list[list[int]] = []
        for _ in range(args.steps):
            if battle.finished:
                break
            action = _legal_action(battle.observation(), args.policy)
            actions.append(action)
            battle.step(action)
        return {
            "schema": "poke_bot.recursive_turn_planner.true_rng_pairing_child_restore/v1",
            "engine_artifact": engine.identity,
            "canonical_abi_sha256": snapshot_abi_sha256(),
            "policy": args.policy,
            "initial_observation_sha256": initial_sha256,
            "actions": actions,
            "transcript_sha256": battle.transcript_sha256,
            "transcript_steps": len(actions),
            "finished": battle.finished,
            "winner": battle.winner,
        }


def _run_child(args: argparse.Namespace, *, policy: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--library-path",
        str(args.library_path),
        "--source-manifest-path",
        str(args.source_manifest_path),
        "--patch-path",
        str(args.patch_path),
        "--build-receipt-path",
        str(args.build_receipt_path),
        "--snapshot-seal-path",
        str(args.snapshot_seal_path),
        "--steps",
        str(args.steps),
        "--policy",
        policy,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RTPPairingSnapshotError(
            "fresh-process snapshot restore failed: " + completed.stderr.strip()
        )
    try:
        loaded = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RTPPairingSnapshotError("fresh-process probe returned invalid JSON") from exc
    if not isinstance(loaded, Mapping):
        raise RTPPairingSnapshotError("fresh-process probe returned a non-object")
    return dict(loaded)


def _first_action(observation: Mapping[str, Any], _: int) -> Sequence[int]:
    return _legal_action(observation, "first")


def _capture_and_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _private_output_root(args.private_output_root)
    artifacts = _artifacts(args)
    engine = RtpPairingSnapshotEngine(args.library_path)
    checked = engine.require_bound_artifacts(artifacts)
    deck0, deck0_identity = _read_deck(args.deck0)
    deck1, deck1_identity = _read_deck(args.deck1)
    case_binding_identity, _ = _verify_case_binding(
        args.case_binding_path,
        seed=args.seed,
        deck0_identity=deck0_identity,
        deck1_identity=deck1_identity,
    )
    with engine.capture_cell_snapshot(deck0, deck1, args.seed) as snapshot:
        snapshot_path = output_root / f"snapshot-{snapshot.fingerprint_sha256[7:]}.bin"
        _write_immutable_bytes(snapshot_path, snapshot.serialized_bytes)
        snapshot_identity = frozen_file_identity(snapshot_path)
        if snapshot_identity["mode"] != 0o444:
            raise RTPPairingSnapshotError("newly sealed snapshot is not mode 0444")
        seal_payload = {
            "schema": SNAPSHOT_SEAL_SCHEMA,
            "status": "sealed",
            "created_at_utc": "1970-01-01T00:00:00Z",
            "engine_artifact_sha256": checked.engine_artifact["sha256"],
            "source_artifact_sha256": checked.source_artifact["sha256"],
            "patch_artifact_sha256": checked.patch_artifact["sha256"],
            "build_artifact_sha256": checked.build_artifact["sha256"],
            "canonical_abi_sha256": snapshot_abi_sha256(),
            "capture_boundary": SNAPSHOT_CAPTURE_BOUNDARY,
            "boundary_tag": SNAPSHOT_BOUNDARY_TAG,
            "rng_kind": "snapshot",
            "requested_seed_audit_only": int(args.seed),
            "requested_seed_is_pairing_proof": False,
            "case_binding_artifact": case_binding_identity,
            "case_binding_artifact_sha256": case_binding_identity["sha256"],
            "snapshot_artifact": snapshot_identity,
        }
        # Emit through the public evidence helper to retain atomic no-clobber
        # publication and lexical symlink rejection.
        from poke_bot.engine_rebuild.rtp_pairing_snapshot import _immutable_json

        seal_path = _immutable_json(
            output_root / f"snapshot-{snapshot.fingerprint_sha256[7:]}.seal.json",
            seal_payload,
        )
        args.snapshot_seal_path = seal_path
        in_process = engine.duplicate_restore_probe(
            snapshot,
            _first_action,
            steps=args.steps,
            delay_seconds=args.delay_seconds,
        )

    first = _run_child(args, policy="first")
    time.sleep(args.delay_seconds)
    second = _run_child(args, policy="first")
    divergent = _run_child(args, policy="last")
    if first != second:
        raise RTPPairingSnapshotError("delayed fresh-process restores diverged")
    if first["engine_artifact"] != checked.engine_artifact:
        raise RTPPairingSnapshotError("fresh worker loaded a different engine artifact")
    if first["canonical_abi_sha256"] != snapshot_abi_sha256():
        raise RTPPairingSnapshotError("fresh worker loaded a different pairing ABI")
    if first["initial_observation_sha256"] != divergent["initial_observation_sha256"]:
        raise RTPPairingSnapshotError("divergent policy did not restore the same initial state")
    if first["actions"] == divergent["actions"]:
        raise RTPPairingSnapshotError("divergent policy did not make a distinct legal choice")
    deterministic_probe = {
        **in_process,
        "cross_process_restore_passed": True,
        "delayed_restore_transcript_passed": True,
        "cross_process_reference_transcript_sha256": first["transcript_sha256"],
        "cross_process_reference_steps": first["transcript_steps"],
        "cross_process_initial_observation_sha256": first["initial_observation_sha256"],
        "cross_process_finished": first["finished"],
        "cross_process_winner": first["winner"],
        "delayed_restore_seconds": float(args.delay_seconds),
    }
    probe_path = emit_true_rng_pairing_probe(
        output_path=output_root / "true-rng-pairing-probe-v2.json",
        artifacts=checked,
        deterministic_probe=deterministic_probe,
        divergent_policy_true_pairing_passed=True,
        all_arms_restored_or_replayed=True,
    )
    capability_path = emit_true_rng_pairing_capability(
        output_path=output_root / "true-rng-pairing-capability-v2.json",
        artifacts=checked,
        probe_path=probe_path,
    )
    return {
        "snapshot_seal": frozen_file_identity(seal_path),
        "probe": frozen_file_identity(probe_path),
        "capability": frozen_file_identity(capability_path),
        "engine": engine.identity,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-path", type=Path, required=True)
    parser.add_argument("--source-manifest-path", type=Path, required=True)
    parser.add_argument("--patch-path", type=Path, required=True)
    parser.add_argument("--build-receipt-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--delay-seconds", type=float, default=0.05)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--snapshot-seal-path", type=Path)
    parser.add_argument("--policy", choices=("first", "last"), default="first")
    parser.add_argument("--private-output-root", type=Path)
    parser.add_argument("--deck0", type=Path)
    parser.add_argument("--deck1", type=Path)
    parser.add_argument("--case-binding-path", type=Path)
    parser.add_argument("--seed", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.steps < 1 or args.steps > 64:
            raise RTPPairingSnapshotError("probe steps must be in [1, 64]")
        if args.delay_seconds <= 0.0 or args.delay_seconds > 5.0:
            raise RTPPairingSnapshotError("probe delay must be in (0, 5] seconds")
        if args.child:
            if args.snapshot_seal_path is None:
                raise RTPPairingSnapshotError("child requires --snapshot-seal-path")
            print(json.dumps(_child_result(args), sort_keys=True))
        else:
            if (
                args.private_output_root is None
                or args.deck0 is None
                or args.deck1 is None
                or args.case_binding_path is None
            ):
                raise RTPPairingSnapshotError(
                    "capture requires --private-output-root, --deck0, --deck1, and --case-binding-path"
                )
            if args.seed is None or isinstance(args.seed, bool) or not 0 <= args.seed <= 0xFFFFFFFF:
                raise RTPPairingSnapshotError("capture requires a uint32 --seed")
            print(json.dumps(_capture_and_probe(args), sort_keys=True))
        return 0
    except RTPPairingSnapshotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
