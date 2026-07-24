#!/usr/bin/env python3
"""Import audited rare-route assets without mutating the live specialist run.

The service waits for Elmo to finish the additive six-day router/corpus work,
copies only checksum-verifiable artifacts, builds a new hard-linked corpus
generation on Inzi, and publishes one atomic receipt.  Handoff controllers may
consume that receipt at a specialist boundary; the active trainer never does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "poke_bot.rare_route_asset_promotion/v1"
AUDIT_SCHEMA = "poke_bot.public_matchup_tree_candidate_audit/v1"
POINTER_SCHEMA = "poke_bot.pinned_expert_corpus/v1"
RARE_ARCHETYPES = (
    "dragapult-blaziken",
    "dragapult-dusknoir",
    "dudunsparce",
    "gardevoir",
    "ns-zoroark",
    "walrein",
)
REQUIRED_TARGETS = (
    "temporal_action_rows",
    "opponent_hand_rows",
    "opponent_remainder_rows",
    "opponent_private_prize_rows",
    "lethal_threat_rows",
    "prize_race_rows",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _run(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(argv)}"
        )


def _remote_ready(host: str, path: str) -> None:
    _run(
        [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"sudo -n test -s {path}",
        ]
    )


def _copy_remote(host: str, source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            completed = subprocess.run(
                [
                    "/usr/bin/ssh",
                    "-o",
                    "BatchMode=yes",
                    host,
                    f"sudo -n cat -- {shlex.quote(source)}",
                ],
                check=False,
                stdout=stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        if completed.returncode:
            raise RuntimeError(
                f"remote copy failed rc={completed.returncode}: {host}:{source}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_remote_directory(host: str, source: str, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    remote = subprocess.Popen(
        [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            host,
            (
                "sudo -n tar -C "
                f"{shlex.quote(source.rstrip('/'))} -cf - ."
            ),
        ],
        stdout=subprocess.PIPE,
    )
    if remote.stdout is None:
        raise RuntimeError("remote archive stream did not open")
    local = subprocess.run(
        ["/usr/bin/tar", "-xf", "-", "-C", str(destination)],
        stdin=remote.stdout,
        check=False,
    )
    remote.stdout.close()
    remote_returncode = remote.wait()
    if local.returncode or remote_returncode:
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError(
            "remote directory copy failed: "
            f"ssh_rc={remote_returncode} tar_rc={local.returncode}"
        )


def _sidecar_dates(directory: Path) -> list[str]:
    dates: list[str] = []
    for sidecar in sorted(directory.glob("*.features.json")):
        payload = _read(sidecar)
        dates.extend(str(value) for value in payload.get("source_dates") or ())
    if not dates or len(dates) != len(set(dates)):
        raise RuntimeError(f"missing or overlapping corpus dates: {directory}")
    return sorted(dates)


def _pointer_has_complete_target_coverage(
    pointer: dict[str, Any],
) -> bool:
    totals = pointer.get("totals") or {}
    decisions = int(totals.get("decisions_kept") or 0)
    coverage = totals.get("target_coverage") or {}
    return decisions > 0 and all(
        int(coverage.get(name) or 0) == decisions
        for name in REQUIRED_TARGETS
    )


def _make_directories_owner_writable(root: Path) -> None:
    """Make a copied immutable tree replaceable without mutating its files.

    Protected corpus generations are deliberately installed read-only.  The
    merge uses hard links for the large shard files, so changing file modes
    here would also change the source generation.  Directory modes are not
    shared by ``copytree``; adding owner write/execute permission only to the
    copied directories lets the atomic builder unlink manifests and replace
    archetype directories while leaving every protected shard untouched.
    """
    for directory in (root, *sorted(path for path in root.rglob("*") if path.is_dir())):
        directory.chmod(directory.stat().st_mode | 0o300)


def _merge_corpora(
    *,
    source_root: Path,
    additive_root: Path,
    output_root: Path,
    python: Path,
    repository: Path,
) -> dict[str, int]:
    if output_root.exists():
        raise FileExistsError(
            f"refusing to replace corpus generation: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", dir=str(output_root.parent)
        )
    )
    try:
        shutil.rmtree(temporary)
        shutil.copytree(source_root, temporary, copy_function=os.link)
        _make_directories_owner_writable(temporary)
        for stale in (
            "SPECIALIST_CORPORA_READY.json",
            "VALIDATED_SPECIALIST_CORPORA.json",
        ):
            temporary.joinpath(stale).unlink(missing_ok=True)

        decisions: dict[str, int] = {}
        for archetype in RARE_ARCHETYPES:
            target = temporary / archetype
            old = source_root / archetype
            additive = additive_root / archetype
            if not additive.joinpath("PROTECTED_EXPERT_CORPUS.json").is_file():
                if old.joinpath("PROTECTED_EXPERT_CORPUS.json").is_file():
                    decisions[archetype] = int(
                        (
                            _read(
                                old / "PROTECTED_EXPERT_CORPUS.json"
                            ).get("totals")
                            or {}
                        ).get("decisions_kept")
                        or 0
                    )
                    continue
                raise RuntimeError(
                    f"both base and additive specialist corpus are missing: "
                    f"{archetype}"
                )
            replacement = temporary / f".{archetype}.merge"
            replacement.mkdir()
            for source in (old, additive):
                if not source.is_dir():
                    continue
                for path in source.glob("*.features*"):
                    destination = replacement / path.name
                    if destination.exists():
                        raise RuntimeError(
                            f"duplicate additive shard name: {destination.name}"
                        )
                    os.link(path, destination)
            dates = _sidecar_dates(replacement)
            command = [
                str(python),
                "-u",
                str(repository / "scripts/assemble_feature_manifest.py"),
                "--staging-dir",
                str(replacement),
                "--out",
                str(replacement / "manifest.json"),
                "--compact-mode",
                "temporal-expert-v1",
                "--expected-max-context",
                "320",
                "--required-archetype",
                archetype,
                "--seal-protected",
                "--require-target-coverage",
                "temporal_action_rows",
            ]
            for day in dates:
                command.extend(("--expected-date", day))
            # Historical public-only additions legitimately omit the
            # opponent's hidden hand/prizes. Requiring those targets on the
            # combined corpus would either reject truthful public evidence or
            # encourage fabricated labels. Preserve full multi-head sealing
            # only when the additive generation itself has complete coverage.
            # Never fabricate those labels.
            additive_pointer = _read(
                additive / "PROTECTED_EXPERT_CORPUS.json"
            )
            if _pointer_has_complete_target_coverage(additive_pointer):
                for target_name in REQUIRED_TARGETS:
                    if target_name != "temporal_action_rows":
                        command.extend(
                            ("--require-target-coverage", target_name)
                        )
            _run(command)
            pointer = _read(replacement / "PROTECTED_EXPERT_CORPUS.json")
            if (
                pointer.get("schema") != POINTER_SCHEMA
                or (pointer.get("selection") or {}).get("value") != archetype
            ):
                raise RuntimeError(
                    f"merged protected corpus identity failed: {archetype}"
                )
            decisions[archetype] = int(
                (pointer.get("totals") or {}).get("decisions_kept") or 0
            )
            shutil.rmtree(target, ignore_errors=True)
            os.replace(replacement, target)
        os.replace(temporary, output_root)
        return decisions
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def promote(args: argparse.Namespace) -> dict[str, Any]:
    remote_root = args.remote_root.rstrip("/")
    remote_audit = f"{remote_root}/public-matchup-tree.audit.json"
    remote_tree = f"{remote_root}/public-matchup-tree.json"
    remote_split = args.remote_split_root.rstrip("/")
    _remote_ready(args.host, remote_audit)
    if not args.router_only:
        _remote_ready(
            args.host, f"{remote_split}/SPECIALIST_CORPORA_READY.json"
        )

    stage = args.stage_root.expanduser().resolve()
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    audit_stage = stage / "public-matchup-tree.audit.json"
    tree_stage = stage / "public-matchup-tree.json"
    split_stage = stage / "specialists-v1"
    _copy_remote(args.host, remote_audit, audit_stage)
    _copy_remote(args.host, remote_tree, tree_stage)
    if not args.router_only:
        _copy_remote_directory(args.host, remote_split, split_stage)

    audit = _read(audit_stage)
    accepted = tuple(str(value) for value in audit.get("accepted_specialist_ids") or ())
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("runtime_enabled") is not False
        or int(audit.get("target_count") or 0) != 22
        or audit.get("artifact_sha256") != _sha256(tree_stage)
        or float(audit.get("minimum_precision") or 0.0) != 0.93
        or int(audit.get("minimum_weighted_support") or 0) != 10_000
        or len(accepted) != len(set(accepted))
        or (
            args.require_all_targets
            and (
                len(accepted) != 22
                or bool(audit.get("rejected_specialists"))
            )
        )
    ):
        raise RuntimeError("Elmo router audit identity failed")

    output_root = args.output_corpus_root.expanduser().resolve()
    source_root = args.source_corpus_root.expanduser().resolve()
    if args.router_only:
        if output_root != source_root or not output_root.is_dir():
            raise RuntimeError(
                "router-only promotion must retain the existing protected "
                "corpus generation"
            )
        decisions = {}
        for archetype in RARE_ARCHETYPES:
            pointer = output_root / archetype / "PROTECTED_EXPERT_CORPUS.json"
            decisions[archetype] = (
                int((_read(pointer).get("totals") or {}).get("decisions_kept") or 0)
                if pointer.is_file()
                else 0
            )
    elif output_root.is_dir():
        decisions = {
            archetype: int(
                (
                    _read(
                        output_root
                        / archetype
                        / "PROTECTED_EXPERT_CORPUS.json"
                    ).get("totals")
                    or {}
                ).get("decisions_kept")
                or 0
            )
            for archetype in RARE_ARCHETYPES
        }
    else:
        decisions = _merge_corpora(
            source_root=args.source_corpus_root.expanduser().resolve(),
            additive_root=split_stage,
            output_root=output_root,
            python=args.python.expanduser().resolve(),
            repository=args.repository.expanduser().resolve(),
        )

    tree_output = args.tree_output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    tree_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tree_stage, tree_output)
    shutil.copy2(audit_stage, audit_output)
    coverage: dict[str, dict[str, Any]] = {}
    for archetype in RARE_ARCHETYPES:
        pointer_path = output_root / archetype / "PROTECTED_EXPERT_CORPUS.json"
        if not pointer_path.is_file():
            coverage[archetype] = {
                "mode": "unavailable",
                "target_coverage": {},
            }
            continue
        pointer = _read(pointer_path)
        manifest = _read(output_root / archetype / str(pointer["manifest"]))
        target_coverage = dict(
            ((manifest.get("totals") or {}).get("target_coverage") or {})
        )
        decisions_kept = int(
            ((manifest.get("totals") or {}).get("decisions_kept") or 0)
        )
        full = all(
            decisions_kept > 0
            and int(target_coverage.get(name) or 0) == decisions_kept
            for name in REQUIRED_TARGETS
        )
        coverage[archetype] = {
            "mode": (
                "full_multihead"
                if full
                else "temporal_policy_with_masked_missing_aux"
            ),
            "target_coverage": target_coverage,
        }
    receipt = {
        "schema": SCHEMA,
        "status": "ready",
        "candidate_tree": str(tree_output),
        "candidate_tree_sha256": _sha256(tree_output),
        "candidate_audit": str(audit_output),
        "candidate_audit_sha256": _sha256(audit_output),
        "accepted_specialist_ids": list(accepted),
        "accepted_count": len(accepted),
        "corpus_root": str(output_root),
        "corpus_mode": (
            "existing_protected_generation"
            if args.router_only
            else "additive_rare_route_generation"
        ),
        "rare_archetype_decisions": decisions,
        "rare_archetype_coverage": coverage,
        "minimum_decisions": 20_000,
        "ready_rare_archetype_ids": sorted(
            archetype
            for archetype, count in decisions.items()
            if count >= 20_000 and archetype in accepted
        ),
        "live_trainer_modified": False,
        "activation_policy": "specialist_boundary_only",
    }
    receipt["identity_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    output = args.receipt.expanduser().resolve()
    if output.is_file():
        if _read(output) != receipt:
            raise RuntimeError("existing rare-route promotion receipt differs")
    else:
        _atomic(output, receipt)
    shutil.rmtree(stage, ignore_errors=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="elmo")
    parser.add_argument(
        "--remote-root",
        default=(
            "/mnt/Main/main/poke-adapter-oracle-v29/output/"
            "public-matchup-tree-calibration-v34"
        ),
    )
    parser.add_argument(
        "--remote-split-root",
        default=(
            "/mnt/Main/main/poke-bot-agent/archive/"
            "rare-route-expert-history-20260626-20260701/specialists-v1"
        ),
    )
    parser.add_argument("--source-corpus-root", type=Path, required=True)
    parser.add_argument("--output-corpus-root", type=Path, required=True)
    parser.add_argument("--tree-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--require-all-targets",
        action="store_true",
        help="Fail closed unless every one of the canonical 22 routes passed.",
    )
    parser.add_argument(
        "--router-only",
        action="store_true",
        help=(
            "Promote an audited router while retaining the existing immutable "
            "expert-corpus generation. This does not claim rare corpora are ready."
        ),
    )
    args = parser.parse_args()
    result = promote(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
