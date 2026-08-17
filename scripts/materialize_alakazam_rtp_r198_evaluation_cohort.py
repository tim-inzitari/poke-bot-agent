#!/usr/bin/env python3
"""Seal the r198 evaluation-only official-control cohort and exclusion proof.

This is deliberately a producer for *evaluation inputs*, not a trainer,
selector, submission tool, or promotion tool.  It reads the completed r197
shadow receipt and the frozen official-four registry, then creates exactly one
4 × 2 × 125 cohort plus a source-exclusion proof below a caller-supplied new
output root.  It never reuses supervised heldout rows as gameplay evidence.

The output directory name is content-addressed from immutable candidate and
registry provenance.  Publication is no-clobber; an interrupted directory is
preserved as forensic evidence rather than overwritten or retried in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


COHORT_SCHEMA = "poke_bot.recursive_turn_planner.r197_evaluation_only_cohort/v1"
PROOF_SCHEMA = "poke_bot.recursive_turn_planner.r197_evaluation_only_source_exclusion/v1"
COMPLETION_SCHEMA = "poke_bot.alakazam_rtp_r197_shadow_candidate/v1"
CANDIDATE_CONTRACT_SHA256 = (
    "sha256:bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
OFFICIAL_PANEL = (
    (
        "iono",
        "sha256:6ba8e818b698774b6e437364e9457600eda950fbefb663d8e4ad39cdaf0371e2",
    ),
    (
        "dragapult-ex",
        "sha256:835dcbcc26366faa04d902db727620d4b12618b6a66d000dccb9c9b86e9d62a0",
    ),
    (
        "mega-abomasnow-ex",
        "sha256:57a9499b2bee493a830abaf5a3e19b8a73faea200faee87aeeb2864bab25c2fb",
    ),
    (
        "mega-lucario-ex",
        "sha256:98f20936d430c6cc60f3eb1da8230392bf6dce8ecacf97773bda4db63f56376a",
    ),
)
RESEARCH_CONTROL_REGISTRY_SHA256 = (
    "sha256:78fd8e52df1464db94e74a49247a67ced41b5d164dc86fafec3229f2c1e47edc"
)
RESEARCH_CONTROL_REGISTRY_BYTES = 2117
REPLICATES_PER_SEAT = 125
SEATS = (0, 1)
PAIRED_CELLS = len(OFFICIAL_PANEL) * len(SEATS) * REPLICATES_PER_SEAT
COHORT_FILENAME = "evaluation_only_cohort.json"
PROOF_FILENAME = "source-exclusion-proof.json"


class CohortError(RuntimeError):
    """Raised when evaluation-only cohort provenance cannot be proved."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _physical_path(path: Path | str, *, label: str, kind: str) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:]):
        current = current / part
        try:
            status = current.lstat()
        except OSError as exc:
            raise CohortError(f"{label} is missing: {absolute}") from exc
        if stat.S_ISLNK(status.st_mode):
            raise CohortError(f"{label} traverses a symlink: {current}")
        if index < len(absolute.parts) - 2 and not stat.S_ISDIR(status.st_mode):
            raise CohortError(f"{label} has a non-directory ancestor: {current}")
    status = absolute.lstat()
    if kind == "file" and not stat.S_ISREG(status.st_mode):
        raise CohortError(f"{label} is not a physical regular file: {absolute}")
    if kind == "directory" and not stat.S_ISDIR(status.st_mode):
        raise CohortError(f"{label} is not a physical directory: {absolute}")
    return absolute


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CohortError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise CohortError(f"{label} must be a JSON object")
    return dict(value)


def _identity(path: Path, *, label: str) -> dict[str, Any]:
    file_path = _physical_path(path, label=label, kind="file")
    return {
        "path": str(file_path),
        "sha256": _sha256(file_path),
        "bytes": int(file_path.stat().st_size),
    }


def _sha256_value(value: Any, *, label: str) -> str:
    text = str(value or "")
    if (
        len(text) != 71
        or not text.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise CohortError(f"{label} must be a SHA-256 identity")
    return text


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CohortError(f"{label} must be a JSON object")
    return dict(value)


def _candidate_provenance(completion_receipt: Path) -> dict[str, Any]:
    receipt_path = _physical_path(
        completion_receipt, label="r197 completion receipt", kind="file"
    )
    receipt = _json_object(receipt_path, label="r197 completion receipt")
    if (
        receipt.get("schema") != COMPLETION_SCHEMA
        or receipt.get("status") != "completed_shadow_only"
        or receipt.get("candidate_contract_sha256") != CANDIDATE_CONTRACT_SHA256
    ):
        raise CohortError("completion receipt is not the exact completed shadow-only r198 candidate")
    authority = _mapping(receipt.get("authority"), label="r197 completion authority")
    if authority.get("shadow_only") is not True or any(
        authority.get(key) is not False
        for key in (
            "serving_eligible",
            "action_authority_enabled",
            "selector_authority",
            "live_checkpoint_publication",
            "submission_eligible",
        )
    ):
        raise CohortError("completion receipt unexpectedly grants evaluation candidate authority")
    contract = _mapping(receipt.get("contract"), label="r197 completion contract")
    corpus = _mapping(
        contract.get("complete_action_corpus"), label="r197 complete action corpus"
    )
    if corpus.get("schema") != "poke_bot.rtp_complete_action_shadow_corpus/v1":
        raise CohortError("completion receipt corpus schema is invalid")
    split = _mapping(corpus.get("split"), label="r197 corpus split")
    if (
        split.get("source_disjoint") is not True
        or split.get("unit") != "episode_id"
        or int(split.get("seed") or -1) != 5_000_000
    ):
        raise CohortError("r197 completion receipt does not bind a source-disjoint episode split")
    selection = _mapping(corpus.get("selection"), label="r197 selection")
    if selection.get("schema") != "poke_bot.recursive_turn_planner.r197_training_selection_plan/v1":
        raise CohortError("r197 completion selection plan schema is invalid")
    train = _mapping(selection.get("train"), label="r197 train selection")
    heldout = _mapping(selection.get("heldout"), label="r197 heldout selection")
    train_cap = _mapping(train.get("batch_cap_selection"), label="r197 train cap")
    heldout_cap = _mapping(heldout.get("batch_cap_selection"), label="r197 heldout cap")
    train_digest = _sha256_value(
        train_cap.get("retained_episode_ids_sha256"), label="r197 train selection digest"
    )
    heldout_digest = _sha256_value(
        heldout_cap.get("retained_episode_ids_sha256"), label="r197 heldout selection digest"
    )
    if (
        _sha256_value(selection.get("train_selection_sha256"), label="r197 plan train digest")
        != train_digest
        or _sha256_value(
            selection.get("heldout_selection_sha256"), label="r197 plan heldout digest"
        )
        != heldout_digest
    ):
        raise CohortError("r197 selection plan does not bind its retained episode IDs")
    training = _mapping(receipt.get("training"), label="r197 completion training")
    if training.get("heldout_is_source_excluded") is not True:
        raise CohortError("r197 completion receipt does not attest source-excluded heldout data")
    return {
        "completion_receipt": _identity(receipt_path, label="r197 completion receipt"),
        "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
        "r197_corpus_manifest_sha256": _sha256_value(
            corpus.get("manifest_sha256"), label="r197 corpus manifest"
        ),
        "r197_corpus_receipt_sha256": _sha256_value(
            corpus.get("receipt_sha256"), label="r197 corpus receipt"
        ),
        "r197_selection_plan_sha256": _sha256_value(
            selection.get("selection_plan_sha256"), label="r197 selection plan"
        ),
        "r197_train_selection_sha256": train_digest,
        "r197_heldout_selection_sha256": heldout_digest,
    }


def _registry_rows(registry_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry_file = _physical_path(
        registry_path, label="official research-control registry", kind="file"
    )
    identity = _identity(registry_file, label="official research-control registry")
    if (
        identity["sha256"] != RESEARCH_CONTROL_REGISTRY_SHA256
        or identity["bytes"] != RESEARCH_CONTROL_REGISTRY_BYTES
    ):
        raise CohortError("official research-control registry identity changed")
    registry = _json_object(registry_file, label="official research-control registry")
    if registry.get("schema") != "poke_bot.research_control_registry/v1":
        raise CohortError("official research-control registry schema is invalid")
    rows = registry.get("controls")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise CohortError("official research-control registry controls must be a list")
    expected = dict(OFFICIAL_PANEL)
    by_id: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, label="official control registry row")
        identifier = str(row.get("opponent_id") or "")
        if identifier in by_id:
            raise CohortError("official research-control registry repeats an opponent")
        by_id[identifier] = row
    if tuple(by_id) != tuple(identifier for identifier, _ in OFFICIAL_PANEL):
        raise CohortError("official research-control registry is not the ordered official-four panel")
    normalized: list[dict[str, Any]] = []
    for identifier, digest in OFFICIAL_PANEL:
        row = by_id[identifier]
        if (
            row.get("content_digest") != digest
            or row.get("training_eligible") is not False
            or row.get("formal_eval") is not False
            or row.get("included_in_gate_pass") is not False
            or float(row.get("gate_weight") or -1.0) != 0.0
            or row.get("source_gate_id") != "legacy-original-four"
        ):
            raise CohortError(f"official control {identifier} has an unsafe role")
        normalized.append(
            {"id": identifier, "content_digest": digest, "training_eligible": False}
        )
    return identity, normalized


def _cohort_payload(
    provenance: Mapping[str, Any], registry_identity: Mapping[str, Any], registry_rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_identity_payload = {
        "schema": "poke_bot.recursive_turn_planner.r198_evaluation_only_source_identity/v1",
        "candidate_contract_sha256": provenance["candidate_contract_sha256"],
        "r197_completion_receipt_sha256": provenance["completion_receipt"]["sha256"],
        "r197_corpus_manifest_sha256": provenance["r197_corpus_manifest_sha256"],
        "r197_corpus_receipt_sha256": provenance["r197_corpus_receipt_sha256"],
        "r197_selection_plan_sha256": provenance["r197_selection_plan_sha256"],
        "r197_train_selection_sha256": provenance["r197_train_selection_sha256"],
        "r197_heldout_selection_sha256": provenance["r197_heldout_selection_sha256"],
        "research_control_registry_sha256": registry_identity["sha256"],
        "registry_rows": [dict(row) for row in registry_rows],
        "evaluation_only": True,
        "replay_eligible": False,
        "training_eligible": False,
        "source_identity_overlap_count": 0,
    }
    source_identity = _canonical_digest(source_identity_payload)
    cases: list[dict[str, Any]] = []
    for identifier, digest in OFFICIAL_PANEL:
        for candidate_seat in SEATS:
            for replicate in range(REPLICATES_PER_SEAT):
                cases.append(
                    {
                        "case_id": f"r198-{identifier}-seat{candidate_seat}-rep{replicate:03d}",
                        "opponent_id": identifier,
                        "content_digest": digest,
                        "candidate_seat": candidate_seat,
                        "replicate": replicate,
                        "evaluation_only": True,
                        "training_eligible": False,
                        "replay_eligible": False,
                    }
                )
    bindings = sorted(
        [
            {
                "case_id": row["case_id"],
                "opponent_id": row["opponent_id"],
                "content_digest": row["content_digest"],
                "candidate_seat": row["candidate_seat"],
                "replicate": row["replicate"],
            }
            for row in cases
        ],
        key=lambda row: str(row["case_id"]),
    )
    binding_digest = _canonical_digest(bindings)
    cohort = {
        "schema": COHORT_SCHEMA,
        "status": "frozen",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "source_identity_sha256": source_identity,
        "registry_rows": [dict(row) for row in registry_rows],
        "cases": cases,
        "case_bindings_sha256": binding_digest,
    }
    proof = {
        "schema": PROOF_SCHEMA,
        "status": "verified",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "all_registry_rows_training_eligible": False,
        "r197_supervised_heldout_calibration_only": True,
        "r197_completion_receipt_sha256": provenance["completion_receipt"]["sha256"],
        "candidate_contract_sha256": provenance["candidate_contract_sha256"],
        "r197_corpus_manifest_sha256": provenance["r197_corpus_manifest_sha256"],
        "r197_corpus_receipt_sha256": provenance["r197_corpus_receipt_sha256"],
        "r197_selection_plan_sha256": provenance["r197_selection_plan_sha256"],
        "r197_train_selection_sha256": provenance["r197_train_selection_sha256"],
        "r197_heldout_selection_sha256": provenance["r197_heldout_selection_sha256"],
        "source_identity_sha256": source_identity,
        "evaluation_case_bindings_sha256": binding_digest,
        "source_identity_overlap_count": 0,
        "registry_rows": [dict(row) for row in registry_rows],
    }
    return cohort, proof


def plan(
    *, completion_receipt: Path, registry_path: Path, output_root: Path
) -> dict[str, Any]:
    """Return deterministic bytes/paths without writing any artifacts."""

    provenance = _candidate_provenance(completion_receipt)
    registry_identity, rows = _registry_rows(registry_path)
    cohort, proof = _cohort_payload(provenance, registry_identity, rows)
    cohort_bytes = (json.dumps(cohort, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    proof["evaluation_only_cohort_sha256"] = "sha256:" + hashlib.sha256(cohort_bytes).hexdigest()
    proof["evaluation_only_cohort_bytes"] = len(cohort_bytes)
    proof_bytes = (json.dumps(proof, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    content = _canonical_digest(
        {
            "cohort_sha256": "sha256:" + hashlib.sha256(cohort_bytes).hexdigest(),
            "proof_sha256": "sha256:" + hashlib.sha256(proof_bytes).hexdigest(),
            "source_identity_sha256": cohort["source_identity_sha256"],
        }
    )
    base = _absolute(output_root)
    return {
        "content_sha256": content,
        "output_dir": base / ("cohort-" + content.removeprefix("sha256:")[:16]),
        "cohort": cohort,
        "proof": proof,
        "cohort_bytes": cohort_bytes,
        "proof_bytes": proof_bytes,
        "provenance": provenance,
        "registry": registry_identity,
    }


def _ensure_output_directory(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            status = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o755)
            except FileExistsError:
                pass
            status = current.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise CohortError(f"{label} has a symlink or non-directory component: {current}")
    return absolute


def _write_exclusive(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CohortError(f"refusing to overwrite immutable cohort artifact: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise CohortError(f"cannot write immutable cohort artifact: {path}") from exc
    os.chmod(path, 0o444)


def _sealed_identity(path: Path, *, expected: bytes, label: str) -> dict[str, Any]:
    file_path = _physical_path(path, label=label, kind="file")
    if stat.S_IMODE(file_path.lstat().st_mode) != 0o444 or file_path.read_bytes() != expected:
        raise CohortError(f"{label} is not the exact sealed expected artifact")
    return _identity(file_path, label=label)


def materialize(
    *, completion_receipt: Path, registry_path: Path, output_root: Path
) -> dict[str, Any]:
    """Create or strictly revalidate the no-clobber cohort/proof pair."""

    prepared = plan(
        completion_receipt=completion_receipt,
        registry_path=registry_path,
        output_root=output_root,
    )
    base = _ensure_output_directory(_absolute(output_root), label="r198 cohort output root")
    target = Path(prepared["output_dir"])
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise CohortError("content-addressed cohort target escapes its output root") from exc
    cohort_path = target / COHORT_FILENAME
    proof_path = target / PROOF_FILENAME
    if target.exists() or target.is_symlink():
        _physical_path(target, label="existing r198 cohort directory", kind="directory")
        if not cohort_path.exists() or not proof_path.exists():
            raise CohortError("existing cohort directory is incomplete and cannot be reused")
        cohort_identity = _sealed_identity(
            cohort_path, expected=prepared["cohort_bytes"], label="existing evaluation-only cohort"
        )
        proof_identity = _sealed_identity(
            proof_path, expected=prepared["proof_bytes"], label="existing source-exclusion proof"
        )
        return {
            "status": "already_materialized",
            "content_sha256": prepared["content_sha256"],
            "output_dir": str(target),
            "evaluation_only_cohort": cohort_identity,
            "source_exclusion_proof": proof_identity,
            "provenance": prepared["provenance"],
            "registry": prepared["registry"],
        }
    try:
        target.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise CohortError(f"cohort target appeared while publishing: {target}") from exc
    try:
        _write_exclusive(cohort_path, prepared["cohort_bytes"])
        _write_exclusive(proof_path, prepared["proof_bytes"])
        os.chmod(target, 0o555)
    except Exception:
        # Preserve incomplete evidence.  Never delete or overwrite it.
        raise
    return {
        "status": "materialized",
        "content_sha256": prepared["content_sha256"],
        "output_dir": str(target),
        "evaluation_only_cohort": _sealed_identity(
            cohort_path, expected=prepared["cohort_bytes"], label="evaluation-only cohort"
        ),
        "source_exclusion_proof": _sealed_identity(
            proof_path, expected=prepared["proof_bytes"], label="source-exclusion proof"
        ),
        "provenance": prepared["provenance"],
        "registry": prepared["registry"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="describe without writing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check:
            prepared = plan(
                completion_receipt=args.completion_receipt,
                registry_path=args.registry,
                output_root=args.output_root,
            )
            print(
                json.dumps(
                    {
                        "status": "checked_no_writes",
                        "content_sha256": prepared["content_sha256"],
                        "pending_output_dir": str(prepared["output_dir"]),
                    },
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps(
                    materialize(
                        completion_receipt=args.completion_receipt,
                        registry_path=args.registry,
                        output_root=args.output_root,
                    ),
                    sort_keys=True,
                )
            )
    except CohortError as exc:
        print(f"r198 evaluation cohort error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
