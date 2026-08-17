"""Immutable, separately mounted baseline payloads for r241.

The r241 executable source snapshot is intentionally code-only.  Public-mix
baseline packages, including the H10 Marnie model/data package, therefore
live in a second content-addressed read-only tree.  This module owns the small
format shared by the publisher and the r241 launcher; it never imports a
baseline's Python entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .r241_direct_policy_runtime import (
    R241_H10_CONTENT_SHA256,
    R241_H10_DIR_NAME,
    R241_H10_MODEL_SHA256,
    R241_H10_MODEL_SIZE_BYTES,
    R241_H10_OPPONENT_ID,
)
from .r241_marnie_direct_policy_adapter import (
    R241_H10_MATCHUP_TREE_SHA256,
    R241_H10_MATCHUP_TREE_SIZE_BYTES,
)


BASELINE_PAYLOAD_SNAPSHOT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_baseline_payload_snapshot/v1"
)
BASELINE_PAYLOAD_STAGING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_baseline_payload_snapshot_staging/v1"
)
CANONICAL_BASELINE_ROSTER_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_canonical_baseline_roster/v1"
)
BASELINE_PAYLOAD_MANIFEST_FILENAME = "r241-baseline-payload-manifest.json"
BASELINE_PAYLOAD_ROOT_PREFIX = "alakazam-new-list-direct-r241-baselines-"
CANDIDATE_ID = "alakazam-new-list-direct-policy-r241"
REVISION = 241
_SHA256_PREFIX = "sha256:"
_FORBIDDEN_COMPONENTS = frozenset({".git", ".venv", "__pycache__", "outputs", "runtime"})


class R241BaselinePayloadError(RuntimeError):
    """The separate r241 baseline payload cannot be admitted."""


def canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R241BaselinePayloadError("baseline payload is not canonical JSON") from exc


def sha256_bytes(payload: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def valid_sha256(value: object, *, label: str) -> str:
    rendered = str(value or "")
    if not rendered.startswith(_SHA256_PREFIX) or len(rendered) != 71:
        raise R241BaselinePayloadError(f"{label} is missing a canonical SHA-256")
    try:
        int(rendered.removeprefix(_SHA256_PREFIX), 16)
    except ValueError as exc:
        raise R241BaselinePayloadError(f"{label} is not a hexadecimal SHA-256") from exc
    return rendered


def regular_directory(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise R241BaselinePayloadError(f"{label} must be a real non-symlink directory: {raw}")
    return raw.resolve()


def regular_file(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241BaselinePayloadError(f"{label} must be a regular non-symlink file: {raw}")
    return raw.resolve()


def _relative(value: str | Path, *, label: str) -> Path:
    text = str(value or "").strip()
    candidate = Path(text)
    if (
        not text
        or candidate.is_absolute()
        or "." in candidate.parts
        or ".." in candidate.parts
    ):
        raise R241BaselinePayloadError(f"{label} is not a safe relative path: {text!r}")
    for part in candidate.parts:
        if part in _FORBIDDEN_COMPONENTS or part.startswith(".env") or part.endswith(".env"):
            raise R241BaselinePayloadError(f"{label} uses forbidden component: {text}")
    return candidate


def _member(root: Path, relative: str | Path, *, label: str) -> Path:
    candidate = root / _relative(relative, label=label)
    if candidate.is_symlink() or not candidate.is_file():
        raise R241BaselinePayloadError(f"{label} is not a regular file: {candidate}")
    resolved = candidate.resolve()
    if root not in resolved.parents:
        raise R241BaselinePayloadError(f"{label} escapes baseline payload root")
    return resolved


def tree_digest(rows: Sequence[Mapping[str, object]]) -> str:
    canonical = [
        {
            "path": str(row["path"]),
            "sha256": valid_sha256(row["sha256"], label="baseline tree member sha"),
            "size_bytes": _exact_int(row["size_bytes"], label="baseline tree member size"),
        }
        for row in sorted(rows, key=lambda item: str(item["path"]))
    ]
    return sha256_bytes(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    )


def _exact_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise R241BaselinePayloadError(f"{label} must be an exact integer")
    return value


def content_digest(root: Path | str) -> str:
    """Match ``baselines_runtime.baseline_content_digest`` without importing code."""

    directory = regular_directory(root, label="baseline package")
    rows: list[dict[str, object]] = []
    for child in sorted(path for path in directory.rglob("*") if path.is_file()):
        if child.is_symlink():
            raise R241BaselinePayloadError(f"baseline package has symlinked member: {child}")
        if "__pycache__" in child.parts or child.suffix in (".pyc", ".log"):
            continue
        resolved = child.resolve()
        if directory not in resolved.parents:
            raise R241BaselinePayloadError(f"baseline package member escapes root: {child}")
        rows.append(
            {
                "path": child.relative_to(directory).as_posix(),
                "size": int(resolved.stat().st_size),
                "digest": sha256_file(resolved),
            }
        )
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def manifest_roster(root: Path | str) -> list[dict[str, str]]:
    """Return the full installed manifest roster with byte-derived digests.

    ``train_pure_rl`` probes every manifest row before it selects the active
    public/gate/research subsets.  The r241 mount must therefore bind every
    installed package, not merely H10 or the currently weighted rows.
    """

    directory = regular_directory(root, label="baseline payload root")
    manifest = _load_json(directory / "manifest.json", label="baseline package manifest")
    agents = manifest.get("agents")
    if not isinstance(agents, list) or not agents:
        raise R241BaselinePayloadError("baseline package manifest has no agent list")
    rows: list[dict[str, str]] = []
    ids: set[str] = set()
    locations: set[tuple[str, str]] = set()
    for raw in agents:
        if not isinstance(raw, Mapping):
            raise R241BaselinePayloadError("baseline package manifest agent is malformed")
        opponent_id = str(raw.get("id") or "").strip()
        group = str(raw.get("group", "community") or "").strip()
        directory_name = str(raw.get("dir") or "").strip()
        if not opponent_id or opponent_id in ids:
            raise R241BaselinePayloadError("baseline package manifest repeats/misses an id")
        # ``group`` and ``dir`` are path components, not arbitrary labels.
        for label, value in (("baseline group", group), ("baseline dir", directory_name)):
            candidate = Path(value)
            if (
                not value
                or candidate.is_absolute()
                or len(candidate.parts) != 1
                or value in {".", ".."}
            ):
                raise R241BaselinePayloadError(f"unsafe {label}: {value!r}")
        location = (group, directory_name)
        if location in locations:
            raise R241BaselinePayloadError("baseline package manifest repeats a package location")
        package = directory / group / directory_name
        rows.append(
            {
                "id": opponent_id,
                "group": group,
                "dir": directory_name,
                "content_digest": content_digest(package),
            }
        )
        ids.add(opponent_id)
        locations.add(location)
    return sorted(rows, key=lambda row: row["id"])


def roster_digest(rows: Sequence[Mapping[str, object]]) -> str:
    canonical = [
        {
            "id": str(row["id"]),
            "group": str(row["group"]),
            "dir": str(row["dir"]),
            "content_digest": valid_sha256(
                row["content_digest"], label="baseline roster content digest"
            ),
        }
        for row in sorted(rows, key=lambda item: str(item["id"]))
    ]
    return sha256_bytes(canonical_json(canonical))


def minimal_manifest_payload(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return the only baseline manifest r241 is allowed to materialize.

    A host's generic baseline library can contain unrelated packages and an
    arbitrary historical manifest.  The mounted r241 payload instead carries
    only the exact checksum-derived union needed by the preserved public mix.
    ``baselines_runtime`` needs only these placement fields; names/source text
    are deliberately not copied from an unrelated mutable manifest.
    """

    roster = normalized_roster(rows)
    return {
        "agents": [
            {
                "id": row["id"],
                "group": row["group"],
                "dir": row["dir"],
            }
            for row in roster
        ]
    }


def minimal_manifest_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return canonical_json(minimal_manifest_payload(rows))


def roster_shape_digest(rows: Sequence[Mapping[str, object]]) -> str:
    """Digest just static manifest identity/placement fields."""

    canonical = [
        {"id": str(row["id"]), "group": str(row["group"]), "dir": str(row["dir"])}
        for row in sorted(rows, key=lambda item: str(item["id"]))
    ]
    return sha256_bytes(canonical_json(canonical))


def normalized_roster(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    """Validate and canonicalize exact package identity rows."""

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    locations: set[tuple[str, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise R241BaselinePayloadError("canonical baseline roster row is malformed")
        opponent_id = str(raw.get("id") or "").strip()
        group = str(raw.get("group") or "").strip()
        directory_name = str(raw.get("dir") or "").strip()
        content = valid_sha256(raw.get("content_digest"), label="canonical roster content")
        if not opponent_id or opponent_id in seen:
            raise R241BaselinePayloadError("canonical baseline roster repeats/misses an id")
        for label, value in (("canonical baseline group", group), ("canonical baseline dir", directory_name)):
            candidate = Path(value)
            if (
                not value
                or candidate.is_absolute()
                or len(candidate.parts) != 1
                or value in {".", ".."}
            ):
                raise R241BaselinePayloadError(f"unsafe {label}: {value!r}")
        location = (group, directory_name)
        if location in locations:
            raise R241BaselinePayloadError("canonical baseline roster repeats a package location")
        normalized.append(
            {"id": opponent_id, "group": group, "dir": directory_name, "content_digest": content}
        )
        seen.add(opponent_id)
        locations.add(location)
    if not normalized:
        raise R241BaselinePayloadError("canonical baseline roster is empty")
    return sorted(normalized, key=lambda row: row["id"])


def validate_canonical_roster_receipt(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    owner_contract_sha256: str,
) -> tuple[Path, dict[str, object]]:
    """Read one externally derived, full r195 public-baseline roster receipt."""

    receipt_path = regular_file(path, label="canonical baseline roster receipt")
    if expected_sha256 is not None and sha256_file(receipt_path) != valid_sha256(
        expected_sha256, label="canonical baseline roster receipt"
    ):
        raise R241BaselinePayloadError("canonical baseline roster receipt checksum drifted")
    payload = _load_json(receipt_path, label="canonical baseline roster receipt")
    if (
        payload.get("schema") != CANONICAL_BASELINE_ROSTER_SCHEMA
        or _exact_int(payload.get("revision"), label="canonical roster revision") != REVISION
        or payload.get("candidate_id") != CANDIDATE_ID
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
        or payload.get("owner_contract_sha256")
        != valid_sha256(owner_contract_sha256, label="canonical roster owner contract")
    ):
        raise R241BaselinePayloadError("canonical baseline roster receipt identity is invalid")
    roster = normalized_roster(list(payload.get("baseline_roster") or []))
    public_contracts = payload.get("public_contract_sha256s")
    if not isinstance(public_contracts, Mapping) or set(public_contracts) != {
        "active_gate_contract",
        "frozen_specialist_registry",
        "research_control_registry",
    }:
        raise R241BaselinePayloadError(
            "canonical baseline roster receipt lacks exact public-contract identities"
        )
    for name, value in public_contracts.items():
        valid_sha256(value, label=f"canonical {name}")
    if (
        payload.get("baseline_roster") != roster
        or payload.get("baseline_roster_sha256") != roster_digest(roster)
        or payload.get("baseline_manifest_sha256")
        != sha256_bytes(minimal_manifest_bytes(roster))
    ):
        raise R241BaselinePayloadError("canonical baseline roster receipt roster identity drifted")
    return receipt_path, payload


def validate_canonical_roster_contract_bindings(
    receipt: Mapping[str, object],
    *,
    active_gate_contract: Path | str,
    frozen_specialist_registry: Path | str,
    research_control_registry: Path | str,
) -> None:
    """Bind the derived full roster to the exact live public-mix contracts.

    The provenance worker, not a local baseline checkout, derives the roster.
    It records the three contract file identities it inspected.  At activation
    the launcher recomputes those identities on the host-specific files named
    by the peak-r195 preservation receipt, so a valid roster cannot be reused
    with a different active-gate, frozen-specialist, or research-control mix.
    """

    expected = receipt.get("public_contract_sha256s")
    if not isinstance(expected, Mapping):
        raise R241BaselinePayloadError(
            "canonical baseline roster receipt lacks public-contract identities"
        )
    actual_paths = {
        "active_gate_contract": regular_file(
            active_gate_contract, label="active public gate contract"
        ),
        "frozen_specialist_registry": regular_file(
            frozen_specialist_registry, label="frozen specialist registry"
        ),
        "research_control_registry": regular_file(
            research_control_registry, label="research-control registry"
        ),
    }
    if set(expected) != set(actual_paths):
        raise R241BaselinePayloadError(
            "canonical baseline roster public-contract identity set drifted"
        )
    for name, path in actual_paths.items():
        digest = valid_sha256(expected.get(name), label=f"canonical {name}")
        if sha256_file(path) != digest:
            raise R241BaselinePayloadError(
                f"canonical baseline roster was derived from another {name}"
            )


def validate_roster_against_public_contracts(
    roster: Sequence[Mapping[str, object]],
    *,
    active_gate: Mapping[str, object],
    frozen_specialists: Mapping[str, object],
    research_controls: Mapping[str, object],
) -> None:
    """Require the canonical full library to preserve every public contract row."""

    rows = {row["id"]: row for row in normalized_roster(roster)}
    gate = dict(active_gate.get("next_gate") or {})
    gate_rows = gate.get("roster")
    frozen_rows = frozen_specialists.get("specialists")
    research_rows = research_controls.get("controls")
    if not isinstance(gate_rows, list) or not isinstance(frozen_rows, list) or not isinstance(research_rows, list):
        raise R241BaselinePayloadError("public baseline contract roster is malformed")
    for raw in [*gate_rows, *research_rows]:
        if not isinstance(raw, Mapping):
            raise R241BaselinePayloadError("public baseline contract row is malformed")
        opponent_id = str(raw.get("opponent_id") or "")
        expected = valid_sha256(raw.get("content_digest"), label="public baseline content")
        actual = rows.get(opponent_id)
        if actual is None or actual["content_digest"] != expected:
            raise R241BaselinePayloadError(
                f"canonical baseline roster does not preserve public package {opponent_id!r}"
            )
    for raw in frozen_rows:
        if not isinstance(raw, Mapping):
            raise R241BaselinePayloadError("frozen baseline contract row is malformed")
        opponent_id = str(raw.get("opponent_id") or "")
        expected = valid_sha256(raw.get("content_digest"), label="frozen baseline content")
        actual = rows.get(opponent_id)
        if (
            actual is None
            or actual["content_digest"] != expected
            or actual["group"] != str(raw.get("baseline_group") or "")
            or actual["dir"] != str(raw.get("baseline_dir") or "")
        ):
            raise R241BaselinePayloadError(
                f"canonical baseline roster does not preserve frozen package {opponent_id!r}"
            )


def inventory(source_root: Path | str) -> list[dict[str, object]]:
    """Inventory every regular payload file using only relative identities."""

    root = regular_directory(source_root, label="baseline source root")
    rows: list[dict[str, object]] = []
    for directory_text, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory = Path(directory_text)
        for name in list(directory_names):
            child = directory / name
            relative = child.relative_to(root).as_posix()
            if child.is_symlink() or name in _FORBIDDEN_COMPONENTS:
                raise R241BaselinePayloadError(
                    f"baseline source root has forbidden directory: {relative}"
                )
            if name.startswith(".env") or name.endswith(".env"):
                raise R241BaselinePayloadError(
                    f"baseline source root has unbound environment directory: {relative}"
                )
        for name in file_names:
            child = directory / name
            relative = child.relative_to(root).as_posix()
            _relative(relative, label="baseline source member")
            if child.is_symlink() or not child.is_file():
                raise R241BaselinePayloadError(
                    f"baseline source root has non-regular file: {relative}"
                )
            rows.append(
                {
                    "path": relative,
                    "sha256": sha256_file(child),
                    "size_bytes": int(child.stat().st_size),
                }
            )
    if not rows:
        raise R241BaselinePayloadError("baseline source root is empty")
    return sorted(rows, key=lambda item: str(item["path"]))


def inventory_exact_roster(
    source_root: Path | str,
    roster: Sequence[Mapping[str, object]],
    *,
    generated_manifest: bytes,
) -> list[dict[str, object]]:
    """Inventory only canonical package directories plus a synthetic manifest.

    This is intentionally not ``inventory(source_root)``: publishing the
    complete generic library would copy unrelated, potentially huge packages
    and let an arbitrary local manifest change the r241 opponent surface.
    """

    root = regular_directory(source_root, label="baseline source root")
    rows: list[dict[str, object]] = [
        {
            "path": "manifest.json",
            "sha256": sha256_bytes(generated_manifest),
            "size_bytes": len(generated_manifest),
        }
    ]
    for roster_row in normalized_roster(roster):
        package = regular_directory(
            root / roster_row["group"] / roster_row["dir"],
            label=f"canonical baseline package {roster_row['id']}",
        )
        # ``regular_directory`` resolves its argument.  Check containment
        # immediately so a symlinked group/dir ancestor cannot make the later
        # inventory/copy pass inspect an arbitrary external package.
        try:
            package.relative_to(root)
        except ValueError as exc:
            raise R241BaselinePayloadError(
                f"canonical baseline package escapes source root: {roster_row['id']}"
            ) from exc
        if content_digest(package) != roster_row["content_digest"]:
            raise R241BaselinePayloadError(
                f"canonical baseline package digest drifted: {roster_row['id']}"
            )
        for directory_text, directory_names, file_names in os.walk(
            package, topdown=True, followlinks=False
        ):
            directory = Path(directory_text)
            for name in list(directory_names):
                child = directory / name
                if child.is_symlink() or name in _FORBIDDEN_COMPONENTS:
                    raise R241BaselinePayloadError(
                        f"canonical baseline package has forbidden directory: {child}"
                    )
                if name.startswith(".env") or name.endswith(".env"):
                    raise R241BaselinePayloadError(
                        f"canonical baseline package has unbound environment directory: {child}"
                    )
            for name in file_names:
                child = directory / name
                relative_file = child.relative_to(package).as_posix()
                _relative(relative_file, label="canonical baseline package member")
                if child.is_symlink() or not child.is_file():
                    raise R241BaselinePayloadError(
                        f"canonical baseline package has non-regular member: {child}"
                    )
                rows.append(
                    {
                        "path": (
                            Path(roster_row["group"])
                            / roster_row["dir"]
                            / relative_file
                        ).as_posix(),
                        "sha256": sha256_file(child),
                        "size_bytes": int(child.stat().st_size),
                    }
                )
    paths = [str(row["path"]) for row in rows]
    if len(paths) != len(set(paths)):
        raise R241BaselinePayloadError("canonical baseline inventory has duplicate paths")
    return sorted(rows, key=lambda item: str(item["path"]))


def manifest_payload(
    *,
    owner_contract_sha256: str,
    rows: Sequence[Mapping[str, object]],
    baseline_manifest_sha256: str,
    baseline_roster: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema": BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "owner_contract_sha256": valid_sha256(
            owner_contract_sha256, label="baseline payload owner contract"
        ),
        "baseline_tree_sha256": tree_digest(rows),
        "baseline_manifest_sha256": valid_sha256(
            baseline_manifest_sha256, label="baseline payload source manifest"
        ),
        "baseline_roster_sha256": roster_digest(baseline_roster),
        "baseline_roster": [dict(row) for row in baseline_roster],
        "authenticated": True,
        "status": "authenticated_immutable_baseline_payload_snapshot",
        "files": [dict(row) for row in rows],
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    source = regular_file(path, label=label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241BaselinePayloadError(f"{label} is unreadable JSON") from exc
    if not isinstance(payload, dict):
        raise R241BaselinePayloadError(f"{label} must contain an object")
    return payload


def _validate_h10(root: Path) -> None:
    manifest = _load_json(root / "manifest.json", label="baseline package manifest")
    agents = manifest.get("agents")
    if not isinstance(agents, list):
        raise R241BaselinePayloadError("baseline package manifest has no agent list")
    matches = [
        dict(item)
        for item in agents
        if isinstance(item, Mapping) and item.get("id") == R241_H10_OPPONENT_ID
    ]
    if len(matches) != 1:
        raise R241BaselinePayloadError("baseline package manifest lacks the exact H10 Marnie row")
    row = matches[0]
    if row.get("group") != "specialists" or row.get("dir") != R241_H10_DIR_NAME:
        raise R241BaselinePayloadError("baseline package manifest changed H10 Marnie placement")
    package = root / "specialists" / R241_H10_DIR_NAME
    package = regular_directory(package, label="H10 Marnie package")
    model = _member(package, "model.pt", label="H10 Marnie model")
    deck = _member(package, "deck.csv", label="H10 Marnie deck")
    tree = _member(package, "matchup_tree.json", label="H10 Marnie Matchup Adapter tree")
    if (
        model.stat().st_size != R241_H10_MODEL_SIZE_BYTES
        or sha256_file(model) != R241_H10_MODEL_SHA256
        or sha256_file(tree) != R241_H10_MATCHUP_TREE_SHA256
        or tree.stat().st_size != R241_H10_MATCHUP_TREE_SIZE_BYTES
        or len([line for line in deck.read_text(encoding="utf-8").splitlines() if line]) != 60
        or content_digest(package) != R241_H10_CONTENT_SHA256
    ):
        raise R241BaselinePayloadError("baseline payload H10 Marnie identity drifted")


def validate_snapshot(
    *,
    root: Path | str,
    manifest_path: Path | str,
    manifest_sha256: str,
    baseline_tree_sha256: str,
    owner_contract_sha256: str,
) -> dict[str, object]:
    """Verify one immutable mounted library and its complete inventory."""

    payload_root = regular_directory(root, label="r241 baseline payload root")
    manifest = regular_file(manifest_path, label="r241 baseline payload manifest")
    if manifest.parent != payload_root or manifest.name != BASELINE_PAYLOAD_MANIFEST_FILENAME:
        raise R241BaselinePayloadError("baseline payload manifest must be at its root")
    expected_manifest = valid_sha256(manifest_sha256, label="baseline payload manifest")
    expected_tree = valid_sha256(baseline_tree_sha256, label="baseline payload tree")
    expected_owner = valid_sha256(owner_contract_sha256, label="baseline payload owner contract")
    if sha256_file(manifest) != expected_manifest:
        raise R241BaselinePayloadError("baseline payload manifest checksum drifted")
    if payload_root.stat().st_mode & 0o222 or manifest.stat().st_mode & 0o222:
        raise R241BaselinePayloadError("baseline payload root/manifest must be read-only")
    if not payload_root.name.startswith(BASELINE_PAYLOAD_ROOT_PREFIX):
        raise R241BaselinePayloadError("baseline payload root is not content-addressed")
    payload = _load_json(manifest, label="r241 baseline payload manifest")
    if (
        payload.get("schema") != BASELINE_PAYLOAD_SNAPSHOT_SCHEMA
        or _exact_int(payload.get("revision"), label="baseline payload revision") != REVISION
        or payload.get("candidate_id") != CANDIDATE_ID
        or payload.get("owner_contract_sha256") != expected_owner
        or payload.get("baseline_tree_sha256") != expected_tree
        or payload.get("authenticated") is not True
        or payload.get("status") != "authenticated_immutable_baseline_payload_snapshot"
    ):
        raise R241BaselinePayloadError("baseline payload manifest identity is invalid")
    expected_suffix = expected_manifest.removeprefix(_SHA256_PREFIX)[:16]
    if payload_root.name != BASELINE_PAYLOAD_ROOT_PREFIX + expected_suffix:
        raise R241BaselinePayloadError("baseline payload root is not manifest-content-addressed")
    raw_rows = payload.get("files")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise R241BaselinePayloadError("baseline payload manifest has no file inventory")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise R241BaselinePayloadError("baseline payload inventory row is malformed")
        relative = str(raw.get("path") or "")
        if relative in seen:
            raise R241BaselinePayloadError("baseline payload inventory has duplicate paths")
        seen.add(relative)
        member = _member(payload_root, relative, label="baseline payload inventory member")
        expected_sha = valid_sha256(raw.get("sha256"), label="baseline payload member sha")
        expected_size = _exact_int(raw.get("size_bytes"), label="baseline payload member size")
        if expected_size < 0 or member.stat().st_mode & 0o222:
            raise R241BaselinePayloadError("baseline payload member is writable or invalid")
        if member.stat().st_size != expected_size or sha256_file(member) != expected_sha:
            raise R241BaselinePayloadError("baseline payload inventory member drifted")
        rows.append({"path": relative, "sha256": expected_sha, "size_bytes": expected_size})
    observed: set[str] = set()
    for directory_text, directory_names, file_names in os.walk(payload_root, topdown=True, followlinks=False):
        directory = Path(directory_text)
        for name in list(directory_names):
            child = directory / name
            if child.is_symlink() or child.stat().st_mode & 0o222:
                raise R241BaselinePayloadError("baseline payload contains unsafe directory")
            _relative(child.relative_to(payload_root).as_posix(), label="baseline payload directory")
        for name in file_names:
            child = directory / name
            relative = child.relative_to(payload_root).as_posix()
            if relative == BASELINE_PAYLOAD_MANIFEST_FILENAME:
                continue
            if child.is_symlink() or not child.is_file() or child.stat().st_mode & 0o222:
                raise R241BaselinePayloadError("baseline payload contains unsafe file")
            _relative(relative, label="baseline payload file")
            observed.add(relative)
    if observed != seen:
        raise R241BaselinePayloadError("baseline payload inventory is not the complete mounted library")
    if tree_digest(rows) != expected_tree:
        raise R241BaselinePayloadError("baseline payload tree checksum drifted")
    mounted_manifest = _member(payload_root, "manifest.json", label="baseline package manifest")
    expected_baseline_manifest = valid_sha256(
        payload.get("baseline_manifest_sha256"), label="baseline payload source manifest"
    )
    mounted_roster = manifest_roster(payload_root)
    if (
        sha256_file(mounted_manifest) != expected_baseline_manifest
        or payload.get("baseline_roster_sha256") != roster_digest(mounted_roster)
        or payload.get("baseline_roster") != mounted_roster
    ):
        raise R241BaselinePayloadError("baseline payload manifest/roster identity drifted")
    _validate_h10(payload_root)
    return {
        "root": str(payload_root),
        "manifest": str(manifest),
        "manifest_sha256": expected_manifest,
        "baseline_tree_sha256": expected_tree,
        "file_inventory_sha256": sha256_bytes(canonical_json(rows)),
        "baseline_manifest_sha256": expected_baseline_manifest,
        "baseline_roster_sha256": roster_digest(mounted_roster),
        "baseline_roster": mounted_roster,
    }


def sealed_copy(
    *,
    source_root: Path | str,
    destination_root: Path,
    rows: Iterable[Mapping[str, object]],
    manifest_bytes: bytes,
    generated_files: Mapping[str, bytes] | None = None,
) -> None:
    """Copy a verified inventory to an exclusively-created payload directory."""

    import shutil

    source = regular_directory(source_root, label="baseline source root")
    generated = dict(generated_files or {})
    for row in rows:
        relative = str(row["path"])
        target = destination_root / _relative(relative, label="baseline destination member")
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in generated:
            data = generated[relative]
            if (
                sha256_bytes(data) != str(row["sha256"])
                or len(data) != _exact_int(row["size_bytes"], label="generated baseline member size")
            ):
                raise R241BaselinePayloadError(
                    f"generated baseline member identity drifted: {relative}"
                )
            target.write_bytes(data)
        else:
            original = _member(source, relative, label="baseline source member")
            shutil.copy2(original, target, follow_symlinks=False)
    (destination_root / BASELINE_PAYLOAD_MANIFEST_FILENAME).write_bytes(manifest_bytes)
    for path in sorted(destination_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise R241BaselinePayloadError("baseline publisher encountered a symlink")
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o444)
        else:
            raise R241BaselinePayloadError("baseline publisher encountered a non-regular member")
    destination_root.chmod(0o555)
