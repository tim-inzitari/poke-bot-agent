#!/usr/bin/env python3
"""Create-only staging for r241's isolated Elmo :8767 checkpoint mount.

This is deliberately a data-only publisher.  It copies the immutable r195
parent checkpoint, its exact E60 matchup tree, and a deterministic runtime
marker into the one flat bind-mount directory expected by the generic remote
job ABI.  It never starts a container, listener, worker, service, game, or
trainer.

The receipt is intentionally deterministic: retrying a completed stage is
safe only when every byte already present is identical.  A conflicting file,
symlink, or unexpected entry fails closed; this tool never overwrites or
deletes a published transport object.  Temporary upload files are removed only
after an atomic create-only hard-link publication succeeds or an already exact
destination is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


R241_REVISION = 241
R241_CANDIDATE_ID = "alakazam-new-list-direct-policy-r241"
R241_OWNER_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241/v1"
R241_OWNER_CLARIFICATION_REVISION = 251
R241_CHECKPOINT_TRANSPORT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_checkpoint_transport/v1"
)
R241_CHECKPOINT_TRANSPORT_STAGING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_checkpoint_transport_staging/v1"
)
R241_CHECKPOINT_TRANSPORT_PRODUCER_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_checkpoint_transport_producer/v1"
)
R241_ENDPOINT_ID = "elmo-r241-official-r236-direct-policy-8767"
R241_ENDPOINT = "elmo:8767"
R241_ENDPOINT_PORT = 8767
R241_CONTAINER_ROOT = PurePosixPath("/workspace/checkpoint")
R241_CHECKPOINT_ENVIRONMENT_KEY = "POKEBOT_REMOTE_CHECKPOINT_ROOT"
R241_FILENAME_SCHEME = "poke_bot.remote_jobs.digest_addressed_basename/v1"
R241_MATCHUP_RUNTIME_MARKER = "matchup-runtime-activation.json"
R241_MATCHUP_RUNTIME_MARKER_SCHEMA = "poke_bot.remote_matchup_runtime_activation/v1"

R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_CHECKPOINT_BYTES = 127_914_385
R195_E60_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
R195_IMMUTABLE_SLOT_PREFIX = 20

ELMO_OUTPUTS_ROOT = PurePosixPath("/srv/poke-bot-agent/outputs")
ELMO_R241_RUNTIME_ROOT = (
    ELMO_OUTPUTS_ROOT
    / "pure_rl/alakazam_new_list_direct_policy_r241/runtime/elmo-8767"
)
DEFAULT_ELMO_TRANSPORT_ROOT = ELMO_R241_RUNTIME_ROOT / "checkpoint-transport"
DEFAULT_ELMO_STAGING_RECEIPT = ELMO_R241_RUNTIME_ROOT / "checkpoint-transport-staging.json"

_SAFE_SSH_TARGET_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.@:-[]"
)
_SAFE_REMOTE_PATH_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-"
)


class R241CheckpointTransportStageError(RuntimeError):
    """The immutable transport cannot be safely created."""


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R241CheckpointTransportStageError(
            "transport receipt is not canonical JSON"
        ) from exc


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    rendered = str(value or "").strip().lower()
    if not rendered.startswith("sha256:") or len(rendered) != 71:
        raise R241CheckpointTransportStageError(f"{label} must be a SHA-256 digest")
    try:
        int(rendered.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise R241CheckpointTransportStageError(
            f"{label} must be hexadecimal"
        ) from exc
    return rendered


def _regular_file(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    try:
        details = raw.lstat()
    except OSError as exc:
        raise R241CheckpointTransportStageError(f"{label} is unavailable: {raw}") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise R241CheckpointTransportStageError(
            f"{label} must be a regular non-symlink file: {raw}"
        )
    return raw.resolve()


def _json_object(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    file = _regular_file(path, label=label)
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241CheckpointTransportStageError(
            f"{label} is unreadable JSON: {file}"
        ) from exc
    if not isinstance(value, dict):
        raise R241CheckpointTransportStageError(f"{label} must contain a JSON object")
    return file, value


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R241CheckpointTransportStageError(f"{label} must be an object")
    return dict(value)


def _digest_addressed_basename(path: Path, digest: str) -> str:
    """Mirror the sealed generic remote-jobs filename ABI exactly."""

    short = _require_sha256(digest, label="checkpoint digest").removeprefix("sha256:")[:16]
    if not short or not path.stem or not path.suffix:
        raise R241CheckpointTransportStageError(
            "checkpoint must have a nonempty stem and suffix for the transport ABI"
        )
    return f"{path.stem}.{short}{path.suffix}"


def _safe_ssh_target(value: str) -> str:
    target = str(value or "").strip()
    if not target or target.startswith("-") or any(
        character not in _SAFE_SSH_TARGET_CHARS for character in target
    ):
        raise R241CheckpointTransportStageError("SSH target is not a safe literal")
    return target


def _safe_remote_path(value: PurePosixPath | str, *, label: str) -> str:
    rendered = str(value or "").strip()
    candidate = PurePosixPath(rendered)
    if (
        not rendered
        or not candidate.is_absolute()
        or candidate == PurePosixPath("/")
        or ".." in candidate.parts
        or any(character not in _SAFE_REMOTE_PATH_CHARS for character in rendered)
    ):
        raise R241CheckpointTransportStageError(f"{label} is not a safe absolute path")
    return rendered


def _require_expected_elmo_paths(
    *, host_root: PurePosixPath | str, receipt_path: PurePosixPath | str
) -> tuple[PurePosixPath, PurePosixPath]:
    root = PurePosixPath(_safe_remote_path(host_root, label="Elmo transport root"))
    receipt = PurePosixPath(
        _safe_remote_path(receipt_path, label="Elmo staging receipt")
    )
    if root != DEFAULT_ELMO_TRANSPORT_ROOT or receipt != DEFAULT_ELMO_STAGING_RECEIPT:
        raise R241CheckpointTransportStageError(
            "r241 checkpoint transport must use its dedicated Elmo :8767 output paths"
        )
    if receipt.parent != root.parent:
        raise R241CheckpointTransportStageError(
            "r241 checkpoint transport receipt must remain beside its dedicated mount root"
        )
    return root, receipt


def _validate_owner_contract(
    *, owner_contract: Path | str, expected_sha256: str | None
) -> tuple[Path, str]:
    path, owner = _json_object(owner_contract, label="r241 owner contract")
    actual = _sha256_file(path)
    if expected_sha256 is not None and actual != _require_sha256(
        expected_sha256, label="r241 owner contract sha256"
    ):
        raise R241CheckpointTransportStageError(
            "r241 owner contract checksum does not match the requested immutable identity"
        )
    parent = _require_mapping(owner.get("parent"), label="r241 owner parent")
    if (
        owner.get("schema") != R241_OWNER_SCHEMA
        or owner.get("latest_owner_clarification_revision")
        != R241_OWNER_CLARIFICATION_REVISION
        or parent.get("checkpoint_sha256") != R195_CHECKPOINT_SHA256
        or parent.get("checkpoint_bytes") != R195_CHECKPOINT_BYTES
        or parent.get("immutable") is not True
    ):
        raise R241CheckpointTransportStageError(
            "r241 owner contract does not bind the current immutable r195 parent"
        )
    return path, actual


def _validate_r195_checkpoint(path: Path | str) -> tuple[Path, str]:
    checkpoint = _regular_file(path, label="immutable r195 checkpoint")
    if checkpoint.stat().st_size != R195_CHECKPOINT_BYTES:
        raise R241CheckpointTransportStageError("r195 checkpoint byte size drifted")
    digest = _sha256_file(checkpoint)
    if digest != R195_CHECKPOINT_SHA256:
        raise R241CheckpointTransportStageError("r195 checkpoint digest drifted")
    return checkpoint, digest


def _runtime_marker_for_e60_tree(tree: Path | str) -> tuple[Path, str, bytes]:
    """Create the exact generic runtime marker from the immutable E60 tree."""

    tree_path = _regular_file(tree, label="r195 E60 matchup tree")
    tree_digest = _sha256_file(tree_path)
    if tree_digest != R195_E60_TREE_SHA256:
        raise R241CheckpointTransportStageError("r195 E60 matchup tree digest drifted")
    try:
        tree_payload = json.loads(tree_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241CheckpointTransportStageError("r195 E60 matchup tree is invalid JSON") from exc
    runtime = _require_mapping(
        _require_mapping(tree_payload, label="r195 E60 matchup tree").get(
            "runtime_contract"
        ),
        label="r195 E60 runtime contract",
    )
    accepted = sorted(str(value) for value in runtime.get("accepted_archetype_ids") or ())
    binding_keys = (
        "adapter_format",
        "route_target_ids",
        "route_physical_slots",
        "physical_slot_capacity",
        "slot_registry_digest",
    )
    if (
        not accepted
        or len(accepted) != len(set(accepted))
        or runtime.get("one_route_per_decision") is not True
        or runtime.get("unknown_route_exact_bypass") is not True
        or int(runtime.get("consecutive_required") or 0) < 1
        or runtime.get("adapter_format") != "poke-bot-matchup-adapter-bank-v6"
        or any(key not in runtime for key in binding_keys)
        or runtime.get("physical_slot_capacity") != 64
        or list(runtime.get("route_physical_slots") or ())
        != list(range(R195_IMMUTABLE_SLOT_PREFIX))
        or len(list(runtime.get("route_target_ids") or ()))
        != R195_IMMUTABLE_SLOT_PREFIX
    ):
        raise R241CheckpointTransportStageError(
            "r195 E60 tree does not preserve the immutable 20-slot runtime contract"
        )
    marker_payload: dict[str, Any] = {
        "schema": R241_MATCHUP_RUNTIME_MARKER_SCHEMA,
        "runtime_enabled": True,
        "tree_file": tree_path.name,
        "tree_digest": tree_digest,
        "accepted_archetype_ids": accepted,
        "continuous_reevaluation": True,
        "one_route_per_decision": True,
        "zero_materialized_adapters_allowed": bool(
            runtime.get("zero_materialized_adapters_allowed")
        ),
        **{key: runtime[key] for key in binding_keys},
    }
    # The generic remote worker serializes this marker with the human-readable
    # format below.  Reusing that exact byte convention avoids a second marker
    # dialect while preserving deterministic content-addressed evidence.
    marker = (json.dumps(marker_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return tree_path, tree_digest, marker


def _transport_receipt(
    *,
    owner_contract_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    tree: Path,
    tree_sha256: str,
    marker_sha256: str,
    host_root: PurePosixPath,
    receipt_path: PurePosixPath,
) -> dict[str, Any]:
    checkpoint_name = _digest_addressed_basename(checkpoint, checkpoint_sha256)
    checkpoint_path = R241_CONTAINER_ROOT / checkpoint_name
    tree_path = R241_CONTAINER_ROOT / tree.name
    marker_path = R241_CONTAINER_ROOT / R241_MATCHUP_RUNTIME_MARKER
    return {
        "schema": R241_CHECKPOINT_TRANSPORT_STAGING_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": R241_CANDIDATE_ID,
        "status": "passed",
        "passed": True,
        "owner_contract_sha256": _require_sha256(
            owner_contract_sha256, label="owner contract sha256"
        ),
        "checkpoint_transport": {
            "schema": R241_CHECKPOINT_TRANSPORT_SCHEMA,
            "endpoint_id": R241_ENDPOINT_ID,
            "host_role": "elmo",
            "verification_endpoint": R241_ENDPOINT,
            "verification_port": R241_ENDPOINT_PORT,
            "host_root": str(host_root),
            "container_root": str(R241_CONTAINER_ROOT),
            "environment_key": R241_CHECKPOINT_ENVIRONMENT_KEY,
            "remote_path_prefix": f"{R241_CONTAINER_ROOT}/",
            "content_addressing": {
                "algorithm": "sha256",
                "filename_scheme": R241_FILENAME_SCHEME,
            },
            "read_only_container_mount": True,
            "same_absolute_source_and_baseline_paths_preserved": True,
        },
        "initial_checkpoint": {
            "container_path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "runtime_companions": {
            "learner_matchup_tree": {
                "container_path": str(tree_path),
                "sha256": tree_sha256,
            },
            "matchup_runtime_activation": {
                "container_path": str(marker_path),
                "sha256": marker_sha256,
                "schema": R241_MATCHUP_RUNTIME_MARKER_SCHEMA,
            },
        },
        "producer": {
            "schema": R241_CHECKPOINT_TRANSPORT_PRODUCER_SCHEMA,
            "source_sha256": _sha256_file(Path(__file__)),
            "execution_mode": "create_only_checksum_verified_ssh",
            "worker_started": False,
            "service_started": False,
            "remote_listener_started": False,
            "receipt_path": str(receipt_path),
        },
    }


def _ssh_base(target: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        target,
    ]


def _run(command: Sequence[str], *, label: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or str(completed.returncode)
        raise R241CheckpointTransportStageError(f"{label} failed: {detail}")
    return completed


def _remote_shell(target: str, script: str, *, label: str) -> str:
    completed = _run([*_ssh_base(target), script], label=label)
    return completed.stdout.strip()


def _remote_file_state(target: str, path: PurePosixPath) -> str:
    safe_path = _safe_remote_path(path, label="remote file")
    script = " ".join(
        (
            "set -eu;",
            f"p={shlex.quote(safe_path)};",
            'if test -L "$p"; then printf "SYMLINK";',
            'elif test -f "$p"; then sha256sum -- "$p" | awk \'{print $1}\';',
            'elif test -e "$p"; then printf "NONREGULAR";',
            'else printf "ABSENT"; fi',
        )
    )
    state = _remote_shell(target, script, label=f"inspect remote file {safe_path}")
    if state in {"ABSENT", "SYMLINK", "NONREGULAR"}:
        return state
    if len(state) == 64 and all(character in "0123456789abcdef" for character in state):
        return "sha256:" + state
    raise R241CheckpointTransportStageError(
        f"remote file inspection returned an invalid digest for {safe_path}: {state!r}"
    )


def _ensure_remote_directory(target: str, path: PurePosixPath) -> None:
    safe_path = _safe_remote_path(path, label="remote directory")
    script = " ".join(
        (
            "set -eu;",
            f"p={shlex.quote(safe_path)};",
            'test ! -L "$p";',
            'if test -e "$p"; then test -d "$p"; else mkdir -p -- "$p"; fi;',
            'test -d "$p";',
        )
    )
    _remote_shell(target, script, label=f"create-only remote directory {safe_path}")


def _stage_remote_file_create_only(
    *, target: str, source: Path, destination: PurePosixPath, digest: str
) -> None:
    expected = _require_sha256(digest, label=f"digest for {source.name}")
    existing = _remote_file_state(target, destination)
    if existing == expected:
        return
    if existing != "ABSENT":
        raise R241CheckpointTransportStageError(
            f"remote destination already exists with a conflicting identity: {destination}"
        )
    _ensure_remote_directory(target, destination.parent)
    token = secrets.token_hex(16)
    temporary = destination.parent / f".{destination.name}.r241-{token}.partial"
    if _remote_file_state(target, temporary) != "ABSENT":
        raise R241CheckpointTransportStageError(
            f"refusing to reuse an unexpected remote staging path: {temporary}"
        )
    _run(
        [
            "scp",
            "-q",
            "-p",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=accept-new",
            str(source),
            f"{target}:{temporary}",
        ],
        label=f"copy {source.name} to isolated Elmo transport",
    )
    safe_destination = _safe_remote_path(destination, label="remote destination")
    safe_temporary = _safe_remote_path(temporary, label="remote temporary")
    expected_hex = expected.removeprefix("sha256:")
    # ``ln`` is the publication primitive: unlike a force move it can never
    # replace a different already-published path.  The only removes below are
    # of this freshly-created private temporary after successful publication
    # or after proving another process published identical bytes first.
    script = " ".join(
        (
            "set -eu;",
            f"tmp={shlex.quote(safe_temporary)};",
            f"dst={shlex.quote(safe_destination)};",
            f"expected={shlex.quote(expected_hex)};",
            'test ! -L "$tmp";',
            'test -f "$tmp";',
            'actual=$(sha256sum -- "$tmp" | awk \'{print $1}\');',
            'test "$actual" = "$expected";',
            'chmod 0444 -- "$tmp";',
            'if test -e "$dst" || test -L "$dst"; then',
            '  test ! -L "$dst"; test -f "$dst";',
            '  current=$(sha256sum -- "$dst" | awk \'{print $1}\');',
            '  test "$current" = "$expected"; rm -- "$tmp"; exit 0;',
            "fi;",
            'if ln -- "$tmp" "$dst"; then rm -- "$tmp"; else',
            '  test -e "$dst"; test ! -L "$dst"; test -f "$dst";',
            '  current=$(sha256sum -- "$dst" | awk \'{print $1}\');',
            '  test "$current" = "$expected"; rm -- "$tmp";',
            "fi;",
            'test -f "$dst"; test ! -L "$dst"; test ! -w "$dst";',
            'actual=$(sha256sum -- "$dst" | awk \'{print $1}\');',
            'test "$actual" = "$expected";',
        )
    )
    _remote_shell(target, script, label=f"create-only publish {safe_destination}")
    if _remote_file_state(target, destination) != expected:
        raise R241CheckpointTransportStageError(
            f"remote published digest drifted for {destination}"
        )


def _verify_remote_transport_directory(
    *, target: str, root: PurePosixPath, expected_names: set[str]
) -> None:
    safe_root = _safe_remote_path(root, label="Elmo transport root")
    script = " ".join(
        (
            "set -eu;",
            f"root={shlex.quote(safe_root)};",
            'test -d "$root"; test ! -L "$root";',
            'find "$root" -mindepth 1 -maxdepth 1 -printf "%f\\n" | LC_ALL=C sort;',
        )
    )
    observed = {
        line.strip()
        for line in _remote_shell(target, script, label="inspect Elmo transport directory").splitlines()
        if line.strip()
    }
    if observed != expected_names:
        raise R241CheckpointTransportStageError(
            "Elmo checkpoint transport root contains unexpected or missing entries: "
            f"expected={sorted(expected_names)} observed={sorted(observed)}"
        )


def _write_local_create_only(path: Path | str, payload: bytes) -> Path:
    target = Path(path).expanduser()
    if target.is_symlink():
        raise R241CheckpointTransportStageError(
            f"local receipt output may not be a symlink: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.is_symlink() or target.read_bytes() != payload:
            raise R241CheckpointTransportStageError(
                f"local receipt output already exists with different bytes: {target}"
            )
        return target.resolve()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not target.is_file() or target.is_symlink() or target.read_bytes() != payload:
                raise R241CheckpointTransportStageError(
                    f"local receipt output already exists with different bytes: {target}"
                )
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target.resolve()


def stage_r241_elmo_checkpoint_transport(
    *,
    checkpoint: Path | str,
    matchup_tree: Path | str,
    owner_contract: Path | str,
    owner_contract_sha256: str | None,
    ssh_target: str,
    elmo_transport_root: PurePosixPath | str = DEFAULT_ELMO_TRANSPORT_ROOT,
    elmo_staging_receipt: PurePosixPath | str = DEFAULT_ELMO_STAGING_RECEIPT,
    controller_receipt: Path | str,
    execute: bool,
) -> dict[str, Any]:
    """Validate and optionally publish the immutable r195 transport bundle."""

    root, remote_receipt = _require_expected_elmo_paths(
        host_root=elmo_transport_root, receipt_path=elmo_staging_receipt
    )
    target = _safe_ssh_target(ssh_target)
    _owner_path, owner_sha256 = _validate_owner_contract(
        owner_contract=owner_contract, expected_sha256=owner_contract_sha256
    )
    checkpoint_path, checkpoint_sha256 = _validate_r195_checkpoint(checkpoint)
    tree_path, tree_sha256, marker = _runtime_marker_for_e60_tree(matchup_tree)
    marker_sha256 = "sha256:" + hashlib.sha256(marker).hexdigest()
    checkpoint_name = _digest_addressed_basename(checkpoint_path, checkpoint_sha256)
    receipt = _transport_receipt(
        owner_contract_sha256=owner_sha256,
        checkpoint=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        tree=tree_path,
        tree_sha256=tree_sha256,
        marker_sha256=marker_sha256,
        host_root=root,
        receipt_path=remote_receipt,
    )
    receipt_bytes = _canonical_json(receipt)
    result: dict[str, Any] = {
        "schema": R241_CHECKPOINT_TRANSPORT_PRODUCER_SCHEMA,
        "revision": R241_REVISION,
        "candidate_id": R241_CANDIDATE_ID,
        "execute": bool(execute),
        "worker_started": False,
        "service_started": False,
        "remote_listener_started": False,
        "ssh_target": target,
        "elmo_transport_root": str(root),
        "elmo_staging_receipt": str(remote_receipt),
        "initial_checkpoint": receipt["initial_checkpoint"],
        "runtime_companions": receipt["runtime_companions"],
        "owner_contract_sha256": owner_sha256,
        "staging_receipt_sha256": "sha256:" + hashlib.sha256(receipt_bytes).hexdigest(),
    }
    if not execute:
        result["status"] = "validated_not_staged"
        return result

    with tempfile.TemporaryDirectory(prefix="r241-elmo-checkpoint-transport-") as temporary_dir:
        marker_path = Path(temporary_dir) / R241_MATCHUP_RUNTIME_MARKER
        marker_path.write_bytes(marker)
        marker_path.chmod(0o444)
        receipt_path = Path(temporary_dir) / remote_receipt.name
        receipt_path.write_bytes(receipt_bytes)
        receipt_path.chmod(0o444)
        _stage_remote_file_create_only(
            target=target,
            source=checkpoint_path,
            destination=root / checkpoint_name,
            digest=checkpoint_sha256,
        )
        _stage_remote_file_create_only(
            target=target,
            source=tree_path,
            destination=root / tree_path.name,
            digest=tree_sha256,
        )
        _stage_remote_file_create_only(
            target=target,
            source=marker_path,
            destination=root / R241_MATCHUP_RUNTIME_MARKER,
            digest=marker_sha256,
        )
        _verify_remote_transport_directory(
            target=target,
            root=root,
            expected_names={checkpoint_name, tree_path.name, R241_MATCHUP_RUNTIME_MARKER},
        )
        _stage_remote_file_create_only(
            target=target,
            source=receipt_path,
            destination=remote_receipt,
            digest=result["staging_receipt_sha256"],
        )
    if _remote_file_state(target, remote_receipt) != result["staging_receipt_sha256"]:
        raise R241CheckpointTransportStageError("Elmo staging receipt digest drifted")
    local_receipt = _write_local_create_only(controller_receipt, receipt_bytes)
    result.update(
        {
            "status": "passed",
            "controller_receipt": str(local_receipt),
            "controller_receipt_sha256": _sha256_file(local_receipt),
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--owner-contract", type=Path, required=True)
    parser.add_argument("--owner-contract-sha256")
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument(
        "--elmo-transport-root",
        default=str(DEFAULT_ELMO_TRANSPORT_ROOT),
    )
    parser.add_argument(
        "--elmo-staging-receipt",
        default=str(DEFAULT_ELMO_STAGING_RECEIPT),
    )
    parser.add_argument(
        "--controller-receipt",
        type=Path,
        required=True,
        help="create-only local/controller copy inspected by the overlay publisher",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the create-only SSH publication (default validates only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = stage_r241_elmo_checkpoint_transport(
            checkpoint=args.checkpoint,
            matchup_tree=args.matchup_tree,
            owner_contract=args.owner_contract,
            owner_contract_sha256=args.owner_contract_sha256,
            ssh_target=args.ssh_target,
            elmo_transport_root=args.elmo_transport_root,
            elmo_staging_receipt=args.elmo_staging_receipt,
            controller_receipt=args.controller_receipt,
            execute=args.execute,
        )
    except R241CheckpointTransportStageError as exc:
        print(f"r241 Elmo checkpoint transport staging failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
