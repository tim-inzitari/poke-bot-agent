"""Read-only, checksum-bound submission provenance for Replay Model Inspector.

The Kaggle replay archive identifies a *submission*, but it deliberately does
not duplicate the submitted model package beside every replay.  This module is
the small trust boundary between those two things.  It accepts only an
explicit, checksum-bound manifest and never guesses a checkpoint from a run
name, an iteration number, or a similarly named file.

The public manifest shape is::

    {
      "schema": "poke_bot.replay_model_inspector_provenance/v1",
      "version": 1,
      "records": [{
        "submission_id": 55300685,
        "status": "verified",
        "checkpoint": {"path": "/read-only/checkpoints/model.pt",
                       "sha256": "sha256:<64 lowercase hex characters>"},
        "bundle": {"path": "/read-only/submissions/submission.tar.gz",
                   "sha256": "sha256:<64 lowercase hex characters>"},
        "matchup_tree": {"path": "/read-only/routes/matchup_tree.json",
                         "sha256": "sha256:<64 lowercase hex characters>"}
        "runtime_package": {"path": "/read-only/submissions/submission.tar.gz",
                            "sha256": "sha256:<64 lowercase hex characters>"},
        "runtime_parity_receipt": {
          "path": "/read-only/provenance/runtime-parity.json",
          "sha256": "sha256:<64 lowercase hex characters>"
        },
        "label": "exact submission text",
        "identity": {
          "label": {
            "value": "exact submission text",
            "availability": "available",
            "evidence": [{
              "path": "/receipt/kaggle-submission-queue.json",
              "sha256": "sha256:<64 lowercase hex characters>",
              "pointer": "/queue/30/label",
              "role": "submission_queue_receipt",
              "key": "label"
            }]
          }
        }
      }]
    }

Paths may be relative to the manifest's directory, but every final resolved
path must stay within a caller-configured source root.  An invalid record is
kept in the returned manifest with explicit availability reasons so a UI can
show the submission without accidentally enabling inference for it.

``label`` is presentation-only and is accepted only when the matching
``identity.label`` value has one or more canonical SHA-256 evidence records.
It does not affect model availability; this lets old, fully verified entries
remain inspectable while accurately reporting ``submission text unavailable``.

``runtime_package`` and ``runtime_parity_receipt`` are optional for legacy
records: parameters remain available from their exact checkpoint.  Dynamic
heads/logits additionally require both fields.  The receipt has schema
``poke_bot.replay_model_inspector_runtime_parity_receipt/v1`` and binds the
submission id, checkpoint digest, bundle digest, runtime package digest, and
canonical extracted source-tree digest, plus an independent-verification
attestation.  The server hashes its configured extracted package root and
checks that its imported ``poke_bot`` came from that root before executing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeAlias

PROVENANCE_SCHEMA = "poke_bot.replay_model_inspector_provenance/v1"
RUNTIME_PARITY_RECEIPT_SCHEMA = (
    "poke_bot.replay_model_inspector_runtime_parity_receipt/v1"
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PathLike: TypeAlias = str | os.PathLike[str]


class ProvenanceError(ValueError):
    """The provenance manifest itself is not a usable v1 manifest."""


class PathContainmentError(ProvenanceError):
    """A candidate artifact escapes the configured read-only source roots."""


def sha256_file(path: PathLike) -> str:
    """Return a canonical SHA-256 for one regular file without mutating it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def sha256_source_tree(path: PathLike) -> str:
    """Hash an extracted runtime tree with a stable, auditable encoding.

    The hash covers relative paths and file bytes, rejects links/special files,
    and ignores Python's derived bytecode cache.  This lets an exact submitted
    package be extracted read-only once and checked at every dynamic-trace
    request without treating a mutable current checkout as equivalent.
    """

    try:
        root = _as_path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError("runtime source tree cannot be resolved") from exc
    if not root.is_dir():
        raise ProvenanceError("runtime source tree is not a directory")
    digest = hashlib.sha256()
    files: list[Path] = []
    try:
        for candidate in root.rglob("*"):
            relative = candidate.relative_to(root)
            if "__pycache__" in relative.parts or candidate.suffix in {".pyc", ".pyo"}:
                continue
            if candidate.is_symlink():
                raise ProvenanceError("runtime source tree contains a symbolic link")
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ProvenanceError("runtime source tree contains a non-regular file")
            files.append(candidate)
    except OSError as exc:
        raise ProvenanceError("runtime source tree cannot be read") from exc
    for candidate in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(b"file\0")
        digest.update(relative)
        digest.update(b"\0")
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _as_path(value: PathLike) -> Path:
    return Path(value).expanduser()


def _normalise_roots(roots: Iterable[PathLike]) -> tuple[Path, ...]:
    """Resolve configured roots once, including an intentional root symlink."""

    unique: list[Path] = []
    for raw in roots:
        resolved = _as_path(raw).resolve(strict=False)
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_contained_path(
    value: PathLike,
    *,
    roots: Iterable[PathLike],
    relative_to: PathLike | None = None,
    require_exists: bool = False,
) -> Path:
    """Resolve ``value`` and reject symlink/``..`` escapes from ``roots``.

    ``Path.resolve`` is intentionally performed before containment checks.  A
    lexical prefix test would incorrectly trust ``root/link-to-outside/file``.
    ``require_exists`` performs a second strict resolution for artifact reads,
    catching a final-component symlink as well.
    """

    try:
        configured_roots = _normalise_roots(roots)
    except (OSError, RuntimeError) as exc:
        raise PathContainmentError("cannot resolve configured source roots") from exc
    if not configured_roots:
        raise PathContainmentError("no configured source roots")

    try:
        raw = _as_path(value)
        if not raw.is_absolute():
            base = (
                _as_path(relative_to).resolve(strict=False)
                if relative_to
                else Path.cwd()
            )
            raw = base / raw
        candidate = raw.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PathContainmentError("cannot resolve candidate source path") from exc
    if not any(_is_within(candidate, root) for root in configured_roots):
        raise PathContainmentError(
            f"path escapes configured source roots: {raw} -> {candidate}"
        )
    if require_exists:
        # ``strict=True`` resolves the final path element too; this matters for
        # a symlink whose target was changed after the lexical configuration was
        # read.
        try:
            candidate = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise PathContainmentError("cannot resolve existing source path") from exc
        if not any(_is_within(candidate, root) for root in configured_roots):
            raise PathContainmentError(
                f"resolved path escapes configured source roots: {candidate}"
            )
    return candidate


def _unique_reasons(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in result:
            result.append(clean)
    return tuple(result)


def _positive_int(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def _canonical_sha256(value: Any) -> str | None:
    # The manifest grammar deliberately requires the canonical lower-case
    # spelling.  Silently normalising a digest would make a malformed manifest
    # appear verified in its public diagnostic output.
    text = str(value or "").strip()
    return text if _SHA256_RE.fullmatch(text) else None


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """One checksum-bearing source claim recorded in a provenance manifest.

    This is descriptive evidence, not a path the inspector will open at
    request time.  In particular, a queue receipt can live outside the model
    artifact roots.  Requiring its immutable digest and JSON pointer prevents
    a bare, hand-written label from looking like a submission-bound fact.
    """

    path: str
    sha256: str
    pointer: str
    role: str
    key: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "pointer": self.pointer,
            "role": self.role,
            "key": self.key,
        }


@dataclass(frozen=True, slots=True)
class SubmissionLabel:
    """Exact display text with its submission-bound evidence, if verified.

    A label is intentionally independent of model artifact availability: an
    unresolved checkpoint record may still have an exact queue/upload label.
    Conversely, an unproven label is never promoted from a team name, file
    name, active selector, or numeric submission id.
    """

    text: str | None
    evidence: tuple[SourceEvidence, ...] = ()
    availability_reasons: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return (
            self.text is not None
            and bool(self.evidence)
            and not self.availability_reasons
        )

    @classmethod
    def unavailable(cls, *reasons: str) -> SubmissionLabel:
        return cls(
            text=None,
            evidence=(),
            availability_reasons=_unique_reasons(reasons),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text if self.available else None,
            "available": self.available,
            "availability_reasons": list(self.availability_reasons),
            "evidence": [source.to_dict() for source in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """One declared artifact and the result of its read-only verification."""

    role: str
    declared_path: str | None
    expected_sha256: str | None
    resolved_path: Path | None
    actual_sha256: str | None
    size_bytes: int | None
    availability_reasons: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return (
            not self.availability_reasons
            and self.resolved_path is not None
            and self.expected_sha256 is not None
            and self.actual_sha256 == self.expected_sha256
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "declared_path": self.declared_path,
            "path": str(self.resolved_path) if self.resolved_path else None,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "size_bytes": self.size_bytes,
            "available": self.available,
            "availability_reasons": list(self.availability_reasons),
        }


@dataclass(frozen=True, slots=True)
class RuntimeParityReceipt:
    """An independently issued binding between one submit and its runtime.

    The manifest hashes this receipt as an artifact.  Its payload, in turn,
    binds the submission checkpoint and bundle to an immutable submitted
    runtime package plus a canonical digest of the extracted ``poke_bot``
    source tree that is allowed to execute dynamic traces.  This deliberately
    separates exact checkpoint inspection from dynamic execution: an older
    record remains useful for parameter inspection even if no parity receipt
    was ever archived.
    """

    artifact: ArtifactReference
    submission_id: int
    checkpoint_sha256: str | None
    bundle_sha256: str | None
    runtime_package_sha256: str | None
    runtime_source_tree_sha256: str | None
    status: str | None
    availability_reasons: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return (
            self.artifact.available
            and self.status == "verified"
            and self.checkpoint_sha256 is not None
            and self.bundle_sha256 is not None
            and self.runtime_package_sha256 is not None
            and self.runtime_source_tree_sha256 is not None
            and not self.availability_reasons
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "submission_id": self.submission_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "bundle_sha256": self.bundle_sha256,
            "runtime_package_sha256": self.runtime_package_sha256,
            "runtime_source_tree_sha256": self.runtime_source_tree_sha256,
            "status": self.status,
            "available": self.available,
            "availability_reasons": list(self.availability_reasons),
        }


@dataclass(frozen=True, slots=True)
class ReplayDigestBinding:
    """One immutable replay-byte identity declared for a submitted game.

    The replay cache is deliberately not an artifact root: it is a downloaded
    Kaggle source.  A binding therefore carries only the episode identity and
    expected bytes.  Its path is diagnostic metadata and is never trusted to
    select a file for serving.
    """

    episode_id: int | None
    expected_sha256: str | None
    declared_path: str | None
    availability_reasons: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return (
            self.episode_id is not None
            and self.expected_sha256 is not None
            and not self.availability_reasons
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "declared_path": self.declared_path,
            "expected_sha256": self.expected_sha256,
            "available": self.available,
            "availability_reasons": list(self.availability_reasons),
        }


@dataclass(frozen=True, slots=True)
class SubmissionProvenance:
    """One exact, immutable submission-to-runtime mapping."""

    submission_id: int
    checkpoint: ArtifactReference
    bundle: ArtifactReference
    matchup_tree: ArtifactReference | None
    runtime_package: ArtifactReference | None
    runtime_parity_receipt: RuntimeParityReceipt | None
    manifest_path: Path
    submission_label: SubmissionLabel
    replay_bindings: tuple[ReplayDigestBinding, ...] = ()
    status: str = "verified"
    availability_reasons: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return (
            not self.availability_reasons
            and self.checkpoint.available
            and self.bundle.available
            and (self.matchup_tree is None or self.matchup_tree.available)
        )

    @property
    def trace_available(self) -> bool:
        """Whether a runtime-parity receipt makes dynamic execution eligible.

        ``available`` intentionally remains the checkpoint/parameter gate.
        Dynamic heads/logits need the stronger runtime proof below.
        """

        if not self.available or self.runtime_package is None:
            return False
        receipt = self.runtime_parity_receipt
        return (
            self.runtime_package.available
            and receipt is not None
            and receipt.available
            and receipt.submission_id == self.submission_id
            and receipt.checkpoint_sha256 == self.checkpoint.expected_sha256
            and receipt.bundle_sha256 == self.bundle.expected_sha256
            and receipt.runtime_package_sha256 == self.runtime_package.expected_sha256
        )

    @property
    def trace_availability_reasons(self) -> tuple[str, ...]:
        if not self.available:
            return self.availability_reasons
        reasons: list[str] = []
        if self.runtime_package is None:
            reasons.append("runtime_package_artifact_missing")
        elif not self.runtime_package.available:
            reasons.extend(self.runtime_package.availability_reasons)
        if self.runtime_parity_receipt is None:
            reasons.append("runtime_parity_receipt_missing")
        elif not self.runtime_parity_receipt.available:
            reasons.extend(self.runtime_parity_receipt.availability_reasons)
        if not reasons and not self.trace_available:
            reasons.append("runtime_parity_binding_invalid")
        return _unique_reasons(reasons)

    def replay_bindings_for_episode(
        self, episode_id: int
    ) -> tuple[ReplayDigestBinding, ...]:
        """Return every explicit binding for one episode without guessing.

        A catalog must require exactly one available binding, so duplicate
        declarations remain visible as an ambiguity rather than picking a
        checksum by list order.
        """

        return tuple(
            binding
            for binding in self.replay_bindings
            if binding.episode_id == int(episode_id)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "manifest_path": str(self.manifest_path),
            "label": (
                self.submission_label.text if self.submission_label.available else None
            ),
            "submission_label": self.submission_label.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "bundle": self.bundle.to_dict(),
            "matchup_tree": (
                self.matchup_tree.to_dict() if self.matchup_tree is not None else None
            ),
            "runtime_package": (
                self.runtime_package.to_dict()
                if self.runtime_package is not None
                else None
            ),
            "runtime_parity_receipt": (
                self.runtime_parity_receipt.to_dict()
                if self.runtime_parity_receipt is not None
                else None
            ),
            "replay_bindings": [binding.to_dict() for binding in self.replay_bindings],
            "status": self.status,
            "available": self.available,
            "availability_reasons": list(self.availability_reasons),
            "trace_available": self.trace_available,
            "trace_availability_reasons": list(self.trace_availability_reasons),
        }


@dataclass(frozen=True, slots=True)
class ProvenanceManifest:
    """Verified, listable entries from one explicit provenance manifest."""

    path: Path
    source_roots: tuple[Path, ...]
    entries: tuple[SubmissionProvenance, ...]
    issues: tuple[str, ...] = ()
    schema: str = PROVENANCE_SCHEMA

    def candidates_for_submission(
        self, submission_id: int
    ) -> tuple[SubmissionProvenance, ...]:
        return tuple(
            entry for entry in self.entries if entry.submission_id == int(submission_id)
        )

    # Short aliases make the object convenient for a small HTTP handler.
    by_submission = candidates_for_submission

    def resolve_submission(self, submission_id: int) -> SubmissionProvenance | None:
        """Return the sole verified mapping, otherwise fail closed with ``None``."""

        candidates = self.candidates_for_submission(submission_id)
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        return candidate if candidate.available else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": str(self.path),
            "source_roots": [str(path) for path in self.source_roots],
            "issues": list(self.issues),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _artifact_payload(row: Mapping[str, Any], role: str) -> Mapping[str, Any] | None:
    """Return only the canonical v1 artifact declaration for ``role``.

    Deliberately do not accept guessed aliases (for example ``tree`` or
    ``checksum``) or a nested generic artifact bag.  The manifest is a trust
    boundary, so permissive migration parsing would turn a typo into an
    eligible model binding.
    """

    value = row.get(role)
    return value if isinstance(value, Mapping) else None


def _tree_explicitly_absent(row: Mapping[str, Any]) -> bool:
    """A packaged submission may validly have no matchup tree at all."""

    runtime = row.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or "matchup_tree_path" not in runtime
        or "matchup_tree_sha256" not in runtime
        or "matchup_tree" in row
    ):
        return False
    path = runtime.get("matchup_tree_path")
    digest = runtime.get("matchup_tree_sha256")
    # Do not use a set membership test here: malformed JSON may put a list or
    # object in ``digest``, and even that malformed record must remain
    # diagnosable instead of crashing a whole catalog refresh.
    return path is None and digest is None


def _nonnegative_int(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _replay_digest_bindings(row: Mapping[str, Any]) -> tuple[ReplayDigestBinding, ...]:
    """Parse explicit per-episode replay digests without trusting locations.

    ``scripts/build_replay_inspector_provenance.py`` emits the canonical
    ``replay.games`` list.  We intentionally do not accept aliases here: a
    missing or malformed declaration should leave the replay browseable while
    making model analysis unavailable, not be interpreted permissively.
    """

    replay = row.get("replay")
    if not isinstance(replay, Mapping):
        return ()
    games = replay.get("games")
    if not isinstance(games, list):
        return (
            ReplayDigestBinding(
                episode_id=None,
                expected_sha256=None,
                declared_path=None,
                availability_reasons=("replay_provenance_games_missing_or_invalid",),
            ),
        )
    bindings: list[ReplayDigestBinding] = []
    for raw_game in games:
        if not isinstance(raw_game, Mapping):
            bindings.append(
                ReplayDigestBinding(
                    episode_id=None,
                    expected_sha256=None,
                    declared_path=None,
                    availability_reasons=("replay_provenance_game_not_an_object",),
                )
            )
            continue
        reasons: list[str] = []
        episode_id = _nonnegative_int(raw_game.get("episode_id"))
        if episode_id is None:
            reasons.append("replay_provenance_episode_id_missing_or_invalid")
        expected = _canonical_sha256(raw_game.get("replay_sha256"))
        if expected is None:
            reasons.append("replay_provenance_sha256_missing_or_invalid")
        declared_path = raw_game.get("replay_path")
        if (
            not isinstance(declared_path, (str, os.PathLike))
            or not str(declared_path).strip()
        ):
            declared_path = None
        else:
            declared_path = os.fspath(declared_path)
        bindings.append(
            ReplayDigestBinding(
                episode_id=episode_id,
                expected_sha256=expected,
                declared_path=declared_path,
                availability_reasons=_unique_reasons(reasons),
            )
        )
    return tuple(bindings)


def _submission_label(row: Mapping[str, Any]) -> SubmissionLabel:
    """Parse only the canonical checksum-backed label declaration.

    ``scripts/build_replay_inspector_provenance.py`` writes a top-level
    ``label`` together with ``identity.label`` evidence.  Both pieces are
    required here.  A manifest may keep an invalid declaration visible for
    diagnostics, but the catalog receives no display text to fall back from.
    """

    raw_text = row.get("label")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return SubmissionLabel.unavailable("submission_label_not_declared")

    identity = row.get("identity")
    identity_label = identity.get("label") if isinstance(identity, Mapping) else None
    if not isinstance(identity_label, Mapping):
        return SubmissionLabel.unavailable("submission_label_identity_missing")
    if identity_label.get("value") != raw_text:
        return SubmissionLabel.unavailable("submission_label_identity_value_mismatch")
    if identity_label.get("availability") != "available":
        return SubmissionLabel.unavailable("submission_label_identity_not_available")

    raw_evidence = identity_label.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        return SubmissionLabel.unavailable("submission_label_evidence_missing")
    evidence: list[SourceEvidence] = []
    reasons: list[str] = []
    for raw_source in raw_evidence:
        if not isinstance(raw_source, Mapping):
            reasons.append("submission_label_evidence_not_an_object")
            continue
        path = raw_source.get("path")
        pointer = raw_source.get("pointer")
        role = raw_source.get("role")
        key = raw_source.get("key")
        digest = _canonical_sha256(raw_source.get("sha256"))
        if not isinstance(path, str) or not path.strip():
            reasons.append("submission_label_evidence_path_invalid")
        if not isinstance(pointer, str):
            reasons.append("submission_label_evidence_pointer_invalid")
        if not isinstance(role, str) or not role.strip():
            reasons.append("submission_label_evidence_role_invalid")
        if not isinstance(key, str) or not key.strip():
            reasons.append("submission_label_evidence_key_invalid")
        if digest is None:
            reasons.append("submission_label_evidence_sha256_invalid")
        if (
            isinstance(path, str)
            and path.strip()
            and isinstance(pointer, str)
            and isinstance(role, str)
            and role.strip()
            and isinstance(key, str)
            and key.strip()
            and digest is not None
        ):
            source = SourceEvidence(
                path=path,
                sha256=digest,
                pointer=pointer,
                role=role,
                key=key,
            )
            if source not in evidence:
                evidence.append(source)
    if reasons or not evidence:
        return SubmissionLabel.unavailable(
            "submission_label_evidence_invalid", *reasons
        )
    return SubmissionLabel(text=raw_text, evidence=tuple(evidence))


def _verify_artifact(
    role: str,
    row: Mapping[str, Any],
    *,
    manifest_dir: Path,
    source_roots: tuple[Path, ...],
) -> ArtifactReference:
    payload = _artifact_payload(row, role)
    declared_path: str | None = None
    raw_digest: Any = None
    reasons: list[str] = []
    if payload is None:
        reasons.append(f"{role}_artifact_missing")
    else:
        raw_path = payload.get("path")
        if isinstance(raw_path, (str, os.PathLike)) and str(raw_path).strip():
            declared_path = os.fspath(raw_path)
        else:
            reasons.append(f"{role}_path_missing")
        raw_digest = payload.get("sha256")

    expected = _canonical_sha256(raw_digest)
    if raw_digest is None or expected is None:
        reasons.append(f"{role}_sha256_invalid_or_missing")
    if declared_path is None:
        return ArtifactReference(
            role=role,
            declared_path=None,
            expected_sha256=expected,
            resolved_path=None,
            actual_sha256=None,
            size_bytes=None,
            availability_reasons=_unique_reasons(reasons),
        )
    if not source_roots:
        reasons.append(f"{role}_source_roots_not_configured")
        return ArtifactReference(
            role=role,
            declared_path=declared_path,
            expected_sha256=expected,
            resolved_path=None,
            actual_sha256=None,
            size_bytes=None,
            availability_reasons=_unique_reasons(reasons),
        )

    try:
        resolved = resolve_contained_path(
            declared_path,
            roots=source_roots,
            relative_to=manifest_dir,
            require_exists=True,
        )
    except FileNotFoundError:
        reasons.append(f"{role}_file_missing")
        resolved = None
    except PathContainmentError:
        reasons.append(f"{role}_path_outside_configured_roots")
        resolved = None
    except OSError:
        reasons.append(f"{role}_path_unreadable")
        resolved = None

    if resolved is None:
        return ArtifactReference(
            role=role,
            declared_path=declared_path,
            expected_sha256=expected,
            resolved_path=None,
            actual_sha256=None,
            size_bytes=None,
            availability_reasons=_unique_reasons(reasons),
        )
    if not resolved.is_file():
        reasons.append(f"{role}_not_a_regular_file")
        return ArtifactReference(
            role=role,
            declared_path=declared_path,
            expected_sha256=expected,
            resolved_path=resolved,
            actual_sha256=None,
            size_bytes=None,
            availability_reasons=_unique_reasons(reasons),
        )

    try:
        size_bytes = int(resolved.stat().st_size)
        actual = sha256_file(resolved)
    except OSError:
        reasons.append(f"{role}_hash_unreadable")
        size_bytes = None
        actual = None
    if expected is not None and actual is not None and actual != expected:
        reasons.append(f"{role}_sha256_mismatch")
    return ArtifactReference(
        role=role,
        declared_path=declared_path,
        expected_sha256=expected,
        resolved_path=resolved,
        actual_sha256=actual,
        size_bytes=size_bytes,
        availability_reasons=_unique_reasons(reasons),
    )


def _verify_runtime_parity_receipt(
    row: Mapping[str, Any],
    *,
    manifest_dir: Path,
    source_roots: tuple[Path, ...],
) -> tuple[ArtifactReference | None, RuntimeParityReceipt | None]:
    """Verify the checksum-bound receipt without trusting its claimed paths.

    A receipt is an immutable evidence document, not service configuration.
    Its only executable binding is the canonical source-tree digest that the
    server compares to its separately configured extracted runtime root.
    """

    if "runtime_parity_receipt" not in row:
        return None, None
    artifact = _verify_artifact(
        "runtime_parity_receipt",
        row,
        manifest_dir=manifest_dir,
        source_roots=source_roots,
    )
    reasons = list(artifact.availability_reasons)
    payload: Mapping[str, Any] | None = None
    if artifact.available and artifact.resolved_path is not None:
        try:
            parsed = json.loads(artifact.resolved_path.read_text(encoding="utf-8"))
            if isinstance(parsed, Mapping):
                payload = parsed
            else:
                reasons.append("runtime_parity_receipt_not_an_object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            reasons.append("runtime_parity_receipt_unreadable_or_invalid")
    if payload is None:
        return artifact, RuntimeParityReceipt(
            artifact=artifact,
            submission_id=0,
            checkpoint_sha256=None,
            bundle_sha256=None,
            runtime_package_sha256=None,
            runtime_source_tree_sha256=None,
            status=None,
            availability_reasons=_unique_reasons(reasons),
        )

    submission_id = _positive_int(payload.get("submission_id"))
    if submission_id is None:
        reasons.append("runtime_parity_receipt_submission_id_invalid")
        submission_id = 0
    if payload.get("schema") != RUNTIME_PARITY_RECEIPT_SCHEMA:
        reasons.append("runtime_parity_receipt_schema_invalid")
    if payload.get("version") != 1 or isinstance(payload.get("version"), bool):
        reasons.append("runtime_parity_receipt_version_invalid")
    status = payload.get("status") if isinstance(payload.get("status"), str) else None
    if status != "verified":
        reasons.append("runtime_parity_receipt_status_not_verified")
    verification = payload.get("verification")
    if not isinstance(verification, Mapping):
        reasons.append("runtime_parity_receipt_verification_missing")
    else:
        method = verification.get("method")
        verifier = verification.get("verified_by")
        verified_at = verification.get("verified_at_utc")
        if method != "independent_exact_runtime_parity":
            reasons.append("runtime_parity_receipt_verification_method_invalid")
        if not isinstance(verifier, str) or not verifier.strip():
            reasons.append("runtime_parity_receipt_verifier_invalid")
        if not isinstance(verified_at, str) or not verified_at.strip():
            reasons.append("runtime_parity_receipt_verified_at_invalid")
    fields = {
        "checkpoint_sha256": _canonical_sha256(payload.get("checkpoint_sha256")),
        "bundle_sha256": _canonical_sha256(payload.get("bundle_sha256")),
        "runtime_package_sha256": _canonical_sha256(
            payload.get("runtime_package_sha256")
        ),
        "runtime_source_tree_sha256": _canonical_sha256(
            payload.get("runtime_source_tree_sha256")
        ),
    }
    for name, value in fields.items():
        if value is None:
            reasons.append(f"runtime_parity_receipt_{name}_invalid_or_missing")
    return artifact, RuntimeParityReceipt(
        artifact=artifact,
        submission_id=submission_id,
        checkpoint_sha256=fields["checkpoint_sha256"],
        bundle_sha256=fields["bundle_sha256"],
        runtime_package_sha256=fields["runtime_package_sha256"],
        runtime_source_tree_sha256=fields["runtime_source_tree_sha256"],
        status=status,
        availability_reasons=_unique_reasons(reasons),
    )


def load_provenance_manifest(
    path: PathLike,
    *,
    source_roots: Iterable[PathLike],
) -> ProvenanceManifest:
    """Load a v1 manifest and verify every referenced artifact read-only.

    A malformed top-level document is rejected because it is not an explicit
    provenance manifest at all.  Problems with an individual submission are
    represented on that entry so callers can list it with a precise reason.
    """

    try:
        manifest_path = _as_path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError(f"cannot resolve provenance manifest: {path}") from exc
    if not manifest_path.is_file():
        raise ProvenanceError(
            f"provenance manifest is not a regular file: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"invalid provenance JSON: {manifest_path}") from exc
    except OSError as exc:
        raise ProvenanceError(
            f"cannot read provenance manifest: {manifest_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProvenanceError("provenance manifest must be a JSON object")
    if payload.get("schema") != PROVENANCE_SCHEMA:
        raise ProvenanceError(
            f"expected schema {PROVENANCE_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    version = payload.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ProvenanceError(f"unsupported provenance manifest version: {version!r}")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ProvenanceError("provenance manifest records must be a list")

    try:
        roots = _normalise_roots(source_roots)
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError("cannot resolve configured source roots") from exc
    entries: list[SubmissionProvenance] = []
    issues: list[str] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            issues.append(f"submission_row_{index}_not_an_object")
            continue
        submission_id = _positive_int(raw_row.get("submission_id"))
        if submission_id is None:
            issues.append(f"submission_row_{index}_invalid_submission_id")
            continue
        checkpoint = _verify_artifact(
            "checkpoint",
            raw_row,
            manifest_dir=manifest_path.parent,
            source_roots=roots,
        )
        bundle = _verify_artifact(
            "bundle",
            raw_row,
            manifest_dir=manifest_path.parent,
            source_roots=roots,
        )
        matchup_tree = (
            None
            if _tree_explicitly_absent(raw_row)
            else _verify_artifact(
                "matchup_tree",
                raw_row,
                manifest_dir=manifest_path.parent,
                source_roots=roots,
            )
        )
        runtime_package = (
            _verify_artifact(
                "runtime_package",
                raw_row,
                manifest_dir=manifest_path.parent,
                source_roots=roots,
            )
            if "runtime_package" in raw_row
            else None
        )
        _receipt_artifact, runtime_parity_receipt = _verify_runtime_parity_receipt(
            raw_row,
            manifest_dir=manifest_path.parent,
            source_roots=roots,
        )
        replay_bindings = _replay_digest_bindings(raw_row)
        submission_label = _submission_label(raw_row)
        raw_status = raw_row.get("status")
        if raw_status == "verified":
            status = "verified"
            status_reason: tuple[str, ...] = ()
        elif isinstance(raw_status, str) and raw_status:
            status = raw_status
            # The raw value is retained for diagnostics but not interpolated
            # into a public reason code, where arbitrary user-controlled
            # punctuation would make machine handling unreliable.
            status_reason = ("provenance_status_not_verified",)
        else:
            status = "missing"
            status_reason = ("provenance_status_missing_or_invalid",)
        entries.append(
            SubmissionProvenance(
                submission_id=submission_id,
                checkpoint=checkpoint,
                bundle=bundle,
                matchup_tree=matchup_tree,
                runtime_package=runtime_package,
                runtime_parity_receipt=runtime_parity_receipt,
                manifest_path=manifest_path,
                submission_label=submission_label,
                replay_bindings=replay_bindings,
                status=status,
                availability_reasons=_unique_reasons(
                    [
                        *checkpoint.availability_reasons,
                        *bundle.availability_reasons,
                        *(
                            matchup_tree.availability_reasons
                            if matchup_tree is not None
                            else ()
                        ),
                        *status_reason,
                    ]
                ),
            )
        )

    duplicate_ids = {
        entry.submission_id
        for entry in entries
        if sum(candidate.submission_id == entry.submission_id for candidate in entries)
        > 1
    }
    if duplicate_ids:
        entries = [
            replace(
                entry,
                availability_reasons=_unique_reasons(
                    (
                        *entry.availability_reasons,
                        "ambiguous_provenance_entries",
                    )
                    if entry.submission_id in duplicate_ids
                    else entry.availability_reasons
                ),
            )
            for entry in entries
        ]
    return ProvenanceManifest(
        path=manifest_path,
        source_roots=roots,
        entries=tuple(entries),
        issues=_unique_reasons(issues),
    )


__all__ = [
    "PROVENANCE_SCHEMA",
    "RUNTIME_PARITY_RECEIPT_SCHEMA",
    "ArtifactReference",
    "PathContainmentError",
    "ProvenanceError",
    "ProvenanceManifest",
    "ReplayDigestBinding",
    "RuntimeParityReceipt",
    "SourceEvidence",
    "SubmissionLabel",
    "SubmissionProvenance",
    "load_provenance_manifest",
    "resolve_contained_path",
    "sha256_file",
    "sha256_source_tree",
]
