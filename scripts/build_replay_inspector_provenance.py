#!/usr/bin/env python3
"""Build a read-only provenance catalogue for the Replay Model Inspector.

The catalogue deliberately does not consult the live selector or infer model
identity from names.  A submission is associated with an immutable model only
through a numeric submission id and checksum-bearing evidence supplied to this
tool.  It is intended to run against a downloaded replay cache and one or
more read-only artifact roots.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "poke_bot.replay_model_inspector_provenance/v1"
_SHA256_RE = re.compile(r"^sha256:([0-9a-fA-F]{64})$")
_INTEGER_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_NUMBER_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")
_RAW_REPLAY_JSON_RE = re.compile(r"^episode-[0-9]+-replay\.json$")
_RAW_AGENT_LOG_JSON_RE = re.compile(r"^episode-[0-9]+-agent-[0-9]+-logs\.json$")
_MAX_METADATA_BYTES = 32 * 1024 * 1024

_SUBMISSION_ID_KEYS = ("submission_id", "kaggle_submission_id", "submissionId")
_CHECKPOINT_ID_KEYS = (
    "checkpoint_checksum",
    "checkpoint_digest",
    "exact_frozen_checkpoint_checksum",
    "frozen_checkpoint_checksum",
    "refresh_checkpoint_checksum",
    "candidate_checkpoint_checksum",
    "learner_checksum",
    "learner_checkpoint_checksum",
)
_CHECKPOINT_LINK_KEYS = (
    "source_passing_checkpoint_digest",
    "source_checkpoint_checksum",
    "original_checkpoint_checksum",
    "parent_checkpoint_checksum",
    "bootstrap_checkpoint_checksum",
)
_CHECKPOINT_PATH_KEYS = (
    "checkpoint_path",
    "learner_checkpoint",
    "frozen_checkpoint_path",
    "model_path",
)
_BUNDLE_ID_KEYS = (
    "bundle_checksum",
    "upload_bundle_checksum",
    "submission_bundle_checksum",
    "submission_file_checksum",
)
_BUNDLE_PATH_KEYS = ("bundle_path", "upload_bundle_path", "submission_bundle_path")
_MODEL_ID_KEYS = ("model_checksum", "model_digest", "learner_model_checksum")
_MODEL_PATH_KEYS = ("model_path", "model_file", "model_file_path")
_MATCHUP_ID_KEYS = ("matchup_tree_checksum", "matchup_tree_digest")
_MATCHUP_PATH_KEYS = ("matchup_tree_path", "matchup_tree")
_RUNTIME_ID_KEYS = (
    "runtime_config_checksum",
    "runtime_registry_checksum",
    "runtime_checksum",
)
_RUNTIME_PATH_KEYS = (
    "runtime_config_path",
    "runtime_registry",
    "runtime_registry_path",
    "runtime_config",
)
_LABEL_KEYS = ("label", "submission_label", "kaggle_label")
_SPECIALIST_KEYS = ("specialist_id", "archetype_id")


class BuildError(RuntimeError):
    """An input cannot safely produce a provenance catalogue."""


@dataclass(frozen=True)
class Document:
    path: Path
    digest: str
    data: Any
    role: str


@dataclass(frozen=True)
class Record:
    document: Document
    pointer: str
    data: Mapping[str, Any]

    def source(self) -> dict[str, str]:
        return {
            "path": str(self.document.path),
            "sha256": self.document.digest,
            "pointer": self.pointer,
            "role": self.document.role,
        }


@dataclass(frozen=True)
class Claim:
    value: str
    record: Record
    key: str

    def source(self) -> dict[str, str]:
        payload = self.record.source()
        payload["key"] = self.key
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _normalise_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _SHA256_RE.fullmatch(value.strip())
    return "sha256:" + match.group(1).lower() if match else None


def _normalise_submission_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and _INTEGER_RE.fullmatch(value.strip()):
        return int(value.strip())
    return None


def _strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if quote == '"' and escaped:
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None:
            return line[:index].rstrip()
    return line.rstrip()


def _split_yaml_key_value(text: str) -> tuple[str, str] | None:
    quote: str | None = None
    escaped = False
    depth = 0
    for index, char in enumerate(text):
        if quote == '"' and escaped:
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if quote is None:
            if char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
            elif char == ":" and depth == 0:
                return text[:index].strip(), text[index + 1 :].strip()
    return None


class _FlowYamlParser:
    """A deliberately small, non-executing parser for YAML flow values."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def parse(self) -> Any:
        value = self._value()
        self._space()
        if self.index != len(self.text):
            raise ValueError("unexpected trailing YAML flow content")
        return value

    def _space(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def _value(self) -> Any:
        self._space()
        if self.index >= len(self.text):
            return None
        marker = self.text[self.index]
        if marker == "[":
            return self._list()
        if marker == "{":
            return self._mapping()
        if marker in {"'", '"'}:
            return self._quoted()
        return _yaml_scalar(self._bare())

    def _list(self) -> list[Any]:
        self.index += 1
        values: list[Any] = []
        self._space()
        if self.index < len(self.text) and self.text[self.index] == "]":
            self.index += 1
            return values
        while True:
            values.append(self._value())
            self._space()
            if self.index >= len(self.text):
                raise ValueError("unterminated YAML flow list")
            marker = self.text[self.index]
            self.index += 1
            if marker == "]":
                return values
            if marker != ",":
                raise ValueError("expected comma in YAML flow list")

    def _mapping(self) -> dict[str, Any]:
        self.index += 1
        values: dict[str, Any] = {}
        self._space()
        if self.index < len(self.text) and self.text[self.index] == "}":
            self.index += 1
            return values
        while True:
            self._space()
            if self.index >= len(self.text):
                raise ValueError("unterminated YAML flow mapping")
            key = (
                self._quoted()
                if self.text[self.index] in {"'", '"'}
                else self._bare(":")
            )
            self._space()
            if self.index >= len(self.text) or self.text[self.index] != ":":
                raise ValueError("expected colon in YAML flow mapping")
            self.index += 1
            values[str(key)] = self._value()
            self._space()
            if self.index >= len(self.text):
                raise ValueError("unterminated YAML flow mapping")
            marker = self.text[self.index]
            self.index += 1
            if marker == "}":
                return values
            if marker != ",":
                raise ValueError("expected comma in YAML flow mapping")

    def _quoted(self) -> str:
        quote = self.text[self.index]
        start = self.index
        self.index += 1
        escaped = False
        while self.index < len(self.text):
            char = self.text[self.index]
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                self.index += 1
                token = self.text[start : self.index]
                if quote == "'":
                    return token[1:-1].replace("''", "'")
                return str(ast.literal_eval(token))
            self.index += 1
        raise ValueError("unterminated YAML quoted scalar")

    def _bare(self, stop: str | None = None) -> str:
        start = self.index
        while self.index < len(self.text):
            char = self.text[self.index]
            if char in ",]}" or (stop is not None and char == stop):
                break
            self.index += 1
        return self.text[start : self.index].strip()


def _yaml_scalar(text: str) -> Any:
    value = text.strip()
    if value == "":
        return ""
    if value.startswith("!"):
        raise ValueError("YAML tags are not accepted")
    if value.startswith(("[", "{")):
        return _FlowYamlParser(value).parse()
    if value.startswith(("'", '"')):
        return _FlowYamlParser(value).parse()
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if _NUMBER_RE.fullmatch(value):
        try:
            return (
                int(value) if "." not in value and "e" not in lowered else float(value)
            )
        except ValueError:
            pass
    return value


def _minimal_yaml_load(text: str) -> Any:
    """Load the safe YAML subset used by project state files without PyYAML.

    It supports ordinary indentation-based maps/lists, quoted and scalar
    values, flow maps/lists, comments, and literal/folded block strings.  Tags,
    anchors, aliases, and executable constructors are rejected rather than
    interpreted.
    """

    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if "\t" in raw[: len(raw) - len(raw.lstrip(" \t"))]:
            raise ValueError("tabs are not accepted for YAML indentation")
        stripped = _strip_yaml_comment(raw)
        if not stripped.strip() or stripped.strip() in {"---", "..."}:
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped[indent:]
        if content.startswith(("&", "*")):
            raise ValueError("YAML anchors and aliases are not accepted")
        lines.append((indent, content))
    if not lines:
        return None

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines) or lines[index][0] < indent:
            return None, index
        is_list = lines[index][0] == indent and (
            lines[index][1] == "-" or lines[index][1].startswith("- ")
        )
        if is_list:
            values: list[Any] = []
            while (
                index < len(lines)
                and lines[index][0] == indent
                and (lines[index][1] == "-" or lines[index][1].startswith("- "))
            ):
                rest = lines[index][1][1:].strip()
                index += 1
                pair = _split_yaml_key_value(rest) if rest else None
                if pair is not None:
                    key, raw_value = pair
                    item: dict[str, Any] = {}
                    if raw_value in {"|", ">", "|-", ">-", "|+", ">+"}:
                        scalar, index = parse_block_scalar(index, indent, raw_value)
                        item[str(_yaml_scalar(key))] = scalar
                    elif raw_value:
                        item[str(_yaml_scalar(key))] = _yaml_scalar(raw_value)
                    elif index < len(lines) and lines[index][0] > indent:
                        child_indent = lines[index][0]
                        child, index = parse_block(index, child_indent)
                        item[str(_yaml_scalar(key))] = child
                    else:
                        item[str(_yaml_scalar(key))] = None
                    if index < len(lines) and lines[index][0] > indent:
                        child_indent = lines[index][0]
                        child, index = parse_block(index, child_indent)
                        if not isinstance(child, dict):
                            raise ValueError(
                                "YAML sequence mapping continuation must be a mapping"
                            )
                        item.update(child)
                    values.append(item)
                elif rest:
                    values.append(_yaml_scalar(rest))
                    if index < len(lines) and lines[index][0] > indent:
                        raise ValueError(
                            "YAML scalar sequence item cannot have nested content"
                        )
                elif index < len(lines) and lines[index][0] > indent:
                    child, index = parse_block(index, lines[index][0])
                    values.append(child)
                else:
                    values.append(None)
            return values, index

        values_map: dict[str, Any] = {}
        while (
            index < len(lines)
            and lines[index][0] == indent
            and not (lines[index][1] == "-" or lines[index][1].startswith("- "))
        ):
            pair = _split_yaml_key_value(lines[index][1])
            if pair is None:
                raise ValueError("expected YAML mapping key")
            raw_key, raw_value = pair
            key_value = _yaml_scalar(raw_key)
            if not isinstance(key_value, (str, int, float, bool)):
                raise TypeError("YAML mapping key must be scalar")
            key = str(key_value)
            index += 1
            if raw_value in {"|", ">", "|-", ">-", "|+", ">+"}:
                value, index = parse_block_scalar(index, indent, raw_value)
            elif raw_value:
                value = _yaml_scalar(raw_value)
            elif index < len(lines) and lines[index][0] > indent:
                value, index = parse_block(index, lines[index][0])
            else:
                value = None
            if key in values_map:
                raise ValueError(f"duplicate YAML mapping key: {key}")
            values_map[key] = value
        return values_map, index

    def parse_block_scalar(
        index: int, parent_indent: int, style: str
    ) -> tuple[str, int]:
        parts: list[str] = []
        child_indent: int | None = None
        while index < len(lines) and lines[index][0] > parent_indent:
            indent, content = lines[index]
            if child_indent is None:
                child_indent = indent
            if indent < child_indent:
                break
            parts.append(
                content
                if indent == child_indent
                else " " * (indent - child_indent) + content
            )
            index += 1
        if style.startswith(">"):
            result = " ".join(part.strip() for part in parts).strip()
        else:
            result = "\n".join(parts)
        if not style.endswith("-"):
            result += "\n"
        return result, index

    value, next_index = parse_block(0, lines[0][0])
    if next_index != len(lines):
        raise ValueError("invalid YAML indentation")
    return value


def _load_structured_file(path: Path, *, role: str, required: bool) -> Document | None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        if required:
            raise BuildError(f"cannot stat {path}: {exc}") from exc
        return None
    if size > _MAX_METADATA_BYTES:
        if required:
            raise BuildError(f"structured source is too large to safely load: {path}")
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            try:
                import yaml  # type: ignore[import-not-found]
            except ModuleNotFoundError:
                data = _minimal_yaml_load(raw)
            else:
                data = yaml.safe_load(raw)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        if required:
            raise BuildError(f"cannot parse structured source {path}: {exc}") from exc
        return None
    return Document(path=path.resolve(), digest=_sha256(path), data=data, role=role)


def _safe_root(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise BuildError(f"{label} must be an existing directory: {path}")
    if path.is_symlink():
        raise BuildError(f"{label} itself may not be a symlink: {path}")
    return path.resolve()


def _under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    return True


def _safe_regular_files(root: Path) -> Iterator[Path]:
    for directory, directories, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        directories[:] = [
            name for name in directories if not (directory_path / name).is_symlink()
        ]
        for filename in filenames:
            candidate = directory_path / filename
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or not _under_root(candidate, root)
            ):
                continue
            yield candidate


def _collect_records(document: Document) -> list[Record]:
    records: list[Record] = []

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, Mapping):
            record = Record(document=document, pointer=pointer, data=value)
            if any(key in value for key in _SUBMISSION_ID_KEYS) or any(
                key in value
                for key in (
                    *_CHECKPOINT_ID_KEYS,
                    *_CHECKPOINT_LINK_KEYS,
                    *_BUNDLE_ID_KEYS,
                )
            ):
                records.append(record)
            for key, nested in value.items():
                visit(nested, pointer + "/" + _json_pointer_token(key))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, pointer + "/" + str(index))

    visit(document.data, "")
    return records


def _claims(
    record: Record, keys: Sequence[str], *, digest: bool = False
) -> list[Claim]:
    found: list[Claim] = []
    for key in keys:
        value = record.data.get(key)
        if digest:
            normalised = _normalise_digest(value)
            if normalised is not None:
                found.append(Claim(normalised, record, key))
        elif isinstance(value, (str, Path)) and str(value).strip():
            found.append(Claim(str(value).strip(), record, key))
    return found


def _exact_text_claims(record: Record, keys: Sequence[str]) -> list[Claim]:
    """Return a nonblank display label without normalising its source text."""

    found: list[Claim] = []
    for key in keys:
        value = record.data.get(key)
        if isinstance(value, str) and value.strip():
            # Labels are operator-visible submission text.  Preserve the exact
            # source value rather than using the generic path/identifier
            # normaliser, which intentionally strips whitespace.
            found.append(Claim(value, record, key))
    return found


def _component_claims(record: Record, component: str, *, digest: bool) -> list[Claim]:
    value = record.data.get(component)
    if isinstance(value, Mapping):
        keys = (
            ("checksum", "sha256", "digest")
            if digest
            else ("path", "file", "file_path")
        )
        found: list[Claim] = []
        for key in keys:
            candidate = value.get(key)
            normalised = (
                _normalise_digest(candidate)
                if digest
                else str(candidate).strip()
                if isinstance(candidate, str)
                else None
            )
            if normalised:
                found.append(Claim(normalised, record, f"{component}.{key}"))
        return found
    if isinstance(value, str):
        normalised = _normalise_digest(value) if digest else value.strip()
        if normalised:
            return [Claim(normalised, record, component)]
    return []


def _field_claims(record: Record, field: str) -> list[Claim]:
    if field == "checkpoint_checksum":
        return _claims(record, _CHECKPOINT_ID_KEYS, digest=True) + _component_claims(
            record, "checkpoint", digest=True
        )
    if field == "checkpoint_link":
        return _claims(record, _CHECKPOINT_LINK_KEYS, digest=True)
    if field == "checkpoint_path":
        return _claims(record, _CHECKPOINT_PATH_KEYS) + _component_claims(
            record, "checkpoint", digest=False
        )
    if field == "bundle_checksum":
        return _claims(record, _BUNDLE_ID_KEYS, digest=True) + _component_claims(
            record, "bundle", digest=True
        )
    if field == "bundle_path":
        return _claims(record, _BUNDLE_PATH_KEYS) + _component_claims(
            record, "bundle", digest=False
        )
    if field == "model_checksum":
        return _claims(record, _MODEL_ID_KEYS, digest=True) + _component_claims(
            record, "model", digest=True
        )
    if field == "model_path":
        return _claims(record, _MODEL_PATH_KEYS) + _component_claims(
            record, "model", digest=False
        )
    if field == "matchup_tree_checksum":
        return _claims(record, _MATCHUP_ID_KEYS, digest=True) + _component_claims(
            record, "matchup_tree", digest=True
        )
    if field == "matchup_tree_path":
        return _claims(record, _MATCHUP_PATH_KEYS) + _component_claims(
            record, "matchup_tree", digest=False
        )
    if field == "runtime_checksum":
        return _claims(record, _RUNTIME_ID_KEYS, digest=True)
    if field == "runtime_path":
        return _claims(record, _RUNTIME_PATH_KEYS)
    if field == "label":
        return _exact_text_claims(record, _LABEL_KEYS)
    if field == "specialist_id":
        return _claims(record, _SPECIALIST_KEYS)
    raise AssertionError(f"unknown provenance field: {field}")


def _record_submission_ids(record: Record) -> list[int]:
    found: list[int] = []
    for key in _SUBMISSION_ID_KEYS:
        value = _normalise_submission_id(record.data.get(key))
        if value is not None:
            found.append(value)
    return sorted(set(found))


def _is_canonical_submission_record(record: Record) -> bool:
    """Whether this is a top-level row of the inspector's v1 manifest."""

    document = record.document.data
    return (
        isinstance(document, Mapping)
        and document.get("schema") == SCHEMA
        and document.get("version") == 1
        and isinstance(document.get("records"), list)
        and re.fullmatch(r"/records/[0-9]+", record.pointer) is not None
    )


def _has_direct_submission_identity(record: Record) -> bool:
    """Whether an evidence mapping directly identifies a submitted model.

    Submission IDs also occur in replay games, player/agent records, and
    provenance pointers.  Those nested references are corroborating context,
    not authority to create an inspector submission row.
    """

    return any(
        _field_claims(record, field)
        for field in (
            "checkpoint_checksum",
            "checkpoint_path",
            "bundle_checksum",
            "bundle_path",
            "model_checksum",
            "model_path",
            "matchup_tree_checksum",
            "matchup_tree_path",
            "label",
            "specialist_id",
        )
    )


def _evidence_submission_seed_records(document: Document) -> list[Record]:
    """Return structural evidence records allowed to create submissions.

    Canonical v1 manifests own submission identity only at ``/records/<n>``.
    Generic evidence must name exactly one submission and contain a direct
    model/submission identity claim.  All other recursively collected records
    remain available for checksum-bound corroboration, but cannot enlarge the
    output submission universe.
    """

    return [
        record
        for record in _collect_records(document)
        if len(_record_submission_ids(record)) == 1
        and (
            _is_canonical_submission_record(record)
            or _has_direct_submission_identity(record)
        )
    ]


def _distinct_claims(claims: Iterable[Claim]) -> dict[str, list[Claim]]:
    result: dict[str, list[Claim]] = {}
    for claim in claims:
        result.setdefault(claim.value, []).append(claim)
    return result


def _sources_for_claims(claims: Iterable[Claim]) -> list[dict[str, str]]:
    unique: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for claim in claims:
        source = claim.source()
        key = (
            source["path"],
            source["sha256"],
            source["pointer"],
            source["role"],
            source["key"],
        )
        unique[key] = source
    return [unique[key] for key in sorted(unique)]


def _claim_values(claims: Iterable[Claim]) -> list[str]:
    return sorted(_distinct_claims(claims))


def _resolve_declared_path(
    value: str, record: Record, roots: Sequence[Path]
) -> Path | None:
    candidate = Path(value)
    attempts: list[Path] = []
    if candidate.is_absolute():
        attempts.append(candidate)
    else:
        attempts.append(record.document.path.parent / candidate)
        attempts.extend(root / candidate for root in roots)
    for attempt in attempts:
        try:
            resolved = attempt.resolve(strict=True)
        except OSError:
            continue
        if any(_under_root(resolved, root) for root in roots) and resolved.is_file():
            return resolved
    return None


def _artifact_digest_index(
    roots: Sequence[Path], desired: set[str]
) -> tuple[dict[str, list[Path]], list[str]]:
    """Find requested immutable artifacts by content, never by a filename."""
    found: dict[str, list[Path]] = {digest: [] for digest in desired}
    errors: list[str] = []
    remaining = set(desired)
    if not remaining:
        return found, errors
    for root in roots:
        for path in _safe_regular_files(root):
            if not remaining:
                return found, errors
            try:
                digest = _sha256(path)
            except OSError as exc:
                errors.append(f"cannot hash artifact {path}: {exc}")
                continue
            if digest in remaining:
                found[digest].append(path.resolve())
                # One verified copy is sufficient to make an immutable digest
                # usable.  Stop hashing once every requested identity has one;
                # this keeps catalog refreshes bounded on a NAS full of old
                # checkpoints without trusting filenames to do so.
                remaining.remove(digest)
    return found, errors


def _document_metadata_under_roots(
    roots: Sequence[Path],
) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    errors: list[str] = []
    for root in roots:
        for path in _safe_regular_files(root):
            if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            document = _load_structured_file(
                path, role="artifact_metadata", required=False
            )
            if document is None:
                try:
                    if path.stat().st_size <= _MAX_METADATA_BYTES:
                        errors.append(f"unreadable artifact metadata: {path}")
                except OSError:
                    errors.append(f"unreadable artifact metadata: {path}")
                continue
            documents.append(document)
    return documents, errors


def _record_replay_entries(
    document: Document, replay_root: Path
) -> tuple[dict[int, list[dict[str, Any]]], list[str]]:
    records = _collect_records(document)
    entries: dict[int, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    for record in records:
        submission_ids = _record_submission_ids(record)
        episode_id = _normalise_submission_id(record.data.get("episode_id"))
        if len(submission_ids) != 1 or episode_id is None:
            continue
        submission_id = submission_ids[0]
        game = {
            "episode_id": episode_id,
            "state": record.data.get("state"),
            "type": record.data.get("type"),
            "create_time": record.data.get("create_time"),
            "end_time": record.data.get("end_time"),
            "own_agent": record.data.get("own_agent"),
            "agents": record.data.get("agents"),
            "metadata_source": record.source(),
        }
        explicit_paths = [
            value
            for key in ("replay_path", "replay_file", "replay_file_path")
            if isinstance((value := record.data.get(key)), str) and value.strip()
        ]
        resolved: list[Path] = []
        for value in explicit_paths:
            path = _resolve_declared_path(value, record, [replay_root])
            if path is not None:
                resolved.append(path)
        if not resolved:
            # The Elmo downloader's cache layout is a location hint only; the
            # submission association still comes from the signed/cache metadata.
            expected = document.path.parent / f"episode-{episode_id}-replay.json"
            if (
                expected.is_file()
                and _under_root(expected, replay_root)
                and not expected.is_symlink()
            ):
                resolved.append(expected.resolve())
                game["path_association"] = "cache_layout_episode_id"
        if len({str(path) for path in resolved}) == 1:
            path = resolved[0]
            game["replay_path"] = str(path)
            game["replay_sha256"] = _sha256(path)
            game["availability"] = "available"
        elif len(resolved) > 1:
            game["availability"] = "unavailable"
            game["unavailable_reason"] = "ambiguous_replay_path_declarations"
            warnings.append(
                f"submission {submission_id} episode {episode_id}: ambiguous replay paths"
            )
        else:
            game["availability"] = "unavailable"
            game["unavailable_reason"] = "replay_file_not_found_within_configured_root"
        entries.setdefault(submission_id, []).append(game)
    return entries, warnings


def _replay_documents(replay_root: Path) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    warnings: list[str] = []
    for path in _safe_regular_files(replay_root):
        if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        # Full replay payloads and captured agent logs are immutable artifacts,
        # not catalogue metadata. Loading every one into Python duplicates the
        # entire replay archive in memory before the builder reaches the small
        # ``episodes.json`` association records. The exact replay bytes are
        # still hashed below after an episode metadata row binds them.
        if _RAW_REPLAY_JSON_RE.fullmatch(path.name) or _RAW_AGENT_LOG_JSON_RE.fullmatch(
            path.name
        ):
            continue
        document = _load_structured_file(
            path, role="replay_cache_metadata", required=False
        )
        if document is not None:
            documents.append(document)
        else:
            try:
                if path.stat().st_size <= _MAX_METADATA_BYTES:
                    warnings.append(f"unreadable replay metadata: {path}")
            except OSError:
                warnings.append(f"unreadable replay metadata: {path}")
    return documents, warnings


def _identity_component(
    *,
    checksum: str | None,
    declared_paths: Sequence[Claim],
    evidence: Sequence[Claim],
    verified_by_digest: Mapping[str, Sequence[Path]],
    roots: Sequence[Path],
    fallback_reason: str,
) -> tuple[dict[str, Any], list[str]]:
    paths = sorted({claim.value for claim in declared_paths})
    path_evidence = _sources_for_claims(declared_paths)
    verified: set[str] = set()
    verified_declared: dict[str, int] = {}
    reasons: list[str] = []
    mismatch = False
    for claim in declared_paths:
        candidate = _resolve_declared_path(claim.value, claim.record, roots)
        if candidate is None:
            reasons.append(
                f"declared path is unavailable outside configured artifact roots: {claim.value}"
            )
            continue
        try:
            actual = _sha256(candidate)
        except OSError as exc:
            reasons.append(f"cannot hash declared artifact path {candidate}: {exc}")
            continue
        if checksum is not None and actual != checksum:
            mismatch = True
            reasons.append(f"declared artifact checksum mismatch at {candidate}")
        elif checksum is None:
            reasons.append(f"artifact path has no checksum-bound identity: {candidate}")
        else:
            verified_path = str(candidate)
            verified.add(verified_path)
            # Explicit current evidence outranks recursively discovered stale
            # catalogue metadata when identical bytes exist in multiple
            # submission packages.  Lexical path ordering must never select a
            # sibling submission's runtime merely because its digest matches.
            evidence_rank = 0 if claim.record.document.role == "evidence" else 1
            previous_rank = verified_declared.get(verified_path)
            if previous_rank is None or evidence_rank < previous_rank:
                verified_declared[verified_path] = evidence_rank
    if checksum is not None:
        verified.update(str(path) for path in verified_by_digest.get(checksum, []))
    if mismatch:
        availability = "unavailable"
    elif checksum is None:
        availability = "unavailable"
        reasons.append(fallback_reason)
    elif verified:
        availability = "available"
    else:
        availability = "unavailable"
        reasons.append(
            "no checksum-verified artifact is available within configured artifact roots"
        )
    return (
        {
            "checksum": checksum,
            "declared_paths": paths,
            "verified_paths": sorted(verified),
            # A submission-bound declaration is stronger path provenance than
            # a generic content-addressed discovery elsewhere in the artifact
            # roots.  Keep every verified path above for audit, but use this
            # one for the v1 canonical artifact field.
            "preferred_path": (
                min(
                    verified_declared,
                    key=lambda path: (verified_declared[path], path),
                )
                if verified_declared
                else (min(verified) if verified else None)
            ),
            "availability": availability,
            "unavailable_reasons": sorted(set(reasons)),
            "evidence": _sources_for_claims(evidence),
            "path_evidence": path_evidence,
        },
        [reason for reason in reasons if mismatch],
    )


def _choose_single_field(
    name: str,
    claims: Sequence[Claim],
    reasons: list[str],
) -> str | None:
    values = _claim_values(claims)
    if len(values) > 1:
        reasons.append(f"conflicting {name}: {', '.join(values)}")
        return None
    return values[0] if values else None


def _unique_records(records: Iterable[Record]) -> list[Record]:
    seen: set[tuple[str, str]] = set()
    result: list[Record] = []
    for record in records:
        key = (str(record.document.path), record.pointer)
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def _build_submission(
    submission_id: int,
    direct_records: Sequence[Record],
    all_records: Sequence[Record],
    games: Sequence[dict[str, Any]],
    artifact_roots: Sequence[Path],
    verified_by_digest: Mapping[str, Sequence[Path]],
) -> dict[str, Any]:
    reasons: list[str] = []
    direct_checkpoint_claims = [
        claim
        for record in direct_records
        for claim in _field_claims(record, "checkpoint_checksum")
    ]
    checkpoint_checksum = _choose_single_field(
        "checkpoint checksum", direct_checkpoint_claims, reasons
    )
    if checkpoint_checksum is None and not direct_checkpoint_claims:
        reasons.append("missing checksum-bound checkpoint provenance")

    related_records: list[Record] = list(direct_records)
    if checkpoint_checksum is not None:
        for record in all_records:
            linked = {
                claim.value
                for claim in (
                    *_field_claims(record, "checkpoint_checksum"),
                    *_field_claims(record, "checkpoint_link"),
                )
            }
            if checkpoint_checksum in linked:
                related_records.append(record)
    related_records = _unique_records(related_records)

    def merged(field: str) -> list[Claim]:
        return [
            claim
            for record in related_records
            for claim in _field_claims(record, field)
        ]

    def submission_scoped_merged(field: str) -> list[Claim]:
        """Do not borrow package identity from a sibling submission.

        Multiple submissions can legitimately share one checkpoint while
        packaging different code, trees, decks, or entrypoints.  A generic
        checkpoint-linked record may corroborate those fields, but a record
        that names another numeric submission may not.
        """

        return [
            claim
            for record in related_records
            if not (record_ids := _record_submission_ids(record))
            or submission_id in record_ids
            for claim in _field_claims(record, field)
        ]

    def preferred_submission_identity(field: str) -> list[Claim]:
        """Keep submission-specific text/identity separate from shared model facts.

        A checkpoint can legitimately be submitted more than once.  Checksum
        and artifact claims remain merged below so conflicting artifacts fail
        closed, but a direct label or specialist claim for this numeric
        submission must not be replaced or conflicted by metadata belonging to
        another submission that happened to use the same checkpoint.
        """

        direct = [
            claim for record in direct_records for claim in _field_claims(record, field)
        ]
        if direct:
            return direct
        # A generic checkpoint-level record can fill an absent submission
        # identity, but an identity declared for another numeric submission
        # cannot.  This avoids silently borrowing a sibling submission's
        # label when only one of two shared-checkpoint submissions declares it.
        return [
            claim
            for record in related_records
            if not (record_ids := _record_submission_ids(record))
            or submission_id in record_ids
            for claim in _field_claims(record, field)
        ]

    bundle_checksum = _choose_single_field(
        "bundle checksum", submission_scoped_merged("bundle_checksum"), reasons
    )
    model_claims = merged("model_checksum")
    model_checksum = _choose_single_field("model checksum", model_claims, reasons)
    if model_checksum is None and not model_claims and checkpoint_checksum is not None:
        model_checksum = checkpoint_checksum
        model_evidence = direct_checkpoint_claims
        model_derived_from_checkpoint = True
    else:
        model_evidence = model_claims
        model_derived_from_checkpoint = False
    matchup_checksum = _choose_single_field(
        "matchup tree checksum",
        submission_scoped_merged("matchup_tree_checksum"),
        reasons,
    )
    label_claims = preferred_submission_identity("label")
    label = _choose_single_field("submission label", label_claims, reasons)
    specialist_claims = preferred_submission_identity("specialist_id")
    specialist_id = _choose_single_field("specialist id", specialist_claims, reasons)
    runtime_checksums = _claim_values(submission_scoped_merged("runtime_checksum"))
    runtime_paths = submission_scoped_merged("runtime_path")

    checkpoint, checkpoint_mismatches = _identity_component(
        checksum=checkpoint_checksum,
        declared_paths=merged("checkpoint_path"),
        evidence=direct_checkpoint_claims,
        verified_by_digest=verified_by_digest,
        roots=artifact_roots,
        fallback_reason="checkpoint checksum is not declared",
    )
    bundle, bundle_mismatches = _identity_component(
        checksum=bundle_checksum,
        declared_paths=submission_scoped_merged("bundle_path"),
        evidence=submission_scoped_merged("bundle_checksum"),
        verified_by_digest=verified_by_digest,
        roots=artifact_roots,
        fallback_reason="bundle checksum is not declared",
    )
    model, model_mismatches = _identity_component(
        checksum=model_checksum,
        declared_paths=merged("model_path") or merged("checkpoint_path"),
        evidence=model_evidence,
        verified_by_digest=verified_by_digest,
        roots=artifact_roots,
        fallback_reason="model checksum is not declared",
    )
    matchup_tree, matchup_mismatches = _identity_component(
        checksum=matchup_checksum,
        declared_paths=submission_scoped_merged("matchup_tree_path"),
        evidence=submission_scoped_merged("matchup_tree_checksum"),
        verified_by_digest=verified_by_digest,
        roots=artifact_roots,
        fallback_reason="matchup tree is not declared by this submission format",
    )
    reasons.extend(
        checkpoint_mismatches
        + bundle_mismatches
        + model_mismatches
        + matchup_mismatches
    )
    if bundle_checksum is None:
        reasons.append("missing checksum-bound bundle provenance")
    if checkpoint_checksum is not None and checkpoint["availability"] != "available":
        reasons.append(
            "checkpoint artifact is not checksum-verified within configured artifact roots"
        )
    if bundle_checksum is not None and bundle["availability"] != "available":
        reasons.append(
            "bundle artifact is not checksum-verified within configured artifact roots"
        )
    if matchup_checksum is not None and matchup_tree["availability"] != "available":
        reasons.append(
            "matchup tree artifact is not checksum-verified within configured artifact roots"
        )
    if not games:
        reasons.append("no replay cache entries found for submission")
    if any(game.get("availability") != "available" for game in games):
        reasons.append("one or more replay files are unavailable or ambiguous")

    runtime_config = {
        "checksums": runtime_checksums,
        "declared_paths": sorted({claim.value for claim in runtime_paths}),
        "evidence": _sources_for_claims([*merged("runtime_checksum"), *runtime_paths]),
        "availability": (
            "available" if runtime_paths or runtime_checksums else "unavailable"
        ),
        "unavailable_reasons": (
            []
            if runtime_paths or runtime_checksums
            else ["runtime configuration was not declared by the available provenance"]
        ),
    }
    status = "verified" if not reasons else "unresolved"
    replay_payload = {
        "games": sorted(games, key=lambda game: int(game["episode_id"])),
        "game_count": len(games),
        "available_game_count": sum(
            game.get("availability") == "available" for game in games
        ),
    }

    def v1_artifact(component: Mapping[str, Any]) -> dict[str, str | None]:
        verified_paths = component.get("verified_paths")
        preferred_path = component.get("preferred_path")
        return {
            "path": (
                preferred_path
                if isinstance(preferred_path, str)
                else verified_paths[0]
                if isinstance(verified_paths, list) and verified_paths
                else None
            ),
            "sha256": (
                component.get("checksum")
                if isinstance(component.get("checksum"), str)
                else None
            ),
        }

    v1_checkpoint = v1_artifact(checkpoint)
    v1_bundle = v1_artifact(bundle)
    v1_matchup_tree = (
        v1_artifact(matchup_tree) if matchup_checksum is not None else None
    )
    # The inspector's v1 loader has an intentional explicit-null form for old
    # bundles that did not ship a matchup tree.  Do not invent a tree path for
    # these formats; the rich identity payload below records why it is absent.
    runtime_v1 = {
        "matchup_tree_path": (
            None if v1_matchup_tree is None else v1_matchup_tree["path"]
        ),
        "matchup_tree_sha256": (
            None if v1_matchup_tree is None else v1_matchup_tree["sha256"]
        ),
        "config_paths": runtime_config["declared_paths"],
        "config_checksums": runtime_config["checksums"],
    }
    return {
        "submission_id": submission_id,
        "status": status,
        "unresolved_reasons": sorted(set(reasons)),
        # These are the canonical flat v1 fields consumed by the standalone
        # inspector.  Keep the detailed identity object too so unavailable
        # records stay explainable rather than disappearing from the cache.
        "checkpoint": v1_checkpoint,
        "bundle": v1_bundle,
        "matchup_tree": v1_matchup_tree,
        "runtime": runtime_v1,
        "label": label,
        "specialist_id": specialist_id,
        "replay": replay_payload,
        "identity": {
            "checkpoint": checkpoint,
            "bundle": bundle,
            "model": {
                **model,
                "derived_from_checkpoint_checksum": model_derived_from_checkpoint,
            },
            "matchup_tree": matchup_tree,
            "label": {
                "value": label,
                "evidence": _sources_for_claims(label_claims),
                "availability": "available" if label is not None else "unavailable",
                "unavailable_reason": (
                    None if label is not None else "submission label is not declared"
                ),
            },
            "specialist_id": {
                "value": specialist_id,
                "evidence": _sources_for_claims(specialist_claims),
                "availability": (
                    "available" if specialist_id is not None else "unavailable"
                ),
                "unavailable_reason": (
                    None
                    if specialist_id is not None
                    else "specialist id is not declared"
                ),
            },
            "runtime_config": runtime_config,
        },
        "evidence_records": [record.source() for record in related_records],
    }


def _validate_output_path(
    output: Path,
    *,
    evidence: Sequence[Path],
    replay_root: Path,
    artifact_roots: Sequence[Path],
) -> Path:
    if not output.parent.exists() or not output.parent.is_dir():
        raise BuildError(f"output parent must already exist: {output.parent}")
    if output.exists() and output.is_symlink():
        raise BuildError(f"output may not be a symlink: {output}")
    resolved = output.resolve(strict=False)
    if any(resolved == source.resolve(strict=False) for source in evidence):
        raise BuildError("output may not overwrite an evidence source")
    if _under_root(resolved, replay_root) or any(
        _under_root(resolved, root) for root in artifact_roots
    ):
        raise BuildError("output must be outside replay and artifact source roots")
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def build_catalog(
    *,
    replay_root: Path,
    evidence_paths: Sequence[Path],
    artifact_roots: Sequence[Path],
    runtime_parity_receipts: Mapping[int, Path] | None = None,
) -> dict[str, Any]:
    replay_root = _safe_root(replay_root, "replay root")
    artifacts = [_safe_root(path, "artifact root") for path in artifact_roots]
    receipt_overrides: dict[int, dict[str, str]] = {}
    for submission_id, requested_path in (runtime_parity_receipts or {}).items():
        if submission_id <= 0:
            raise BuildError("runtime parity receipt submission id must be positive")
        try:
            receipt_path = requested_path.resolve(strict=True)
        except OSError as exc:
            raise BuildError("runtime parity receipt cannot be resolved") from exc
        if (
            not receipt_path.is_file()
            or receipt_path.is_symlink()
            or not any(_under_root(receipt_path, root) for root in artifacts)
        ):
            raise BuildError(
                "runtime parity receipt must be a regular file under an artifact root"
            )
        receipt_document = _load_structured_file(
            receipt_path, role="runtime_parity_receipt", required=True
        )
        assert receipt_document is not None
        receipt = receipt_document.data
        if not isinstance(receipt, Mapping):
            raise BuildError("runtime parity receipt must be a JSON/YAML object")
        declared_id = _normalise_submission_id(receipt.get("submission_id"))
        package_digest = _normalise_digest(receipt.get("runtime_package_sha256"))
        if declared_id != submission_id or package_digest is None:
            raise BuildError(
                "runtime parity receipt submission id or package checksum is invalid"
            )
        receipt_overrides[submission_id] = {
            "path": str(receipt_path),
            "sha256": receipt_document.digest,
            "runtime_package_sha256": package_digest,
        }
    if not evidence_paths:
        raise BuildError("at least one --evidence source is required")
    evidence_documents: list[Document] = []
    for path in evidence_paths:
        if not path.exists() or not path.is_file() or path.is_symlink():
            raise BuildError(
                f"evidence source must be a regular non-symlink file: {path}"
            )
        document = _load_structured_file(path, role="evidence", required=True)
        assert document is not None
        evidence_documents.append(document)

    artifact_documents, artifact_warnings = _document_metadata_under_roots(artifacts)
    evidence_records = [
        record
        for document in evidence_documents
        for record in _collect_records(document)
    ]
    evidence_submission_records = [
        record
        for document in evidence_documents
        for record in _evidence_submission_seed_records(document)
    ]
    artifact_records = [
        record
        for document in artifact_documents
        for record in _collect_records(document)
    ]
    # Artifact metadata can mention arbitrary historical or embedded
    # submission ids.  It may corroborate an already selected checksum, but
    # it must never create a new catalogue submission on its own.  Only the
    # replay cache and structural explicit-evidence records own the submission
    # universe.  In particular, a nested replay opponent id is not a direct
    # submission record merely because a canonical manifest was supplied as
    # evidence for a rebuild.
    all_records = [*evidence_records, *artifact_records]
    records_by_submission: dict[int, list[Record]] = {}
    for record in evidence_submission_records:
        for submission_id in _record_submission_ids(record):
            records_by_submission.setdefault(submission_id, []).append(record)

    replay_documents, replay_warnings = _replay_documents(replay_root)
    replay_by_submission: dict[int, list[dict[str, Any]]] = {}
    for document in replay_documents:
        entries, warnings = _record_replay_entries(document, replay_root)
        replay_warnings.extend(warnings)
        for submission_id, games in entries.items():
            replay_by_submission.setdefault(submission_id, []).extend(games)
    for submission_id, games in list(replay_by_submission.items()):
        unique: dict[int, dict[str, Any]] = {}
        for game in games:
            episode_id = int(game["episode_id"])
            if episode_id in unique:
                previous = unique[episode_id]
                if previous.get("replay_sha256") != game.get("replay_sha256"):
                    previous["availability"] = "unavailable"
                    previous["unavailable_reason"] = "conflicting_replay_cache_metadata"
                continue
            unique[episode_id] = game
        replay_by_submission[submission_id] = list(unique.values())

    desired_digests: set[str] = set()
    for record in all_records:
        for field in (
            "checkpoint_checksum",
            "bundle_checksum",
            "model_checksum",
            "matchup_tree_checksum",
        ):
            desired_digests.update(
                claim.value for claim in _field_claims(record, field)
            )
    desired_digests.update(
        override["runtime_package_sha256"] for override in receipt_overrides.values()
    )
    verified_by_digest, artifact_hash_errors = _artifact_digest_index(
        artifacts, desired_digests
    )

    submission_ids = sorted(set(records_by_submission) | set(replay_by_submission))
    submissions = [
        _build_submission(
            submission_id,
            records_by_submission.get(submission_id, []),
            all_records,
            replay_by_submission.get(submission_id, []),
            artifacts,
            verified_by_digest,
        )
        for submission_id in submission_ids
    ]
    for item in submissions:
        submission_id = int(item["submission_id"])
        override = receipt_overrides.get(submission_id)
        if override is None:
            continue
        package_paths = verified_by_digest.get(override["runtime_package_sha256"], [])
        item["runtime_parity_receipt"] = {
            "path": override["path"],
            "sha256": override["sha256"],
        }
        if len(package_paths) == 1:
            item["runtime_package"] = {
                "path": str(package_paths[0]),
                "sha256": override["runtime_package_sha256"],
            }
        else:
            item["runtime_package"] = None
            item.setdefault("unresolved_reasons", []).append(
                "runtime parity receipt package is not uniquely checksum-verified"
            )
    return {
        "schema": SCHEMA,
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "selection_authority": "submission_id_plus_checksum_bound_evidence_only",
        "replay_root": str(replay_root),
        "artifact_roots": [str(root) for root in artifacts],
        "evidence_sources": [
            {
                "path": str(document.path),
                "sha256": document.digest,
                "format": document.path.suffix.lower().lstrip("."),
            }
            for document in evidence_documents
        ],
        "records": submissions,
        "summary": {
            "submission_count": len(submissions),
            "verified_submission_count": sum(
                item["status"] == "verified" for item in submissions
            ),
            "unresolved_submission_count": sum(
                item["status"] != "verified" for item in submissions
            ),
            "replay_game_count": sum(
                item["replay"]["game_count"] for item in submissions
            ),
            "available_replay_game_count": sum(
                item["replay"]["available_game_count"] for item in submissions
            ),
        },
        "catalog_warnings": sorted(
            {*artifact_warnings, *replay_warnings, *artifact_hash_errors}
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument(
        "--evidence",
        required=True,
        action="append",
        type=Path,
        help="JSON or safe YAML evidence source; repeatable",
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        type=Path,
        default=[],
        help="read-only root containing model/bundle/tree artifacts; repeatable",
    )
    parser.add_argument(
        "--runtime-parity-receipt",
        action="append",
        default=[],
        metavar="SUBMISSION_ID=PATH",
        help=(
            "checksum-bound independent runtime-parity receipt for one "
            "submission; repeatable"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        replay_root = _safe_root(args.replay_root, "replay root")
        artifact_roots = [
            _safe_root(path, "artifact root") for path in args.artifact_root
        ]
        runtime_parity_receipts: dict[int, Path] = {}
        for raw_value in args.runtime_parity_receipt:
            submission_text, separator, path_text = str(raw_value).partition("=")
            if not separator or not _INTEGER_RE.fullmatch(submission_text):
                raise BuildError("--runtime-parity-receipt must be SUBMISSION_ID=PATH")
            submission_id = int(submission_text)
            if submission_id <= 0 or not path_text.strip():
                raise BuildError(
                    "--runtime-parity-receipt must use a positive id and path"
                )
            if submission_id in runtime_parity_receipts:
                raise BuildError("duplicate runtime parity receipt submission id")
            runtime_parity_receipts[submission_id] = Path(path_text)
        output = _validate_output_path(
            args.output,
            evidence=args.evidence,
            replay_root=replay_root,
            artifact_roots=artifact_roots,
        )
        catalog = build_catalog(
            replay_root=replay_root,
            evidence_paths=args.evidence,
            artifact_roots=artifact_roots,
            runtime_parity_receipts=runtime_parity_receipts,
        )
        _atomic_json(output, catalog)
    except BuildError as exc:
        parser.error(str(exc))
    print(json.dumps(catalog["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
