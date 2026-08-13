from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .errors import IntegrityError, ValidationError


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise IntegrityError(f"referenced file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise IntegrityError(f"referenced file does not exist: {path}") from exc

    records: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"invalid JSONL record in {path}:{line_number}: {exc}"
            ) from exc
    return records


def resolve_local_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValidationError(f"references must be portable relative paths: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValidationError(f"reference escapes the goal package: {relative}")
    return resolved


def verify_reference(root: Path, reference: dict[str, Any], *, label: str) -> Path:
    path_value = reference.get("path")
    checksum = reference.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValidationError(f"{label}.path must be a non-empty string")
    if not is_sha256(checksum):
        raise ValidationError(f"{label}.sha256 must use sha256:<hex>")
    path = resolve_local_path(root, path_value)
    actual = sha256_file(path)
    if actual != checksum:
        raise IntegrityError(
            f"{label} checksum mismatch: expected {checksum}, observed {actual}"
        )
    return path


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json_lines(records: Iterable[Any]) -> str:
    return "".join(canonical_json(record) + "\n" for record in records)
