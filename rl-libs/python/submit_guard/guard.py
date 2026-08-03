"""Generic one-shot submit grant: consume before upload, receipt after."""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


AUTH_SCHEMA = "rl_libs.submit_authorization/v1"
ATTEMPT_SCHEMA = "rl_libs.submit_attempt/v1"


class AuthorizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SubmitIdentity:
    competition: str
    file_path: str
    file_sha256: str
    message: str
    nonce: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _value(argv: Sequence[str], short: str, long: str) -> str | None:
    for index, token in enumerate(argv):
        if token in {short, long} and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(long + "="):
            return token.split("=", 1)[1]
    return None


def default_is_submit(argv: Sequence[str]) -> bool:
    lowered = [token.lower() for token in argv]
    return any(
        lowered[i] in {"competition", "competitions"} and lowered[i + 1] == "submit"
        for i in range(len(lowered) - 1)
    )


def validate_authorization(
    authorization: Mapping[str, Any],
    argv: Sequence[str],
    *,
    extra_checks: Optional[Callable[[Mapping[str, Any], Path, str], tuple[bool, str]]] = None,
) -> tuple[bool, str, SubmitIdentity | None]:
    competition = _value(list(argv), "-c", "--competition")
    file_raw = _value(list(argv), "-f", "--file")
    message = _value(list(argv), "-m", "--message")
    if not competition or not file_raw or message is None:
        return False, "submission argv missing competition/file/message", None
    file_path = Path(file_raw).expanduser().resolve()
    try:
        digest = sha256_file(file_path)
    except OSError as exc:
        return False, f"submission file cannot be hashed: {exc}", None
    now = time.time()
    expires_at = authorization.get("expires_at_epoch")
    checks = {
        "schema": authorization.get("schema") == AUTH_SCHEMA,
        "explicit_user_approval": authorization.get("explicit_user_approval") is True,
        "one_shot": int(authorization.get("remaining_uses") or 0) == 1,
        "nonce": bool(authorization.get("nonce")),
        "not_expired": isinstance(expires_at, (int, float)) and now <= float(expires_at),
        "competition": str(authorization.get("competition") or "") == competition,
        "file_sha256": str(authorization.get("file_sha256") or "") == digest,
        "message": str(authorization.get("message") or "") == message,
    }
    if extra_checks is not None:
        ok, reason = extra_checks(authorization, file_path, digest)
        checks["extra"] = ok
        if not ok:
            failed = [name for name, passed in checks.items() if not passed]
            return False, ", ".join(failed) + f" ({reason})", None
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return False, ", ".join(failed), None
    identity = SubmitIdentity(
        competition=competition,
        file_path=str(file_path),
        file_sha256=digest,
        message=message,
        nonce=str(authorization.get("nonce") or ""),
    )
    return True, "authorized", identity


def consume_and_run(
    *,
    real: Path,
    argv: Sequence[str],
    authorization_path: Path,
    lock_path: Path,
    receipts_dir: Path,
    is_submit: Callable[[Sequence[str]], bool] = default_is_submit,
    extra_checks: Optional[Callable[[Mapping[str, Any], Path, str], tuple[bool, str]]] = None,
) -> int:
    """If argv is a submit: consume one-shot grant then exec/run ``real``.

    Non-submit commands exec ``real`` directly (no grant required).
    """
    real = Path(real).resolve()
    if not is_submit(argv):
        os.execv(str(real), [str(real), *argv])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        authorization = _read_json(authorization_path)
        valid, reason, identity = validate_authorization(
            authorization, argv, extra_checks=extra_checks
        )
        if not valid or identity is None:
            raise AuthorizationError(f"submission blocked: {reason}")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_nonce = "".join(ch for ch in identity.nonce if ch.isalnum() or ch in "-_")
        consumed_path = receipts_dir / f"{stamp}-{safe_nonce or uuid.uuid4().hex}.authorization-consumed.json"
        consumed = {
            **authorization,
            "remaining_uses": 0,
            "consumed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "consumed_before_upload": True,
            "identity": identity.as_dict(),
        }
        _atomic_json(consumed_path, consumed)
        try:
            authorization_path.unlink()
        except FileNotFoundError:
            pass
        started = time.time()
        completed = subprocess.run([str(real), *argv], check=False).returncode
        receipt = {
            "schema": ATTEMPT_SCHEMA,
            "nonce": identity.nonce,
            "authorization_consumed": str(consumed_path),
            "started_at_epoch": started,
            "completed_at_epoch": time.time(),
            "returncode": completed,
            "identity": identity.as_dict(),
            "retry_requires_new_explicit_user_approval": True,
        }
        _atomic_json(
            receipts_dir / f"{stamp}-{safe_nonce or uuid.uuid4().hex}.json", receipt
        )
        return int(completed)
