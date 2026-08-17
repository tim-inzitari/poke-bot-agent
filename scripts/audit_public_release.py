"""Fail closed when private-development material enters the public source tree."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 10 * 1024 * 1024
FORBIDDEN_TOP_LEVEL = {
    ".cursor-loop",
    ".r241-local-staging",
    ".r274-expert-relay-v1",
    ".r274-expert-relay-v2",
    "evidence",
    "goals",
    "output",
    "outputs",
    "runtime",
    "state",
    "tmp",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".dll",
    ".dylib",
    ".features",
    ".partial",
    ".pt",
    ".pth",
    ".so",
    ".whl",
}
PRIVATE_MARKERS = (
    b"/Users/tsinzitari",
    b"/home/inzi",
    b"/mnt/Main/main/poke-bot-agent",
    b"192.168.1.143",
    b"192.168.1.151",
    b"192.168.1.158",
    b"tsinzitari@",
    b"LicenseRef-PTCG-ABC-Competition-Use-Only",
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[opsu]_[A-Za-z0-9]{30,}"),
    re.compile(rb"(?<![A-Za-z0-9_-])sk-(?:proj-)?[A-Za-z0-9_-]{24,}"),
    re.compile(rb"hf_[A-Za-z0-9]{24,}"),
)


def tracked_files() -> list[Path]:
    try:
        raw = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and ".git" not in path.relative_to(ROOT).parts
            and "__pycache__" not in path.relative_to(ROOT).parts
        )
    return [ROOT / value.decode("utf-8") for value in raw.split(b"\0") if value]


def main() -> int:
    errors: list[str] = []
    for path in tracked_files():
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in FORBIDDEN_TOP_LEVEL:
            errors.append(f"forbidden top-level path: {rel}")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact type: {rel}")
            continue
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            errors.append(f"missing tracked file: {rel}")
            continue
        if size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 10 MiB: {rel} ({size} bytes)")
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if rel == Path("scripts/audit_public_release.py"):
            continue
        for marker in PRIVATE_MARKERS:
            if marker in payload:
                errors.append(f"private/competition-only marker in {rel}: {marker!r}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(payload):
                errors.append(f"possible credential in {rel}: {pattern.pattern!r}")
        if b"teamPassword" in payload and b"config.js" in payload:
            errors.append(f"third-party client credential extraction in {rel}")

    if errors:
        print("public release audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"public release audit passed ({len(tracked_files())} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
