"""Receipt-backed retirement of large artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    try:
        from rl_io import sha256_file as _fast

        return _fast(str(path))
    except Exception:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return "sha256:" + digest.hexdigest()


def retire_with_receipt(
    path: str | Path,
    receipt: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> int:
    """Write an exclusive receipt then unlink ``path``. Returns reclaimed bytes."""
    src = Path(path)
    dst = Path(receipt)
    if not src.is_file():
        return 0
    row = {
        "schema": 1,
        "path": str(src),
        "bytes": int(src.stat().st_size),
        "sha256": sha256_file(src),
        **(extra or {}),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(row, indent=2, sort_keys=True) + "\n"
    try:
        with dst.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = json.loads(dst.read_text(encoding="utf-8"))
        if existing != row:
            raise RuntimeError(f"artifact retirement receipt changed: {dst}")
    verified = json.loads(dst.read_text(encoding="utf-8"))
    if verified != row:
        raise RuntimeError(f"artifact receipt verification failed: {dst}")
    size = int(row["bytes"])
    src.unlink()
    return size
