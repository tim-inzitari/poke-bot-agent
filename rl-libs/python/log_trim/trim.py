"""Delete old/large files under a directory; optionally write retirement receipts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from artifact_registry.retention import retire_with_receipt


@dataclass
class TrimResult:
    scanned: int = 0
    deleted: int = 0
    reclaimed_bytes: int = 0
    paths: list[str] = field(default_factory=list)


def trim_directory(
    root: str | Path,
    *,
    max_age_s: Optional[float] = None,
    max_total_bytes: Optional[int] = None,
    glob: str = "*",
    receipt_dir: str | Path | None = None,
    dry_run: bool = False,
) -> TrimResult:
    """Trim files matching ``glob`` under ``root``.

    Deletes oldest files first when enforcing ``max_total_bytes``.
    """
    base = Path(root)
    result = TrimResult()
    if not base.is_dir():
        return result
    now = time.time()
    files = [p for p in base.rglob(glob) if p.is_file()]
    result.scanned = len(files)
    files.sort(key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)

    def _delete(path: Path) -> None:
        nonlocal total
        size = int(path.stat().st_size)
        if dry_run:
            result.deleted += 1
            result.reclaimed_bytes += size
            result.paths.append(str(path))
            total -= size
            return
        if receipt_dir is not None:
            receipt = Path(receipt_dir) / (path.name + ".retired.json")
            n = retire_with_receipt(path, receipt, extra={"reason": "log_trim"})
            result.reclaimed_bytes += n
        else:
            path.unlink()
            result.reclaimed_bytes += size
        result.deleted += 1
        result.paths.append(str(path))
        total -= size

    if max_age_s is not None:
        for path in list(files):
            age = now - path.stat().st_mtime
            if age > max_age_s:
                _delete(path)
                files.remove(path)

    if max_total_bytes is not None:
        for path in list(files):
            if total <= max_total_bytes:
                break
            if path.exists():
                _delete(path)
                if path in files:
                    files.remove(path)
    return result
