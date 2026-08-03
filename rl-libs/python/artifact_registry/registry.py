"""JSON-backed artifact registry with digest binding."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from .retention import sha256_file


@dataclass
class ArtifactRecord:
    name: str
    path: str
    digest: str
    kind: str = "blob"
    meta: Optional[dict[str, Any]] = None
    registered_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArtifactRegistry:
    """Append-friendly registry stored as one JSON document."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._docs: dict[str, Any] = {"schema": 1, "artifacts": {}}
        if self.path.is_file():
            self._docs = json.loads(self.path.read_text(encoding="utf-8"))

    def register(
        self,
        name: str,
        path: str | Path,
        *,
        kind: str = "blob",
        meta: Optional[dict[str, Any]] = None,
        digest: Optional[str] = None,
    ) -> ArtifactRecord:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        record = ArtifactRecord(
            name=name,
            path=str(resolved),
            digest=digest or sha256_file(resolved),
            kind=kind,
            meta=dict(meta or {}),
            registered_at=time.time(),
        )
        self._docs.setdefault("artifacts", {})[name] = record.as_dict()
        self._save()
        return record

    def get(self, name: str) -> Optional[ArtifactRecord]:
        row = (self._docs.get("artifacts") or {}).get(name)
        if not row:
            return None
        return ArtifactRecord(**row)

    def list(self) -> list[ArtifactRecord]:
        return [ArtifactRecord(**row) for row in (self._docs.get("artifacts") or {}).values()]

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps(self._docs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp, self.path)
