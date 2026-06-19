from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    if not (root / "requirements.txt").exists() and (root.parent / "requirements.txt").exists():
        root = root.parent.resolve()
    os.chdir(root)
    return root


def print_runtime_info(root: Path) -> None:
    print("repo", root)
    print("python", sys.version.split()[0])
