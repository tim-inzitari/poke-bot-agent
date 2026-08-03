"""Atomic Torch checkpoint publish helpers without domain contracts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def _as_path(p: PathLike) -> Path:
    return Path(p).expanduser()


def atomic_torch_save(obj: Any, path: PathLike) -> Path:
    """Write ``obj`` via a temp file then atomically replace ``path``."""
    import torch

    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def immutable_torch_save(obj: Any, path: PathLike) -> Path:
    """Create ``path`` atomically and refuse to replace an existing artifact."""
    import torch

    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.link(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return path


def checkpoint_digest(path: PathLike, algorithm: str = "sha256") -> str:
    """Return a content digest used to bind evaluation to exact weights."""
    try:
        from rl_io import sha256_file

        if algorithm == "sha256":
            return sha256_file(str(_as_path(path)))
    except Exception:
        pass
    h = hashlib.new(algorithm)
    with _as_path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return f"{algorithm}:{h.hexdigest()}"
