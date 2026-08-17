#!/usr/bin/env python3
"""Seed Elmo's durable active-checkpoint record without starting a worker.

This is an explicit one-time production rollout step. An existing record is
validated and preserved, never overwritten, so rerunning the preflight cannot
roll a live iteration back to ``model.pt``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_remote_worker import (  # noqa: E402
    REMOTE_ACTIVE_CHECKPOINT_FILE_ENV,
    _persist_active_checkpoint,
    _raw_sha256_digest,
    _select_startup_checkpoint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)

    raw_state = os.environ.get(REMOTE_ACTIVE_CHECKPOINT_FILE_ENV, "").strip()
    if not raw_state:
        print(
            f"ERROR: {REMOTE_ACTIVE_CHECKPOINT_FILE_ENV} is not configured",
            file=sys.stderr,
        )
        return 64
    state_file = Path(raw_state).expanduser()
    try:
        if state_file.exists() or state_file.is_symlink():
            selected, digest = _select_startup_checkpoint(args.checkpoint)
            print(
                f"active checkpoint state already valid; preserved path={selected} "
                f"digest={digest}",
                flush=True,
            )
            return 0

        checkpoint = Path(args.checkpoint).expanduser()
        digest = _raw_sha256_digest(checkpoint)
        published = _persist_active_checkpoint(checkpoint, digest)
        selected, selected_digest = _select_startup_checkpoint(args.checkpoint)
    except (OSError, ValueError) as exc:
        print(
            f"ERROR: active checkpoint seed failed closed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 78

    print(
        f"seeded active checkpoint state={published} path={selected} "
        f"digest={selected_digest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
