#!/usr/bin/env python3
"""Move a never-started specialist to a fresh immutable run identity."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def _write_atomic(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".partial.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except TypeError:
            if Path(temporary).exists():
                Path(temporary).unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--specialist", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--new-run-name", required=True)
    parser.add_argument("--new-log", type=Path, required=True)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    specialist = dict(registry["specialists"][args.specialist])
    old_name = str(specialist["run_name"])
    old_run = args.run_root / old_name
    state_path = old_run / "loop_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    if not (
        int(state.get("next_iteration", -1)) == 0
        and int(state.get("last_completed_iteration", -2)) == -1
        and not state.get("history")
        and not list((old_run / "commits").glob("iter_*.json"))
        and not list((old_run / "shards").glob("iter_*.jsonl"))
        and not list((old_run / "checkpoints").glob("iter_*.pt"))
    ):
        raise SystemExit(
            f"refusing to retarget non-empty run {old_run}: "
            "iteration progress or training artifacts exist"
        )
    new_run = args.run_root / args.new_run_name
    if new_run.exists():
        raise SystemExit(f"new run identity already exists: {new_run}")

    specialist["run_name"] = args.new_run_name
    specialist["log"] = str(args.new_log)
    registry["specialists"][args.specialist] = specialist
    _write_atomic(args.registry, registry)
    print(
        f"RETARGETED specialist={args.specialist} old={old_name} "
        f"new={args.new_run_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
