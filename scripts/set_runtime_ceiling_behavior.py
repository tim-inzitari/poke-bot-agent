#!/usr/bin/env python3
"""Set the canonical reusable specialist ceiling handoff policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


BEHAVIOR = "freeze_submit_and_continue_without_false_pass"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    path = args.registry.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "poke_bot.specialist_runtime_registry/v1":
        raise RuntimeError("specialist runtime registry schema changed")
    handler = payload.get("pass_handler")
    if not isinstance(handler, dict):
        raise RuntimeError("specialist runtime registry lacks pass_handler")
    handler["ceiling_behavior"] = BEHAVIOR
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    print(f"CEILING_BEHAVIOR_OK {BEHAVIOR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
