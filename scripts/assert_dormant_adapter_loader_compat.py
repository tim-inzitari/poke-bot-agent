#!/usr/bin/env python3
"""Publish an immutable Inzi/Elmo/Bert/submission dormant-loader receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.dormant_adapter_compat import (  # noqa: E402
    FLEET_ROLES,
    build_fleet_compatibility_receipt,
)


def _role_root(value: str) -> tuple[str, Path]:
    role, separator, raw_path = str(value).partition("=")
    role = role.strip().lower()
    if not separator or role not in FLEET_ROLES or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            f"expected ROLE=PATH with ROLE in {FLEET_ROLES}, got {value!r}"
        )
    return role, Path(raw_path).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=ROOT)
    parser.add_argument("--root", action="append", type=_role_root, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = dict(args.root)
    if len(roots) != len(args.root):
        raise SystemExit("duplicate loader role")
    output = build_fleet_compatibility_receipt(
        checkpoint_path=args.checkpoint,
        reference_root=args.reference_root,
        roots=roots,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": "ALL_DORMANT_LOADERS_COMPATIBLE",
                "receipt": str(output),
                "roles": list(FLEET_ROLES),
                "runtime_enabled": False,
                "training_enabled": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
