#!/usr/bin/env python3
"""Resolve checksum-bound registry inputs and atomically activate Marnie family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetype_family_activation import validate_activation_request  # noqa: E402
from scripts.activate_marnie_archetype_family import activate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--environment-drop-in", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    args = parser.parse_args()
    request = args.request.expanduser().resolve()
    payload = json.loads(request.read_text(encoding="utf-8"))
    validate_activation_request(payload)
    bindings = dict(payload.get("bindings") or {})
    before = Path(str((bindings.get("registry") or {}).get("path") or "")).resolve()
    candidate = Path(
        str((bindings.get("candidate_registry") or {}).get("path") or "")
    ).resolve()
    result = activate(
        pause=args.pause.expanduser().resolve(),
        request=request,
        before_registry=before,
        candidate_registry=candidate,
        output_registry=args.output_registry.expanduser().resolve(),
        environment_drop_in=args.environment_drop_in.expanduser().resolve(),
        receipt=args.receipt.expanduser().resolve(),
        python_executable=args.python_executable.expanduser().resolve(),
        launcher=args.launcher.expanduser().resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
