#!/usr/bin/env python3
"""Validate and seal the final r253 BO1000 evidence and human review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from poke_bot.r253_bo1000_completion_audit import (
    build_completion_audit,
    create_once,
    render_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--audit-name", default="r253-completion-audit.json")
    parser.add_argument("--review-name", default="r253-final-review.md")
    args = parser.parse_args(argv)

    root = args.run_root.resolve()
    audit = build_completion_audit(root)
    encoded = (
        json.dumps(audit, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    audit_sha = "sha256:" + hashlib.sha256(encoded).hexdigest()
    review = render_markdown(audit, audit_sha256=audit_sha).encode()
    create_once(root / args.audit_name, encoded)
    create_once(root / args.review_name, review)
    print(json.dumps({
        "audit_path": str(root / args.audit_name),
        "audit_sha256": audit_sha,
        "review_path": str(root / args.review_name),
        "review_sha256": "sha256:" + hashlib.sha256(review).hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
