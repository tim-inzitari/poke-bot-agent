#!/usr/bin/env python3
"""Build an anonymous public-deck prior for submission-time belief search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "poke_bot.submission_belief_decks/v1"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build(sources: list[Path]) -> dict[str, Any]:
    unique: set[tuple[int, ...]] = set()
    source_digests: list[str] = []
    for source in sources:
        source = source.expanduser().resolve()
        value = json.loads(source.read_text(encoding="utf-8"))
        decks = value.get("decks")
        if not isinstance(decks, dict):
            raise RuntimeError(f"representative source has no decks: {source}")
        source_digests.append(_sha256(source))
        for row in decks.values():
            cards = tuple(sorted(int(card) for card in row.get("card_ids") or ()))
            if len(cards) != 60 or any(card <= 0 for card in cards):
                raise RuntimeError("every posterior hypothesis must contain 60 cards")
            unique.add(cards)
    if len(unique) < 8:
        raise RuntimeError("anonymous belief prior has too few deck hypotheses")
    return {
        "schema": SCHEMA,
        "anonymous": True,
        "contains_opponent_identity": False,
        "deck_count": len(unique),
        "deck_lists": [list(cards) for cards in sorted(unique)],
        "source_sha256": sorted(source_digests),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    args = parser.parse_args()
    payload = build(list(args.source))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        f"OK: anonymous belief posterior decks={payload['deck_count']} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
