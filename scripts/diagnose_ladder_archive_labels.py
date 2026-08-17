#!/usr/bin/env python3
"""Print classifier label coverage for a bounded archive prefix."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.ladder_replay import LadderReplayClassifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--card-csv", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    classifier = LadderReplayClassifier.from_paths(
        args.mix, args.representatives, card_csv=args.card_csv
    )
    labels: collections.Counter[str] = collections.Counter()
    errors: collections.Counter[str] = collections.Counter()
    with zipfile.ZipFile(args.archive) as archive:
        members = [
            name for name in archive.namelist() if name.endswith(".json")
        ][: max(0, args.limit)]
        for member in members:
            try:
                payload = json.loads(archive.read(member))
                _decks, classified = classifier.classify_episode(payload)
                labels.update(str(label) for label in classified)
            except Exception as exc:  # diagnostic aggregation
                errors[type(exc).__name__] += 1
    print(
        json.dumps(
            {
                "members": len(members),
                "labels": dict(labels),
                "errors": dict(errors),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
