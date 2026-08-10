#!/usr/bin/env python3
"""Resolve owner Alakazam archetype list card IDs (r175)."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CSV_PATH = Path("/home/inzi/poke-bot-agent/cards/EN_Card_Data.csv")
OUT_CSV = Path(
    "/home/inzi/poke-bot-agent/decks/archetype-samples/"
    "alakazam-owner-rtp-pilot-r175.csv"
)

# Prefer current Alakazam-lineage printings already used in H10 decks.
PREFERRED_IDS = {
    "Abra": 741,
    "Kadabra": 742,
    "Alakazam": 743,
    "Dunsparce": 305,
    "Dudunsparce": 66,
    "Fezandipiti ex": 140,
    "Shaymin": 343,
    "Battle Cage": 1264,
    "Buddy-Buddy Poffin": 1086,
    "Dawn": 1231,
    "Enhanced Hammer": 1081,
    "Hilda": 1225,
    "Poké Pad": 1152,
    "Rare Candy": 1079,
    "Night Stretcher": 1097,
    "Sacred Ash": 1129,
    "Telepath Psychic Energy": 19,
    "Enriching Energy": 13,
}

OWNER_COUNTS = {
    "Abra": 4,
    "Kadabra": 4,
    "Alakazam": 3,
    "Dunsparce": 3,
    # Owner prose said 2× Dudunsparce but also Pokémon(19) and a 60-card
    # list; 2× yields 18 Pokémon / 59 cards. Promote to 3× to satisfy the
    # stated 19/60 invariants (matches prior H10 Alakazam dudunsparce lines).
    "Dudunsparce": 3,
    "Fezandipiti ex": 1,
    "Shaymin": 1,
    "Battle Cage": 4,
    "Buddy-Buddy Poffin": 4,
    "Dawn": 4,
    "Enhanced Hammer": 3,
    "Hilda": 4,
    "Poké Pad": 4,
    "Rare Candy": 3,
    "Night Stretcher": 2,
    "Boss's Orders": 2,
    "Xerosic's Mechinations": 2,  # owner spelling; data may use Machinations
    "Lana's Aid": 1,
    "Sacred Ash": 1,
    "Telepath Psychic Energy": 4,
    "Basic Psychic Energy": 2,
    "Enriching Energy": 1,
}

OWNER_ALIASES = {
    "Xerosic's Mechinations": [
        "Xerosic's Mechinations",
        "Xerosic's Machinations",
        "Xerosic\u2019s Machinations",
        "Xerosic\u2019s Mechinations",
    ],
    "Boss's Orders": [
        "Boss's Orders",
        "Boss\u2019s Orders",
    ],
    "Lana's Aid": [
        "Lana's Aid",
        "Lana\u2019s Aid",
    ],
    "Basic Psychic Energy": [
        "Basic Psychic Energy",
        "Psychic Energy",
        "Basic {P} Energy",
    ],
    "Poké Pad": ["Poké Pad", "Poke Pad"],
}

# Known canonical IDs for aliases that NFKC/apostrophe normalization alone
# cannot uniquely resolve (engine uses brace energy notation).
PREFERRED_IDS.update(
    {
        "Basic Psychic Energy": 5,
        "Boss's Orders": 1182,
        "Lana's Aid": 1184,
        "Xerosic's Mechinations": 1197,
    }
)


def norm(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("`", "'")
    )
    return re.sub(r"\s+", " ", text).strip().casefold()


def main() -> int:
    id_to_name: dict[int, str] = {}
    name_to_ids: dict[str, list[int]] = defaultdict(list)
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_id = row.get("Card ID")
            name = row.get("Card Name")
            if not raw_id or not name:
                continue
            try:
                card_id = int(raw_id)
            except ValueError:
                continue
            if card_id in id_to_name:
                continue
            id_to_name[card_id] = name
            name_to_ids[name].append(card_id)

    norm_map = {
        norm(name): (name, sorted(set(ids)))
        for name, ids in name_to_ids.items()
    }

    resolved: list[int] = []
    report: list[dict[str, object]] = []
    for owner_name, count in OWNER_COUNTS.items():
        aliases = OWNER_ALIASES.get(owner_name, [owner_name])
        found = None
        for alias in aliases:
            hit = norm_map.get(norm(alias))
            if hit is not None:
                found = hit
                break
        if found is None:
            raise SystemExit(f"FAIL_CLOSED missing card name: {owner_name!r}")
        data_name, ids = found
        preferred = PREFERRED_IDS.get(owner_name)
        if preferred is not None and preferred in ids:
            card_id = preferred
        elif len(ids) == 1:
            card_id = ids[0]
        else:
            raise SystemExit(
                f"FAIL_CLOSED ambiguous card {owner_name!r}: {ids} "
                f"(data_name={data_name!r})"
            )
        # Bind remaining previously-missing names into prefer map for report.
        if owner_name not in PREFERRED_IDS:
            PREFERRED_IDS[owner_name] = card_id
        resolved.extend([card_id] * count)
        report.append(
            {
                "owner_name": owner_name,
                "data_name": data_name,
                "count": count,
                "card_id": card_id,
                "spelling_mismatch": owner_name != data_name,
            }
        )

    if len(resolved) != 60:
        raise SystemExit(f"FAIL_CLOSED deck size {len(resolved)} != 60")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text("\n".join(str(card_id) for card_id in resolved) + "\n")
    print("deck_path", OUT_CSV)
    print("card_count", len(resolved))
    print("multiset", dict(sorted(Counter(resolved).items())))
    for row in report:
        flag = " SPELLING_MISMATCH" if row["spelling_mismatch"] else ""
        print(
            f"{row['count']}x {row['card_id']} "
            f"owner={row['owner_name']!r} data={row['data_name']!r}{flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
