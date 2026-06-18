"""Resolver: PTCGL decklist export text -> validated list of 60 in-pool Card IDs.

Card master: cards/EN_Card_Data.csv (UTF-8 BOM). A Card ID is legal iff it
appears there. See docs/decklist-collection-spec.md.
"""
import csv
import os
import re
import unicodedata

CARD_CSV = os.path.join(os.path.dirname(__file__), "..", "cards", "EN_Card_Data.csv")

# Basic energy: Card ID 1-8 (SVE 1-8). Map energy-type words -> Card ID.
ENERGY_NAME_TO_ID = {
    "grass": 1, "g": 1,
    "fire": 2, "r": 2,
    "water": 3, "w": 3,
    "lightning": 4, "l": 4,
    "psychic": 5, "p": 5,
    "fighting": 6, "f": 6,
    "darkness": 7, "dark": 7, "d": 7,
    "metal": 8, "m": 8,
}


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    return " ".join(s.lower().split())


class Resolver:
    def __init__(self, card_csv: str = CARD_CSV):
        self.by_setnum = {}          # (EXP_upper, num_str) -> Card ID
        self.by_name = {}            # normalized name -> set of Card IDs
        self.id_to_name = {}         # Card ID -> raw Card Name
        with open(card_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                cid = int(row["Card ID"])
                if cid in self.id_to_name:
                    continue  # dedupe: multiple rows per card (one per move)
                name = row["Card Name"]
                exp = row["Expansion"].strip().upper()
                num = row["Collection No."].strip()
                self.id_to_name[cid] = name
                if exp and num:
                    self.by_setnum[(exp, num)] = cid
                self.by_name.setdefault(normalize_name(name), set()).add(cid)

    def _resolve_basic_energy(self, name: str):
        """Return Card ID for a basic energy line, else None."""
        n = normalize_name(name)
        # forms: "basic fire energy", "fire energy", "basic {r} energy", "fire"
        n = n.replace("{", " ").replace("}", " ")
        n = re.sub(r"\benergy\b", " ", n)
        n = re.sub(r"\bbasic\b", " ", n)
        words = n.split()
        for w in words:
            if w in ENERGY_NAME_TO_ID:
                return ENERGY_NAME_TO_ID[w]
        return None

    def resolve_card(self, name: str, setcode: str, number: str):
        """Resolve one card line -> (card_id, None) or (None, reason)."""
        setcode_u = (setcode or "").strip().upper()
        number_s = (number or "").strip()
        # 1. primary key (Expansion, Collection No.)
        if setcode_u and number_s and (setcode_u, number_s) in self.by_setnum:
            return self.by_setnum[(setcode_u, number_s)], None
        # 2. basic energy by type (SVE basics, or set-less energy lines)
        nm = normalize_name(name)
        if "energy" in nm and ("basic" in nm or len(nm.split()) <= 2):
            eid = self._resolve_basic_energy(name)
            if eid is not None:
                return eid, None
        # 3. name fallback (exactly one in-pool ID)
        ids = self.by_name.get(nm)
        if ids and len(ids) == 1:
            return next(iter(ids)), None
        if ids and len(ids) > 1:
            return None, f'ambiguous name "{name}" {setcode} {number} -> {len(ids)} IDs'
        return None, f'unresolved: "{name}" {setcode} {number} not in pool'


LINE_RE = re.compile(r"^\s*(?:\*\s*)?(\d+)\s+(.+?)\s+([A-Z]{2,4})\s+([A-Za-z]?\d+[A-Za-z]?)\s*$")
LINE_NOSET_RE = re.compile(r"^\s*(?:\*\s*)?(\d+)\s+(.+?)\s*$")
HEADER_RE = re.compile(r"^\s*(Pok[eé]?mon|Trainer|Energy|Total)\b.*:?\s*\d*\s*$", re.IGNORECASE)


def parse_export(text: str):
    """Parse PTCGL export text -> list of (qty, name, setcode, number).

    setcode/number may be '' for set-less energy lines. Returns (cards, parse_errors).
    """
    cards = []
    errors = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if HEADER_RE.match(line):
            continue
        if line.lower().startswith(("pokemon", "pokémon", "trainer", "energy", "total", "deck", "####")):
            # category header without count
            if re.match(r"^[A-Za-z][A-Za-z ]+:?\s*\d*$", line):
                continue
        m = LINE_RE.match(line)
        if m:
            qty, name, setcode, number = m.groups()
            cards.append((int(qty), name.strip(), setcode, number))
            continue
        m = LINE_NOSET_RE.match(line)
        if m and "energy" in m.group(2).lower():
            qty, name = m.groups()
            cards.append((int(qty), name.strip(), "", ""))
            continue
        errors.append(f"unparseable line: {raw!r}")
    return cards, errors


def resolve_deck(text: str, resolver: Resolver):
    """Resolve a full deck export. Returns dict with ids, total, errors, ok."""
    cards, parse_errors = parse_export(text)
    errors = list(parse_errors)
    ids = []
    name_counts = {}  # normalized name -> qty (for 4-copy rule, energy exempt)
    for qty, name, setcode, number in cards:
        cid, reason = resolver.resolve_card(name, setcode, number)
        if reason:
            errors.append(reason)
            continue
        ids.extend([cid] * qty)
        if not (1 <= cid <= 8):  # basic energy exempt from 4-copy rule
            name_counts[normalize_name(name)] = name_counts.get(normalize_name(name), 0) + qty
    total = sum(qty for qty, *_ in cards)
    over = {n: c for n, c in name_counts.items() if c > 4}
    if over:
        errors.append("over-4-copies: " + ", ".join(f"{n}={c}" for n, c in over.items()))
    if total != 60:
        errors.append(f"deck has {total} cards, not 60")
    if len(ids) != 60 and not errors:
        errors.append(f"resolved {len(ids)} ids, not 60")
    ok = (not errors) and len(ids) == 60
    return {"ids": ids, "total": total, "errors": errors, "ok": ok}


if __name__ == "__main__":
    r = Resolver()
    print("setnum index:", len(r.by_setnum), "names:", len(r.by_name), "ids:", len(r.id_to_name))
