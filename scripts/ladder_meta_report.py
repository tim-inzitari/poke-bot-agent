#!/usr/bin/env python
"""Ladder meta report from daily episode datasets (Phase 6 priority by prevalence).

Labels each seat:
  1. ``poke_bot.archetypes.classify_deck`` on reconstructed 60-card setup decks
  2. Deck-signature families for non-Dragapult meta (Starmie, Lucario, …)
  3. Fallback: highest-HP ex Pokemon played (busyaprime-style), support ex demoted

Writes ``outputs/reports/ladder_meta.md`` + ``.csv`` and refreshes
``outputs/notes/phase6_priority.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import archetypes
from poke_bot.replay_import import classify_episode_seats, extract_setup_decks

SEED = 12345
DEFAULT_MAX_GAMES = 5000
CARD_CSV = ROOT / "cards" / "EN_Card_Data.csv"
REPORT_MD = ROOT / "outputs" / "reports" / "ladder_meta.md"
REPORT_CSV = ROOT / "outputs" / "reports" / "ladder_meta.csv"
PHASE6_NOTE = ROOT / "outputs" / "notes" / "phase6_priority.md"

# Support / tech ex that should not define the archetype ace.
SUPPORT_EX_IDS = {
    140,   # Fezandipiti ex
    1071,  # Meowth ex
    184,   # Latias ex
    754,   # Mega Latias ex
    272,   # Lillie's Clefairy ex
}

# Non-Dragapult deck signatures (checked after classify_deck). Order = specificity.
# Each entry: (archetype_id, required_any_of_card_ids, optional_extra_any)
META_SIGNATURES: list[tuple[str, set[int], set[int]]] = [
    ("starmie", {1031}, set()),                          # Mega Starmie ex
    ("lucario", {678}, set()),                           # Mega Lucario ex
    ("garchomp", {381}, set()),                          # Cynthia's Garchomp ex
    ("rockets-mewtwo", {431}, set()),                    # Team Rocket's Mewtwo ex
    ("ns-zoroark", {293}, set()),                        # N's Zoroark ex
    ("hydrapple", {150}, set()),                         # Hydrapple ex
    ("raging-bolt", {63}, set()),                        # Raging Bolt ex
    ("gardevoir", {747}, set()),                         # Mega Gardevoir ex
    ("abomasnow", {723}, set()),                         # Mega Abomasnow ex
    ("lopunny", {849}, set()),                           # Mega Lopunny ex
    ("charizard", {790, 928}, set()),                    # Mega Charizard X/Y
    ("gengar", {772}, set()),                            # Mega Gengar ex
    ("ogerpon-meganium", {96}, set()),                   # Teal Mask Ogerpon (meganium shells often)
    ("cornerstone-ogerpon", {117}, set()),               # Cornerstone Mask Ogerpon
    ("clefairy", {272}, set()),                          # Lillie's Clefairy as lead
    ("okidogi", {116, 138, 890}, {1052}),                # Okidogi (+ Barbaracle hint)
    ("crustle", {345, 533}, set()),
    ("slowking", {163}, set()),
    ("alakazam", {245, 743}, set()),
    ("festival-lead", {1245}, set()),                    # Festival Grounds stadium
    ("slaking", {232}, set()),
    ("mamoswine", {283}, set()),
    ("palafin", {107}, set()),
]

# Ace-name → archetype_id for busyaprime fallback.
ACE_NAME_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"dragapult", re.I), "dragapult"),  # rare if classify_deck missed
    (re.compile(r"starmie", re.I), "starmie"),
    (re.compile(r"lucario", re.I), "lucario"),
    (re.compile(r"garchomp", re.I), "garchomp"),
    (re.compile(r"mewtwo", re.I), "rockets-mewtwo"),
    (re.compile(r"zoroark", re.I), "ns-zoroark"),
    (re.compile(r"hydrapple", re.I), "hydrapple"),
    (re.compile(r"raging\s*bolt", re.I), "raging-bolt"),
    (re.compile(r"gardevoir", re.I), "gardevoir"),
    (re.compile(r"abomasnow", re.I), "abomasnow"),
    (re.compile(r"lopunny", re.I), "lopunny"),
    (re.compile(r"charizard", re.I), "charizard"),
    (re.compile(r"gengar", re.I), "gengar"),
    (re.compile(r"froslass", re.I), "starmie"),  # often paired; starmie family
    (re.compile(r"teal\s*mask\s*ogerpon", re.I), "ogerpon-meganium"),
    (re.compile(r"cornerstone\s*mask\s*ogerpon", re.I), "cornerstone-ogerpon"),
    (re.compile(r"ogerpon", re.I), "ogerpon"),
    (re.compile(r"clefairy", re.I), "clefairy"),
    (re.compile(r"okidogi", re.I), "okidogi"),
    (re.compile(r"slaking", re.I), "slaking"),
    (re.compile(r"mamoswine", re.I), "mamoswine"),
    (re.compile(r"palafin", re.I), "palafin"),
    (re.compile(r"hariyama", re.I), "lucario"),
]


def load_card_db(path: Path = CARD_CSV) -> tuple[dict[int, str], dict[int, float], dict[int, bool]]:
    id2name: dict[int, str] = {}
    id2hp: dict[int, float] = {}
    is_ace: dict[int, bool] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                cid = int(row["Card ID"])
            except (ValueError, TypeError, KeyError):
                continue
            nm = str(row.get("Card Name", "") or "")
            try:
                hp = float(row.get("HP") or 0)
            except ValueError:
                hp = 0.0
            id2name[cid] = nm
            id2hp[cid] = hp
            is_ace[cid] = bool(re.search(r"\bex\b", nm, re.I)) and hp > 0
    return id2name, id2hp, is_ace


def classify_meta_deck(card_ids: Iterable[int]) -> str:
    """Signature classify for non-Dragapult families (after classify_deck fails)."""
    counts = Counter(int(c) for c in card_ids)
    for arch_id, required, bonus in META_SIGNATURES:
        if not any(counts.get(cid, 0) > 0 for cid in required):
            continue
        if bonus and not any(counts.get(cid, 0) > 0 for cid in bonus):
            # bonus is a soft hint — still accept if required matched alone
            # (okidogi works without barbaracle)
            pass
        return arch_id
    return archetypes.UNKNOWN


def label_seat(
    deck: Optional[list[int]],
    played: list[int],
    id2name: dict[int, str],
    id2hp: dict[int, float],
    is_ace: dict[int, bool],
) -> tuple[str, str]:
    """Return (archetype_id, method)."""
    if deck is not None:
        arch = archetypes.classify_deck(deck)
        if arch != archetypes.UNKNOWN:
            return arch, "classify_deck"
        meta = classify_meta_deck(deck)
        if meta != archetypes.UNKNOWN:
            return meta, "deck_signature"

    # Ace fallback from cards played (busyaprime), demote support ex.
    primary: list[tuple[float, int]] = []
    support: list[tuple[float, int]] = []
    for cid in played:
        if not is_ace.get(cid):
            continue
        pair = (id2hp.get(cid, 0.0), cid)
        if cid in SUPPORT_EX_IDS:
            support.append(pair)
        else:
            primary.append(pair)
    pool = primary or support
    if not pool:
        return archetypes.UNKNOWN, "no_ace"

    ace_id = max(pool)[1]
    ace_name = id2name.get(ace_id, f"card_{ace_id}")
    for pat, arch_id in ACE_NAME_MAP:
        if pat.search(ace_name):
            # If ace says dragapult but deck wasn't classified, treat as pure dragapult
            # unless hammer/tech lines appear in played cards.
            if arch_id == "dragapult" and deck is not None:
                return archetypes.classify_deck(deck) if archetypes.classify_deck(deck) != archetypes.UNKNOWN else "dragapult", "ace_fallback"
            return arch_id, "ace_fallback"
    # Slugify ace name
    slug = re.sub(r"[^a-z0-9]+", "-", ace_name.lower()).strip("-")
    return slug or archetypes.UNKNOWN, "ace_name"


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (p_hat, lo, hi)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return p, max(0.0, lo), min(1.0, hi)


def collect_episode_paths(sources: list[Path]) -> list[Path]:
    files: list[Path] = []
    for src in sources:
        if src.is_file() and src.suffix == ".zip":
            continue  # handled separately via ZipMember
        if src.is_dir():
            for p in src.rglob("*.json"):
                stem = p.stem
                if stem.isdigit():
                    files.append(p)
    return sorted(set(files))


def iter_zip_members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return [
            n
            for n in zf.namelist()
            if n.endswith(".json") and Path(n).stem.isdigit()
        ]


def load_payload_from_zip(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    return json.loads(zf.read(name))


def parse_episode(
    payload: dict[str, Any],
    id2name: dict[int, str],
    id2hp: dict[int, float],
    is_ace: dict[int, bool],
) -> Optional[dict[str, Any]]:
    rewards = payload.get("rewards") or []
    if len(rewards) != 2 or rewards[0] is None or rewards[1] is None or rewards[0] == rewards[1]:
        return None

    winner = 0 if float(rewards[0]) > float(rewards[1]) else 1
    steps = payload.get("steps") or []
    total_steps = len(steps)
    played: list[list[int]] = [[], []]
    for st in steps:
        if not isinstance(st, list):
            continue
        for i in range(min(2, len(st))):
            seat = st[i]
            if not isinstance(seat, dict):
                continue
            obs = seat.get("observation") or {}
            for lg in obs.get("logs") or []:
                if lg.get("playerIndex") == i and lg.get("cardId") is not None:
                    played[i].append(int(lg["cardId"]))

    decks, _ = classify_episode_seats(payload)
    labels: list[str] = []
    methods: list[str] = []
    for i in range(2):
        arch, method = label_seat(decks[i], played[i], id2name, id2hp, is_ace)
        labels.append(arch)
        methods.append(method)

    return {
        "total_steps": total_steps,
        "winner": winner,
        "arch": labels,
        "method": methods,
        "has_deck": [d is not None for d in decks],
    }


def fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "—"
    return f"{100.0 * x:.1f}%"


def _iter_day_entries(episode_dir: Optional[Path], day_label: Optional[str]) -> tuple[list[tuple[str, Any]], str, dict[Path, zipfile.ZipFile]]:
    """Build a streaming entry list for a single day.

    Prefers an extracted directory; otherwise falls back to the daily zip. Returns
    ``(entries, source_desc, zip_handles)`` where each entry is ``("file", path)``
    or ``("zip", (zip_path, member))``.
    """
    zip_handles: dict[Path, zipfile.ZipFile] = {}
    # 1) explicit dir, 2) shared extracted day dir, 3) daily zip on disk.
    if episode_dir is None:
        cand = Path("/tmp/replay-len/day1")
        episode_dir = cand if cand.is_dir() else None

    if episode_dir is not None and episode_dir.is_dir():
        files = collect_episode_paths([episode_dir])
        if files:
            return [("file", fp) for fp in files], str(episode_dir), zip_handles

    if day_label:
        zp = ROOT / "data" / "episodes" / "raw" / f"pokemon-tcg-ai-battle-episodes-{day_label}.zip"
        if zp.is_file():
            zip_handles[zp] = zipfile.ZipFile(zp)
            return [("zip", (zp, m)) for m in iter_zip_members(zp)], str(zp), zip_handles

    return [], "", zip_handles


def day_report_main(args: argparse.Namespace) -> int:
    """Scoped, streaming archetype-prevalence report for a single ladder day."""
    id2name, id2hp, is_ace = load_card_db()

    entries, src_desc, zip_handles = _iter_day_entries(args.episode_dir, args.day_label)
    if not entries:
        print("No episode sources found for day-mode.", file=sys.stderr)
        return 1

    day_label = args.day_label or "unknown-day"
    n_available = len(entries)
    entries = sorted(entries, key=lambda e: str(e[1]))
    # Optional bound (default DEFAULT_MAX_GAMES is 5000; raise via --max-games for a full day).
    sampled = False
    if args.max_games and args.max_games > 0 and len(entries) > args.max_games:
        rng = random.Random(args.seed)
        entries = sorted(rng.sample(entries, args.max_games), key=lambda e: str(e[1]))
        sampled = True

    print(f"# day={day_label} source={src_desc}")
    print(f"# available={n_available} processing={len(entries)} sampled={sampled}")

    n_games = 0
    dropped = 0
    seat_appearances: Counter[str] = Counter()   # both-seat appearances
    games_present: Counter[str] = Counter()       # distinct games featuring archetype
    wins: Counter[str] = Counter()                # seat-level wins
    method_counts: Counter[str] = Counter()
    has_deck_seats = 0
    total_seats = 0
    matchup_pairs: Counter[tuple[str, str]] = Counter()  # unordered pair frequency

    for k, (kind, ref) in enumerate(entries, 1):
        try:
            if kind == "file":
                payload = json.loads(Path(ref).read_text(encoding="utf-8"))
            else:
                zp, member = ref
                payload = load_payload_from_zip(zip_handles[zp], member)
        except (OSError, json.JSONDecodeError, KeyError, zipfile.BadZipFile):
            dropped += 1
            continue
        parsed = parse_episode(payload, id2name, id2hp, is_ace)
        del payload  # keep RAM bounded — do not retain episodes
        if parsed is None:
            dropped += 1
            continue
        n_games += 1
        a0, a1 = parsed["arch"]
        w = parsed["winner"]
        for i, arch in enumerate((a0, a1)):
            seat_appearances[arch] += 1
            total_seats += 1
            if w == i:
                wins[arch] += 1
            if parsed["has_deck"][i]:
                has_deck_seats += 1
        for arch in {a0, a1}:
            games_present[arch] += 1
        for m in parsed["method"]:
            method_counts[m] += 1
        matchup_pairs[tuple(sorted((a0, a1)))] += 1
        if k % 500 == 0:
            print(f"  parsed {k}/{len(entries)} decisive={n_games} dropped={dropped}", flush=True)

    for zf in zip_handles.values():
        zf.close()

    if n_games == 0:
        print("No decisive games parsed.", file=sys.stderr)
        return 1

    UNK = archetypes.UNKNOWN
    rows: list[dict[str, Any]] = []
    for arch, seats in seat_appearances.most_common():
        w = wins.get(arch, 0)
        p, lo, hi = wilson_interval(w, seats)
        rows.append({
            "archetype": arch,
            "seats": seats,
            "share": seats / total_seats,
            "games": games_present.get(arch, 0),
            "game_share": games_present.get(arch, 0) / n_games,
            "wins": w,
            "wr": p,
            "wr_lo": lo,
            "wr_hi": hi,
        })

    unknown_seats = seat_appearances.get(UNK, 0)
    unknown_games = games_present.get(UNK, 0)
    # Registered/recognised labels = Dragapult family + hammer + META_SIGNATURES ids.
    recognised = set(archetypes.archetype_ids()) | {sig[0] for sig in META_SIGNATURES}
    longtail_seats = sum(
        s for a, s in seat_appearances.items() if a != UNK and a not in recognised
    )
    longtail_labels = sum(
        1 for a in seat_appearances if a != UNK and a not in recognised
    )

    # --- write CSV ---
    report_csv = args.report_csv_out or (ROOT / "outputs" / "reports" / f"archetype_meta_{day_label}.csv")
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "archetype", "seats", "share", "games", "game_share", "wins", "wr", "wr_lo", "wr_hi"],
        )
        writer.writeheader()
        for i, r in enumerate(rows, 1):
            writer.writerow({
                "rank": i,
                "archetype": r["archetype"],
                "seats": r["seats"],
                "share": f"{r['share']:.6f}",
                "games": r["games"],
                "game_share": f"{r['game_share']:.6f}",
                "wins": r["wins"],
                "wr": f"{r['wr']:.6f}",
                "wr_lo": f"{r['wr_lo']:.6f}",
                "wr_hi": f"{r['wr_hi']:.6f}",
            })

    # --- markdown ---
    report_md = args.report_md or (ROOT / "outputs" / "reports" / f"archetype_meta_{day_label}.md")
    lines: list[str] = []
    lines.append(f"# Archetype prevalence report — {day_label}")
    lines.append("")
    lines.append(
        f"Full archetype-prevalence breakdown for the most recent ladder day available on disk "
        f"(`{day_label}`). One row per archetype; prevalence is share of **both-seat appearances** "
        f"across all decisive games."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Ladder day (window) | **{day_label}** (single UTC calendar day) |")
    lines.append(f"| Dataset | `kaggle/pokemon-tcg-ai-battle-episodes-{day_label}` |")
    lines.append(f"| Source | `{src_desc}` |")
    lines.append(f"| Episodes available | {n_available} |")
    lines.append(f"| Episodes processed | {len(entries)}{' (sampled)' if sampled else ' (full day)'} |")
    lines.append(f"| Decisive games | **{n_games}** (dropped tie/crash/malformed: {dropped}) |")
    lines.append(f"| Both-seat appearances (total) | **{total_seats}** (= 2 × decisive games) |")
    lines.append(f"| Seats with reconstructed 60-card deck | {has_deck_seats} ({fmt_pct(has_deck_seats/total_seats)}) |")
    lines.append(
        "| Labeling | `archetypes.classify_deck` → deck signatures → highest-HP ex played "
        "(support ex demoted) |"
    )
    lines.append(
        "| Label methods | " + ", ".join(f"`{m}`={c}" for m, c in method_counts.most_common()) + " |"
    )
    lines.append("")
    lines.append(
        "Dragapult variants are kept **separate** (`dragapult`, `dragapult-dudunsparce`, "
        "`dragapult-dusknoir`, `dragapult-blaziken`, `hammer-pult`)."
    )
    lines.append("")
    lines.append("## Archetype prevalence (all archetypes, by both-seat appearances)")
    lines.append("")
    lines.append("| Rank | Archetype | Seat appearances | Prevalence (share) | Games featuring | Game share | Win rate | Wilson 95% |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{r['archetype']}` | {r['seats']} | {fmt_pct(r['share'])} | "
            f"{r['games']} | {fmt_pct(r['game_share'])} | {fmt_pct(r['wr'])} | "
            f"{fmt_pct(r['wr_lo'])}–{fmt_pct(r['wr_hi'])} |"
        )
    lines.append("")
    lines.append("## Unclassified / other")
    lines.append("")
    lines.append(
        f"- **Unclassified (`unknown`)**: {unknown_seats} seat appearances "
        f"({fmt_pct(unknown_seats/total_seats)} of seats), in {unknown_games} games "
        f"({fmt_pct(unknown_games/n_games)} of games). These are seats with no reconstructable "
        f"deck and no ex ace to fall back on."
    )
    lines.append(
        f"- **Long-tail / \"other\" named aces** (labeled by highest-HP ex but not a registered "
        f"Dragapult-family or deck-signature archetype): {longtail_seats} seat appearances "
        f"({fmt_pct(longtail_seats/total_seats)}) spread across {longtail_labels} distinct ace labels."
    )
    lines.append(
        f"- **Recognised archetypes** (Dragapult family + tracked meta signatures): "
        f"{fmt_pct((total_seats - unknown_seats - longtail_seats)/total_seats)} of seats."
    )
    lines.append("")
    lines.append("## Matchup frequency (most common pairings)")
    lines.append("")
    lines.append("| Rank | Matchup | Games | % of games |")
    lines.append("|---:|---|---:|---:|")
    for i, (pair, c) in enumerate(matchup_pairs.most_common(15), 1):
        a, b = pair
        label = f"`{a}` mirror" if a == b else f"`{a}` vs `{b}`"
        lines.append(f"| {i} | {label} | {c} | {fmt_pct(c/n_games)} |")
    lines.append("")
    lines.append(f"Full ranked table: [`{report_csv.name}`]({report_csv.name})")
    lines.append("")

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_md}")
    print(f"wrote {report_csv}")

    print(f"\n=== ARCHETYPE PREVALENCE — {day_label} (n_games={n_games}, seats={total_seats}) ===")
    for i, r in enumerate(rows, 1):
        print(
            f"{i:2d}. {r['archetype']:26s} seats={r['seats']:5d} "
            f"share={fmt_pct(r['share']):>6s} games={r['games']:5d} WR={fmt_pct(r['wr']):>6s}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sources",
        nargs="*",
        type=Path,
        default=None,
        help="Dirs of episode JSON and/or daily zips (default: raw zip + /tmp/replay-len/day1).",
    )
    p.add_argument("--max-games", type=int, default=DEFAULT_MAX_GAMES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-download", action="store_true", default=True)
    # --- scoped single-day report mode (does NOT touch the fixed ladder_meta.* /
    #     phase6_priority.md outputs). Streams one episode at a time. ---
    p.add_argument("--day-mode", action="store_true", help="Emit a scoped archetype-prevalence report for one day.")
    p.add_argument("--day-label", type=str, default=None, help="Calendar day label, e.g. 2026-07-12.")
    p.add_argument("--episode-dir", type=Path, default=None, help="Directory of extracted episode JSON for the day.")
    p.add_argument("--report-md", type=Path, default=None, help="Output markdown path (day-mode).")
    p.add_argument("--report-csv-out", type=Path, default=None, help="Output CSV path (day-mode).")
    args = p.parse_args(argv)

    if args.day_mode:
        return day_report_main(args)

    id2name, id2hp, is_ace = load_card_db()

    default_zip = ROOT / "data" / "episodes" / "raw" / "pokemon-tcg-ai-battle-episodes-2026-07-12.zip"
    default_day1 = Path("/tmp/replay-len/day1")
    sources = args.sources or ([default_day1] if default_day1.is_dir() else [default_zip])

    # Prefer extracted dir for speed; fall back to zip.
    file_paths = collect_episode_paths([s for s in sources if s.is_dir()])
    zip_paths = [s for s in sources if s.is_file() and s.suffix == ".zip"]
    if not file_paths and default_zip.is_file() and default_zip not in zip_paths:
        zip_paths.append(default_zip)

    # Build unified sample list: either Path or (zip, member)
    entries: list[tuple[str, Any]] = [("file", fp) for fp in file_paths]
    zip_handles: dict[Path, zipfile.ZipFile] = {}
    for zp in zip_paths:
        members = iter_zip_members(zp)
        # Avoid double-counting if we already have extracted files from same day.
        if file_paths and len(file_paths) >= min(args.max_games, len(members) // 2):
            continue
        zip_handles[zp] = zipfile.ZipFile(zp)
        for m in members:
            entries.append(("zip", (zp, m)))

    if not entries:
        print("No episode sources found.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    if len(entries) > args.max_games:
        entries = rng.sample(entries, args.max_games)
    entries = sorted(entries, key=lambda e: str(e[1]))

    print(f"# sources={sources}")
    print(f"# sampled={len(entries)} seed={args.seed}")

    games: list[dict[str, Any]] = []
    dropped = 0
    method_counts: Counter[str] = Counter()
    for k, (kind, ref) in enumerate(entries, 1):
        try:
            if kind == "file":
                payload = json.loads(Path(ref).read_text(encoding="utf-8"))
            else:
                zp, member = ref
                payload = load_payload_from_zip(zip_handles[zp], member)
        except (OSError, json.JSONDecodeError, KeyError, zipfile.BadZipFile):
            dropped += 1
            continue
        parsed = parse_episode(payload, id2name, id2hp, is_ace)
        if parsed is None:
            dropped += 1
        else:
            games.append(parsed)
            for m in parsed["method"]:
                method_counts[m] += 1
        if k % 250 == 0:
            print(f"  parsed {k}/{len(entries)} decisive={len(games)} dropped={dropped}", flush=True)

    for zf in zip_handles.values():
        zf.close()

    n_games = len(games)
    print(f"# decisive games={n_games} dropped={dropped}")
    if n_games == 0:
        return 1

    # Per-seat rows
    seat_arch: list[str] = []
    seat_win: list[int] = []
    seat_steps: list[int] = []  # game length attached to each seat
    for g in games:
        for i in range(2):
            seat_arch.append(g["arch"][i])
            seat_win.append(1 if g["winner"] == i else 0)
            seat_steps.append(g["total_steps"])

    n_seats = len(seat_arch)
    usage = Counter(seat_arch)

    # Aggregate per archetype
    rows: list[dict[str, Any]] = []
    for arch, count in usage.most_common():
        wins = sum(seat_win[i] for i, a in enumerate(seat_arch) if a == arch)
        steps = [seat_steps[i] for i, a in enumerate(seat_arch) if a == arch]
        p, lo, hi = wilson_interval(wins, count)
        steps_sorted = sorted(steps)
        med = steps_sorted[len(steps_sorted) // 2] if steps_sorted else float("nan")
        share = count / n_seats
        rows.append(
            {
                "archetype": arch,
                "seats": count,
                "share": share,
                "wins": wins,
                "wr": p,
                "wr_lo": lo,
                "wr_hi": hi,
                "median_steps": med,
            }
        )

    # Matchup matrix (arch_i vs arch_j): wins for i when facing j
    match_wins: dict[tuple[str, str], list[int]] = defaultdict(list)
    for g in games:
        a0, a1 = g["arch"]
        w = g["winner"]
        match_wins[(a0, a1)].append(1 if w == 0 else 0)
        match_wins[(a1, a0)].append(1 if w == 1 else 0)

    # Expected WR vs field = overall WR (already have). Highlight notable matchups.
    top_arch = [r["archetype"] for r in rows[:12] if r["archetype"] != archetypes.UNKNOWN]
    matchup_highlights: list[str] = []
    for a in top_arch[:8]:
        vs_stats = []
        for b in top_arch:
            if a == b:
                continue
            outcomes = match_wins.get((a, b), [])
            if len(outcomes) < 20:
                continue
            p, lo, hi = wilson_interval(sum(outcomes), len(outcomes))
            vs_stats.append((b, len(outcomes), p, lo, hi))
        vs_stats.sort(key=lambda x: -x[2])
        if vs_stats:
            best = vs_stats[0]
            worst = vs_stats[-1]
            matchup_highlights.append(
                f"- **{a}**: best vs `{best[0]}` {fmt_pct(best[2])} "
                f"(n={best[1]}, Wilson {fmt_pct(best[3])}–{fmt_pct(best[4])}); "
                f"worst vs `{worst[0]}` {fmt_pct(worst[2])} (n={worst[1]})"
            )

    # Phase 6 order: prevalence among buildable agents after pure dragapult.
    # Keep dragapult variants + hammer + other meta separate; primary stays dragapult.
    primary = "dragapult"
    phase6_candidates = [
        r for r in rows
        if r["archetype"] not in (primary, archetypes.UNKNOWN)
        and r["seats"] >= 30  # min sample for priority list
    ]
    # Soft hint: note starmie/lucario ranks but do not boost unless high.
    soft_pref = {"starmie", "lucario"}

    # Write CSV
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank", "archetype", "seats", "share", "wins", "wr", "wr_lo", "wr_hi", "median_steps",
            ],
        )
        writer.writeheader()
        for i, r in enumerate(rows, 1):
            writer.writerow(
                {
                    "rank": i,
                    "archetype": r["archetype"],
                    "seats": r["seats"],
                    "share": f"{r['share']:.6f}",
                    "wins": r["wins"],
                    "wr": f"{r['wr']:.6f}",
                    "wr_lo": f"{r['wr_lo']:.6f}",
                    "wr_hi": f"{r['wr_hi']:.6f}",
                    "median_steps": r["median_steps"],
                }
            )

    # Source description
    src_desc = ", ".join(str(s) for s in sources)
    day_label = "2026-07-12"
    if "2026-07-12" in src_desc or "day1" in src_desc:
        day_label = "2026-07-12"

    top10 = rows[:10]
    lines: list[str] = []
    lines.append("# Ladder meta report")
    lines.append("")
    lines.append(f"**Generated for Phase 6 priority.** Sample of ladder episodes from `{day_label}`.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Sources | `{src_desc}` |")
    lines.append(f"| Sample size (requested) | {args.max_games} (seed `{args.seed}`) |")
    lines.append(f"| Decisive games | **{n_games}** (dropped tie/crash/malformed: {dropped}) |")
    lines.append(f"| Per-seat rows | **{n_seats}** |")
    lines.append(
        "| Labeling | `classify_deck` → deck signatures → highest-HP ex played "
        "(support ex demoted) |"
    )
    lines.append(
        f"| Label methods | "
        + ", ".join(f"`{m}`={c}" for m, c in method_counts.most_common())
        + " |"
    )
    lines.append("")
    lines.append(
        "Dragapult variants are kept **separate** (`dragapult`, `dragapult-dudunsparce`, "
        "`dragapult-dusknoir`, `dragapult-blaziken`, `hammer-pult`) whenever the 60-card "
        "list or Hammer signature is available."
    )
    lines.append("")
    lines.append("## Archetype prevalence (top 25)")
    lines.append("")
    lines.append(
        "| Rank | Archetype | Seats | Share | WR | Wilson 95% | Median steps |"
    )
    lines.append("|---:|---|---:|---:|---:|---|---:|")
    for i, r in enumerate(rows[:25], 1):
        lines.append(
            f"| {i} | `{r['archetype']}` | {r['seats']} | {fmt_pct(r['share'])} | "
            f"{fmt_pct(r['wr'])} | {fmt_pct(r['wr_lo'])}–{fmt_pct(r['wr_hi'])} | "
            f"{r['median_steps']} |"
        )
    lines.append("")
    lines.append("## Matchup highlights (top archetypes, n≥20)")
    lines.append("")
    if matchup_highlights:
        lines.extend(matchup_highlights)
    else:
        lines.append("_Insufficient pairwise volume for highlights._")
    lines.append("")
    lines.append("## Expected WR vs field")
    lines.append("")
    lines.append(
        "Per-archetype WR above is the empirical win rate across all opposing seats "
        "(≈ expected WR vs the sampled field). Wilson intervals widen for rare decks."
    )
    lines.append("")
    # Compact field-EV table for top 10
    lines.append("| Archetype | Field WR | Wilson lo | n |")
    lines.append("|---|---:|---:|---:|")
    for r in top10:
        lines.append(
            f"| `{r['archetype']}` | {fmt_pct(r['wr'])} | {fmt_pct(r['wr_lo'])} | {r['seats']} |"
        )
    lines.append("")
    lines.append("## Recommended Phase 6 build order (by prevalence)")
    lines.append("")
    lines.append(
        "Primary remains **pure `dragapult`** (Phase 3–5 in progress). "
        "Next Phase 6 baseline-strength agents ordered by **ladder seat share** "
        "(prevalence wins; Starmie/Lucario soft preference only noted if they rank high)."
    )
    lines.append("")
    lines.append("| # | Archetype | Share | Field WR | Median steps | Notes |")
    lines.append("|---:|---|---:|---:|---:|---|")
    lines.append(
        f"| 1 | `dragapult` | "
        f"{fmt_pct(next((r['share'] for r in rows if r['archetype']=='dragapult'), float('nan')))} | "
        f"{fmt_pct(next((r['wr'] for r in rows if r['archetype']=='dragapult'), float('nan')))} | "
        f"{next((r['median_steps'] for r in rows if r['archetype']=='dragapult'), '—')} | "
        f"**Current primary** (Phase 3–5) |"
    )
    for i, r in enumerate(phase6_candidates, 2):
        note = ""
        if r["archetype"] in soft_pref:
            note = "soft-pref (user); ranks high by prevalence"
        elif r["archetype"].startswith("dragapult") or r["archetype"] == "hammer-pult":
            note = "Dragapult family variant — keep separate"
        lines.append(
            f"| {i} | `{r['archetype']}` | {fmt_pct(r['share'])} | {fmt_pct(r['wr'])} | "
            f"{r['median_steps']} | {note} |"
        )
    lines.append("")
    lines.append(
        "Phase 6 specialist nets stay **3080-Ti-optimized** for self-play / leaf-eval speed "
        "(not Blackwell-scale); see plan hardware section."
    )
    lines.append("")
    lines.append("## CSV")
    lines.append("")
    lines.append(f"Full table: [`ladder_meta.csv`]({REPORT_CSV.name})")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_MD}")
    print(f"wrote {REPORT_CSV}")

    # Phase 6 note
    p6: list[str] = []
    p6.append("# Phase 6 archetype priority")
    p6.append("")
    p6.append(
        "**Status:** next-up only — do **not** divert Phase 3–5 primary training from pure Dragapult."
    )
    p6.append("")
    p6.append(
        f"**Source:** ladder meta sample `{day_label}` "
        f"({n_games} decisive games, seed {args.seed}) — see "
        f"[`outputs/reports/ladder_meta.md`](../reports/ladder_meta.md)."
    )
    p6.append("")
    p6.append("## Generation order (baseline-strength agents)")
    p6.append("")
    p6.append("| # | Archetype ID | Share | Why |")
    p6.append("|---:|---|---:|---|")
    p6.append(
        "| 1 | `dragapult` | "
        f"{fmt_pct(next((r['share'] for r in rows if r['archetype']=='dragapult'), float('nan')))} | "
        "**Current / Phase 3–5 primary** — Campinas 2nd pure Dragapult |"
    )
    for i, r in enumerate(phase6_candidates, 2):
        why = "ladder prevalence"
        if r["archetype"] in soft_pref:
            why += " (+ soft user hint; already high-share)"
        if r["archetype"].startswith("dragapult") or r["archetype"] == "hammer-pult":
            why += "; keep separate from pure Dragapult"
        p6.append(f"| {i} | `{r['archetype']}` | {fmt_pct(r['share'])} | {why} |")
    p6.append("")
    p6.append("## Soft preference note")
    p6.append("")
    for pref in ("starmie", "lucario"):
        rank = next((i for i, r in enumerate(phase6_candidates, 2) if r["archetype"] == pref), None)
        share = next((r["share"] for r in rows if r["archetype"] == pref), None)
        if rank is None:
            p6.append(
                f"- `{pref}`: earlier user soft preference — **not** in top prevalence "
                f"cut (or n&lt;30); prevalence wins, so it is not forced upward."
            )
        else:
            p6.append(
                f"- `{pref}`: soft preference **aligns** with prevalence "
                f"(Phase 6 #{rank}, share {fmt_pct(share or float('nan'))})."
            )
    p6.append("")
    p6.append("## Pipeline per archetype (same as Dragapult)")
    p6.append("")
    p6.append("1. Signature-filtered ladder bootstrap → JSONL (info-set only)")
    p6.append("2. Supervised bootstrap train + early stop")
    p6.append("3. Short MCTS round-robin vs full `baselines/manifest.json` until roughly baseline-strength")
    p6.append("4. Register in local deck/agent pool for self-play + matchup-ID routing")
    p6.append("")
    p6.append("## Hardware")
    p6.append("")
    p6.append(
        "Phase 6 specialist models are **sized for RTX 3080 Ti self-play speed** "
        "(leaf eval + collect on `device.leaf_eval_device`), not Blackwell-scale training width. "
        "Keep Blackwell busy on generalist / pure-Dragapult while specialists roll on the 3080 Ti."
    )
    p6.append("")
    p6.append("See also: plan Phase 6; `outputs/notes/archetype_pivot.md`; `outputs/reports/ladder_meta.md`.")
    p6.append("")
    PHASE6_NOTE.parent.mkdir(parents=True, exist_ok=True)
    PHASE6_NOTE.write_text("\n".join(p6) + "\n", encoding="utf-8")
    print(f"wrote {PHASE6_NOTE}")

    # Print summary for agent return
    print("\n=== TOP 10 BY PREVALENCE ===")
    for i, r in enumerate(top10, 1):
        print(f"{i:2d}. {r['archetype']:28s} share={fmt_pct(r['share']):>6s}  "
              f"WR={fmt_pct(r['wr']):>6s}  n={r['seats']}")
    print("\n=== PHASE 6 ORDER ===")
    print("1. dragapult (primary, in progress)")
    for i, r in enumerate(phase6_candidates, 2):
        print(f"{i}. {r['archetype']} ({fmt_pct(r['share'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
