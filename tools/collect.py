"""Collect competitive Standard decklists from Limitless TCG, resolve to in-pool
Card IDs, validate, tier, and write per docs/decklist-collection-spec.md.

Reliable path: fetch HTML directly and parse data-set/data-number/card-count/
card-name attributes (no model transcription). Resumable via progress.json.

Usage: python tools/collect.py <tournament_id> [<tournament_id> ...]
Tournament metadata (date, field size, name) is read from TOURNAMENTS below.
"""
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from resolver import Resolver, resolve_deck, normalize_name  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = os.path.join(ROOT, "decks", "competitive")
HIGH = os.path.join(COMP, "high_performing")
REST = os.path.join(COMP, "the_rest")
INDEX = os.path.join(COMP, "index.csv")
REJECT = os.path.join(COMP, "rejected.log")
PROGRESS = os.path.join(COMP, "progress.json")
CACHE = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE, exist_ok=True)

# Tournament metadata: id -> (name, date YYYY-MM-DD, field_size, event_slug)
TOURNAMENTS = {
    "518": ("NAIC 2026, New Orleans", "2026-06-10", 3752, "naic-2026-new-orleans"),
    "540": ("Special Event Turin", "2026-06-06", 2033, "se-turin-2026"),
    "559": ("Regional Indianapolis", "2026-05-30", 1974, "regional-indianapolis-2026"),
    "536": ("Special Event Lima", "2026-05-23", 499, "se-lima-2026"),
    "550": ("Regional Melbourne", "2026-05-23", 959, "regional-melbourne-2026"),
    "535": ("Regional Utrecht", "2026-05-16", 2150, "regional-utrecht-2026"),
    "544": ("Regional Campinas", "2026-05-16", 1725, "regional-campinas-2026"),
    "558": ("Regional Los Angeles", "2026-05-09", 1849, "regional-la-2026"),
    "565": ("Korean League Season 4", "2026-04-25", 397, "korean-league-s4-2026"),
    "539": ("Regional Prague", "2026-04-25", 1370, "regional-prague-2026"),
}

UA = {"User-Agent": "Mozilla/5.0 (poke-bot-agent research; decklist collection)"}


def fetch(url: str) -> str:
    key = hashlib.sha1(url.encode()).hexdigest()
    cpath = os.path.join(CACHE, key + ".html")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            return f.read()
    req = urllib.request.Request(url, headers=UA)
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    with open(cpath, "w", encoding="utf-8") as f:
        f.write(html)
    time.sleep(0.4)  # be polite
    return html


def unescape(s: str) -> str:
    return (s.replace("&#039;", "'").replace("&amp;", "&").replace("&quot;", '"')
            .replace("&lt;", "<").replace("&gt;", ">").replace("&eacute;", "é"))


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "unknown"


ROW_RE = re.compile(
    r'<tr data-rank="(\d+)"\s+data-name="([^"]*)"[^>]*?data-deck="([^"]*)"[^>]*?>(.*?)</tr>',
    re.S)
LIST_RE = re.compile(r"/decks/list/(\d+)")
CARD_RE = re.compile(
    r'<div class="decklist-card" data-set="([^"]*)" data-number="([^"]*)"[^>]*?>'
    r'.*?<span class="card-count">(\d+)</span>'
    r'\s*<span class="card-name">([^<]*)</span>', re.S)


def parse_standings(html: str):
    """-> list of dicts {rank, player, archetype, list_id}."""
    out = []
    for m in ROW_RE.finditer(html):
        rank, name, deck, body = m.groups()
        lm = LIST_RE.search(body)
        if not lm:
            continue  # no public decklist for this player
        out.append({
            "rank": int(rank),
            "player": unescape(name),
            "archetype": unescape(deck),
            "list_id": lm.group(1),
        })
    return out


def parse_decklist(html: str):
    """-> list of (qty, name, setcode, number)."""
    cards = []
    for m in CARD_RE.finditer(html):
        setc, num, cnt, name = m.groups()
        cards.append((int(cnt), unescape(name).strip(), setc.strip(), num.strip()))
    return cards


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def tier(rank: int, field: int) -> str:
    if rank <= 2:
        return "high_performing"
    if field >= 256 and rank <= 32:
        return "high_performing"
    if 64 <= field <= 255 and rank <= 8:
        return "high_performing"
    return "the_rest"


def load_progress():
    if os.path.exists(PROGRESS):
        with open(PROGRESS, encoding="utf-8") as f:
            return json.load(f)
    return {"processed_event_urls": [], "processed_deck_urls": [],
            "written_deck_hashes": {}, "notes": []}


def save_progress(p):
    with open(PROGRESS, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)


def log_reject(reason, url, event, placement):
    with open(REJECT, "a", encoding="utf-8") as f:
        f.write(f"{reason} | url={url} event={event} placement={placement}\n")


def append_index(row):
    with open(INDEX, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def deck_hash(ids):
    return hashlib.sha1(",".join(map(str, sorted(ids))).encode()).hexdigest()


def main(tids):
    r = Resolver()
    p = load_progress()
    processed = set(p["processed_deck_urls"])
    hashes = p["written_deck_hashes"]  # hash -> filename
    stats = {"written": 0, "rejected": 0, "dup": 0, "skipped_seen": 0}

    for tid in tids:
        if tid not in TOURNAMENTS:
            print(f"!! unknown tournament {tid}, skipping")
            continue
        name, date, field, eslug = TOURNAMENTS[tid]
        ym = date[:7]
        turl = f"https://limitlesstcg.com/tournaments/{tid}"
        print(f"\n=== {name} ({tid}) field={field} ===")
        rows = parse_standings(fetch(turl))
        print(f"  {len(rows)} public decklists")
        if turl not in p["processed_event_urls"]:
            p["processed_event_urls"].append(turl)
        rows.sort(key=lambda x: x["rank"])
        for row in rows:
            durl = f"https://limitlesstcg.com/decks/list/{row['list_id']}"
            if durl in processed:
                stats["skipped_seen"] += 1
                continue
            placement = row["rank"]
            try:
                cards = parse_decklist(fetch(durl))
            except Exception as e:
                log_reject(f"fetch-error: {e}", durl, name, placement)
                stats["rejected"] += 1
                processed.add(durl)
                continue
            if not cards:
                log_reject("no cards parsed (list may be hidden/empty)", durl, name, placement)
                stats["rejected"] += 1
                processed.add(durl)
                continue
            export = "\n".join(f"{q} {n} {s} {num}".strip() for q, n, s, num in cards)
            res = resolve_deck(export, r)
            if not res["ok"]:
                log_reject("invalid: " + "; ".join(res["errors"]), durl, name, placement)
                stats["rejected"] += 1
                processed.add(durl)
                continue
            h = deck_hash(res["ids"])
            if h in hashes:
                log_reject(f"duplicate of {hashes[h]}", durl, name, placement)
                stats["dup"] += 1
                processed.add(durl)
                continue
            t = tier(placement, field)
            aslug = slugify(row["archetype"])
            fname = f"{ym}_{eslug}_{ordinal(placement)}_{aslug}.csv"
            outdir = HIGH if t == "high_performing" else REST
            fpath = os.path.join(outdir, fname)
            # guard against rare filename collision (same event/place/archetype)
            if os.path.exists(fpath):
                fname = f"{ym}_{eslug}_{ordinal(placement)}_{aslug}_{row['list_id']}.csv"
                fpath = os.path.join(outdir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("\n".join(str(i) for i in res["ids"]) + "\n")
            append_index([fname, t, name, date, field, placement, row["archetype"], durl])
            hashes[h] = fname
            processed.add(durl)
            stats["written"] += 1
        p["processed_deck_urls"] = sorted(processed)
        p["written_deck_hashes"] = hashes
        save_progress(p)
        print(f"  running totals: {stats}")
    print("\nDONE", stats)


if __name__ == "__main__":
    args = sys.argv[1:] or list(TOURNAMENTS.keys())
    main(args)
