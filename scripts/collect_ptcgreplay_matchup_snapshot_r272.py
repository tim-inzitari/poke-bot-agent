#!/usr/bin/env python3
"""Seal the completed PTCGReplay Meta snapshot for r272 adapter preparation.

Credentials are consumed only in memory from the site's runtime configuration.
The output intentionally excludes credentials, sessions, action labels, and
curation rules.  Snapshot decklists are guide/reference evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CONFIG_URL = "https://ptcgreplay.netlify.app/config.js"
SOURCE_PAGE = "https://ptcgreplay.netlify.app/"
SCHEMA = "poke_bot.ptcgreplay_matchup_meta_snapshot_r272/v1"
GOAL_REVISION = 272
HIGHEST_WIN_RATE_DECK_MINIMUM_GAMES = 20
SNAPSHOT_MINIMUM_GAMES = 30
PAGE_SIZE = 1000
HTTP_TIMEOUT_SECONDS = int(os.environ.get("PTCGREPLAY_HTTP_TIMEOUT_SECONDS", "180"))
HTTP_MAX_ATTEMPTS = int(os.environ.get("PTCGREPLAY_HTTP_MAX_ATTEMPTS", "6"))
HTTP_RETRY_BASE_SECONDS = float(os.environ.get("PTCGREPLAY_HTTP_RETRY_BASE_SECONDS", "2"))


class SnapshotError(RuntimeError):
    """Fail closed when source or snapshot invariants drift."""


def canonical_json(value: object, *, newline: bool = True) -> bytes:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + ("\n" if newline else "")).encode("utf-8")


def semantic_digest(payload: dict[str, Any]) -> str:
    detached = dict(payload)
    detached.pop("snapshot_sha256", None)
    return "sha256:" + hashlib.sha256(canonical_json(detached)).hexdigest()


def _request_json(
    url: str,
    *,
    headers: dict[str, str],
    data: bytes | None = None,
    method: str | None = None,
    timeout: int = HTTP_TIMEOUT_SECONDS,
) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", **headers},
        data=data,
        method=method,
    )
    for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == HTTP_MAX_ATTEMPTS:
                raise SnapshotError(
                    f"PTCGReplay request failed after {attempt} attempts: {url}"
                ) from exc
            time.sleep(min(30.0, HTTP_RETRY_BASE_SECONDS * (2 ** (attempt - 1))))
    raise AssertionError("unreachable retry loop")


def _runtime_config() -> dict[str, Any]:
    request = urllib.request.Request(CONFIG_URL, headers={"User-Agent": "Mozilla/5.0"})
    text = None
    for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                text = response.read().decode("utf-8")
            break
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == HTTP_MAX_ATTEMPTS:
                raise SnapshotError(
                    f"PTCGReplay runtime config failed after {attempt} attempts"
                ) from exc
            time.sleep(min(30.0, HTTP_RETRY_BASE_SECONDS * (2 ** (attempt - 1))))
    if text is None:
        raise AssertionError("unreachable retry loop")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise SnapshotError("PTCGReplay runtime config has no JSON object")
    config = json.loads(match.group(0))
    required = {"supabaseUrl", "anonKey", "teamEmail", "teamPassword"}
    if not required.issubset(config) or any(
        not isinstance(config[key], str) or not config[key] for key in required
    ):
        raise SnapshotError("PTCGReplay runtime config is incomplete")
    return config


class Client:
    """Small authenticated, read-only PostgREST client."""

    def __init__(self) -> None:
        config = _runtime_config()
        self.base = str(config["supabaseUrl"]).rstrip("/")
        anon = str(config["anonKey"])
        body = canonical_json(
            {
                "email": config["teamEmail"],
                "password": config["teamPassword"],
            },
            newline=False,
        )
        auth = _request_json(
            self.base + "/auth/v1/token?grant_type=password",
            headers={"apikey": anon, "Content-Type": "application/json"},
            data=body,
            method="POST",
        )
        token = auth.get("access_token") if isinstance(auth, dict) else None
        if not isinstance(token, str) or not token:
            raise SnapshotError("PTCGReplay authentication returned no access token")
        self.headers = {"apikey": anon, "Authorization": "Bearer " + token}

    def one(self, path: str) -> Any:
        return _request_json(
            self.base + "/rest/v1/" + path,
            headers=self.headers,
        )

    def all(self, path: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for start in range(0, 10_000_000, PAGE_SIZE):
            batch = _request_json(
                self.base + "/rest/v1/" + path,
                headers={
                    **self.headers,
                    "Range": f"{start}-{start + PAGE_SIZE - 1}",
                },
            )
            if not isinstance(batch, list) or any(
                not isinstance(row, dict) for row in batch
            ):
                raise SnapshotError(f"PTCGReplay query is not a row list: {path}")
            rows.extend(batch)
            if len(batch) < PAGE_SIZE:
                return rows
        raise SnapshotError(f"PTCGReplay pagination did not terminate: {path}")


def decode_facts(blob: dict[str, Any]) -> tuple[str, list[str], list[tuple[int, int, int, int, int]]]:
    base = blob.get("base")
    decks = blob.get("decks")
    raw = blob.get("facts")
    if not isinstance(base, str) or not isinstance(decks, list) or not isinstance(raw, str):
        raise SnapshotError("match_facts payload is malformed")
    tokens = raw.split(",") if raw else []
    if len(tokens) % 5:
        raise SnapshotError("match_facts values are not exact quintuples")
    facts: list[tuple[int, int, int, int, int]] = []
    for offset in range(0, len(tokens), 5):
        try:
            row = tuple(int(value) for value in tokens[offset : offset + 5])
        except ValueError as exc:
            raise SnapshotError("match_facts contains a non-integer") from exc
        day, deck0, deck1, winner, first = row
        if (
            day < 0
            or deck0 < 0
            or deck1 < 0
            or deck0 >= len(decks)
            or deck1 >= len(decks)
            or winner not in {-1, 0, 1}
            or first not in {-1, 0, 1}
        ):
            raise SnapshotError("match_facts contains an out-of-domain value")
        facts.append((day, deck0, deck1, winner, first))
    return base, [str(value) for value in decks], facts


def _rate(wins: int, decided: int) -> float | None:
    if not decided:
        return None
    result = wins / decided
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SnapshotError("computed win rate is invalid")
    return result


def _card_multiset(
    card_ids: Iterable[int],
    *,
    card_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = Counter(int(value) for value in card_ids)
    if sum(counts.values()) != 60:
        raise SnapshotError("guide decklist is not exactly 60 cards")
    rows: list[dict[str, Any]] = []
    for card_id in sorted(counts):
        card = card_by_id.get(card_id)
        if not card or not isinstance(card.get("name"), str):
            raise SnapshotError(f"guide decklist references unknown card {card_id}")
        rows.append(
            {
                "card_id": card_id,
                "name": card["name"],
                "count": counts[card_id],
                "stage_type": card.get("stage_type"),
                "type": card.get("type"),
            }
        )
    return rows


def build_snapshot(
    *,
    run: dict[str, Any],
    archetypes: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    decks: list[dict[str, Any]],
    fact_blob: dict[str, Any],
) -> dict[str, Any]:
    if not run.get("finished_at"):
        raise SnapshotError("newest ingest is not complete")
    base, fact_decks, facts = decode_facts(fact_blob)
    if base != run.get("window_start"):
        raise SnapshotError("match_facts base differs from completed ingest")
    if int(run.get("days_ingested") or 0) <= 0:
        raise SnapshotError("completed ingest has no days")
    max_day = max((row[0] for row in facts), default=-1)
    if max_day + 1 != int(run["days_ingested"]):
        raise SnapshotError("match_facts day coverage differs from completed ingest")

    card_by_id = {int(row["card_id"]): row for row in cards}
    deck_by_hash = {str(row["deck_hash"]): row for row in decks}
    arch_by_id = {int(row["id"]): row for row in archetypes}
    deck_arch: list[int | None] = []
    deck_cards: list[list[int] | None] = []
    for deck_hash in fact_decks:
        row = deck_by_hash.get(deck_hash)
        if row is None:
            deck_arch.append(None)
            deck_cards.append(None)
            continue
        arch_id = row.get("archetype_id")
        deck_arch.append(int(arch_id) if arch_id is not None else None)
        raw_cards = row.get("card_ids")
        deck_cards.append(
            [int(value) for value in raw_cards]
            if isinstance(raw_cards, list)
            else None
        )

    deck_games = [0] * len(fact_decks)
    deck_decided = [0] * len(fact_decks)
    deck_wins = [0] * len(fact_decks)
    arch_games: Counter[int] = Counter()
    arch_decided: Counter[int] = Counter()
    arch_wins: Counter[int] = Counter()
    for _day, d0, d1, winner, _first in facts:
        for seat, deck_index in ((0, d0), (1, d1)):
            deck_games[deck_index] += 1
            arch_id = deck_arch[deck_index]
            if arch_id is not None:
                arch_games[arch_id] += 1
            if winner >= 0:
                deck_decided[deck_index] += 1
                if arch_id is not None:
                    arch_decided[arch_id] += 1
                if winner == seat:
                    deck_wins[deck_index] += 1
                    if arch_id is not None:
                        arch_wins[arch_id] += 1

    total_appearances = sum(arch_games.values())
    roster: list[dict[str, Any]] = []
    rogue: dict[str, Any] | None = None
    for arch_id in sorted(arch_by_id):
        arch = arch_by_id[arch_id]
        name = str(arch.get("name") or "").strip()
        if not name:
            raise SnapshotError(f"archetype {arch_id} has no exact name")
        indices = [
            index
            for index, candidate in enumerate(deck_arch)
            if candidate == arch_id and deck_games[index] > 0
        ]
        deck_rows = [
            {
                "index": index,
                "deck_hash": fact_decks[index],
                "games": deck_games[index],
                "decided": deck_decided[index],
                "wins": deck_wins[index],
                "win_rate": _rate(deck_wins[index], deck_decided[index]),
            }
            for index in indices
        ]
        popular = max(
            deck_rows,
            key=lambda row: int(row["games"]),
            default=None,
        )
        eligible = [
            row
            for row in deck_rows
            if int(row["games"]) >= HIGHEST_WIN_RATE_DECK_MINIMUM_GAMES
            and row["win_rate"] is not None
        ]
        best_pool = eligible or deck_rows
        best = max(
            best_pool,
            key=lambda row: (
                float(row["win_rate"]) if row["win_rate"] is not None else -1.0
            ),
            default=None,
        )

        def guide_deck(row: dict[str, Any] | None) -> dict[str, Any] | None:
            if row is None:
                return None
            index = int(row["index"])
            card_ids = deck_cards[index]
            if card_ids is None:
                raise SnapshotError("selected guide deck lacks a card list")
            return {
                key: value for key, value in row.items() if key != "index"
            } | {
                "card_count": len(card_ids),
                "card_multiset": _card_multiset(card_ids, card_by_id=card_by_id),
                "role": "guide_reference_only",
                "hard_requirement": False,
            }

        core_values = arch.get("core_cards") or []
        if not isinstance(core_values, list):
            raise SnapshotError(f"archetype {arch_id} core cards are malformed")
        core_cards: list[dict[str, Any]] = []
        for raw_core in core_values:
            if isinstance(raw_core, dict):
                raw_card_id = raw_core.get("card_id")
                inclusion = raw_core.get("inclusion")
            else:
                raw_card_id = raw_core
                inclusion = None
            try:
                core_card_id = int(raw_card_id)
            except (TypeError, ValueError) as exc:
                raise SnapshotError(
                    f"archetype {arch_id} has a malformed core-card identity"
                ) from exc
            card = card_by_id.get(core_card_id)
            if card is None or not isinstance(card.get("name"), str):
                raise SnapshotError(
                    f"archetype {arch_id} core card {core_card_id} is unknown"
                )
            if inclusion is not None:
                inclusion = float(inclusion)
                if not math.isfinite(inclusion) or not 0.0 <= inclusion <= 1.0:
                    raise SnapshotError(
                        f"archetype {arch_id} core-card inclusion is invalid"
                    )
            core_cards.append(
                {
                    "card_id": core_card_id,
                    "name": card["name"],
                    "inclusion": inclusion,
                }
            )
        entry = {
            "source_id": arch_id,
            "exact_name": name,
            "description": arch.get("description"),
            "cluster_key": arch.get("cluster_key"),
            "appearances": arch_games[arch_id],
            "decided": arch_decided[arch_id],
            "wins": arch_wins[arch_id],
            "win_rate": _rate(arch_wins[arch_id], arch_decided[arch_id]),
            "appearance_share": (
                arch_games[arch_id] / total_appearances if total_appearances else 0.0
            ),
            "core_cards": core_cards,
            "guide_decklists": {
                "most_popular": guide_deck(popular),
                "highest_win_rate_minimum_20_games": guide_deck(best),
                "authority": "guide_reference_only_not_hard_requirement",
            },
        }
        if name == "Rogue / Other":
            rogue = {**entry, "excluded_reason": "pooled_non_archetype_bucket"}
        else:
            roster.append(entry)

    if len(archetypes) != 32 or len(roster) != 31 or rogue is None:
        raise SnapshotError("completed Meta snapshot is not exact 31-real-plus-Rogue roster")
    if len(facts) != 148_997:
        raise SnapshotError("completed Meta snapshot match count drifted")
    if run.get("window_start") != "2026-07-10" or run.get("window_end") != "2026-08-10":
        raise SnapshotError("completed Meta snapshot window drifted")
    if any(row["appearances"] < SNAPSHOT_MINIMUM_GAMES for row in roster):
        raise SnapshotError("a real snapshot archetype is below the visible floor")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "goal_revision": GOAL_REVISION,
        "status": "sealed_completed_meta_snapshot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE_PAGE,
        "source_view": "Meta snapshot",
        "info_view": {
            "window_start": run["window_start"],
            "window_end": run["window_end"],
            "days": int(run["days_ingested"]),
            "last_updated": run["finished_at"],
            "ingest_id": int(run["id"]),
            "matches_analyzed": len(facts),
            "archetype_rows_including_rogue": len(archetypes),
        },
        "snapshot_filters": {
            "archetypes_shown": "all",
            "rank_by": "play_rate",
            "going": "first_or_second",
            "minimum_games": SNAPSHOT_MINIMUM_GAMES,
        },
        "allocation_order": "source_numeric_id_ascending",
        "real_archetype_count": len(roster),
        "real_archetypes": roster,
        "excluded_rows": [rogue],
        "decklist_contract": {
            "role": "guide_reference_only",
            "exact_deck_or_hard_signature_requirement": False,
            "action_outcome_label_or_gate_authority": False,
            "independent_checksum_backed_replay_support_required": True,
            "most_popular_semantics": "maximum_appearances_in_window",
            "highest_win_rate_minimum_games": HIGHEST_WIN_RATE_DECK_MINIMUM_GAMES,
        },
        "credential_or_session_material_included": False,
    }
    payload["snapshot_sha256"] = semantic_digest(payload)
    return payload


def collect() -> dict[str, Any]:
    client = Client()
    run_rows = client.all(
        "ingest_runs?select=*&finished_at=not.is.null&order=id.desc&limit=1"
    )
    if len(run_rows) != 1:
        raise SnapshotError("newest completed ingest query did not return one row")
    return build_snapshot(
        run=run_rows[0],
        archetypes=client.all(
            "archetypes?select=id,name,description,cluster_key,core_cards&order=name"
        ),
        cards=client.all("cards?select=card_id,name,stage_type,type"),
        decks=client.all("decks?select=deck_hash,archetype_id,card_ids"),
        fact_blob=client.one("rpc/match_facts"),
    )


def write_create_only(path: Path, payload: dict[str, Any]) -> None:
    target = path.expanduser()
    if target.exists() or target.is_symlink():
        raise SnapshotError(f"snapshot output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + target.name + ".",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
        os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = collect()
    write_create_only(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "snapshot_sha256": payload["snapshot_sha256"],
                "ingest_id": payload["info_view"]["ingest_id"],
                "matches": payload["info_view"]["matches_analyzed"],
                "real_archetypes": payload["real_archetype_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
