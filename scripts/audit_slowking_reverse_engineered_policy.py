#!/usr/bin/env python3
"""Measure the causal Slowking heuristic surrogate on recovered replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.slowking_reverse_engineered_policy import (
    POLICY_VERSION,
    SCHEMA_VERSION,
    audit_decision,
)
from scripts.distill_slowking_top_replays import (
    acting_frames,
    action_confirmed,
    card_id,
    resolve_card,
    setup_decks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("state/slowking_archetype_learning_index_v1.json"),
    )
    parser.add_argument("--archive-dir", type=Path, default=Path("data/episodes/raw"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("state/slowking_reverse_engineered_policy_audit_v1.json"),
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def _archive_path(archive_dir: Path, date: str) -> Path:
    return archive_dir / f"pokemon-tcg-ai-battle-episodes-{date}.zip"


def _bucket() -> Counter[str]:
    return Counter(
        games=0,
        single_select_prompts=0,
        covered=0,
        agreed=0,
        disagreed=0,
        invalid_or_missing_episode=0,
    )


def _audit_date(
    date: str,
    rows: list[dict[str, Any]],
    archive_dir: Path,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Counter[str]]]:
    archive = _archive_path(archive_dir, date)
    totals = _bucket()
    stage_counts: dict[str, Counter[str]] = defaultdict(_bucket)
    decisions: list[dict[str, Any]] = []
    if not archive.exists():
        totals["invalid_or_missing_episode"] += len(rows)
        return decisions, totals, stage_counts

    with zipfile.ZipFile(archive) as zipped:
        members = set(zipped.namelist())
        for row in rows:
            episode_id = str(row["episode_id"])
            member = f"{episode_id}.json"
            if member not in members:
                matches = [name for name in members if name.endswith("/" + member)]
                member = matches[0] if len(matches) == 1 else ""
            if not member:
                totals["invalid_or_missing_episode"] += 1
                continue
            try:
                payload = json.loads(zipped.read(member))
            except (json.JSONDecodeError, KeyError, OSError):
                totals["invalid_or_missing_episode"] += 1
                continue
            seat = int(row["seat"])
            decks = setup_decks(payload)
            deck = decks[seat] if seat < len(decks) else None
            if deck is None:
                totals["invalid_or_missing_episode"] += 1
                continue
            totals["games"] += 1
            for env_step, entry in acting_frames(payload, seat):
                observation = entry.get("observation") or {}
                select = observation.get("select") or {}
                options = select.get("option") or []
                action = entry.get("action") or []
                if (
                    select.get("minCount") != 1
                    or select.get("maxCount") != 1
                    or len(action) != 1
                    or len(options) < 2
                ):
                    continue
                totals["single_select_prompts"] += 1
                chosen_option = options[int(action[0])]
                chosen_card_id = card_id(resolve_card(observation, chosen_option))
                if not action_confirmed(
                    payload,
                    seat,
                    env_step,
                    chosen_option,
                    chosen_card_id,
                ):
                    continue
                combos = [(index,) for index in range(len(options))]
                audit = audit_decision(observation, combos, deck=deck)
                if audit is None:
                    continue
                stage = str(audit["stage_class"])
                preferred = combos[int(audit["preferred_combo_index"])][0]
                chosen = int(action[0])
                agreed = preferred == chosen
                totals["covered"] += 1
                totals["agreed" if agreed else "disagreed"] += 1
                stage_counts[stage]["covered"] += 1
                stage_counts[stage]["agreed" if agreed else "disagreed"] += 1
                decisions.append(
                    {
                        "date": date,
                        "episode_id": episode_id,
                        "seat": seat,
                        "team_name": row["team_name"],
                        "split": row["split"],
                        "deck_fingerprint": row["deck_fingerprint"],
                        "env_step": env_step,
                        "stage_class": stage,
                        "chosen_option": chosen,
                        "preferred_option": preferred,
                        "agreement": agreed,
                        "margin": audit["margin"],
                        "option_scores": audit["scores"],
                        "option_rule_ids": audit["combo_rule_ids"],
                        "chosen_card_ids": audit["combo_resolved_card_ids"][chosen],
                        "preferred_card_ids": audit["combo_resolved_card_ids"][preferred],
                    }
                )
    return decisions, totals, stage_counts


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _render_bucket(bucket: Counter[str]) -> dict[str, Any]:
    result = dict(sorted(bucket.items()))
    result["coverage_of_single_select"] = _rate(
        bucket["covered"], bucket["single_select_prompts"]
    )
    result["agreement_on_covered"] = _rate(bucket["agreed"], bucket["covered"])
    return result


def main() -> None:
    args = parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    rows = index.get("rows") or []
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[str(row["date"])].append(row)

    all_decisions: list[dict[str, Any]] = []
    totals = _bucket()
    stages: dict[str, Counter[str]] = defaultdict(_bucket)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            date: pool.submit(_audit_date, date, date_rows, args.archive_dir)
            for date, date_rows in sorted(by_date.items())
        }
        for date in sorted(futures):
            decisions, date_totals, date_stages = futures[date].result()
            all_decisions.extend(decisions)
            totals.update(date_totals)
            for stage, counts in date_stages.items():
                stages[stage].update(counts)

    strata: dict[str, dict[str, Counter[str]]] = {
        "split": defaultdict(_bucket),
        "team_name": defaultdict(_bucket),
        "deck_fingerprint": defaultdict(_bucket),
    }
    for decision in all_decisions:
        for field, buckets in strata.items():
            bucket = buckets[str(decision[field])]
            bucket["covered"] += 1
            bucket["agreed" if decision["agreement"] else "disagreed"] += 1

    output = {
        "schema": "poke_bot.slowking_reverse_engineered_policy_audit/v1",
        "status": "research_only_complete",
        "policy": {
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "module": "poke_bot/slowking_reverse_engineered_policy.py",
            "module_sha256": digest(ROOT / "poke_bot/slowking_reverse_engineered_policy.py"),
            "runtime_authority": "none",
            "future_or_result_inputs": [],
        },
        "source": {
            "auditor": "scripts/audit_slowking_reverse_engineered_policy.py",
            "auditor_sha256": digest(Path(__file__).resolve()),
            "index": str(args.index),
            "index_sha256": digest(args.index),
            "requested_games": len(rows),
            "dates": sorted(by_date),
        },
        "overall": _render_bucket(totals),
        "by_stage": {stage: _render_bucket(counts) for stage, counts in sorted(stages.items())},
        "by_stratum": {
            field: {key: _render_bucket(counts) for key, counts in sorted(buckets.items())}
            for field, buckets in strata.items()
        },
        "covered_decisions": sorted(
            all_decisions,
            key=lambda row: (row["date"], row["episode_id"], row["seat"], row["env_step"]),
        ),
        "interpretation": {
            "agreement_is_imitation_not_win_rate": True,
            "abstentions_are_masked": True,
            "exact_list_is_evaluation_stratum_only": True,
            "safe_use": "teacher feature, confidence mask, or offline baseline",
            "unsafe_use": "unreviewed serving or promotion authority",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({"out": str(args.out), "overall": output["overall"], "by_stage": output["by_stage"]}, indent=2))


if __name__ == "__main__":
    main()
