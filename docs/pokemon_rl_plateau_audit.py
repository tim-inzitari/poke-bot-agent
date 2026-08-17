"""Reproducible calculations for the Pokemon pure-RL plateau audit.

The per-iteration aggregates and reconstructed deck results are a frozen
snapshot of Inzi run v12 through iteration 181, queried on 2026-07-19.  The
ladder mix is read from the versioned repository artifact so its provenance
and weights remain independently checkable.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LADDER_MIX_PATH = ROOT / "data" / "training_mixes" / "top_ladder.v1.json"

PERIOD_METRICS = [
    {
        "metric": "Fixed held-out win rate",
        "first_20": 0.115875,
        "last_20": 0.105500,
        "definition": "Mean of each iteration's 200-game greedy evaluation against four official bots.",
    },
    {
        "metric": "Policy action accuracy",
        "first_20": 0.487500,
        "last_20": 0.986550,
        "definition": "Accuracy on the held-out split of the run's own freshly generated action data.",
    },
    {
        "metric": "Agreement with previous policy",
        "first_20": 0.941880,
        "last_20": 0.991710,
        "definition": "Action agreement between candidate and previous policy on replay decisions.",
    },
]

# Deck identity is not retained in heldout_rows.  These rates were reconstructed
# from job_index using the deterministic balanced-eval schedule in
# scripts/train_pure_rl.py: deck_i = floor(floor(job_index / 2) / 4) mod 17.
DECK_HELDOUT = [
    ("cornerstone-ogerpon", 0.2624, 0.2156, 0.2344),
    ("crustle", 0.2280, 0.2031, 0.2500),
    ("rockets-mewtwo", 0.1365, 0.1469, 0.1438),
    ("marnie-s-grimmsnarl-ex", 0.1154, 0.1250, 0.1219),
    ("alakazam", 0.1120, 0.1563, 0.0938),
    ("archaludon-ex", 0.0948, 0.1063, 0.1063),
    ("gardevoir", 0.0886, 0.1250, 0.0688),
    ("lucario", 0.0845, 0.1250, 0.0438),
    ("garchomp", 0.0804, 0.1063, 0.0688),
    ("hammer-pult", 0.0632, 0.0688, 0.0563),
    ("dragapult", 0.0601, 0.0750, 0.0438),
    ("festival-lead", 0.0591, 0.1500, 0.0625),
    ("dragapult-dudunsparce", 0.0549, 0.0500, 0.0625),
    ("ns-zoroark", 0.0536, 0.0438, 0.0750),
    ("lopunny", 0.0536, 0.0500, 0.0938),
    ("starmie", 0.0522, 0.0547, 0.0500),
    ("raging-bolt", 0.0240, 0.0188, 0.0438),
]

LATEST_OFFICIAL_OPPONENTS = [
    ("mega-lucario-ex", 13, 50),
    ("dragapult-ex", 4, 50),
    ("mega-abomasnow-ex", 3, 50),
    ("iono", 1, 50),
]


def balanced_eval_game_counts(n_games: int, n_specs: int, n_decks: int) -> list[int]:
    """Mirror the evaluator's job-index schedule exactly."""
    counts: Counter[int] = Counter()
    for game_i in range(n_games):
        pair_i = game_i // 2
        deck_i = (pair_i // n_specs) % n_decks
        counts[deck_i] += 1
    return [counts[i] for i in range(n_decks)]


def build_results() -> dict:
    ladder = json.loads(LADDER_MIX_PATH.read_text(encoding="utf-8"))
    ladder_by_deck = {row["deck_id"]: row for row in ladder["decks"]}
    deck_rows = []
    for deck_id, overall, first_20, last_20 in DECK_HELDOUT:
        meta = ladder_by_deck[deck_id]
        deck_rows.append(
            {
                "deck_id": deck_id,
                "heldout_overall": overall,
                "heldout_first_20": first_20,
                "heldout_last_20": last_20,
                "heldout_change": last_20 - first_20,
                "ladder_train_weight": meta["train_weight"],
                "ladder_game_share": meta["game_share"],
                "ladder_win_rate": meta["win_rate"],
                "ladder_observed_count": meta["observed_count"],
            }
        )

    schedule_counts = balanced_eval_game_counts(200, 4, 17)
    latest_opponents = [
        {
            "opponent": opponent,
            "wins": wins,
            "games": games,
            "win_rate": wins / games,
        }
        for opponent, wins, games in LATEST_OFFICIAL_OPPONENTS
    ]

    return {
        "run_snapshot": {
            "first_iteration": 0,
            "last_iteration": 181,
            "committed_iterations": 182,
            "games_per_iteration": 2048,
            "scheduled_source_games": 182 * 2048,
            "heldout_first_20": PERIOD_METRICS[0]["first_20"],
            "heldout_last_20": PERIOD_METRICS[0]["last_20"],
            "heldout_change_percentage_points": 100
            * (PERIOD_METRICS[0]["last_20"] - PERIOD_METRICS[0]["first_20"]),
            "latest_heldout": 0.105,
            "best_iteration_heldout": 0.145,
            "best_iteration": 33,
            "promotions": 46,
            "latest_relative_update_norm_l2": 1.83967e-5,
        },
        "period_metrics": PERIOD_METRICS,
        "deck_rows": deck_rows,
        "latest_official_opponents": latest_opponents,
        "balanced_eval_game_counts_by_deck_index": schedule_counts,
        "ladder_mix": {
            "mix_id": ladder["mix_id"],
            "artifact_sha256": ladder["artifact_sha256"],
            "episodes_processed": ladder["source"]["episodes_processed"],
            "decisive_games": ladder["source"]["decisive_games"],
            "recognized_seat_appearances": ladder["coverage"]["recognized_seat_appearances"],
            "weight_sum": sum(row["train_weight"] for row in ladder["decks"]),
        },
    }


def validate(results: dict) -> None:
    decks = results["deck_rows"]
    assert len(decks) == 17
    assert len({row["deck_id"] for row in decks}) == 17
    assert abs(results["ladder_mix"]["weight_sum"] - 1.0) < 1e-9
    assert results["balanced_eval_game_counts_by_deck_index"] == [16] * 8 + [8] * 9
    for row in decks:
        for key in (
            "heldout_overall",
            "heldout_first_20",
            "heldout_last_20",
            "ladder_train_weight",
            "ladder_game_share",
            "ladder_win_rate",
        ):
            assert 0.0 <= row[key] <= 1.0, (row["deck_id"], key, row[key])


def summary(results: dict) -> dict:
    by_last = sorted(results["deck_rows"], key=lambda row: row["heldout_last_20"], reverse=True)
    by_ladder = sorted(results["deck_rows"], key=lambda row: row["ladder_train_weight"], reverse=True)
    selected = {row["deck_id"]: row for row in results["deck_rows"]}
    return {
        "run_snapshot": results["run_snapshot"],
        "top_heldout_last_20": [row["deck_id"] for row in by_last[:5]],
        "top_ladder_weight": [row["deck_id"] for row in by_ladder[:5]],
        "lucario": selected["lucario"],
        "crustle": selected["crustle"],
        "cornerstone_ogerpon": selected["cornerstone-ogerpon"],
        "rockets_mewtwo": selected["rockets-mewtwo"],
        "eval_games_per_deck_index": results["balanced_eval_game_counts_by_deck_index"],
    }


if __name__ == "__main__":
    audit_results = build_results()
    validate(audit_results)
    print(json.dumps(summary(audit_results), indent=2, sort_keys=True))
