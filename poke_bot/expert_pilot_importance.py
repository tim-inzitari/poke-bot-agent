"""Checksum-bound expert-pilot importance for revision 138.

The policy changes sampling frequency only.  Replay actions, targets, and the
validation partition are never modified.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "poke_bot.expert_pilot_importance/v1"
TARGET_SCHEMA = "poke_bot.expert_pilot_importance_targets/v1"
PILOT_MAP_SCHEMA = "poke_bot.expert_pilot_map/v1"
SNAPSHOT_SCHEMA = "poke_bot.ptcgreplay_leaderboard_snapshot/v1"
OWNER_DECISION_REVISION = 138
TIERS = (
    (1, 31, 1.5),
    (32, 127, 2.0),
    (128, 511, 3.0),
    (512, 1023, 4.0),
    (1024, 2047, 7.0),
    (2048, None, 10.0),
)
MAX_IMPORTANCE_WEIGHT = max(float(weight) for _, _, weight in TIERS)


def canonical_digest(value: object) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def weight_for_support(support: int) -> float:
    value = int(support)
    for minimum, maximum, weight in TIERS:
        if value >= minimum and (maximum is None or value <= maximum):
            return float(weight)
    return 1.0


def identity_key(row: Mapping[str, object]) -> tuple[str, int]:
    episode_id = str(row.get("episode_id") or "")
    seat = int(row.get("seat", -1))
    if not episode_id or seat not in (0, 1):
        raise ValueError("expert identity row has invalid episode_id/seat")
    return episode_id, seat


def materialize_importance_index(
    *,
    targets: Mapping[str, Any],
    pilot_map: Mapping[str, Any],
    leaderboard_snapshot: Mapping[str, Any],
    targets_digest: str,
    pilot_map_digest: str,
    leaderboard_snapshot_digest: str,
) -> dict[str, Any]:
    if targets.get("schema") != TARGET_SCHEMA:
        raise ValueError("invalid expert importance target schema")
    if pilot_map.get("schema") != PILOT_MAP_SCHEMA:
        raise ValueError("invalid expert pilot-map schema")
    if leaderboard_snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("invalid leaderboard snapshot schema")

    train_rows = list(targets.get("train_rows") or ())
    validation_rows = list(targets.get("validation_rows") or ())
    train_keys = [identity_key(row) for row in train_rows]
    if len(train_keys) != len(set(train_keys)):
        raise ValueError("training episode/seat identities are not unique")

    pilots: dict[tuple[str, int], str] = {}
    for row in list(pilot_map.get("rows") or ()):
        key = identity_key(row)
        team_name = str(row.get("team_name") or "")
        if not team_name:
            raise ValueError("pilot-map row has an empty exact team name")
        if key in pilots and pilots[key] != team_name:
            raise ValueError("pilot-map has conflicting exact team names")
        pilots[key] = team_name

    leaderboard: dict[str, int] = {}
    for row in list(leaderboard_snapshot.get("rows") or ()):
        team_name = str(row.get("team_name") or "")
        rank = int(row.get("rank", 0))
        if not team_name or not 1 <= rank <= 100:
            raise ValueError("leaderboard snapshot row is not top-100")
        if team_name in leaderboard:
            raise ValueError("leaderboard snapshot has duplicate team_name")
        leaderboard[team_name] = rank
    if len(leaderboard) != 100 or set(leaderboard.values()) != set(range(1, 101)):
        raise ValueError("leaderboard snapshot must contain exact ranks 1..100")

    # Support is deliberately computed from training rows only.  Validation
    # membership therefore cannot influence sample importance.
    support = Counter(
        pilots[key]
        for key in train_keys
        if key in pilots and pilots[key] in leaderboard
    )
    weights: list[float] = []
    weighted_rows: list[dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    matched = 0
    for index, (row, key) in enumerate(zip(train_rows, train_keys, strict=True)):
        team_name = pilots.get(key)
        count = int(support.get(team_name or "", 0))
        weight = weight_for_support(count) if team_name in leaderboard else 1.0
        weights.append(weight)
        label = f"{weight:.1f}x"
        tier_counts[label] += 1
        if weight > 1.0:
            matched += 1
        weighted_rows.append(
            {
                "train_index": index,
                "episode_id": key[0],
                "seat": key[1],
                "team_name": team_name,
                "top_100_rank": leaderboard.get(team_name or ""),
                "same_training_corpus_support": count,
                "weight": weight,
            }
        )

    per_team = [
        {
            "rank": leaderboard[team],
            "team_name": team,
            "training_games": int(support.get(team, 0)),
            "weight": weight_for_support(int(support.get(team, 0))),
        }
        for team in sorted(leaderboard, key=leaderboard.__getitem__)
    ]
    train_identity_digest = canonical_digest(train_rows)
    validation_identity_digest = canonical_digest(validation_rows)
    return {
        "schema": SCHEMA,
        "owner_decision_revision": OWNER_DECISION_REVISION,
        "status": "ready",
        "corpus_manifest": targets.get("corpus_manifest"),
        "corpus_manifest_sha256": targets.get("corpus_manifest_sha256"),
        "split_seed": int(targets.get("split_seed", -1)),
        "validation_fraction": float(targets.get("validation_fraction", -1.0)),
        "max_context": int(targets.get("max_context", -1)),
        "support_partition": "training_only",
        "join_key": ["episode_id", "seat", "exact_team_name"],
        "tiers": [
            {"minimum_games": lo, "maximum_games": hi, "weight": weight}
            for lo, hi, weight in TIERS
        ],
        "unmatched_or_unverifiable_weight": 1.0,
        "actions_and_labels_unchanged": True,
        "validation_unweighted": True,
        "kaggle_evaluation_replays_excluded": True,
        "targets_sha256": targets_digest,
        "pilot_map_sha256": pilot_map_digest,
        "leaderboard_snapshot_sha256": leaderboard_snapshot_digest,
        "train_identity_sha256": train_identity_digest,
        "validation_identity_sha256": validation_identity_digest,
        "train_games": len(train_rows),
        "validation_games": len(validation_rows),
        "matched_top_100_train_games": matched,
        "unmatched_or_unverifiable_train_games": len(train_rows) - matched,
        "effective_training_weight_mass": float(sum(weights)),
        "tier_counts": dict(sorted(tier_counts.items())),
        "per_team_support": per_team,
        "train_game_weights": weights,
        "weighted_train_rows": weighted_rows,
        "train_game_weights_sha256": canonical_digest(weights),
    }


def load_aligned_training_weights(
    path: Path,
    *,
    expected_manifest_digest: str,
    train_identity_rows: Sequence[Mapping[str, object]],
    validation_identity_rows: Sequence[Mapping[str, object]],
) -> tuple[list[float], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "ready"
        or int(payload.get("owner_decision_revision", -1))
        != OWNER_DECISION_REVISION
        or payload.get("corpus_manifest_sha256") != expected_manifest_digest
        or payload.get("actions_and_labels_unchanged") is not True
        or payload.get("validation_unweighted") is not True
        or payload.get("kaggle_evaluation_replays_excluded") is not True
    ):
        raise ValueError("expert pilot importance index contract changed")
    if payload.get("train_identity_sha256") != canonical_digest(
        list(train_identity_rows)
    ) or payload.get("validation_identity_sha256") != canonical_digest(
        list(validation_identity_rows)
    ):
        raise ValueError("expert pilot importance index is misaligned")
    weights = [float(value) for value in payload.get("train_game_weights") or ()]
    if len(weights) != len(train_identity_rows):
        raise ValueError("expert pilot importance weight count changed")
    if canonical_digest(weights) != payload.get("train_game_weights_sha256"):
        raise ValueError("expert pilot importance weights changed")
    if any(value < 1.0 or value > MAX_IMPORTANCE_WEIGHT for value in weights):
        raise ValueError("expert pilot importance weight is out of bounds")
    contract = {
        key: payload[key]
        for key in (
            "schema",
            "owner_decision_revision",
            "actions_and_labels_unchanged",
            "validation_unweighted",
            "kaggle_evaluation_replays_excluded",
            "corpus_manifest_sha256",
            "targets_sha256",
            "pilot_map_sha256",
            "leaderboard_snapshot_sha256",
            "train_identity_sha256",
            "validation_identity_sha256",
            "train_game_weights_sha256",
            "train_games",
            "validation_games",
            "matched_top_100_train_games",
            "unmatched_or_unverifiable_train_games",
            "effective_training_weight_mass",
            "tier_counts",
        )
    }
    contract["importance_index_sha256"] = file_digest(Path(path))
    return weights, contract


def load_training_weights_for_corpus(
    path: Path,
    *,
    expected_manifest_digest: str,
    split_seed: int,
    validation_fraction: float,
    max_context: int,
    train_games: int,
    validation_games: int,
) -> tuple[list[float], dict[str, Any]]:
    """Load an index against the exact deterministic resident-pack contract."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "ready"
        or int(payload.get("owner_decision_revision", -1))
        != OWNER_DECISION_REVISION
        or payload.get("corpus_manifest_sha256") != expected_manifest_digest
        or int(payload.get("split_seed", -1)) != int(split_seed)
        or float(payload.get("validation_fraction", -1.0))
        != float(validation_fraction)
        or int(payload.get("max_context", -1)) != int(max_context)
        or int(payload.get("train_games", -1)) != int(train_games)
        or int(payload.get("validation_games", -1))
        != int(validation_games)
        or payload.get("support_partition") != "training_only"
        or payload.get("actions_and_labels_unchanged") is not True
        or payload.get("validation_unweighted") is not True
        or payload.get("kaggle_evaluation_replays_excluded") is not True
    ):
        raise ValueError("expert pilot importance corpus contract changed")
    weights = [float(value) for value in payload.get("train_game_weights") or ()]
    if len(weights) != int(train_games):
        raise ValueError("expert pilot importance weight count changed")
    if canonical_digest(weights) != payload.get("train_game_weights_sha256"):
        raise ValueError("expert pilot importance weights changed")
    if any(value < 1.0 or value > MAX_IMPORTANCE_WEIGHT for value in weights):
        raise ValueError("expert pilot importance weight is out of bounds")
    contract = {
        key: payload[key]
        for key in (
            "schema",
            "owner_decision_revision",
            "actions_and_labels_unchanged",
            "validation_unweighted",
            "kaggle_evaluation_replays_excluded",
            "corpus_manifest_sha256",
            "split_seed",
            "validation_fraction",
            "max_context",
            "support_partition",
            "targets_sha256",
            "pilot_map_sha256",
            "leaderboard_snapshot_sha256",
            "train_identity_sha256",
            "validation_identity_sha256",
            "train_game_weights_sha256",
            "train_games",
            "validation_games",
            "matched_top_100_train_games",
            "unmatched_or_unverifiable_train_games",
            "effective_training_weight_mass",
            "tier_counts",
        )
    }
    contract["importance_index_sha256"] = file_digest(Path(path))
    return weights, contract
