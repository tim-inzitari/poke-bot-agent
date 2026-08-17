from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace

from poke_bot.authoritative_visual_trace import _json_digest
from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION, GameSequence
from poke_bot.feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
)
from poke_bot.features import FEATURE_SCHEMA_VERSION
from poke_bot.ladder_deck_mix import load_ladder_deck_representatives
from poke_bot.ladder_replay import LadderReplayClassifier
from poke_bot.matchup_adapters import EXPERT_IDS
from poke_bot.public_matchup_router import PUBLIC_TREE_SCHEMA
from poke_bot.pure_rl.matchup_adapter_corpus import sha256_file
from scripts.build_matchup_adapter_oracle import build_oracle


ROOT = Path(__file__).resolve().parents[1]
MIX = ROOT / "data/training_mixes/top_ladder.v1.json"
REPS = ROOT / "data/training_mixes/top_ladder_representatives.v1.json"


def test_builder_binds_filtered_row_to_raw_member_and_full_deck(tmp_path: Path) -> None:
    reps = load_ladder_deck_representatives(REPS)
    our_deck = list(reps.decks["alakazam"]["card_ids"])
    opponent_deck = list(reps.decks["lucario"]["card_ids"])
    archive_dir = tmp_path / "raw"
    archive_dir.mkdir()
    archive = archive_dir / "pokemon-tcg-ai-battle-episodes-2026-07-20.zip"
    payload = {
        "info": {"EpisodeId": "episode-1", "TeamNames": ["Expert A", "Expert B"]},
        "steps": [[{"action": our_deck}, {"action": opponent_deck}]],
    }
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("episode-1.json", json.dumps(payload))

    classifier = LadderReplayClassifier.from_paths(MIX, REPS)
    classifier_digest = _json_digest(classifier.contract)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    feature = feature_dir / "day.alakazam.features"
    sequence = GameSequence(
        episode_id="episode-1",
        seat=0,
        archetype="alakazam",
        opp_archetype="lucario",
        deck=our_deck,
        value=1.0,
        decisions=[],
    )
    with feature.open("wb") as stream:
        pickle.dump(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                "required_archetype": "alakazam",
                "opponent_routes_only": True,
                "source_dates": ["2026-07-20"],
                "source_archive": archive.name,
                "source_archive_sha256": sha256_file(archive),
                "classifier_sha256": classifier_digest,
            },
            stream,
        )
        pickle.dump(sequence, stream)
        pickle.dump(
            {
                "format": SHARD_FORMAT + "-footer",
                "format_version": SHARD_FORMAT_VERSION,
                "stats": {"records_kept": 1, "decisions_kept": 0},
            },
            stream,
        )
    manifest = feature_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                "selection": {
                    "value": "alakazam",
                    "opponent_routes_only": True,
                },
                "shards": [
                    {
                        "path": feature.name,
                        "sha256": sha256_file(feature),
                        "stats": {"records_kept": 1, "decisions_kept": 0},
                    }
                ],
            }
        )
        + "\n"
    )
    oracle = build_oracle(
        manifest,
        archive_dir,
        tmp_path / "oracle",
        mix=MIX,
        representatives=REPS,
    )
    oracle_payload = json.loads(oracle.read_text())
    index = oracle.parent / oracle_payload["shards"][0]["path"]
    row = json.loads(index.read_text().splitlines()[1])
    assert row["episode_id"] == "episode-1"
    assert row["acting_archetype"] == "alakazam"
    assert row["opponent_archetype"] == "lucario"
    assert row["opponent_id"].startswith("expert-ladder:")
    assert row["formal_eval"] is False
    registry = json.loads((oracle.parent / "opponent-registry.json").read_text())
    assert registry["identity_kind"] == (
        "official-ladder-archive-agent-and-full-deck/v1"
    )
    assert registry["packages"][0]["opponent_id"] == row["opponent_id"]


def test_public_route_rows_align_by_feature_env_step_not_raw_action_count(
    tmp_path: Path,
) -> None:
    reps = load_ladder_deck_representatives(REPS)
    our_deck = list(reps.decks["alakazam"]["card_ids"])
    opponent_deck = list(reps.decks["lucario"]["card_ids"])
    archive_dir = tmp_path / "raw"
    archive_dir.mkdir()
    archive = archive_dir / "pokemon-tcg-ai-battle-episodes-2026-07-20.zip"
    decision_observation = {
        "select": {"option": [0], "minCount": 1, "maxCount": 1},
        "current": {"yourIndex": 0, "players": [{}, {}]},
    }
    payload = {
        "info": {"EpisodeId": "episode-1", "TeamNames": ["Expert A", "Expert B"]},
        "steps": [
            [{"action": our_deck}, {"action": opponent_deck}],
            [
                # Deliberately incompatible under the current raw action-frame
                # filter.  The already-built feature row remains authoritative.
                {"observation": decision_observation, "action": [2]},
                {},
            ],
            [
                {"observation": decision_observation, "action": [3]},
                {},
            ],
        ],
    }
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("episode-1.json", json.dumps(payload))

    classifier = LadderReplayClassifier.from_paths(MIX, REPS)
    classifier_digest = _json_digest(classifier.contract)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    feature = feature_dir / "day.alakazam.features"
    sequence = GameSequence(
        episode_id="episode-1",
        seat=0,
        archetype="alakazam",
        opp_archetype="lucario",
        deck=our_deck,
        value=1.0,
        decisions=[SimpleNamespace(env_step=1), SimpleNamespace(env_step=2)],
    )
    with feature.open("wb") as stream:
        pickle.dump(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                "required_archetype": "alakazam",
                "opponent_routes_only": True,
                "source_dates": ["2026-07-20"],
                "source_archive": archive.name,
                "source_archive_sha256": sha256_file(archive),
                "classifier_sha256": classifier_digest,
            },
            stream,
        )
        pickle.dump(sequence, stream)
        pickle.dump(
            {
                "format": SHARD_FORMAT + "-footer",
                "format_version": SHARD_FORMAT_VERSION,
                "stats": {"records_kept": 1, "decisions_kept": 2},
            },
            stream,
        )
    manifest = feature_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                "selection": {"value": "alakazam", "opponent_routes_only": True},
                "shards": [
                    {
                        "path": feature.name,
                        "sha256": sha256_file(feature),
                        "stats": {"records_kept": 1, "decisions_kept": 2},
                    }
                ],
            }
        )
        + "\n"
    )
    lucario = EXPERT_IDS.index("lucario")
    counts = [0.0] * (len(EXPERT_IDS) + 1)
    counts[lucario] = 10.0
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(
        json.dumps(
            {
                "schema": PUBLIC_TREE_SCHEMA,
                "targets": list(EXPERT_IDS),
                "runtime_enabled": False,
                "prediction_contract": {
                    "route_class_names": list(EXPERT_IDS),
                    "route_output_width": len(EXPERT_IDS),
                    "adapter_count": len(EXPERT_IDS),
                    "unknown_is_separate_abstention": True,
                    "unknown_class_index": len(EXPERT_IDS),
                },
                "tree": {
                    "node_count": 1,
                    "class_names": list(EXPERT_IDS) + ["unknown"],
                    "children_left": [-1],
                    "children_right": [-1],
                    "feature_card_id": [-2],
                    "threshold": [-2.0],
                    "weighted_class_counts": [counts],
                },
            }
        )
        + "\n"
    )
    oracle = build_oracle(
        manifest,
        archive_dir,
        tmp_path / "oracle",
        mix=MIX,
        representatives=REPS,
        public_tree_path=tree_path,
    )
    oracle_payload = json.loads(oracle.read_text())
    index = oracle.parent / oracle_payload["shards"][0]["path"]
    row = json.loads(index.read_text().splitlines()[1])
    assert row["public_decisions"] == 2
    assert row["public_route_segments"] == [[1, 2]]
