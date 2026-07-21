from __future__ import annotations

from pathlib import Path

from scripts.train_pure_rl import _build_collect_jobs, _record_to_compact_game


def test_record_compaction_keeps_exact_hidden_targets() -> None:
    record = {
        "episode_id": "all-heads",
        "seat": 0,
        "archetype": "alakazam",
        "opp_archetype": "crustle",
        "deck": [1] * 60,
        "value": 1.0,
        "steps": [
            {
                "env_step": 4,
                "observation": {"public": True},
                "action": [0],
                "aux_labels": {
                    "opp_hand": [2, 3],
                    "opp_deck_order": [4, 5],
                    "opp_prizes": [6],
                    "privileged_label_source": (
                        "training_fork_exact_same_state"
                    ),
                },
            }
        ],
        "factorized_policy_targets": [
            [{"selected_index": 0, "action_combos": [[0]]}]
        ],
    }
    compact = _record_to_compact_game(record)
    assert compact is not None
    assert compact.opp_archetype == "crustle"
    labels = compact.decisions[0].aux_labels
    assert labels["opp_hand"] == [2, 3]
    assert labels["opp_deck_order"] == [4, 5]
    assert labels["opp_prizes"] == [6]


def test_curriculum_selfplay_requests_exact_belief_labels() -> None:
    jobs, public = _build_collect_jobs(
        n_games=1,
        ckpt=Path("/tmp/model.pt"),
        digest="sha256:abc",
        model_generation=1,
        decks=[("alakazam", [1] * 60)],
        specs=[],
        seed=7,
        game_timeout_s=60,
        mode="specialist",
        self_play_frac=1.0,
        opponent_pool=[
            {"path": "/tmp/model.pt", "digest": "sha256:abc"}
        ],
    )
    assert not public
    assert len(jobs) == 1
    assert jobs[0]["collect_privileged_belief"] is True
    assert jobs[0]["opp_archetype"] == "alakazam"

