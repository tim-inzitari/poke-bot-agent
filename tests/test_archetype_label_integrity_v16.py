from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot.train import _archetype_label
from scripts.train_round_robin import _build_selfplay_record


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts/train_pure_rl.py"
SPEC = importlib.util.spec_from_file_location(
    "train_pure_rl_v16_archetype_labels", TRAINER_PATH
)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


GATE_ARCHETYPES = {
    "pilkwang-meta-20260708": "crustle",
    "yaminh-ai-challenge": "lucario",
    "aman-crustle-fighting": "lucario",
    "penguin-public-scores-915": "lucario",
    "archaludon-ex": "archaludon-ex",
    "yaroslav-lucario-v2-crustle": "lucario",
    "makthanithin-1084-5": "lucario",
    "lucifer19-battlecore": "archaludon-ex",
}
def _history_target() -> dict:
    return {
        "observation": {
            "select": {
                "option": [{"type": 14}, {"type": 14}],
                "minCount": 1,
                "maxCount": 1,
            }
        },
        "action": [0],
        "factorized_stages": [
            {
                "action_combos": [[0], [1]],
                "policy": [0.75, 0.25],
            }
        ],
        "target_source": "history_policy",
    }


def _record(*, opponent_id: str, opponent_archetype: str | None) -> dict:
    record = _build_selfplay_record(
        [_history_target()],
        our_deck=[1] * 60,
        our_seat=0,
        value=1.0,
        opp_id=opponent_id,
        opp_archetype=opponent_archetype,
        archetype="alakazam",
        seed=17,
        target_provenance={"pure_rl": True},
    )
    assert record is not None
    return record


def _stats() -> dict:
    return {
        "ok": 0,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": 0,
        "self_play": 0,
        "leaf_remote": 0,
        "multi_env_games": 0,
        "leaf_modes": {},
    }


class _Writer:
    def __init__(self) -> None:
        self.games = []

    @property
    def n_decisions(self) -> int:
        return sum(len(game.decisions) for game in self.games)

    def write_game(self, game) -> None:
        self.games.append(game)


def test_worker_record_preserves_package_and_canonical_archetype_separately() -> None:
    record = _record(
        opponent_id="yaminh-ai-challenge",
        opponent_archetype="lucario",
    )
    assert record["opp_archetype"] == "lucario"
    assert record["target_provenance"]["opponent_id"] == "yaminh-ai-challenge"
    assert record["target_provenance"]["opponent_archetype_id"] == "lucario"


def test_forced_go_first_row_does_not_create_a_mixed_policy_source() -> None:
    forced = {
        "observation": {
            "select": {
                "option": [{"type": 14}, {"type": 14}],
                "minCount": 1,
                "maxCount": 1,
            }
        },
        "action": [0],
        "diagnostics": {
            "target_source": "forced_go_first_contract",
            "trusted": True,
        },
    }
    record = _build_selfplay_record(
        [forced, _history_target()],
        our_deck=[1] * 60,
        our_seat=0,
        value=1.0,
        opp_id="yaminh-ai-challenge",
        opp_archetype="lucario",
        archetype="alakazam",
        seed=18,
        target_provenance={"pure_rl": True},
    )
    assert record is not None
    assert len(record["steps"]) == 2
    assert record["target_provenance"]["target_source"] == "history_policy"
    assert record["factorized_policy_targets"][0][0]["policy"] == [1.0, 0.0]


def test_unmapped_package_identity_cannot_become_a_known_archetype_label() -> None:
    record = _record(
        opponent_id="unknown-public-package-v7",
        opponent_archetype=None,
    )
    assert record["opp_archetype"] == "unknown-public-package-v7"
    assert _archetype_label(record["opp_archetype"]) is None


def test_all_gate_jobs_pin_package_and_canonical_archetype_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer, "_spec_payload", lambda spec: {"id": spec.id})
    specs = [SimpleNamespace(id=value) for value in GATE_ARCHETYPES]
    _self_jobs, public_jobs = trainer._build_collect_jobs(
        n_games=80,
        ckpt=Path("/tmp/candidate.pt"),
        digest="sha256:candidate",
        model_generation=3,
        decks=[("alakazam", [1] * 60)],
        specs=[],
        seed=20_000,
        game_timeout_s=30,
        mode="specialist",
        self_play_frac=0.0,
        iteration=2,
        priority_specs=specs,
        priority_frac=1.0,
        priority_weights={value: 1.0 for value in GATE_ARCHETYPES},
        priority_group=trainer.STRONG_PUBLIC_PRACTICE_GROUP,
        priority_temperature=0.35,
        priority_archetypes=GATE_ARCHETYPES,
        priority_context={
            "active_gate_id": "alakazam-strong-public-roster-v1",
            "formal_eval": False,
            "seed_namespace": "train/strong-public-practice-v1",
        },
    )
    assert {job["opponent_id"] for job in public_jobs} == set(GATE_ARCHETYPES)
    for job in public_jobs:
        opponent_id = job["opponent_id"]
        provenance = job["target_provenance"]
        assert job["opp_archetype"] == GATE_ARCHETYPES[opponent_id]
        assert provenance["opponent_id"] == opponent_id
        assert (
            provenance["opponent_archetype_id"]
            == GATE_ARCHETYPES[opponent_id]
        )


def test_coordinator_repairs_stale_remote_label_before_compaction() -> None:
    stale = _record(
        opponent_id="yaminh-ai-challenge",
        opponent_archetype=None,
    )
    writer = _Writer()
    stats = _stats()
    seen: set[int] = set()
    successful: set[int] = set()
    written: set[int] = set()
    trainer._consume_results(
        [
            {
                "job_index": 41,
                "opponent_id": "yaminh-ai-challenge",
                "our_seat": 0,
                "winner": 0,
                "record_json": stale,
            }
        ],
        writer,
        [],
        stats,
        practice_record_contracts={
            41: {
                "opponent_id": "yaminh-ai-challenge",
                "opponent_archetype_id": "lucario",
                "active_gate_id": "alakazam-strong-public-roster-v1",
                "our_seat": "0",
            }
        },
        practice_seen_indices=seen,
        practice_successful_indices=successful,
        practice_written_indices=written,
    )
    assert seen == successful == written == {41}
    assert stats["strong_public_practice_records_repaired"] == 1
    assert len(writer.games) == 1
    game = writer.games[0]
    assert game.opp_archetype == "lucario"
    assert game.target_provenance["opponent_id"] == "yaminh-ai-challenge"
    assert game.target_provenance["opponent_archetype_id"] == "lucario"


def test_coordinator_rejects_wrong_or_duplicate_practice_result_identity() -> None:
    contract = {
        7: {
            "opponent_id": "pilkwang-meta-20260708",
            "opponent_archetype_id": "crustle",
            "active_gate_id": "alakazam-strong-public-roster-v1",
            "our_seat": "1",
        }
    }
    with pytest.raises(RuntimeError, match="result identity mismatch"):
        trainer._consume_results(
            [
                {
                    "job_index": 7,
                    "opponent_id": "yaminh-ai-challenge",
                    "our_seat": 1,
                }
            ],
            _Writer(),
            [],
            _stats(),
            practice_record_contracts=contract,
            practice_seen_indices=set(),
            practice_successful_indices=set(),
            practice_written_indices=set(),
        )

    valid = {
        "job_index": 7,
        "opponent_id": "pilkwang-meta-20260708",
        "our_seat": 1,
        "winner": 1,
        "record_json": _record(
            opponent_id="pilkwang-meta-20260708",
            opponent_archetype="crustle",
        ),
    }
    with pytest.raises(RuntimeError, match="duplicate strong-public practice result"):
        trainer._consume_results(
            [valid, valid],
            _Writer(),
            [],
            _stats(),
            practice_record_contracts=contract,
            practice_seen_indices=set(),
            practice_successful_indices=set(),
            practice_written_indices=set(),
        )
