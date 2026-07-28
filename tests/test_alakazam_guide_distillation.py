from __future__ import annotations

import pytest
import torch

from poke_bot import alakazam_heuristics, dataset, features
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.pure_rl import dataset_bridge
from poke_bot.pure_rl.shards import CompactDecision, CompactGame
from poke_bot.train import (
    count_usable_alakazam_guide_rows,
    masked_alakazam_guide_ce,
)


def _sparse(words: int) -> features.SparseVector:
    value = features.SparseVector()
    for i in range(words):
        value.word_start()
        value.add(i, 1.0)
    return value


def _stage(
    target: int = -1,
    confidence: float = 0.0,
) -> PolicyStage:
    return PolicyStage(
        options=_sparse(3),
        action_combos=[[0], [1], [2]],
        target_index=0,
        guide_target_index=target,
        guide_confidence=confidence,
    )


def _sequence(stage: PolicyStage) -> GameSequence:
    decision = DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS),
        options=stage.options,
        action=[0],
        action_combo_index=0,
        action_combos=stage.action_combos,
        env_step=0,
        policy_stages=[stage],
    )
    return GameSequence(
        episode_id="guide",
        seat=0,
        archetype="alakazam",
        opp_archetype="baseline",
        deck=[1] * 60,
        value=1.0,
        decisions=[decision],
    )


def test_masked_guide_ce_is_confidence_weighted() -> None:
    logits = torch.tensor([[0.1, -0.2, 4.0]], requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss, rows = masked_alakazam_guide_ce(
        log_probs,
        [0],
        [0.5],
        [3],
    )
    assert rows == 1
    assert float(loss.detach()) == pytest.approx(
        0.5 * -float(log_probs[0, 0].detach())
    )
    loss.backward()
    assert float(logits.grad[0, 0]) < 0.0
    assert float(logits.grad[0, 1]) > 0.0
    assert float(logits.grad[0, 2]) > 0.0


def test_guide_ce_masks_absent_zero_confidence_and_singleton_rows() -> None:
    logits = torch.log_softmax(torch.tensor([[1.0, 0.0, -1.0]]), dim=-1)
    for target, confidence, count in (
        (-1, 1.0, 3),
        (0, 0.0, 3),
        (0, 1.0, 1),
    ):
        loss, rows = masked_alakazam_guide_ce(
            logits,
            [target],
            [confidence],
            [count],
        )
        assert rows == 0
        assert float(loss) == 0.0


def test_guide_row_count_requires_a_real_comparison() -> None:
    assert count_usable_alakazam_guide_rows([_sequence(_stage())]) == 0
    assert (
        count_usable_alakazam_guide_rows(
            [_sequence(_stage(0, 0.75))]
        )
        == 1
    )


def test_featurize_step_attaches_deck_gated_aligned_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset.features,
        "factorized_teacher_forcing_stages",
        lambda _obs, _action: [([[0], [1]], 0)],
    )
    monkeypatch.setattr(
        dataset.features,
        "build_option_tokens",
        lambda _obs, combos: _sparse(len(combos)),
    )
    monkeypatch.setattr(
        dataset.features,
        "build_board_tokens",
        lambda _obs, _deck: _sparse(features.NUM_BOARD_TOKENS),
    )
    monkeypatch.setattr(alakazam_heuristics, "enabled", lambda: True)

    seen: dict[str, object] = {}

    def _guide(_obs, combos, *, deck):
        seen["combos"] = combos
        seen["deck"] = deck
        return [1.5, 0.0]

    monkeypatch.setattr(alakazam_heuristics, "guide_scores", _guide)
    sample = dataset.featurize_step(
        {"observation": {"public": True}, "action": [0]},
        [743] * 60,
        verify_info_set=False,
    )
    stage = sample.policy_stages[0]
    assert seen == {"combos": [[0], [1]], "deck": [743] * 60}
    assert stage.guide_target_index == 0
    assert stage.guide_confidence == 1.0


def test_featurize_step_masks_tied_or_partial_guide_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset.features,
        "factorized_teacher_forcing_stages",
        lambda _obs, _action: [([[0], [1]], 0)],
    )
    monkeypatch.setattr(
        dataset.features,
        "build_option_tokens",
        lambda _obs, combos: _sparse(len(combos)),
    )
    monkeypatch.setattr(
        dataset.features,
        "build_board_tokens",
        lambda _obs, _deck: _sparse(features.NUM_BOARD_TOKENS),
    )
    monkeypatch.setattr(alakazam_heuristics, "enabled", lambda: True)
    for scores in ([1.0, 1.0], [2.0, float("nan")]):
        monkeypatch.setattr(
            alakazam_heuristics,
            "guide_scores",
            lambda *_args, _scores=scores, **_kwargs: _scores,
        )
        sample = dataset.featurize_step(
            {"observation": {"public": True}, "action": [0]},
            [743] * 60,
            verify_info_set=False,
        )
        assert sample.policy_stages[0].guide_target_index == -1
        assert sample.policy_stages[0].guide_confidence == 0.0


def test_featurize_core_does_not_call_guide_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset.features,
        "factorized_teacher_forcing_stages",
        lambda _obs, _action: [([[0], [1]], 0)],
    )
    monkeypatch.setattr(
        dataset.features,
        "build_option_tokens",
        lambda _obs, combos: _sparse(len(combos)),
    )
    monkeypatch.setattr(
        dataset.features,
        "build_board_tokens",
        lambda _obs, _deck: _sparse(features.NUM_BOARD_TOKENS),
    )
    monkeypatch.setattr(alakazam_heuristics, "enabled", lambda: False)
    monkeypatch.setattr(
        alakazam_heuristics,
        "guide_scores",
        lambda *_args, **_kwargs: pytest.fail("core called guide scorer"),
    )
    sample = dataset.featurize_step(
        {"observation": {"public": True}, "action": [0]},
        [1] * 60,
        verify_info_set=False,
    )
    assert sample.policy_stages[0].guide_target_index == -1


def test_compact_bridge_preserves_guide_labels_when_rebuilding_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage(0, 0.8)
    sample = _sequence(stage).decisions[0]
    monkeypatch.setattr(dataset_bridge, "featurize_step", lambda *_a, **_k: sample)
    game = CompactGame(
        episode_id="compact-guide",
        seat=0,
        archetype="alakazam",
        opp_archetype="baseline",
        deck=[743] * 60,
        value=1.0,
        decisions=[
            CompactDecision(
                env_step=0,
                selected_index=1,
                n_options=3,
                action=[1],
                observation={"public": True},
            )
        ],
    )
    rebuilt = dataset_bridge.compact_game_to_sequence(game)
    assert rebuilt is not None
    rebuilt_stage = rebuilt.decisions[0].policy_stages[0]
    assert rebuilt_stage.target_index == 1
    assert rebuilt_stage.guide_target_index == 0
    assert rebuilt_stage.guide_confidence == pytest.approx(0.8)


def test_compact_bridge_preserves_exact_aux_and_attaches_tactical_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []
    sample = _sequence(_stage(0, 0.8)).decisions[0]

    def _featurize(step, *_args, **_kwargs):
        captured.append(step)
        sample.aux_labels = dict(step.get("aux_labels") or {})
        return sample

    monkeypatch.setattr(dataset_bridge, "featurize_step", _featurize)
    monkeypatch.setattr(
        dataset_bridge,
        "attach_blackwell_strategy_labels",
        lambda steps: steps[0]["aux_labels"].update(
            {"lethal_threat": 1.0, "prize_race": [0.5, 1.0]}
        ),
    )
    exact = {
        "opp_hand": [3, 5],
        "opp_deck_order": [7, 11],
        "privileged_label_source": "training_fork_exact_same_state",
    }
    game = CompactGame(
        episode_id="compact-aux",
        seat=0,
        archetype="alakazam",
        opp_archetype="crustle",
        deck=[743] * 60,
        value=1.0,
        decisions=[
            CompactDecision(
                env_step=0,
                selected_index=0,
                n_options=3,
                action=[0],
                observation={"public": True},
                aux_labels=exact,
            )
        ],
    )
    rebuilt = dataset_bridge.compact_game_to_sequence(game)
    assert rebuilt is not None
    labels = rebuilt.decisions[0].aux_labels
    assert labels["opp_hand"] == [3, 5]
    assert labels["opp_deck_order"] == [7, 11]
    assert labels["lethal_threat"] == 1.0
    assert labels["prize_race"] == [0.5, 1.0]
    assert captured[0]["observation"] == {"public": True}
