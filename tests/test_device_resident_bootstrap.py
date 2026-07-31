from pathlib import Path

import pytest
import torch

from poke_bot import archetypes, checkpoint, config, features
from poke_bot.dataset import (
    BootstrapDataset,
    DecisionSample,
    GameSequence,
    PolicyStage,
)
from poke_bot.aux_label_contract import AuxLabelContractError
from poke_bot.device_corpus import (
    DEVICE_CORPUS_PACKING_SCHEMA_VERSION,
    DeviceResidentBootstrapCorpus,
    _validated_exact_aux_targets,
)
from poke_bot.model import build_model
from poke_bot.train import (
    batch_losses,
    is_allowed_missing_belief_head_key,
    device_batch_losses,
    device_temporal_greedy_policy_targets,
    device_temporal_batch_losses,
    device_exact_batch_losses,
    device_exact_value_predictions,
    supervised_rehearsal_step,
    temporal_batches_for_game_ids,
)


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    sv = features.SparseVector()
    for word in range(words):
        sv.word_start()
        sv.add((offset + word) % 48, 1.0 + word / 100.0)
        if word % 2:
            sv.add((offset + word + 7) % 48, 0.25)
    return sv


def _decision(index: int, *, staged: bool) -> DecisionSample:
    options = _sparse(2, index + 3)
    stages = []
    if staged:
        stages = [
            PolicyStage(options=options, action_combos=[[0], [1]], target_index=1),
            PolicyStage(
                options=_sparse(3, index + 9),
                action_combos=[[0], [1], [2]],
                target_index=2,
            ),
        ]
    return DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS, index),
        options=options,
        action=[index % 2],
        action_combo_index=index % 2,
        action_combos=[[0], [1]],
        env_step=index,
        action_token=_sparse(1, index + 17),
        policy_stages=stages,
    )


def _game() -> GameSequence:
    return GameSequence(
        episode_id="device-corpus",
        seat=0,
        archetype="",
        opp_archetype="unknown",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0, staged=True), _decision(1, staged=False)],
    )


def _exact_temporal_game(*, include_hand: bool = True) -> GameSequence:
    game = _game()
    game.opp_archetype = "alakazam"
    for decision in game.decisions:
        decision.aux_labels = {
            "opp_hidden_remainder": [3, 4, 5],
            "lethal_threat": 1.0,
            "prize_race": [0.25, 0.75],
        }
        if include_hand:
            decision.aux_labels["opp_hand"] = [1, 2]
        if not decision.policy_stages:
            decision.policy_stages = [
                PolicyStage(
                    options=decision.options,
                    action_combos=decision.action_combos,
                    target_index=decision.action_combo_index,
                )
            ]
        for stage in decision.policy_stages:
            stage.guide_target_index = 0
            stage.guide_confidence = 0.75
    return game


def test_resident_aux_packer_uses_strict_shared_label_contract() -> None:
    assert DEVICE_CORPUS_PACKING_SCHEMA_VERSION == 6
    hand, has_hand, remainder, has_remainder, lethal, race = (
        _validated_exact_aux_targets(
            {
                "opp_hand": [],
                "opp_deck_order": [2, 3],
                "opp_prizes": [4],
                "lethal_threat": 1.0,
                "prize_race": [0.5, 1.0],
            },
            8,
        )
    )
    assert hand == [] and has_hand is True
    assert remainder == [2, 3, 4] and has_remainder is True
    assert lethal == 1.0 and race == (0.5, 1.0)

    with pytest.raises(AuxLabelContractError, match="opp_hand"):
        _validated_exact_aux_targets({"opp_hand": [0]}, 8)
    with pytest.raises(ValueError, match="lethal_threat"):
        _validated_exact_aux_targets({"lethal_threat": float("nan")}, 8)
    with pytest.raises(ValueError, match="prize_race"):
        _validated_exact_aux_targets({"prize_race": [0.5]}, 8)


def _history_model(*, seed: int) -> torch.nn.Module:
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        dense_card2vec=False,
        dropout=0.0,
    )
    torch.manual_seed(seed)
    return build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=len(archetypes.archetype_ids()) + 1,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )


def _gradient_norm(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum().item())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_device_resident_loss_matches_existing_stateless_hard_target_path() -> None:
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=0,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="stateless",
        kv_cache=False,
        dense_card2vec=False,
        dropout=0.0,
    )
    torch.manual_seed(7)
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
    )
    model.eval()
    game = _game()
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [game], [], device=torch.device("cpu")
    )
    sample_ids = torch.arange(corpus.train_samples)

    reference, reference_metrics = batch_losses(
        model,
        [game],
        value_weight=1.0,
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        lethal_threat_weight=0.0,
        prize_race_weight=0.0,
    )
    resident, resident_metrics = device_batch_losses(
        model, corpus, sample_ids, value_weight=1.0
    )

    assert corpus.decisions == 2
    assert corpus.train_samples == 3
    torch.testing.assert_close(resident, reference, rtol=1e-5, atol=1e-6)
    assert resident_metrics.n_decisions == reference_metrics.n_decisions == 3
    assert resident_metrics.policy_acc == reference_metrics.policy_acc
    assert abs(resident_metrics.policy_loss - reference_metrics.policy_loss) < 1e-5
    assert abs(resident_metrics.value_loss - reference_metrics.value_loss) < 1e-5


def test_device_resident_temporal_loss_matches_full_game_reference_path() -> None:
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        dense_card2vec=False,
        dropout=0.0,
    )
    torch.manual_seed(19)
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
    )
    model.eval()
    first = _game()
    second = _game()
    second.episode_id = "device-corpus-2"
    second.value = -1.0
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [first, second], [], device=torch.device("cpu")
    )

    reference, reference_metrics = batch_losses(
        model,
        [first, second],
        value_weight=1.0,
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        lethal_threat_weight=0.0,
        prize_race_weight=0.0,
    )
    resident, resident_metrics = device_temporal_batch_losses(
        model,
        corpus,
        torch.tensor([0, 1]),
        value_weight=1.0,
    )

    assert corpus.has_temporal_layout
    assert corpus.game_decision_offset.tolist() == [0, 2, 4]
    assert corpus.game_sample_offset.tolist() == [0, 3, 6]
    torch.testing.assert_close(resident, reference, rtol=1e-5, atol=1e-6)
    assert resident_metrics.n_games == reference_metrics.n_games == 2
    assert resident_metrics.n_decisions == reference_metrics.n_decisions == 6
    assert resident_metrics.policy_acc == reference_metrics.policy_acc
    assert resident_metrics.policy_loss == pytest.approx(
        reference_metrics.policy_loss, rel=1e-5, abs=1e-6
    )
    assert resident_metrics.value_loss == pytest.approx(
        reference_metrics.value_loss, rel=1e-5, abs=1e-6
    )


def test_temporal_resident_rehearsal_trains_every_available_head() -> None:
    model = _history_model(seed=23)
    model.eval()
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [_exact_temporal_game()],
        [],
        device=torch.device("cpu"),
        exact_card_vocab=64,
    )
    game_ids = torch.tensor([0])
    weights = {
        "aux_weight": 0.11,
        "opp_hand_weight": 0.12,
        "opp_remainder_weight": 0.13,
        "lethal_threat_weight": 0.14,
        "prize_race_weight": 0.15,
        "alakazam_guide_weight": 0.16,
    }

    policy_value, _ = device_temporal_batch_losses(
        model, corpus, game_ids, value_weight=1.0
    )
    total, metrics = device_temporal_batch_losses(
        model,
        corpus,
        game_ids,
        value_weight=1.0,
        **weights,
    )

    for field in (
        "aux_loss",
        "opp_hand_loss",
        "opp_remainder_loss",
        "lethal_threat_loss",
        "prize_race_loss",
        "alakazam_guide_loss",
    ):
        assert getattr(metrics, field) > 0.0, field
    archetype_label = archetypes.archetype_ids().index("alakazam")
    assert corpus.sample_aux_class is not None
    assert corpus.sample_aux_class.tolist() == [
        archetype_label
    ] * corpus.train_samples
    assert metrics.n_archetype_rows == corpus.train_samples
    assert metrics.n_opp_hand_rows == corpus.train_samples
    assert metrics.n_opp_remainder_rows == corpus.train_samples
    assert metrics.n_lethal_threat_rows == corpus.train_samples
    assert metrics.n_prize_race_rows == corpus.train_samples
    assert metrics.n_alakazam_guide_rows == corpus.train_samples
    expected = policy_value.detach() + sum(
        float(weights[weight_name]) * getattr(metrics, metric_name)
        for weight_name, metric_name in (
            ("aux_weight", "aux_loss"),
            ("opp_hand_weight", "opp_hand_loss"),
            ("opp_remainder_weight", "opp_remainder_loss"),
            ("lethal_threat_weight", "lethal_threat_loss"),
            ("prize_race_weight", "prize_race_loss"),
            ("alakazam_guide_weight", "alakazam_guide_loss"),
        )
    )
    torch.testing.assert_close(total.detach(), expected, rtol=1e-6, atol=1e-6)

    model.zero_grad(set_to_none=True)
    total.backward()
    for head in (
        model.aux_head,
        model.opp_hand_head,
        model.opp_remainder_head,
        model.lethal_threat_head,
        model.prize_race_head,
    ):
        assert _gradient_norm(head) > 0.0

    # Subtract the unchanged policy/value objective so this gradient is due to
    # guide distillation itself, not the selected-action policy target.
    model.zero_grad(set_to_none=True)
    guide_total, _ = device_temporal_batch_losses(
        model,
        corpus,
        game_ids,
        value_weight=1.0,
        alakazam_guide_weight=1.0,
    )
    base_total, _ = device_temporal_batch_losses(
        model, corpus, game_ids, value_weight=1.0
    )
    (guide_total - base_total).backward()
    assert _gradient_norm(model.policy_head) > 0.0

    def _guide_only_policy_gradient(weight: float) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        weighted_total, _ = device_temporal_batch_losses(
            model,
            corpus,
            game_ids,
            value_weight=1.0,
            alakazam_guide_weight=weight,
        )
        unweighted_total, _ = device_temporal_batch_losses(
            model,
            corpus,
            game_ids,
            value_weight=1.0,
        )
        (weighted_total - unweighted_total).backward()
        return torch.cat(
            [
                parameter.grad.detach().flatten()
                for parameter in model.policy_head.parameters()
                if parameter.grad is not None
            ]
        )

    quarter_weight_gradient = _guide_only_policy_gradient(0.25)
    half_weight_gradient = _guide_only_policy_gradient(0.50)
    assert float(quarter_weight_gradient.abs().sum()) > 0.0
    torch.testing.assert_close(
        half_weight_gradient,
        quarter_weight_gradient * 2.0,
        rtol=1e-5,
        atol=1e-6,
    )


def test_temporal_resident_teacher_policy_targets_are_masked_and_weighted() -> None:
    model = _history_model(seed=29)
    model.eval()
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [_game()], [], device=torch.device("cpu")
    )
    game_ids = torch.tensor([0])
    base, _ = device_temporal_batch_losses(model, corpus, game_ids)
    sample_ids, greedy = device_temporal_greedy_policy_targets(
        model, corpus, game_ids
    )
    assert sample_ids.tolist() == list(range(corpus.total_samples))
    assert bool((greedy < corpus.n_options.to(dtype=torch.long)).all())
    explicit_batches = temporal_batches_for_game_ids(
        corpus, game_ids, batch_size=2
    )
    assert [value for batch in explicit_batches for value in batch.tolist()] == [
        0
    ]
    teacher_targets = torch.zeros(
        corpus.total_samples, dtype=torch.long
    )
    teacher_targets[-1] = -1
    total, metrics = device_temporal_batch_losses(
        model,
        corpus,
        game_ids,
        teacher_policy_targets=teacher_targets,
        teacher_policy_weight=0.5,
    )
    assert metrics.n_teacher_policy_rows == corpus.total_samples - 1
    assert metrics.teacher_policy_loss > 0.0
    assert total.detach().item() == pytest.approx(
        base.detach().item() + 0.5 * metrics.teacher_policy_loss,
        rel=1e-5,
        abs=1e-6,
    )


def test_expert_rehearsal_materializes_and_trains_legacy_missing_heads(
    tmp_path: Path,
) -> None:
    card_vocab = features.card_vocab_size()
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        dense_card2vec=False,
        dropout=0.0,
    )
    torch.manual_seed(31)
    legacy = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=len(archetypes.archetype_ids()) + 1,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=card_vocab,
    )
    payload = checkpoint.build_checkpoint(model=legacy, model_config=cfg)
    payload["model_state_dict"] = {
        key: value
        for key, value in payload["model_state_dict"].items()
        if not is_allowed_missing_belief_head_key(key)
    }
    base = checkpoint.atomic_torch_save(payload, tmp_path / "legacy.pt")
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [_exact_temporal_game()],
        [],
        device=torch.device("cpu"),
        exact_card_vocab=card_vocab,
    )
    output = tmp_path / "rehearsed.pt"
    result = supervised_rehearsal_step(
        corpus,
        base_ckpt=base,
        output_path=output,
        parent_digest=checkpoint.checkpoint_digest(base),
        rehearsal_iteration=6,
        manifest_identity={"digest": "sha256:test"},
        epochs=1,
        lr=2e-5,
        requested_batch_size=8,
        seed=37,
        corpus_split_seed=41,
        device=torch.device("cpu"),
        aux_loss_weight=0.05,
        opp_hand_loss_weight=0.05,
        opp_remainder_loss_weight=0.05,
        lethal_threat_loss_weight=0.025,
        prize_race_loss_weight=0.025,
        alakazam_guide_loss_weight=0.05,
    )

    assert result["candidate_path"] == str(output)
    saved = checkpoint.load_checkpoint(output, map_location="cpu")
    state = saved["model_state_dict"]
    for name in (
        "opp_hand_head",
        "opp_remainder_head",
        "lethal_threat_head",
        "prize_race_head",
    ):
        assert f"{name}.weight" in state
    rehearsal = saved["extra"]["expert_rehearsal"]
    assert set(rehearsal["warm_started_belief_heads_before"]) == {
        "opp_hand_head",
        "opp_remainder_head",
        "lethal_threat_head",
        "prize_race_head",
    }
    assert rehearsal["warm_started_belief_heads_remaining"] == []
    assert rehearsal["optimizer_state_restored"] is False
    assert saved["provenance"]["warm_started_belief_heads"] == []
    assert rehearsal["train_metrics"]["n_opp_hand_rows"] > 0
    assert rehearsal["train_metrics"]["n_opp_remainder_rows"] > 0


def test_temporal_resident_absent_exact_hand_rows_stay_masked() -> None:
    model = _history_model(seed=29)
    model.eval()
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [_exact_temporal_game(include_hand=False)],
        [],
        device=torch.device("cpu"),
        exact_card_vocab=64,
    )
    game_ids = torch.tensor([0])

    hand_total, metrics = device_temporal_batch_losses(
        model,
        corpus,
        game_ids,
        opp_hand_weight=1.0,
    )
    base_total, _ = device_temporal_batch_losses(model, corpus, game_ids)

    assert metrics.n_opp_hand_rows == 0
    assert metrics.opp_hand_loss == 0.0
    # The remainder target proves this is an all-head exact pack rather than a
    # legacy policy/value-only corpus whose hand tensors are wholly absent.
    assert metrics.n_opp_remainder_rows == corpus.train_samples
    torch.testing.assert_close(hand_total.detach(), base_total.detach())
    model.zero_grad(set_to_none=True)
    (hand_total - base_total).backward()
    assert _gradient_norm(model.opp_hand_head) == 0.0


def test_streamed_exact_pack_labels_every_sample_archetype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _load_one_game(
        cls: type[BootstrapDataset],
        _path: Path,
        **_kwargs: object,
    ) -> BootstrapDataset:
        return cls([_exact_temporal_game()])

    monkeypatch.setattr(
        BootstrapDataset,
        "from_jsonl",
        classmethod(_load_one_game),
    )
    corpus = DeviceResidentBootstrapCorpus.from_exact_shards(
        [tmp_path / "train.jsonl"],
        [tmp_path / "val.jsonl"],
        cache_dir=tmp_path / "cache",
        max_context=8,
        card_vocab=64,
        device=torch.device("cpu"),
    )

    archetype_label = archetypes.archetype_ids().index("alakazam")
    assert corpus.sample_aux_class is not None
    assert corpus.sample_aux_class.tolist() == [
        archetype_label
    ] * corpus.total_samples


def test_exact_resident_awr_belief_and_strategy_targets_match_reference_path() -> None:
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=0,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="stateless",
        kv_cache=False,
        dense_card2vec=False,
        dropout=0.0,
    )
    torch.manual_seed(11)
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=len(archetypes.archetype_ids()) + 1,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )
    model.eval()
    game = _game()
    game.opp_archetype = "alakazam"
    for decision in game.decisions:
        decision.aux_labels = {
            "opp_hand": [1, 2],
            "opp_deck_order": [3, 4],
            "opp_prizes": [5],
            "lethal_threat": 1.0,
            "prize_race": [0.5, 1.0],
        }
        for stage in decision.policy_stages:
            stage.guide_target_index = 0
            stage.guide_confidence = 0.75

    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [game],
        [],
        device=torch.device("cpu"),
        exact_card_vocab=64,
    )
    sample_ids = torch.arange(corpus.train_samples)
    cache: dict[tuple[int, int, int], float] = {}
    batch_losses(
        model,
        [game],
        pure_rl=True,
        awr_capture_baseline=cache,
        aux_weight=0.0,
        opp_hand_weight=0.05,
        opp_remainder_weight=0.05,
        lethal_threat_weight=0.025,
        prize_race_weight=0.025,
        alakazam_guide_weight=0.05,
        entropy_bonus=0.01,
    )
    reference, reference_metrics = batch_losses(
        model,
        [game],
        pure_rl=True,
        awr_baseline_cache=cache,
        aux_weight=0.0,
        opp_hand_weight=0.05,
        opp_remainder_weight=0.05,
        lethal_threat_weight=0.025,
        prize_race_weight=0.025,
        alakazam_guide_weight=0.05,
        entropy_bonus=0.01,
    )
    baseline = device_exact_value_predictions(model, corpus, sample_ids)
    resident, resident_metrics = device_exact_batch_losses(
        model,
        corpus,
        sample_ids,
        baseline_pred=baseline,
        aux_weight=0.0,
        opp_hand_weight=0.05,
        opp_remainder_weight=0.05,
        lethal_threat_weight=0.025,
        prize_race_weight=0.025,
        alakazam_guide_weight=0.05,
    )

    for field in (
        "policy_loss",
        "value_loss",
        "opp_hand_loss",
        "opp_remainder_loss",
        "lethal_threat_loss",
        "prize_race_loss",
        "alakazam_guide_loss",
    ):
        assert getattr(resident_metrics, field) == pytest.approx(
            getattr(reference_metrics, field), rel=1e-5, abs=1e-6
        ), field
    torch.testing.assert_close(resident, reference, rtol=1e-5, atol=1e-6)
    assert resident_metrics.n_decisions == reference_metrics.n_decisions
    # Resident packs intentionally supervise the constant opponent archetype
    # on every valid sample.  The older object path uses one row per game.
    assert resident_metrics.n_archetype_rows == corpus.train_samples
    assert reference_metrics.n_archetype_rows == 1
    assert resident_metrics.n_opp_hand_rows == reference_metrics.n_opp_hand_rows
    assert (
        resident_metrics.n_opp_remainder_rows
        == reference_metrics.n_opp_remainder_rows
    )
    assert (
        resident_metrics.n_lethal_threat_rows
        == reference_metrics.n_lethal_threat_rows
    )
    assert resident_metrics.n_prize_race_rows == reference_metrics.n_prize_race_rows
    assert (
        resident_metrics.n_alakazam_guide_rows
        == reference_metrics.n_alakazam_guide_rows
    )
