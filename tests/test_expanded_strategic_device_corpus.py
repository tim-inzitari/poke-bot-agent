from __future__ import annotations

import copy

import pytest
import torch

from poke_bot import archetypes, checkpoint, config, features
from poke_bot.dataset import (
    BootstrapDataset,
    DecisionSample,
    GameSequence,
    PolicyStage,
)
from poke_bot.device_corpus import (
    DEVICE_CORPUS_PACKING_SCHEMA_VERSION,
    DeviceResidentBootstrapCorpus,
)
from poke_bot.model import EXPANDED_HEAD_NAMES, build_model
from poke_bot.pure_rl.expert_cpu_pack import (
    EXPERT_CPU_PACK_SCHEMA_VERSION,
    ExpertCpuPackCache,
    ExpertCpuPackError,
    ExpertCpuPackKey,
    validate_cpu_corpus,
)
from poke_bot.strategic_heads import (
    ACTION_FACTOR_NAMES,
    ACTION_UTILITY_NAMES,
    EXPANDED_STRATEGIC_KEY,
    EXPANDED_STRATEGIC_SCHEMA,
    EXPANDED_STRATEGIC_SCHEMA_DIGEST,
    EXPANDED_STRATEGIC_SCHEMA_VERSION,
    OPPONENT_RESPONSE_NAMES,
    RESOURCE_FORECAST_NAMES,
    TACTICAL_HORIZONS,
    TACTICAL_OUTCOME_NAMES,
)
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS
from poke_bot.strategic_schedule import expanded_head_epoch_plan
from poke_bot.train import (
    device_temporal_batch_losses,
    supervised_rehearsal_step,
)
from scripts.run_starmie_expert_bootstrap import load_expanded_head_contract


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    result = features.SparseVector()
    for word in range(words):
        result.word_start()
        result.add((offset + word) % 32, 1.0)
    return result


def _target(factors: list[dict[str, bool]]) -> dict:
    action_utility = [
        float(index + 1) for index in range(len(ACTION_UTILITY_NAMES))
    ]
    action_utility[-1] = 1.0
    tactical = [
        [
            float(10 * horizon + column)
            for column in range(len(TACTICAL_OUTCOME_NAMES))
        ]
        for horizon in range(len(TACTICAL_HORIZONS))
    ]
    for row in tactical:
        row[2] = 1.0
        row[3] = 0.0
    resources = [
        float(index) for index in range(len(RESOURCE_FORECAST_NAMES))
    ]
    resources[4] = 1.0
    resources[5] = 0.0
    return {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "version": EXPANDED_STRATEGIC_SCHEMA_VERSION,
        "digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
        "action_factors": factors,
        "action_utility": {
            "values": action_utility,
            "mask": [True] * len(ACTION_UTILITY_NAMES),
        },
        "tactical_outcomes": {
            "values": tactical,
            "mask": [
                [True] * len(TACTICAL_OUTCOME_NAMES)
                for _ in TACTICAL_HORIZONS
            ],
        },
        "opponent_response": {
            "values": [float(index % 2) for index in range(len(OPPONENT_RESPONSE_NAMES))],
            "mask": [True] * len(OPPONENT_RESPONSE_NAMES),
        },
        "resource_forecast": {
            "values": resources,
            "mask": [True] * len(RESOURCE_FORECAST_NAMES),
        },
        "game_phase": 2,
        "outcome_class": 2,
        "remaining_turns_log1p": 1.25,
        "provenance": {
            "trajectory": "full_untruncated_same_seat",
            "transition_after": "explicit_public_post_action",
            "terminal_complete": True,
            "target_only": True,
        },
    }


def _decision(
    index: int,
    *,
    stages: list[PolicyStage],
    expanded: dict | None,
) -> DecisionSample:
    first = stages[0]
    return DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS, index),
        options=first.options,
        action=[0],
        action_combo_index=first.target_index,
        action_combos=first.action_combos,
        env_step=index,
        action_token=_sparse(1, index + 16),
        policy_stages=stages,
        aux_labels=(
            {} if expanded is None else {EXPANDED_STRATEGIC_KEY: expanded}
        ),
    )


def _game() -> GameSequence:
    first_stages = [
        PolicyStage(
            options=_sparse(2, 3),
            action_combos=[[0], [1]],
            target_index=1,
        ),
        PolicyStage(
            options=_sparse(3, 7),
            action_combos=[[0], [1], [2]],
            target_index=2,
        ),
    ]
    second_stages = [
        PolicyStage(
            options=_sparse(2, 11),
            action_combos=[[0], [1]],
            target_index=0,
        )
    ]
    factors = [
        {"action_type": True, "target": False, "resource": False},
        {"action_type": False, "target": True, "resource": True},
    ]
    return GameSequence(
        episode_id="expanded-resident",
        seat=0,
        archetype="",
        opp_archetype="unknown",
        deck=[1] * 60,
        value=1.0,
        decisions=[
            _decision(0, stages=first_stages, expanded=_target(factors)),
            _decision(1, stages=second_stages, expanded=None),
        ],
    )


def test_expanded_targets_preserve_sample_and_decision_alignment() -> None:
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [_game()],
        [],
        device=torch.device("cpu"),
    )

    assert DEVICE_CORPUS_PACKING_SCHEMA_VERSION == 4
    assert corpus.has_expanded_strategic_targets
    assert corpus.expanded_strategic_schema == EXPANDED_STRATEGIC_SCHEMA
    assert (
        corpus.expanded_strategic_schema_digest
        == EXPANDED_STRATEGIC_SCHEMA_DIGEST
    )
    assert corpus.sample_board.tolist() == [0, 0, 1]
    assert corpus.strategic_action_q_target.tolist() == [1.0, 1.0, 0.0]
    assert corpus.strategic_action_q_mask.tolist() == [1, 1, 0]
    assert corpus.strategic_action_factor_mask.tolist() == [
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]
    assert corpus.strategic_action_utility_mask.tolist() == [
        [0] * len(ACTION_UTILITY_NAMES),
        [1] * len(ACTION_UTILITY_NAMES),
        [0] * len(ACTION_UTILITY_NAMES),
    ]
    assert corpus.strategic_tactical_outcome_target.shape == (
        2,
        len(TACTICAL_HORIZONS),
        len(TACTICAL_OUTCOME_NAMES),
    )
    assert bool(torch.all(corpus.strategic_tactical_outcome_mask[0] == 1))
    assert bool(torch.all(corpus.strategic_tactical_outcome_mask[1] == 0))
    assert corpus.strategic_game_phase_target.tolist() == [2, -1]
    assert corpus.strategic_outcome_class_target.tolist() == [2, -1]
    assert corpus.strategic_remaining_turns_target.tolist() == [1.25, 0.0]
    validate_cpu_corpus(corpus)


def test_temporal_resident_path_trains_all_enabled_expanded_heads_once() -> None:
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [_game()],
        [],
        device=torch.device("cpu"),
    )
    model_cfg = config.ModelConfig(
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
        expanded_heads_enabled=True,
    )
    torch.manual_seed(20260724)
    model = build_model(
        model_cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=len(archetypes.archetype_ids()) + 1,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )
    model.train()
    total, metrics = device_temporal_batch_losses(
        model,
        corpus,
        torch.tensor([0], dtype=torch.long),
        expanded_head_weights={name: 1.0 for name in EXPANDED_HEAD_IDS},
    )
    expanded = metrics.expanded_head_metrics
    assert expanded["labeled"] == {
        "action_q": 2,
        "action_type": 1,
        "action_target": 1,
        "action_resource": 1,
        "action_utility": 1,
        "tactical_outcome": 1,
        "opponent_response": 1,
        "resource_forecast": 1,
        "game_phase": 1,
        "outcome_distribution": 1,
        "remaining_turns": 1,
    }
    total.backward()
    for module_name in EXPANDED_HEAD_NAMES:
        module = getattr(model, module_name)
        assert module is not None
        assert any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.detach().abs().sum().item()) > 0.0
            for parameter in module.parameters()
        ), module_name


def test_supervised_rehearsal_persists_stage_scoped_expanded_contract(
    tmp_path,
) -> None:
    train_game = _game()
    val_game = copy.deepcopy(train_game)
    val_game.episode_id = "expanded-resident-validation"
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [train_game],
        [val_game],
        device=torch.device("cpu"),
    )
    model_cfg = config.ModelConfig(
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
        expanded_heads_enabled=True,
    )
    torch.manual_seed(20260725)
    source_model = build_model(
        model_cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=len(archetypes.archetype_ids()) + 1,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )
    base_payload = checkpoint.build_checkpoint(
        model=source_model,
        model_config=model_cfg,
        archetype_id="core",
        model_id="expanded-stage-source",
    )
    base_path = checkpoint.atomic_torch_save(
        base_payload, tmp_path / "expanded-source.pt"
    )
    raw_schedule, _identity = load_expanded_head_contract()
    plan = expanded_head_epoch_plan(raw_schedule, 1)
    output = tmp_path / "expanded-stage-1.pt"
    result = supervised_rehearsal_step(
        corpus,
        base_ckpt=base_path,
        output_path=output,
        parent_digest=checkpoint.checkpoint_digest(base_path),
        rehearsal_iteration=1,
        manifest_identity={
            "digest": "sha256:" + "d" * 64,
            "expanded_strategic_targets": {
                "schema": EXPANDED_STRATEGIC_SCHEMA,
                "digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
            },
        },
        epochs=1,
        lr=2e-5,
        requested_batch_size=8,
        seed=20260726,
        corpus_split_seed=20260727,
        device=torch.device("cpu"),
        expanded_head_loss_weights=dict(plan.loss_weights),
        expanded_head_schedule=plan.as_dict(),
        output_archetype_id="test-expanded",
        output_model_id="test-expanded.epoch01",
    )
    contract = result["expanded_head_training"]
    assert contract["runtime_enabled_heads"] == []
    assert contract["shadow_only"] is True
    assert contract["flat_policy_authoritative"] is True
    assert contract["authoritative_action_path"] == "flat_policy"
    assert contract["gradient_enabled_heads"] == list(plan.enabled_heads)
    assert contract["trained_this_epoch"] == list(plan.enabled_heads)
    assert contract["target_schema_digest"] == EXPANDED_STRATEGIC_SCHEMA_DIGEST
    assert all(
        int(contract["heads"][name]["train_labeled_rows"]) > 0
        and int(contract["heads"][name]["validation_labeled_rows"]) > 0
        for name in plan.enabled_heads
    )

    saved = checkpoint.load_checkpoint(output, map_location="cpu")
    source_state = base_payload["model_state_dict"]
    saved_state = saved["model_state_dict"]
    enabled_modules = {f"{name}_head" for name in plan.enabled_heads}
    for module_name in EXPANDED_HEAD_NAMES:
        keys = [
            key
            for key in source_state
            if key.startswith(f"{module_name}.")
        ]
        assert keys
        changed = any(
            not torch.equal(source_state[key], saved_state[key]) for key in keys
        )
        assert changed is (module_name in enabled_modules)


def test_supervised_rehearsal_records_fused_policy_as_authoritative(
    tmp_path,
) -> None:
    train_game = _game()
    val_game = copy.deepcopy(train_game)
    val_game.episode_id = "expanded-fused-validation"
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [train_game],
        [val_game],
        device=torch.device("cpu"),
    )
    model_cfg = config.ModelConfig(
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
        expanded_heads_enabled=True,
        decision_fusion_enabled=True,
        decision_fusion_runtime_enabled=True,
    )
    source_model = build_model(
        model_cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=len(archetypes.archetype_ids()) + 1,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )
    base_path = checkpoint.atomic_torch_save(
        checkpoint.build_checkpoint(
            model=source_model,
            model_config=model_cfg,
            archetype_id="core",
            model_id="expanded-fused-source",
        ),
        tmp_path / "expanded-fused-source.pt",
    )
    raw_schedule, _identity = load_expanded_head_contract()
    plan = expanded_head_epoch_plan(raw_schedule, 1)
    result = supervised_rehearsal_step(
        corpus,
        base_ckpt=base_path,
        output_path=tmp_path / "expanded-fused-stage-1.pt",
        parent_digest=checkpoint.checkpoint_digest(base_path),
        rehearsal_iteration=1,
        manifest_identity={
            "digest": "sha256:" + "e" * 64,
            "expanded_strategic_targets": {
                "schema": EXPANDED_STRATEGIC_SCHEMA,
                "digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
            },
        },
        epochs=1,
        lr=2e-5,
        requested_batch_size=8,
        seed=20260726,
        corpus_split_seed=20260727,
        device=torch.device("cpu"),
        expanded_head_loss_weights=dict(plan.loss_weights),
        expanded_head_schedule=plan.as_dict(),
        output_archetype_id="test-expanded-fused",
        output_model_id="test-expanded-fused.epoch01",
    )
    contract = result["expanded_head_training"]
    assert contract["shadow_only"] is False
    assert contract["flat_policy_authoritative"] is False
    assert contract["authoritative_action_path"] == "fused_policy"


def test_present_stage_count_mismatch_fails_closed() -> None:
    game = _game()
    target = game.decisions[0].aux_labels[EXPANDED_STRATEGIC_KEY]
    target["action_factors"] = target["action_factors"][:1]
    with pytest.raises(ValueError, match="do not align"):
        DeviceResidentBootstrapCorpus.from_splits(
            [game],
            [],
            device=torch.device("cpu"),
        )


def test_streamed_exact_shards_keep_expanded_alignment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_one_game(
        cls: type[BootstrapDataset],
        _path,
        **_kwargs,
    ) -> BootstrapDataset:
        return cls([_game()])

    monkeypatch.setattr(
        BootstrapDataset,
        "from_jsonl",
        classmethod(load_one_game),
    )
    corpus = DeviceResidentBootstrapCorpus.from_exact_shards(
        [tmp_path / "train.jsonl"],
        [tmp_path / "val.jsonl"],
        cache_dir=tmp_path / "cache",
        max_context=320,
        card_vocab=64,
        device=torch.device("cpu"),
    )
    assert corpus.train_samples == corpus.val_samples == 3
    assert corpus.decisions == 4
    assert corpus.strategic_action_factor_mask.tolist() == [
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]
    assert corpus.strategic_tactical_outcome_target.shape[0] == 4


def test_cache_identity_pins_pack_and_target_schema() -> None:
    key = ExpertCpuPackKey(
        manifest_digest="sha256:" + "a" * 64,
        split_seed=7,
        val_frac=0.1,
        max_context=320,
    )
    contract = key.contract()
    assert contract["packing_schema"] == DEVICE_CORPUS_PACKING_SCHEMA_VERSION
    assert contract["cache_schema"] == EXPERT_CPU_PACK_SCHEMA_VERSION
    assert contract["expanded_strategic_schema"] == EXPANDED_STRATEGIC_SCHEMA
    assert (
        contract["expanded_strategic_schema_digest"]
        == EXPANDED_STRATEGIC_SCHEMA_DIGEST
    )
    with pytest.raises(ValueError, match="packing schema is incompatible"):
        ExpertCpuPackKey(
            manifest_digest="sha256:" + "a" * 64,
            split_seed=7,
            val_frac=0.1,
            max_context=320,
            packing_schema=DEVICE_CORPUS_PACKING_SCHEMA_VERSION - 1,
        ).contract()


def test_expanded_targets_survive_durable_cpu_pack_round_trip(tmp_path) -> None:
    key = ExpertCpuPackKey(
        manifest_digest="sha256:" + "b" * 64,
        split_seed=11,
        val_frac=0.1,
        max_context=320,
    )
    cache = ExpertCpuPackCache(tmp_path)
    _, first = cache.load_or_build(
        key,
        lambda: DeviceResidentBootstrapCorpus.from_splits(
            [_game()],
            [],
            device=torch.device("cpu"),
        ),
    )
    assert first["cache_hit"] is False
    loaded, second = ExpertCpuPackCache(tmp_path).load_or_build(
        key,
        lambda: (_ for _ in ()).throw(
            AssertionError("expanded strategic CPU pack unexpectedly rebuilt")
        ),
    )
    assert second["cache_hit"] is True
    assert loaded.has_expanded_strategic_targets
    assert loaded.strategic_action_factor_mask.tolist() == [
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]
    assert (
        loaded.expanded_strategic_schema_digest
        == EXPANDED_STRATEGIC_SCHEMA_DIGEST
    )


def test_cpu_validator_rejects_partial_or_misaligned_strategic_layout() -> None:
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [_game()],
        [],
        device=torch.device("cpu"),
    )
    partial_tensors = corpus.tensor_state()
    partial_tensors.pop("strategic_action_q_mask")
    partial = DeviceResidentBootstrapCorpus.from_packed_state(
        tensors=partial_tensors,
        scalars={
            **corpus.scalar_state(),
            "input_bytes": sum(
                int(value.numel()) * int(value.element_size())
                for value in partial_tensors.values()
            ),
        },
    )
    with pytest.raises(ExpertCpuPackError, match="partial expanded-strategic"):
        validate_cpu_corpus(partial)

    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [_game()],
        [],
        device=torch.device("cpu"),
    )
    tensors = copy.copy(corpus.tensor_state())
    tensors["strategic_action_factor_mask"] = torch.zeros(
        corpus.total_samples + 1,
        len(ACTION_FACTOR_NAMES),
        dtype=torch.uint8,
    )
    malformed = DeviceResidentBootstrapCorpus.from_packed_state(
        tensors=tensors,
        scalars={
            **corpus.scalar_state(),
            "input_bytes": sum(
                int(value.numel()) * int(value.element_size())
                for value in tensors.values()
            ),
        },
    )
    with pytest.raises(ExpertCpuPackError, match="shape mismatch"):
        validate_cpu_corpus(malformed)
