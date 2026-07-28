from __future__ import annotations

import copy

import pytest
import torch

from poke_bot import archetypes, features
from poke_bot.config import ModelConfig
from poke_bot.dormant_adapter_compat import validate_zero_dormant_checkpoint
from poke_bot.matchup_adapters import (
    EXPERT_IDS,
    ZERO_DORMANT_CHECKPOINT_SCHEMA,
    MatchupAdapterBank,
)
from poke_bot.matchup_adapters_v6 import (
    ADAPTER_CHECKPOINT_FORMAT,
    LEGACY_V5_PREFIX_LENGTH,
    SLOT_CAPACITY,
    MatchupAdapterBankV6,
    allocate_archetype,
    load_slot_registry,
    migrate_v5_adapter_state_dict,
    migrate_v5_checkpoint_payload,
    migrate_v5_named_optimizer_state,
    migrate_v5_positional_optimizer_state,
    project_v6_adapter_state_to_v5,
    registry_digest,
    resolve_ptcgreplay_mapping,
    retire_archetype,
    route_for_archetype,
)
from poke_bot.model import TemporalCabtTransformer
from poke_bot.train import load_append_only_matchup_adapter_optimizer_state


def test_registry_is_fixed_capacity_and_preserves_v5_prefix() -> None:
    registry = load_slot_registry()
    assert registry["slot_capacity"] == SLOT_CAPACITY == 64
    assert tuple(
        row["archetype_id"]
        for row in registry["slots"][:LEGACY_V5_PREFIX_LENGTH]
    ) == EXPERT_IDS
    assert len(registry["active_expert_ids"]) == 18
    assert sum(row["status"] == "unused" for row in registry["slots"]) == 46
    assert registry_digest(registry).startswith("sha256:")


def test_model_selects_v6_only_from_explicit_serialized_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(features, "decoder_binding_offset", lambda: 16)
    monkeypatch.setattr(features, "card_vocab_size", lambda: 32)
    monkeypatch.setattr(features, "attack_vocab_size", lambda: 32)
    common = dict(
        d_model=96,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=192,
        dense_card2vec=False,
    )
    legacy = TemporalCabtTransformer(
        ModelConfig(**common),
        encoder_vocab=32,
        decoder_vocab=32,
        belief_card_vocab=32,
    )
    assert isinstance(legacy.matchup_adapter_bank, MatchupAdapterBank)
    v6 = TemporalCabtTransformer(
        ModelConfig(
            **common,
            matchup_adapter_format=ADAPTER_CHECKPOINT_FORMAT,
            matchup_adapter_registry=load_slot_registry(),
        ),
        encoder_vocab=32,
        decoder_vocab=32,
        belief_card_vocab=32,
    )
    assert isinstance(v6.matchup_adapter_bank, MatchupAdapterBankV6)
    assert len(v6.matchup_adapter_bank.experts) == SLOT_CAPACITY


def test_v5_to_v6_copies_prefix_bit_exact_and_zeroes_unused_slots() -> None:
    torch.manual_seed(13)
    v5 = MatchupAdapterBank()
    with torch.no_grad():
        for route, expert in enumerate(v5.experts):
            expert.down.bias.fill_(route + 0.125)
            expert.up.bias.fill_(route + 0.25)
    migrated = migrate_v5_adapter_state_dict(v5.state_dict())
    v6 = MatchupAdapterBankV6()
    v6.load_state_dict(migrated, strict=True)

    for name, value in v5.state_dict().items():
        assert torch.equal(value, v6.state_dict()[name])
    for name, value in v6.state_dict().items():
        route = int(name.split(".")[1])
        if route >= LEGACY_V5_PREFIX_LENGTH:
            assert torch.count_nonzero(value).item() == 0
    assert not any(parameter.requires_grad for parameter in v6.parameters())


def test_registry_add_and_retire_never_change_physical_bank_shape() -> None:
    registry = load_slot_registry()
    original = MatchupAdapterBankV6(registry=registry)
    original_keys = tuple(original.state_dict())
    original_shapes = {
        name: tuple(value.shape)
        for name, value in original.state_dict().items()
    }

    added = allocate_archetype(registry, "future-archetype")
    assert added["slots"][18]["archetype_id"] == "future-archetype"
    assert added["slots"][18]["status"] == "dormant"
    assert route_for_archetype("future-archetype", registry=added) == 18
    expanded = MatchupAdapterBankV6(registry=added)
    assert tuple(expanded.state_dict()) == original_keys
    assert {
        name: tuple(value.shape)
        for name, value in expanded.state_dict().items()
    } == original_shapes

    retired = retire_archetype(added, "future-archetype")
    assert retired["slots"][18]["status"] == "retired"
    assert retired["slots"][18]["archetype_id"] == "future-archetype"
    assert route_for_archetype("future-archetype", registry=retired) == -1
    second = allocate_archetype(retired, "later-archetype")
    assert second["slots"][18]["status"] == "retired"
    assert second["slots"][19]["archetype_id"] == "later-archetype"


def test_only_authorized_slots_receive_gradients() -> None:
    registry = allocate_archetype(load_slot_registry(), "future-archetype")
    bank = MatchupAdapterBankV6(enabled=True, registry=registry)
    assert bank.authorize_slots_for_training(["future-archetype"]) == (18,)
    assert all(parameter.requires_grad for parameter in bank.experts[18].parameters())
    assert not any(parameter.requires_grad for parameter in bank.experts[0].parameters())
    assert torch.count_nonzero(bank.experts[18].down.weight).item() > 0
    assert torch.count_nonzero(bank.experts[18].up.weight).item() == 0


def test_v5_projection_is_guarded_against_v6_only_identity_or_weights() -> None:
    registry = load_slot_registry()
    v6 = MatchupAdapterBankV6(registry=registry)
    projected = project_v6_adapter_state_to_v5(v6.state_dict(), registry=registry)
    assert set(projected) == set(MatchupAdapterBank().state_dict())

    added = allocate_archetype(registry, "future-archetype")
    with pytest.raises(ValueError, match="active V6-only"):
        project_v6_adapter_state_to_v5(v6.state_dict(), registry=added)

    retired = retire_archetype(added, "future-archetype")
    trained = copy.deepcopy(v6.state_dict())
    trained["experts.18.up.bias"].fill_(1)
    with pytest.raises(ValueError, match="trained V6-only"):
        project_v6_adapter_state_to_v5(trained, registry=retired)


def test_checkpoint_migration_preserves_non_adapter_state_and_metadata() -> None:
    v5 = MatchupAdapterBank()
    base = torch.randn(3, 4)
    payload = {
        "model_state_dict": {
            "trunk.weight": base.clone(),
            **{
                f"matchup_adapter_bank.{name}": value.clone()
                for name, value in v5.state_dict().items()
            },
        },
        "rng_state": {"token": "unchanged"},
        "rl_iteration": 9,
        "extra": {
            "matchup_adapter_config": v5.config_dict(),
            "matchup_adapter_optimizer_included": False,
        },
    }
    migrated = migrate_v5_checkpoint_payload(payload)
    assert torch.equal(migrated["model_state_dict"]["trunk.weight"], base)
    assert migrated["rng_state"] == payload["rng_state"]
    assert migrated["rl_iteration"] == 9
    assert (
        migrated["extra"]["matchup_adapter_config"]["format"]
        == "poke-bot-matchup-adapter-bank-v6"
    )
    assert payload["extra"]["matchup_adapter_config"]["format"].endswith("roster18")


def test_positional_optimizer_without_name_map_fails_closed() -> None:
    v5 = MatchupAdapterBank()
    payload = {
        "model_state_dict": {
            f"matchup_adapter_bank.{name}": value.clone()
            for name, value in v5.state_dict().items()
        },
        "extra": {
            "matchup_adapter_config": v5.config_dict(),
            "matchup_adapter_optimizer_included": True,
        },
    }
    with pytest.raises(ValueError, match="name-keyed"):
        migrate_v5_checkpoint_payload(payload)


def test_name_keyed_v5_optimizer_moments_are_preserved_exactly() -> None:
    source = {
        "experts.0.down.weight": {
            "step": torch.tensor(7),
            "exp_avg": torch.randn(8, 96),
            "exp_avg_sq": torch.rand(8, 96),
        }
    }
    migrated = migrate_v5_named_optimizer_state(source)
    assert migrated is not source
    assert migrated["experts.0.down.weight"] is not source[
        "experts.0.down.weight"
    ]
    for key, value in source["experts.0.down.weight"].items():
        assert torch.equal(value, migrated["experts.0.down.weight"][key])


def test_positional_dormant_optimizer_expands_without_changing_v5_moments() -> None:
    source = {
        "state": {
            0: {
                "step": torch.tensor(7),
                "exp_avg": torch.randn(8, 96),
                "exp_avg_sq": torch.rand(8, 96),
            },
            71: {
                "step": torch.tensor(3),
                "exp_avg": torch.randn(96),
                "exp_avg_sq": torch.rand(96),
            },
        },
        "param_groups": [
            {
                "lr": 1e-4,
                "params": list(range(18 * 4)),
            }
        ],
    }
    migrated = migrate_v5_positional_optimizer_state(source)
    assert migrated["param_groups"][0]["params"] == list(range(64 * 4))
    assert set(migrated["state"]) == {0, 71}
    for parameter_id, moments in source["state"].items():
        for name, value in moments.items():
            assert torch.equal(value, migrated["state"][parameter_id][name])
    assert source["param_groups"][0]["params"] == list(range(18 * 4))


def test_positional_dormant_optimizer_fails_closed_on_noncanonical_ids() -> None:
    source = {
        "state": {},
        "param_groups": [{"params": list(range(1, 18 * 4 + 1))}],
    }
    with pytest.raises(ValueError, match="canonical order"):
        migrate_v5_positional_optimizer_state(source)


def test_runtime_optimizer_loader_accepts_exact_migrated_v6_bank() -> None:
    bank = MatchupAdapterBankV6()
    optimizer = torch.optim.AdamW(bank.parameters(), lr=1e-4)
    source = {
        "state": {
            0: {
                "step": torch.tensor(2),
                "exp_avg": torch.randn_like(bank.experts[0].down.weight),
                "exp_avg_sq": torch.rand_like(bank.experts[0].down.weight),
            }
        },
        "param_groups": [
            {
                **optimizer.state_dict()["param_groups"][0],
                "params": list(range(18 * 4)),
            }
        ],
    }
    migrated = migrate_v5_positional_optimizer_state(source)
    assert load_append_only_matchup_adapter_optimizer_state(
        optimizer,
        migrated,
    ) == SLOT_CAPACITY
    restored = optimizer.state_dict()
    assert len(restored["param_groups"][0]["params"]) == SLOT_CAPACITY * 4
    assert set(restored["state"]) == {0}
    assert torch.equal(
        restored["state"][0]["exp_avg"],
        source["state"][0]["exp_avg"],
    )


def test_checkpoint_migration_expands_dormant_optimizer_and_bank_metadata() -> None:
    v5 = MatchupAdapterBank()
    payload = {
        "model_state_dict": {
            f"matchup_adapter_bank.{name}": value.clone()
            for name, value in v5.state_dict().items()
        },
        "extra": {
            "matchup_adapter_config": v5.config_dict(),
            "matchup_adapter_optimizer_included": False,
            "dormant_matchup_adapter_bank": {
                "schema": "poke_bot.trained_dormant_matchup_adapter/v1",
                "adapter_config": v5.config_dict(),
                "parameter_count": sum(
                    value.numel() for value in v5.state_dict().values()
                ),
            },
            "dormant_matchup_adapter_optimizer_state": {
                "state": {},
                "param_groups": [{"params": list(range(18 * 4))}],
            },
        },
    }
    migrated = migrate_v5_checkpoint_payload(payload)
    extra = migrated["extra"]
    assert migrated["model_config"]["matchup_adapter_format"] == (
        ADAPTER_CHECKPOINT_FORMAT
    )
    assert migrated["model_config"]["matchup_adapter_registry"] == (
        load_slot_registry()
    )
    assert extra["dormant_matchup_adapter_bank"]["adapter_config"] == extra[
        "matchup_adapter_config"
    ]
    assert extra["dormant_matchup_adapter_bank"]["parameter_count"] == sum(
        value.numel() for value in MatchupAdapterBankV6().state_dict().values()
    )
    assert extra["dormant_matchup_adapter_optimizer_state"]["param_groups"][0][
        "params"
    ] == list(range(64 * 4))
    assert payload["extra"]["dormant_matchup_adapter_bank"]["parameter_count"] < extra[
        "dormant_matchup_adapter_bank"
    ]["parameter_count"]


def test_v6_zero_dormant_checkpoint_passes_shared_loader_validation(
    tmp_path,
) -> None:
    v5 = MatchupAdapterBank()
    payload = {
        "model_config": {"matchup_adapters_enabled": False},
        "model_state_dict": {
            f"matchup_adapter_bank.{name}": value.clone()
            for name, value in v5.state_dict().items()
        },
        "extra": {
            "matchup_adapter_config": v5.config_dict(),
            "matchup_adapters_runtime_enabled": False,
            "matchup_adapter_training_enabled": False,
            "matchup_adapter_optimizer_included": False,
            "dormant_matchup_adapter_bank": {
                "schema": ZERO_DORMANT_CHECKPOINT_SCHEMA,
                "runtime_enabled": False,
                "training_enabled": False,
                "optimizer_imported": False,
                "optimizer_included": False,
                "frozen": True,
                "zero_output": True,
                "parameter_count": sum(
                    value.numel() for value in v5.state_dict().values()
                ),
                "adapter_config": v5.config_dict(),
            },
        },
    }
    migrated = migrate_v5_checkpoint_payload(payload)
    path = tmp_path / "zero-v6.pt"
    torch.save(migrated, path)
    validated = validate_zero_dormant_checkpoint(path)
    assert validated["adapter_config"]["format"] == ADAPTER_CHECKPOINT_FORMAT
    assert validated["parameter_count"] == sum(
        value.numel() for value in MatchupAdapterBankV6().state_dict().values()
    )


def test_meta_crosswalk_requires_exact_identity_guards() -> None:
    exact = resolve_ptcgreplay_mapping(source_id=55, source_name="Crustle")
    assert exact == {"status": "exact", "archetype_id": "crustle"}
    assert resolve_ptcgreplay_mapping(
        source_id=55,
        source_name="crustle",
    ) == {"status": "missing", "archetype_id": None}

    hammer_cards = [
        archetypes.CRUSHING_HAMMER,
        archetypes.CRUSHING_HAMMER,
        archetypes.CRUSHING_HAMMER,
        next(iter(archetypes.MUNKIDORI)),
        archetypes.BUDEW,
        archetypes.UNFAIR_STAMP,
        next(iter(archetypes.DRAGAPULT_LINE)),
    ]
    signature = resolve_ptcgreplay_mapping(
        source_id=58,
        source_name="Dragapult ex",
        card_ids=hammer_cards,
        classifier=archetypes.classify_deck,
    )
    assert signature == {
        "status": "card_signature",
        "archetype_id": "hammer-pult",
    }
    wrong_family = resolve_ptcgreplay_mapping(
        source_id=55,
        source_name="Crustle",
        card_ids=hammer_cards,
        classifier=archetypes.classify_deck,
    )
    assert wrong_family["archetype_id"] == "crustle"
