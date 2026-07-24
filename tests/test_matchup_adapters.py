from __future__ import annotations

import copy
import hashlib
import json
from enum import IntEnum
from pathlib import Path

import pytest
import torch

from poke_bot import batched_infer, cg_env, checkpoint, config, features
from poke_bot import train as train_module
from poke_bot.dataset import DecisionSample, GameSequence
from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.dormant_adapter_compat import validate_zero_dormant_checkpoint
from poke_bot.matchup_adapters import (
    EXPERT_IDS,
    HIDDEN_DIM,
    UNKNOWN_ROUTE,
    MatchupAdapterBank,
    archetype_for_route,
    route_for_archetype,
    routes_for_archetypes,
)
from poke_bot.matchup_adapter_activation import (
    ActivationReceipt,
    ShadowMatchupAdapterRouter,
    TRAINING_TICKET_SCHEMA,
    build_activation_receipt,
    build_adapter_rehearsal_authorization,
    materialize_zero_dormant_adapter_checkpoint,
    validate_activation_receipt,
)
from poke_bot.model import build_model
from poke_bot.train import (
    TrainConfig,
    _train_dormant_matchup_adapter_phase,
    assert_matchup_adapter_isolation_guard,
    assert_matchup_adapter_training_contract,
    batch_losses,
    build_matchup_adapter_optimizer,
    device_temporal_batch_losses,
    load_append_only_matchup_adapter_optimizer_state,
    load_model_from_checkpoint,
    matchup_adapter_base_state,
    prepare_matchup_adapter_isolation_guard,
)


class _SelectContext(IntEnum):
    MAIN = 0
    RECOVER_SPECIAL_CONDITION = 48


@pytest.fixture(autouse=True)
def _fake_feature_vocab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features, "_CARD_COUNT", 8)
    monkeypatch.setattr(features, "_ATTACK_COUNT", 8)
    monkeypatch.setattr(features, "_CARD_TABLE", {})
    monkeypatch.setitem(cg_env.__dict__, "SelectContext", _SelectContext)


def _cfg() -> config.ModelConfig:
    return config.ModelConfig(
        d_model=HIDDEN_DIM,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=128,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        matchup_adapters_enabled=False,
        dropout=0.0,
    )


def _model():
    return build_model(
        _cfg(),
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=16,
    )


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    sparse = features.SparseVector()
    for index in range(words):
        sparse.word_start()
        sparse.add((offset + index) % 32, 1.0)
    return sparse


def _decision(offset: int = 0) -> DecisionSample:
    return DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS, offset),
        options=_sparse(2, offset + 3),
        action=[0],
        action_combo_index=0,
        action_combos=[[0], [1]],
        env_step=offset,
        action_token=_sparse(1, offset + 7),
    )


def _sequence(
    matchup: str,
    *,
    offset: int = 0,
    acting_archetype: str = "alakazam",
    source: str = "",
    package_id: str | None = None,
) -> GameSequence:
    sequence = GameSequence(
        episode_id=f"episode-{offset}",
        seat=0,
        archetype=acting_archetype,
        opp_archetype=matchup,
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(offset)],
        source=source,
        target_provenance={"package_id": package_id} if package_id else {},
    )
    if matchup in EXPERT_IDS:
        route = route_for_archetype(matchup)
        for decision in sequence.decisions:
            decision.matchup_adapter_oracle_route = route
        sequence.matchup_adapter_training_ticket = {
            "schema": TRAINING_TICKET_SCHEMA,
            "opponent_id": package_id or f"package-{matchup}",
            "package_digest": "sha256:" + "a" * 64,
            "archetype_id": matchup,
            "route": route,
            "corpus_manifest_digest": "sha256:" + "b" * 64,
            "gate_contract_digest": "sha256:" + "c" * 64,
            "episode_id": sequence.episode_id,
            "seat": sequence.seat,
            "acting_archetype_id": acting_archetype,
        }
    return sequence


def _activation_receipt() -> ActivationReceipt:
    return ActivationReceipt(
        path=Path("/tmp/adapter-activation.json"),
        commit_path=Path("/tmp/iter_00015.json"),
        commit_digest="sha256:" + "1" * 64,
        parent_checkpoint=Path("/tmp/iter15-learner.pt"),
        parent_checkpoint_digest="sha256:" + "2" * 64,
        completed_iteration=15,
        first_eligible_iteration=16,
    )


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _optimizer_snapshot(
    optimizer: torch.optim.Optimizer,
) -> dict[int, dict[str, object]]:
    return {
        id(parameter): {
            name: value.detach().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
            for name, value in state.items()
        }
        for parameter, state in optimizer.state.items()
    }


def _assert_optimizer_snapshot_equal(
    left: dict[int, dict[str, object]],
    right: dict[int, dict[str, object]],
) -> None:
    assert left.keys() == right.keys()
    for parameter_id, left_state in left.items():
        right_state = right[parameter_id]
        assert left_state.keys() == right_state.keys()
        for name, left_value in left_state.items():
            right_value = right_state[name]
            if isinstance(left_value, torch.Tensor):
                assert isinstance(right_value, torch.Tensor)
                assert torch.equal(left_value, right_value)
            else:
                assert left_value == right_value


def test_adapter_architecture_initialization_and_exact_routes() -> None:
    bank = MatchupAdapterBank()

    assert EXPERT_IDS == (
        "crustle",
        "marnie-s-grimmsnarl-ex",
        "garchomp",
        "rockets-mewtwo",
        "starmie",
        "hammer-pult",
        "alakazam",
        "lucario",
        "archaludon-ex",
        "dragapult-dudunsparce",
        "dragapult",
        "dudunsparce",
        "hops-trevenant",
        "walrein",
        "dragapult-dusknoir",
        "dragapult-blaziken",
        "thwackey",
        "team-rockets-spidops",
    )
    assert len(bank.experts) == 18
    assert [
        sum(parameter.numel() for parameter in expert.parameters())
        for expert in bank.experts
    ] == [1_640] * len(EXPERT_IDS)
    parameters_per_head = sum(
        parameter.numel() for parameter in bank.experts[0].parameters()
    )
    assert sum(parameter.numel() for parameter in bank.parameters()) == (
        len(EXPERT_IDS) * parameters_per_head
    )
    for expert in bank.experts:
        assert expert.down.in_features == 96
        assert expert.down.out_features == 8
        assert expert.up.in_features == 8
        assert expert.up.out_features == 96
        assert torch.count_nonzero(expert.up.weight).item() == 0
        assert torch.count_nonzero(expert.up.bias).item() == 0
        assert not any(
            "norm" in name for name, _parameter in expert.named_parameters()
        )

    assert routes_for_archetypes(
        [
            EXPERT_IDS[0],
            None,
            EXPERT_IDS[-1],
            "CRUSTLE",
            " crustle",
            "",
            "abstain",
        ]
    ).tolist() == [
        0,
        UNKNOWN_ROUTE,
        len(EXPERT_IDS) - 1,
        UNKNOWN_ROUTE,
        UNKNOWN_ROUTE,
        UNKNOWN_ROUTE,
        UNKNOWN_ROUTE,
    ]
    assert route_for_archetype("crustle") == 0
    # The specialist's mirror and the two current strong-public archetypes
    # each own a distinct appended expert.  Keep these explicit so a future
    # registry cleanup cannot silently collapse Lucario into a generic route.
    assert route_for_archetype("alakazam") == 6
    assert route_for_archetype("lucario") == 7
    assert archetype_for_route(7) == "lucario"
    assert route_for_archetype("archaludon-ex") == 8


def test_v1_bank_requires_explicit_identity_migration() -> None:
    source = MatchupAdapterBank()
    legacy_state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if int(name.split(".")[1]) < 7
    }
    with torch.no_grad():
        legacy_state["experts.0.up.bias"].fill_(0.25)

    expanded = MatchupAdapterBank()
    with pytest.raises(RuntimeError, match="Missing key"):
        expanded.load_state_dict(legacy_state, strict=True)


def test_v2_bank_requires_explicit_identity_migration() -> None:
    source = MatchupAdapterBank()
    legacy_state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if int(name.split(".")[1]) < 10
    }
    with torch.no_grad():
        legacy_state["experts.7.up.bias"].fill_(0.375)

    expanded = MatchupAdapterBank()
    with pytest.raises(RuntimeError, match="Missing key"):
        expanded.load_state_dict(legacy_state, strict=True)


def test_v3_bank_requires_explicit_identity_migration() -> None:
    source = MatchupAdapterBank()
    legacy_state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if int(name.split(".")[1]) < 15
    }
    with torch.no_grad():
        legacy_state["experts.14.up.bias"].fill_(0.5)

    expanded = MatchupAdapterBank()
    with pytest.raises(RuntimeError, match="Missing key"):
        expanded.load_state_dict(legacy_state, strict=True)


def test_bank_disabled_unknown_and_zero_initialized_paths_are_exact_noops() -> None:
    torch.manual_seed(7)
    state = torch.randn(3, HIDDEN_DIM)
    bank = MatchupAdapterBank(enabled=False)

    assert bank(state, torch.tensor([0, 1, 2])) is state
    bank.enabled = True
    assert torch.equal(bank(state, torch.tensor([0, 1, 2])), state)
    unknown = torch.full((3,), UNKNOWN_ROUTE, dtype=torch.long)
    assert bank(state, unknown) is state

    with torch.no_grad():
        bank.experts[0].up.bias.fill_(100.0)
    bounded = bank(state[:1], torch.tensor([0])) - state[:1]
    assert float(bounded.detach().abs().max()) <= 0.250_001


def test_model_forward_adapts_only_policy_value_and_keeps_aux_state_raw() -> None:
    torch.manual_seed(11)
    model = _model().eval()
    board = _decision().board
    options = _decision().options
    route = torch.tensor([0])

    baseline = model(board, options, n_options=[2])
    disabled_routed = model(
        board,
        options,
        n_options=[2],
        matchup_routes=route,
    )
    for key in ("policy_logits", "value", "state_vec", "aux_logits"):
        assert torch.equal(baseline[key], disabled_routed[key])

    model.matchup_adapter_bank.enabled = True
    with torch.no_grad():
        model.matchup_adapter_bank.experts[0].up.bias.copy_(
            torch.linspace(-0.75, 0.75, HIDDEN_DIM)
        )
    adapted = model(
        board,
        options,
        n_options=[2],
        matchup_routes=route,
    )

    assert torch.equal(baseline["state_vec"], adapted["state_vec"])
    assert torch.equal(baseline["aux_logits"], adapted["aux_logits"])
    assert not torch.equal(baseline["value"], adapted["value"])
    assert not torch.equal(baseline["policy_logits"], adapted["policy_logits"])


def test_recognized_shadow_route_is_bit_exact_policy_value_noop() -> None:
    torch.manual_seed(111)
    model = _model().eval()
    model.matchup_adapter_bank.enabled = True
    route = route_for_archetype("marnie-s-grimmsnarl-ex")
    with torch.no_grad():
        model.matchup_adapter_bank.experts[route].up.weight.normal_(0.0, 0.1)
        model.matchup_adapter_bank.experts[route].up.bias.fill_(0.25)

    observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [{"id": 646}], "bench": [], "discard": []},
            ],
        }
    }
    shadow = ShadowMatchupAdapterRouter()
    assert shadow.observe(observation).route == UNKNOWN_ROUTE
    assert shadow.observe(observation).route == route
    assert shadow.model_route == UNKNOWN_ROUTE

    board = _decision(4).board
    options = _decision(4).options
    baseline = model(board, options, n_options=[2])
    shadowed = model(
        board,
        options,
        n_options=[2],
        matchup_routes=[shadow.model_route],
    )
    assert torch.equal(baseline["policy_logits"], shadowed["policy_logits"])
    assert torch.equal(baseline["value"], shadowed["value"])


@pytest.mark.parametrize("decision_context", ["history", "stateless"])
def test_batched_leaf_forward_hard_pins_every_model_route_unknown(
    decision_context: str,
) -> None:
    class SpyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.decision_context = decision_context
            self.routes = None

        def _output(self, rows: int) -> dict[str, torch.Tensor]:
            return {
                "policy_logits": torch.zeros(rows, 2),
                "value": torch.zeros(rows),
            }

        def forward_history_batch(self, histories, opts, **kwargs):
            self.routes = list(kwargs["matchup_routes"])
            return self._output(len(opts))

        def forward(self, boards, opts, **kwargs):
            self.routes = list(kwargs["matchup_routes"])
            return self._output(len(opts))

    model = SpyModel()
    rows = 3
    batched_infer._forward_chunk(
        model,
        [object()] * rows,
        [object()] * rows,
        [2] * rows,
        [0] * rows,
        [0] * rows,
        None,
        histories=[[object()]] * rows,
        previous_action_histories=[[None]] * rows,
    )
    assert model.routes == [UNKNOWN_ROUTE] * rows


def test_mixed_rows_use_only_their_selected_expert_and_unknown_is_gradient_free() -> None:
    torch.manual_seed(12)
    bank = MatchupAdapterBank(enabled=True)
    reference = copy.deepcopy(bank)
    with torch.no_grad():
        bank.experts[0].up.bias.copy_(torch.linspace(-0.4, 0.4, HIDDEN_DIM))
        bank.experts[3].up.bias.copy_(torch.linspace(0.6, -0.6, HIDDEN_DIM))
        reference.load_state_dict(bank.state_dict())

    state = torch.randn(4, HIDDEN_DIM)
    routes = torch.tensor([0, 3, UNKNOWN_ROUTE, 0])
    mixed = bank(state, routes)
    assert torch.equal(mixed[2], state[2])
    assert torch.equal(mixed[[0, 3]], reference(state[[0, 3]], torch.tensor([0, 0])))
    assert torch.equal(mixed[1:2], reference(state[1:2], torch.tensor([3])))

    mixed.sum().backward()
    mixed_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in bank.named_parameters()
        if parameter.grad is not None
    }
    reference.zero_grad(set_to_none=True)
    reference(state[[0, 1, 3]], torch.tensor([0, 3, 0])).sum().backward()
    supported_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }
    assert mixed_gradients.keys() == supported_gradients.keys()
    assert all(
        torch.equal(gradient, supported_gradients[name])
        for name, gradient in mixed_gradients.items()
    )
    assert all(
        name.startswith("experts.0.") or name.startswith("experts.3.")
        for name in mixed_gradients
    )


def test_mixed_all_routes_and_unknown_match_independent_row_references() -> None:
    torch.manual_seed(120)
    bank = MatchupAdapterBank(enabled=True)
    with torch.no_grad():
        for route, expert in enumerate(bank.experts):
            expert.up.weight.normal_(mean=0.0, std=0.03)
            expert.up.bias.copy_(
                torch.linspace(-0.08, 0.08, HIDDEN_DIM) * float(route + 1)
            )
    references = [copy.deepcopy(bank) for _ in EXPERT_IDS]

    state = torch.randn(len(EXPERT_IDS) + 1, HIDDEN_DIM, requires_grad=True)
    routes = torch.tensor([*range(len(EXPERT_IDS)), UNKNOWN_ROUTE])
    cotangent = torch.randn_like(state)
    mixed = bank(state, routes)

    assert torch.equal(mixed[-1], state[-1])
    (mixed * cotangent).sum().backward()
    assert state.grad is not None
    assert torch.equal(state.grad[-1], cotangent[-1])

    for route, reference in enumerate(references):
        row = state.detach()[route : route + 1].clone().requires_grad_(True)
        single = reference(row, torch.tensor([route]))
        assert torch.equal(mixed[route : route + 1], single)
        (single * cotangent[route : route + 1]).sum().backward()

        assert row.grad is not None
        assert torch.equal(state.grad[route : route + 1], row.grad)
        for parameter_name, parameter in reference.experts[route].named_parameters():
            mixed_parameter = dict(
                bank.experts[route].named_parameters()
            )[parameter_name]
            assert mixed_parameter.grad is not None
            assert parameter.grad is not None
            assert torch.equal(mixed_parameter.grad, parameter.grad)
        assert all(
            parameter.grad is None
            for other_route, expert in enumerate(reference.experts)
            if other_route != route
            for parameter in expert.parameters()
        )


def test_legacy_checkpoint_without_adapter_state_loads_strict_and_exact(
    tmp_path: Path,
) -> None:
    torch.manual_seed(13)
    model = _model().eval()
    board = _decision(1).board
    options = _decision(1).options
    expected = model(board, options, n_options=[2], matchup_routes=torch.tensor([0]))

    payload = checkpoint.build_checkpoint(model=model, model_config=model.cfg)
    payload["model_config"].pop("matchup_adapters_enabled")
    payload["model_state_dict"] = {
        key: value
        for key, value in payload["model_state_dict"].items()
        if not key.startswith("matchup_adapter_bank.")
    }
    payload.get("extra", {}).pop("matchup_adapter_config", None)
    if not payload.get("extra"):
        payload.pop("extra", None)
    path = checkpoint.atomic_torch_save(payload, tmp_path / "legacy.pt")

    loaded = load_model_from_checkpoint(path, device=torch.device("cpu"))
    actual = loaded(
        board,
        options,
        n_options=[2],
        matchup_routes=torch.tensor([0]),
    )
    assert loaded.matchup_adapter_bank.enabled is False
    for key in (
        "policy_logits",
        "value",
        "state_vec",
        "aux_logits",
        "opp_hand_logits",
    ):
        assert torch.equal(expected[key], actual[key]), key

    fresh = _model()
    incompatible = fresh.load_state_dict(payload["model_state_dict"], strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_materialized_zero_bank_has_exact_full_forward_parity(
    tmp_path: Path,
) -> None:
    torch.manual_seed(131)
    parent_model = _model().eval()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in parent_model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    payload = checkpoint.build_checkpoint(
        model=parent_model,
        optimizer=optimizer,
        model_config=parent_model.cfg,
        extra={"pure_rl": True, "matchup_adapters_runtime_enabled": False},
    )
    payload["model_config"].pop("matchup_adapters_enabled")
    payload["model_state_dict"] = {
        name: value
        for name, value in payload["model_state_dict"].items()
        if not name.startswith("matchup_adapter_bank.")
    }
    payload["extra"].pop("matchup_adapter_config", None)
    payload["extra"].pop("dormant_matchup_adapter_bank", None)
    payload["extra"].pop("matchup_adapter_training_enabled", None)
    payload["extra"].pop("matchup_adapter_optimizer_included", None)
    parent = checkpoint.atomic_torch_save(payload, tmp_path / "parent.pt")
    digest = "sha256:" + hashlib.sha256(parent.read_bytes()).hexdigest()
    state = {
        "version": 2,
        "run_name": "parity",
        "mode": "specialist",
        "last_completed_iteration": 15,
        "next_iteration": 16,
        "learner": {"path": str(parent.resolve()), "digest": digest},
    }
    run = tmp_path / "run"
    (run / "commits").mkdir(parents=True)
    serialized = json.dumps(state, sort_keys=True) + "\n"
    (run / "commits" / "iter_00015.json").write_text(serialized)
    (run / "loop_state.json").write_text(serialized)
    receipt = tmp_path / "activation.json"
    build_activation_receipt(run_dir=run, output_path=receipt)
    migrated = tmp_path / "parent.zero-dormant.pt"
    materialize_zero_dormant_adapter_checkpoint(
        parent_checkpoint=parent,
        activation_receipt=receipt,
        output_path=migrated,
    )

    before = load_model_from_checkpoint(parent, device=torch.device("cpu")).eval()
    after = load_model_from_checkpoint(migrated, device=torch.device("cpu")).eval()
    board = _decision(3).board
    options = _decision(3).options
    history = [[_decision(1).board, board]]
    action_history = [[_decision(1).action_token, _decision(3).action_token]]
    with torch.inference_mode():
        parent_single = before(board, options, n_options=[2])
        migrated_single = after(board, options, n_options=[2])
        parent_history = before.forward_history_batch(
            history,
            [options],
            n_options=[2],
            previous_action_histories=action_history,
        )
        migrated_history = after.forward_history_batch(
            history,
            [options],
            n_options=[2],
            previous_action_histories=action_history,
        )
    for expected, actual in (
        (parent_single, migrated_single),
        (parent_history, migrated_history),
    ):
        assert expected.keys() == actual.keys()
        for name in expected:
            if isinstance(expected[name], torch.Tensor):
                assert torch.equal(expected[name], actual[name]), name
            else:
                assert expected[name] == actual[name], name
    parent.unlink()
    recovered = validate_activation_receipt(
        receipt,
        parent_checkpoint=parent,
    )
    assert recovered.parent_checkpoint_digest == digest
    assert after.matchup_adapter_bank.enabled is False
    assert all(
        parameter.requires_grad is False
        for parameter in after.matchup_adapter_bank.parameters()
    )
    resumed_optimizer = torch.optim.AdamW(
        (parameter for parameter in after.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    checkpoint.apply_checkpoint(
        checkpoint.load_checkpoint(migrated, map_location="cpu"),
        model=after,
        optimizer=resumed_optimizer,
        restore_rng=False,
        strict=True,
    )
    optimizer_parameter_ids = {
        id(candidate)
        for group in resumed_optimizer.param_groups
        for candidate in group["params"]
    }
    assert not any(
        id(parameter) in optimizer_parameter_ids
        for parameter in after.matchup_adapter_bank.parameters()
    )


@pytest.mark.parametrize("later_boundary", [False, True])
def test_activation_receipt_allows_bankless_nonparent_runtime_anchors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    later_boundary: bool,
) -> None:
    def legacy_checkpoint(name: str, seed: int) -> Path:
        torch.manual_seed(seed)
        model = _model().eval()
        payload = checkpoint.build_checkpoint(
            model=model,
            model_config=model.cfg,
            extra={"pure_rl": True},
            rl_iteration=15,
        )
        payload["model_config"].pop("matchup_adapters_enabled")
        payload["model_state_dict"] = {
            key: value
            for key, value in payload["model_state_dict"].items()
            if not key.startswith("matchup_adapter_bank.")
        }
        for key in (
            "matchup_adapter_config",
            "matchup_adapters_runtime_enabled",
            "matchup_adapter_training_enabled",
            "matchup_adapter_optimizer_included",
            "dormant_matchup_adapter_bank",
        ):
            payload["extra"].pop(key, None)
        return checkpoint.atomic_torch_save(payload, tmp_path / name)

    parent = legacy_checkpoint("iter15-parent.pt", 145)
    anchor = legacy_checkpoint("protected-anchor.pt", 146)
    parent_digest = checkpoint.checkpoint_digest(parent)
    completed_iteration = 26 if later_boundary else 15
    state = {
        "version": 2,
        "run_name": "multi-anchor-loader",
        "mode": "specialist",
        "last_completed_iteration": completed_iteration,
        "next_iteration": completed_iteration + 1,
        "learner": {"path": str(parent.resolve()), "digest": parent_digest},
    }
    run = tmp_path / "run"
    (run / "commits").mkdir(parents=True)
    serialized = json.dumps(state, sort_keys=True) + "\n"
    (run / "commits" / f"iter_{completed_iteration:05d}.json").write_text(
        serialized
    )
    (run / "loop_state.json").write_text(serialized)
    receipt = tmp_path / "activation.json"
    if later_boundary:
        build_adapter_rehearsal_authorization(
            run_dir=run,
            completed_iteration=completed_iteration,
            output_path=receipt,
        )
    else:
        build_activation_receipt(run_dir=run, output_path=receipt)
    monkeypatch.setenv("POKEBOT_MATCHUP_ADAPTER_BOUNDARY_RECEIPT", str(receipt))

    parent_model = load_model_from_checkpoint(parent, device=torch.device("cpu"))
    anchor_model = load_model_from_checkpoint(anchor, device=torch.device("cpu"))
    parent_provenance = parent_model.matchup_adapter_bank.dormant_provenance
    anchor_provenance = anchor_model.matchup_adapter_bank.dormant_provenance

    assert parent_provenance["activation_parent_match"] is True
    assert anchor_provenance["activation_parent_match"] is False
    assert anchor_provenance["parent_checkpoint"] == str(anchor.resolve())
    assert (
        anchor_provenance["activation_parent_checkpoint"]
        == str(parent.resolve())
    )
    assert (
        anchor_provenance["activation_parent_checkpoint_digest"]
        == parent_digest
    )
    assert parent_model.matchup_adapter_bank.enabled is False
    assert anchor_model.matchup_adapter_bank.enabled is False
    assert all(
        not parameter.requires_grad
        for model in (parent_model, anchor_model)
        for parameter in model.matchup_adapter_bank.parameters()
    )


def test_bankless_parent_dynamically_persists_frozen_zero_bank_next_checkpoint(
    tmp_path: Path,
) -> None:
    torch.manual_seed(132)
    source_model = _model().eval()
    source_optimizer = torch.optim.AdamW(
        (parameter for parameter in source_model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    payload = checkpoint.build_checkpoint(
        model=source_model,
        optimizer=source_optimizer,
        model_config=source_model.cfg,
        extra={"pure_rl": True},
        rl_iteration=15,
    )
    payload["model_config"].pop("matchup_adapters_enabled")
    payload["model_state_dict"] = {
        name: value
        for name, value in payload["model_state_dict"].items()
        if not name.startswith("matchup_adapter_bank.")
    }
    for key in (
        "matchup_adapter_config",
        "matchup_adapters_runtime_enabled",
        "matchup_adapter_training_enabled",
        "matchup_adapter_optimizer_included",
        "dormant_matchup_adapter_bank",
    ):
        payload["extra"].pop(key, None)
    parent = checkpoint.atomic_torch_save(payload, tmp_path / "iter15-parent.pt")
    parent_bytes = parent.read_bytes()
    parent_digest = checkpoint.checkpoint_digest(parent)

    loaded = load_model_from_checkpoint(parent, device=torch.device("cpu"))
    loaded.train()
    assert loaded.matchup_adapter_bank.enabled is False
    assert all(
        parameter.requires_grad is False
        for parameter in loaded.matchup_adapter_bank.parameters()
    )
    ordinary_optimizer = torch.optim.AdamW(
        (parameter for parameter in loaded.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    candidate_payload = checkpoint.build_checkpoint(
        model=loaded,
        optimizer=ordinary_optimizer,
        model_config=loaded.cfg,
        extra={"pure_rl": True, "parent_digest": parent_digest},
        rl_iteration=16,
    )
    candidate = checkpoint.atomic_torch_save(
        candidate_payload,
        tmp_path / "iter16-candidate.pt",
    )

    assert parent.read_bytes() == parent_bytes
    saved = checkpoint.load_checkpoint(candidate, map_location="cpu")
    extra = saved["extra"]
    dormant = extra["dormant_matchup_adapter_bank"]
    assert saved["model_config"]["matchup_adapters_enabled"] is False
    assert extra["matchup_adapters_runtime_enabled"] is False
    assert extra["matchup_adapter_training_enabled"] is False
    assert extra["matchup_adapter_optimizer_included"] is False
    assert "matchup_adapter_training_contract" not in extra
    assert dormant["materialization"] == "legacy_bankless_dynamic_zero_init"
    assert dormant["parent_checkpoint"] == str(parent.resolve())
    assert dormant["parent_checkpoint_digest"] == parent_digest
    assert dormant["runtime_enabled"] is False
    assert dormant["training_enabled"] is False
    assert dormant["optimizer_included"] is False
    assert dormant["optimizer_imported"] is False
    assert dormant["frozen"] is True
    assert dormant["zero_output"] is True
    parameters_per_head = sum(
        parameter.numel()
        for parameter in loaded.matchup_adapter_bank.experts[0].parameters()
    )
    assert dormant["parameter_count"] == len(EXPERT_IDS) * parameters_per_head
    assert any(
        name.startswith("matchup_adapter_bank.")
        for name in saved["model_state_dict"]
    )
    assert all(
        int(value.count_nonzero().item()) == 0
        for name, value in saved["model_state_dict"].items()
        if name.startswith("matchup_adapter_bank.")
        and (name.endswith("up.weight") or name.endswith("up.bias"))
    )
    restored = load_model_from_checkpoint(candidate, device=torch.device("cpu"))
    assert restored.matchup_adapter_bank.dormant_provenance == dormant


def test_ordinary_pure_rl_checkpoint_rejects_adapter_optimizer_or_nonzero_bank() -> None:
    model = _model()
    leaked_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="leaked into the ordinary optimizer"):
        checkpoint.build_checkpoint(
            model=model,
            optimizer=leaked_optimizer,
            model_config=model.cfg,
            extra={"pure_rl": True},
        )
    with torch.no_grad():
        model.matchup_adapter_bank.experts[0].up.bias.fill_(1.0)
    clean_optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    with pytest.raises(ValueError, match="cannot persist non-zero"):
        checkpoint.build_checkpoint(
            model=model,
            optimizer=clean_optimizer,
            model_config=model.cfg,
            extra={"pure_rl": True},
        )


def test_live_dormant_phase_trains_lucario_and_mirror_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    base_before = matchup_adapter_base_state(model)
    receipt_path = tmp_path / "activation.json"
    receipt_path.write_text(
        json.dumps({"parent_checkpoint": "/tmp/iter15-learner.pt"})
    )
    initial = _activation_receipt()
    activation = ActivationReceipt(
        path=initial.path,
        commit_path=Path("/tmp/iter_00026.json"),
        commit_digest=initial.commit_digest,
        parent_checkpoint=initial.parent_checkpoint,
        parent_checkpoint_digest=initial.parent_checkpoint_digest,
        completed_iteration=26,
        first_eligible_iteration=27,
    )
    monkeypatch.setattr(
        "poke_bot.train.validate_adapter_training_authorization",
        lambda path, parent_checkpoint, **kwargs: activation,
    )
    cfg = TrainConfig.pure_rl_defaults(
        games_per_batch=2,
        max_decisions_per_batch=32,
        dormant_matchup_adapter_epochs=1,
        dormant_matchup_adapter_lr=1e-2,
        dormant_matchup_adapter_activation_receipt=str(receipt_path),
    )

    fit, optimizer_state = _train_dormant_matchup_adapter_phase(
        model,
        [_sequence("lucario", offset=8), _sequence("alakazam", offset=9)],
        cfg=cfg,
        # Runtime checkpoint migrations can preserve an older internal
        # counter.  The outer receipt-backed iteration is authoritative.
        base_rl_iteration=25,
        target_rl_iteration=27,
        awr_baseline_cache=None,
        seed=3,
    )

    assert fit["runtime_enabled"] is False
    assert fit["route_sequences"]["lucario"] == 1
    assert fit["route_sequences"]["alakazam"] == 1
    assert fit["route_sequences"]["archaludon-ex"] == 0
    assert fit["steps"] == 1
    assert fit["rows"] > 0
    assert optimizer_state
    assert model.matchup_adapter_bank.enabled is False
    assert model.cfg.matchup_adapters_enabled is False
    assert not any(
        parameter.requires_grad
        for parameter in model.matchup_adapter_bank.parameters()
    )
    for name, value in matchup_adapter_base_state(model).items():
        assert torch.equal(value, base_before[name])
    changed_routes = {
        route
        for route, expert in enumerate(model.matchup_adapter_bank.experts)
        if any(
            int(value.count_nonzero().item()) > 0
            for name, value in expert.state_dict().items()
            if name.startswith("up.")
        )
    }
    assert changed_routes == {
        route_for_archetype("alakazam"),
        route_for_archetype("lucario"),
    }
    second_fit, _ = _train_dormant_matchup_adapter_phase(
        model,
        [_sequence("archaludon-ex", offset=10)],
        cfg=cfg,
        base_rl_iteration=26,
        target_rl_iteration=28,
        awr_baseline_cache=None,
        seed=4,
        prior_optimizer_state=optimizer_state,
        prior_fit=fit,
    )
    assert second_fit["phase_route_decisions"]["archaludon-ex"] > 0
    assert second_fit["phase_route_decisions"]["alakazam"] == 0
    assert second_fit["route_decisions"]["alakazam"] == fit["route_decisions"]["alakazam"]
    assert second_fit["route_decisions"]["lucario"] == fit["route_decisions"]["lucario"]
    assert second_fit["route_decisions"]["archaludon-ex"] > 0
    assert second_fit["epochs"] == 2
    assert second_fit["phase_epochs"] == 1
    assert set(second_fit["trained_archetype_ids"]) == {
        "alakazam",
        "archaludon-ex",
        "lucario",
    }
    ordinary_optimizer = torch.optim.AdamW(
        (
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("matchup_adapter_bank.")
            and parameter.requires_grad
        ),
        lr=1e-3,
    )
    payload = checkpoint.build_checkpoint(
        model=model,
        optimizer=ordinary_optimizer,
        model_config=model.cfg,
        extra={
            "pure_rl": True,
            "dormant_matchup_adapter_fit": fit,
            "dormant_matchup_adapter_optimizer_state": optimizer_state,
        },
    )
    trained_path = checkpoint.atomic_torch_save(
        payload, tmp_path / "trained-dormant.pt"
    )
    validated = validate_zero_dormant_checkpoint(
        trained_path, allow_trained=True
    )
    assert validated["trained"] is True
    with pytest.raises(RuntimeError, match="frozen dormant bank"):
        validate_zero_dormant_checkpoint(trained_path)

def test_checkpoint_resume_pins_route_order_and_rejects_mapping_drift(
    tmp_path: Path,
) -> None:
    model = _model()
    with torch.no_grad():
        for route, expert in enumerate(model.matchup_adapter_bank.experts):
            expert.up.bias.fill_(float(route + 1))
    adapter_config = model.matchup_adapter_bank.config_dict()
    payload = checkpoint.build_checkpoint(
        model=model,
        model_config=model.cfg,
        extra={"matchup_adapter_config": adapter_config},
    )
    assert payload["extra"]["matchup_adapter_config"] == adapter_config
    path = checkpoint.atomic_torch_save(payload, tmp_path / "adapter-resume.pt")

    resumed = load_model_from_checkpoint(path, device=torch.device("cpu"))
    assert resumed.matchup_adapter_bank.expert_ids == EXPERT_IDS
    for route, matchup_id in enumerate(EXPERT_IDS):
        assert route_for_archetype(matchup_id) == route
        assert torch.equal(
            resumed.matchup_adapter_bank.experts[route].up.bias,
            torch.full((HIDDEN_DIM,), float(route + 1)),
        )

    drifted = copy.deepcopy(payload)
    drifted["extra"]["matchup_adapter_config"]["expert_ids"] = list(
        reversed(EXPERT_IDS)
    )
    drifted_path = checkpoint.atomic_torch_save(
        drifted,
        tmp_path / "adapter-drifted.pt",
    )
    with pytest.raises(ValueError, match="routing contract mismatch"):
        load_model_from_checkpoint(drifted_path, device=torch.device("cpu"))

    missing_contract = copy.deepcopy(payload)
    missing_contract["extra"].pop("matchup_adapter_config")
    missing_contract_path = checkpoint.atomic_torch_save(
        missing_contract,
        tmp_path / "adapter-missing-contract.pt",
    )
    with pytest.raises(ValueError, match="missing.*routing contract"):
        load_model_from_checkpoint(
            missing_contract_path,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="missing.*routing contract"):
        checkpoint.apply_checkpoint(
            missing_contract,
            model=_model(),
            restore_rng=False,
        )


def test_legacy_bootstrap_optimizer_state_loads_with_dormant_bank() -> None:
    legacy_model = _model()
    legacy_base_parameters = [
        parameter
        for name, parameter in legacy_model.named_parameters()
        if not name.startswith("matchup_adapter_bank.")
    ]
    legacy_optimizer = torch.optim.AdamW(legacy_base_parameters, lr=1e-3)
    legacy_base_parameters[0].grad = torch.ones_like(legacy_base_parameters[0])
    legacy_optimizer.step()
    payload = checkpoint.build_checkpoint(
        model=legacy_model,
        optimizer=legacy_optimizer,
        model_config=legacy_model.cfg,
    )
    payload["model_config"].pop("matchup_adapters_enabled")
    payload["model_state_dict"] = {
        key: value
        for key, value in payload["model_state_dict"].items()
        if not key.startswith("matchup_adapter_bank.")
    }
    payload.get("extra", {}).pop("matchup_adapter_config", None)
    if not payload.get("extra"):
        payload.pop("extra", None)

    resumed_model = _model()
    resumed_optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in resumed_model.parameters()
            if parameter.requires_grad
        ),
        lr=1e-3,
    )
    checkpoint.apply_checkpoint(
        payload,
        model=resumed_model,
        optimizer=resumed_optimizer,
        restore_rng=False,
        strict=True,
    )

    assert len(resumed_optimizer.param_groups[0]["params"]) == len(
        legacy_base_parameters
    )
    resumed_first = resumed_optimizer.param_groups[0]["params"][0]
    assert torch.equal(
        resumed_optimizer.state[resumed_first]["step"],
        legacy_optimizer.state[legacy_base_parameters[0]]["step"],
    )
    assert all(
        not parameter.requires_grad
        for parameter in resumed_model.matchup_adapter_bank.parameters()
    )


def test_adamw_step_updates_only_routes_present_on_that_step() -> None:
    torch.manual_seed(17)
    bank = MatchupAdapterBank(enabled=True)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=1e-2, weight_decay=0.2)

    optimizer.zero_grad(set_to_none=True)
    bank(torch.randn(4, HIDDEN_DIM), torch.zeros(4, dtype=torch.long)).sum().backward()
    optimizer.step()
    after_route_zero = _snapshot(bank)
    route_zero_state = _optimizer_snapshot(optimizer)
    route_zero_parameter_ids = {
        id(parameter) for parameter in bank.experts[0].parameters()
    }
    assert route_zero_state.keys() == route_zero_parameter_ids

    optimizer.zero_grad(set_to_none=True)
    route_one = torch.ones(4, dtype=torch.long)
    bank(torch.randn(4, HIDDEN_DIM), route_one).square().sum().backward()
    for route, expert in enumerate(bank.experts):
        gradients = [parameter.grad for parameter in expert.parameters()]
        if route == 1:
            assert all(gradient is not None for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)
    optimizer.step()
    after_route_one = _snapshot(bank)
    after_route_one_state = _optimizer_snapshot(optimizer)

    _assert_optimizer_snapshot_equal(
        route_zero_state,
        {
            parameter_id: after_route_one_state[parameter_id]
            for parameter_id in route_zero_parameter_ids
        },
    )
    assert after_route_one_state.keys() == route_zero_parameter_ids | {
        id(parameter) for parameter in bank.experts[1].parameters()
    }

    assert any(
        not torch.equal(after_route_one[name], value)
        for name, value in after_route_zero.items()
        if name.startswith("experts.1.")
    )
    assert all(
        torch.equal(after_route_one[name], value)
        for name, value in after_route_zero.items()
        if not name.startswith("experts.1.")
    )

    optimizer.zero_grad(set_to_none=True)
    before_unknown = _snapshot(bank)
    optimizer_before_unknown = _optimizer_snapshot(optimizer)
    unknown_state = torch.randn(3, HIDDEN_DIM)
    assert bank(
        unknown_state,
        torch.full((3,), UNKNOWN_ROUTE, dtype=torch.long),
    ) is unknown_state
    optimizer.step()
    assert all(
        torch.equal(bank.state_dict()[name], value)
        for name, value in before_unknown.items()
    )
    _assert_optimizer_snapshot_equal(
        optimizer_before_unknown,
        _optimizer_snapshot(optimizer),
    )


def test_sequential_all_route_adamw_steps_preserve_inactive_parameters_and_state() -> None:
    torch.manual_seed(170)
    bank = MatchupAdapterBank(enabled=True)
    optimizer = torch.optim.AdamW(
        bank.parameters(),
        lr=5e-2,
        betas=(0.55, 0.77),
        weight_decay=3.0,
    )

    for route in range(len(EXPERT_IDS)):
        optimizer.zero_grad(set_to_none=True)
        parameters_before = _snapshot(bank)
        optimizer_before = _optimizer_snapshot(optimizer)
        current_parameter_ids = {
            id(parameter) for parameter in bank.experts[route].parameters()
        }
        assert current_parameter_ids.isdisjoint(optimizer_before)

        state = torch.randn(5, HIDDEN_DIM)
        cotangent = torch.randn_like(state)
        output = bank(
            state,
            torch.full((state.shape[0],), route, dtype=torch.long),
        )
        (output * cotangent).sum().backward()

        for expert_route, expert in enumerate(bank.experts):
            gradients = [parameter.grad for parameter in expert.parameters()]
            if expert_route == route:
                assert all(gradient is not None for gradient in gradients)
            else:
                assert all(gradient is None for gradient in gradients)

        optimizer.step()
        parameters_after = _snapshot(bank)
        optimizer_after = _optimizer_snapshot(optimizer)
        changed_routes = {
            expert_route
            for expert_route in range(len(EXPERT_IDS))
            if any(
                not torch.equal(parameters_before[name], value)
                for name, value in parameters_after.items()
                if name.startswith(f"experts.{expert_route}.")
            )
        }
        assert changed_routes == {route}

        prior_parameter_ids = {
            id(parameter)
            for prior_route in range(route)
            for parameter in bank.experts[prior_route].parameters()
        }
        _assert_optimizer_snapshot_equal(
            {
                parameter_id: optimizer_before[parameter_id]
                for parameter_id in prior_parameter_ids
            },
            {
                parameter_id: optimizer_after[parameter_id]
                for parameter_id in prior_parameter_ids
            },
        )
        assert current_parameter_ids.issubset(optimizer_after)
        assert all(
            id(parameter) not in optimizer_after
            for future_route in range(route + 1, len(EXPERT_IDS))
            for parameter in bank.experts[future_route].parameters()
        )


def test_ordinary_training_ignores_all_adapter_routing_metadata_exactly() -> None:
    """Oracle tickets must be inert unless adapter-training mode is explicit."""

    torch.manual_seed(18)
    model = _model().eval()
    clean = _sequence(EXPERT_IDS[1], offset=18)
    poisoned = copy.deepcopy(clean)
    poisoned.matchup_adapter_training_ticket.update(
        {
            "archetype_id": EXPERT_IDS[5],
            "route": True,
            "seat": 99,
        }
    )
    poisoned.decisions[0].matchup_adapter_oracle_route = True
    poisoned.decisions[0].matchup_adapter_public_route = route_for_archetype(
        EXPERT_IDS[4]
    )

    clean_loss, clean_metrics = batch_losses(
        model,
        [clean],
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        matchup_adapter_training=False,
    )
    poisoned_loss, poisoned_metrics = batch_losses(
        model,
        [poisoned],
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        matchup_adapter_training=False,
    )

    assert torch.equal(clean_loss, poisoned_loss)
    assert clean_metrics == poisoned_metrics
    poisoned_loss.backward()
    assert all(
        parameter.grad is None
        for parameter in model.matchup_adapter_bank.parameters()
    )


def test_adapter_only_zero_weight_auxiliary_targets_are_not_materialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route fleet must not build millions of unused aux-label tensors."""

    model = _model().eval()
    build_matchup_adapter_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.0,
        activation_receipt=_activation_receipt(),
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("zero-weight auxiliary path was evaluated")

    monkeypatch.setattr(type(model), "belief_aux_logits", unexpected)
    monkeypatch.setattr(
        train_module, "belief_multihots_from_aux_labels", unexpected
    )
    monkeypatch.setattr(train_module, "lethal_target_from_aux", unexpected)
    monkeypatch.setattr(train_module, "prize_race_target_from_aux", unexpected)

    loss, metrics = batch_losses(
        model,
        [_sequence(EXPERT_IDS[2], offset=29)],
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        lethal_threat_weight=0.0,
        prize_race_weight=0.0,
        alakazam_guide_weight=0.0,
        matchup_adapter_training=True,
    )

    assert torch.isfinite(loss)
    assert metrics.n_opp_hand_rows == 0
    assert metrics.n_opp_remainder_rows == 0
    assert metrics.n_lethal_threat_rows == 0
    assert metrics.n_prize_race_rows == 0


def test_bootstrap_uses_ground_truth_route_and_frozen_adapter_optimizer() -> None:
    torch.manual_seed(19)
    model = _model().train()
    optimizer = build_matchup_adapter_optimizer(
        model,
        lr=1e-2,
        weight_decay=0.2,
        activation_receipt=_activation_receipt(),
    )
    base_before = matchup_adapter_base_state(model)
    adapters_before = _snapshot(model.matchup_adapter_bank)
    aux_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith(
            (
                "aux_head.",
                "opp_hand_head.",
                "opp_remainder_head.",
                "lethal_threat_head.",
                "prize_race_head.",
            )
        )
    }

    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized_ids == {
        id(parameter) for parameter in model.matchup_adapter_bank.parameters()
    }
    assert optimized_ids.isdisjoint(
        id(parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("matchup_adapter_bank.")
    )

    optimizer.zero_grad(set_to_none=True)
    loss, metrics = batch_losses(
        model,
        [
            _sequence(
                EXPERT_IDS[2],
                offset=2,
                acting_archetype="alakazam",
                source=EXPERT_IDS[1],
                package_id=EXPERT_IDS[4],
            ),
            _sequence(EXPERT_IDS[5], offset=3),
        ],
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        matchup_adapter_training=True,
    )
    assert metrics.n_matchup_adapter_rows == 2
    loss.backward()

    for route, expert in enumerate(model.matchup_adapter_bank.experts):
        gradients = [parameter.grad for parameter in expert.parameters()]
        if route in {2, 5}:
            assert all(gradient is not None for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)
    assert_matchup_adapter_training_contract(model, optimizer=optimizer)
    optimizer.step()
    assert_matchup_adapter_training_contract(
        model,
        optimizer=optimizer,
        base_state=base_before,
    )

    adapters_after = _snapshot(model.matchup_adapter_bank)
    assert any(
        not torch.equal(adapters_after[name], value)
        for name, value in adapters_before.items()
        if name.startswith(("experts.2.", "experts.5."))
    )
    assert all(
        torch.equal(adapters_after[name], value)
        for name, value in adapters_before.items()
        if not name.startswith(("experts.2.", "experts.5."))
    )
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in aux_before.items()
    )

    optimizer.zero_grad(set_to_none=True)
    before_unsupported = _snapshot(model.matchup_adapter_bank)
    unsupported_sequence = _sequence("unsupported-matchup", offset=6)
    with pytest.raises(RuntimeError, match="lacks an audited"):
        batch_losses(
            model,
            [unsupported_sequence],
            aux_weight=0.0,
            opp_hand_weight=0.0,
            opp_remainder_weight=0.0,
            matchup_adapter_training=True,
        )
    optimizer.step()
    assert all(
        torch.equal(model.matchup_adapter_bank.state_dict()[name], value)
        for name, value in before_unsupported.items()
    )


def test_packed_adapter_temporal_batches_match_legacy_loss_and_gradients() -> None:
    torch.manual_seed(29)
    legacy = _model().eval()
    packed = copy.deepcopy(legacy).eval()
    build_matchup_adapter_optimizer(
        legacy,
        lr=1e-3,
        weight_decay=0.0,
        activation_receipt=_activation_receipt(),
    )
    build_matchup_adapter_optimizer(
        packed,
        lr=1e-3,
        weight_decay=0.0,
        activation_receipt=_activation_receipt(),
    )

    sequences = [
        _sequence(EXPERT_IDS[2], offset=20),
        _sequence(EXPERT_IDS[5], offset=30),
    ]
    sequences[0].decisions.extend([_decision(21), _decision(22)])
    sequences[1].decisions.append(_decision(31))
    for sequence in sequences:
        route = route_for_archetype(sequence.opp_archetype)
        for decision in sequence.decisions:
            decision.matchup_adapter_oracle_route = route

    legacy_loss, legacy_metrics = batch_losses(
        legacy,
        sequences,
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        matchup_adapter_training=True,
        pack_temporal_games=False,
    )
    packed_loss, packed_metrics = batch_losses(
        packed,
        sequences,
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        matchup_adapter_training=True,
        pack_temporal_games=True,
    )
    torch.testing.assert_close(packed_loss, legacy_loss, atol=1e-6, rtol=1e-5)
    assert packed_metrics.n_decisions == legacy_metrics.n_decisions
    assert packed_metrics.n_matchup_adapter_rows == (
        legacy_metrics.n_matchup_adapter_rows
    )
    assert packed_metrics.total_loss == pytest.approx(
        legacy_metrics.total_loss, abs=1e-6, rel=1e-5
    )

    legacy_loss.backward()
    packed_loss.backward()
    legacy_parameters = dict(legacy.matchup_adapter_bank.named_parameters())
    packed_parameters = dict(packed.matchup_adapter_bank.named_parameters())
    assert legacy_parameters.keys() == packed_parameters.keys()
    for name in legacy_parameters:
        legacy_grad = legacy_parameters[name].grad
        packed_grad = packed_parameters[name].grad
        if legacy_grad is None or packed_grad is None:
            assert legacy_grad is None and packed_grad is None
        else:
            torch.testing.assert_close(
                packed_grad, legacy_grad, atol=2e-6, rtol=2e-5
            )


def test_resident_adapter_route_matches_object_loss_and_gradients() -> None:
    torch.manual_seed(31)
    route = 2
    sequences = [
        _sequence(EXPERT_IDS[route], offset=40),
        _sequence(EXPERT_IDS[route], offset=50),
    ]
    sequences[0].decisions.extend([_decision(41), _decision(42)])
    sequences[1].decisions.append(_decision(51))
    validation = _sequence(EXPERT_IDS[route], offset=60)
    for sequence in [*sequences, validation]:
        for decision in sequence.decisions:
            decision.matchup_adapter_oracle_route = route

    corpus = DeviceResidentBootstrapCorpus.from_splits(
        sequences,
        [validation],
        device=torch.device("cpu"),
        matchup_adapter_route=route,
    )
    object_model = _model().eval()
    resident_model = copy.deepcopy(object_model).eval()
    build_matchup_adapter_optimizer(
        object_model,
        lr=1e-3,
        weight_decay=0.0,
        activation_receipt=_activation_receipt(),
    )
    build_matchup_adapter_optimizer(
        resident_model,
        lr=1e-3,
        weight_decay=0.0,
        activation_receipt=_activation_receipt(),
    )

    object_loss, object_metrics = batch_losses(
        object_model,
        sequences,
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        matchup_adapter_training=True,
        pack_temporal_games=True,
    )
    resident_loss, resident_metrics = device_temporal_batch_losses(
        resident_model,
        corpus,
        torch.tensor([0, 1], dtype=torch.long),
        matchup_adapter_route=route,
    )
    torch.testing.assert_close(
        resident_loss, object_loss, atol=1e-6, rtol=1e-5
    )
    assert resident_metrics.n_decisions == object_metrics.n_decisions
    assert resident_metrics.n_matchup_adapter_rows == object_metrics.n_decisions

    object_loss.backward()
    resident_loss.backward()
    object_parameters = dict(
        object_model.matchup_adapter_bank.named_parameters()
    )
    resident_parameters = dict(
        resident_model.matchup_adapter_bank.named_parameters()
    )
    for name, parameter in object_parameters.items():
        observed = resident_parameters[name].grad
        if parameter.grad is None or observed is None:
            assert parameter.grad is None and observed is None
        else:
            torch.testing.assert_close(
                observed, parameter.grad, atol=2e-6, rtol=2e-5
            )


def test_optimizer_restore_preserves_exact_canonical_moments() -> None:
    source = _model().train()
    source_optimizer = build_matchup_adapter_optimizer(
        source,
        lr=1e-3,
        weight_decay=0.1,
        activation_receipt=_activation_receipt(),
    )
    for parameter in source.matchup_adapter_bank.parameters():
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()
    full = source_optimizer.state_dict()
    canonical_parameter_ids = list(full["param_groups"][0]["params"])
    v2 = copy.deepcopy(full)

    target = _model().train()
    target_optimizer = build_matchup_adapter_optimizer(
        target,
        lr=9e-4,
        weight_decay=0.2,
        activation_receipt=_activation_receipt(),
    )
    restored_experts = load_append_only_matchup_adapter_optimizer_state(
        target_optimizer, v2
    )
    migrated = target_optimizer.state_dict()

    assert restored_experts == len(EXPERT_IDS)
    assert len(migrated["param_groups"][0]["params"]) == 4 * len(EXPERT_IDS)
    assert len(migrated["state"]) == 4 * len(EXPERT_IDS)
    for index, old_id in enumerate(canonical_parameter_ids):
        new_id = migrated["param_groups"][0]["params"][index]
        assert torch.equal(
            migrated["state"][new_id]["exp_avg"],
            v2["state"][old_id]["exp_avg"],
        )
        assert torch.equal(
            migrated["state"][new_id]["exp_avg_sq"],
            v2["state"][old_id]["exp_avg_sq"],
        )


def test_step_guard_hard_fails_cross_route_gradient_or_optimizer_drift() -> None:
    torch.manual_seed(31)
    model = _model().eval()
    optimizer = build_matchup_adapter_optimizer(
        model,
        lr=1e-2,
        weight_decay=0.2,
        activation_receipt=_activation_receipt(),
    )
    sequence = _sequence(EXPERT_IDS[2], offset=21)
    optimizer.zero_grad(set_to_none=True)
    guard = prepare_matchup_adapter_isolation_guard(model, optimizer, [sequence])
    loss, _metrics = batch_losses(
        model,
        [sequence],
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        matchup_adapter_training=True,
    )
    loss.backward()
    assert_matchup_adapter_isolation_guard(
        model, optimizer, guard, after_step=False
    )
    optimizer.step()
    assert_matchup_adapter_isolation_guard(
        model, optimizer, guard, after_step=True
    )

    with torch.no_grad():
        model.matchup_adapter_bank.experts[0].up.bias.add_(0.5)
    with pytest.raises(AssertionError, match="inactive adapter route changed"):
        assert_matchup_adapter_isolation_guard(
            model, optimizer, guard, after_step=True
        )
