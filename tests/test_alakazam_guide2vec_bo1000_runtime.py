"""Focused no-game coverage for the isolated r212 runtime bridge."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "poke_bot/guide2vec_bo1000_runtime.py"
RUNNER_PATH = ROOT / "scripts/run_alakazam_guide2vec_no_mcts_bo1000_r212.py"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_r212_runtime_bridge_has_no_legacy_bo_runner_import() -> None:
    """The bridge may use neutral native primitives, never legacy BO runners."""

    forbidden = ("r205", "r207", "r214", "r215", "r216")
    for path in (RUNTIME_PATH, RUNNER_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not [
            name for name in imported if any(token in name.casefold() for token in forbidden)
        ], path


def test_runner_exposes_an_explicit_non_service_execution_boundary() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'mode.add_argument("--plan"' in source
    assert 'mode.add_argument("--preflight"' in source
    assert 'mode.add_argument("--run"' in source
    assert "run_guide2vec_bo1000(" in source
    assert ".service" not in source


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for Guide2Vec graph tests")
def test_control_graph_rejects_a_guide_module_or_guide_linear_transform() -> None:
    import torch.nn as nn

    from poke_bot.guide2vec_bo1000_runtime import (
        Guide2VecBO1000RuntimeError,
        inspect_control_graph,
    )

    class PlainPolicy:
        pass

    audit = inspect_control_graph(model=nn.Sequential(nn.Linear(2, 2)), policy=PlainPolicy())
    assert audit.module_instance_count == 0
    assert audit.parameter_count == 0
    assert audit.state_dict_key_count == 0
    assert audit.forward_hook_count == 0
    assert audit.linear_transform_count == 0

    blocked = nn.Module()
    blocked.guide_linear = nn.Linear(2, 2)
    with pytest.raises(Guide2VecBO1000RuntimeError, match="forbidden Guide2Vec"):
        inspect_control_graph(model=blocked, policy=PlainPolicy())


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for Guide2Vec graph tests")
def test_candidate_graph_binds_one_frozen_head_to_its_state_digest() -> None:
    from poke_bot.guide2vec import Guide2VecHead, state_dict_sha256
    from poke_bot.guide2vec_bo1000_runtime import (
        CandidateArtifact,
        R212_RUNTIME_GRAPH_SCHEMA,
        canonical_sha256,
        inspect_candidate_graph,
    )

    head = Guide2VecHead().eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    state_sha = state_dict_sha256(head.state_dict())
    checkpoint_sha = canonical_sha256({"checkpoint": "candidate"})
    runtime_sha = canonical_sha256({"runtime": "candidate"})
    base_sha = canonical_sha256({"base": "r195"})
    component_sha = canonical_sha256(
        {
            "schema": R212_RUNTIME_GRAPH_SCHEMA,
            "kind": "frozen_guide2vec_component",
            "checkpoint_sha256": checkpoint_sha,
            "runtime_config_sha256": runtime_sha,
            "parameter_count": head.parameter_count,
            "state_dict_sha256": state_sha,
            "base_identity_sha256": base_sha,
            "max_logit_bonus": 0.05,
            "frozen": True,
        }
    )
    artifact = CandidateArtifact(
        checkpoint_sha256=checkpoint_sha,
        training_receipt_sha256=canonical_sha256({"receipt": "candidate"}),
        runtime_config_sha256=runtime_sha,
        parameter_count=head.parameter_count,
        state_dict_sha256=state_sha,
        base_identity_sha256=base_sha,
        source_snapshot_sha256=canonical_sha256({"source": "candidate"}),
        component_graph_sha256=component_sha,
        feature_schema_sha256=canonical_sha256({"features": "candidate"}),
        model_config_sha256=canonical_sha256({"model": "r195"}),
    )
    audit = inspect_candidate_graph(head=head, expected=artifact)
    assert audit.module_instance_count == 1
    assert audit.parameter_count == head.parameter_count
    assert audit.frozen is True


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="runtime module imports torch")
def test_native_pair_binding_round_trips_without_opening_an_engine() -> None:
    import poke_bot.guide2vec_bo1000_runtime as runtime
    from poke_bot.guide2vec_bo1000_runtime import NativePairBinding, canonical_sha256
    from poke_bot.seeded_mirror_harness import (
        PairFirstPlayerSeal,
        build_seeded_seat_swapped_schedule,
    )

    native_seed_identity = canonical_sha256({"seed": "unit"})
    games = build_seeded_seat_swapped_schedule(
        evaluation_id=f"{runtime.GUIDE2VEC_EVALUATION_ID}-native-seed",
        seed_identity_sha256=native_seed_identity,
        pair_count=1,
    )
    seal = PairFirstPlayerSeal(
        evaluation_id=games[0].evaluation_id,
        pair_index=games[0].pair_index,
        pair_id=games[0].pair_id,
        pair_nonce_sha256=games[0].pair_nonce_sha256,
        engine_seed_u32=games[0].engine_seed_u32,
        deck_order_seed_u32=games[0].deck_order_seed_u32,
        first_player_seat=0,
        post_turn_order_observation_sha256=canonical_sha256({"first": 0}),
    )
    binding = NativePairBinding(
        r212_pair_id="r212-unit-pair",
        r212_pair_index=0,
        r212_pair_nonce_sha256=canonical_sha256({"r212": "pair"}),
        r212_pair_initial_rng_sha256=canonical_sha256({"r212": "rng"}),
        r212_pair_deck_order_rng_sha256=canonical_sha256({"r212": "deck"}),
        sealed_initial_first_actor_seat=0,
        seed_attempt=0,
        native_seed_identity_sha256=native_seed_identity,
        native_games=(games[0], games[1]),
        native_first_player_seal=seal,
        setup_actions_sha256=canonical_sha256([]),
    )
    assert NativePairBinding.from_payload(binding.as_payload()) == binding
    # Construction/round-trip above uses only schedule and snapshot objects;
    # it deliberately does not call the runtime's native environment factory.


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="runtime module imports torch")
def test_plan_shape_is_exactly_500_seat_swapped_pairs_without_starting_games(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import poke_bot.guide2vec_bo1000_runtime as runtime

    candidate = runtime.CandidateArtifact(
        checkpoint_sha256=runtime.canonical_sha256({"checkpoint": "candidate"}),
        training_receipt_sha256=runtime.canonical_sha256({"receipt": "candidate"}),
        runtime_config_sha256=runtime.canonical_sha256({"config": "candidate"}),
        parameter_count=155_468,
        state_dict_sha256=runtime.canonical_sha256({"state": "candidate"}),
        base_identity_sha256=runtime.canonical_sha256({"base": "r195"}),
        source_snapshot_sha256=runtime.canonical_sha256({"source": "candidate"}),
        component_graph_sha256=runtime.canonical_sha256({"component": "candidate"}),
        feature_schema_sha256=runtime.canonical_sha256({"features": "candidate"}),
        model_config_sha256=runtime.canonical_sha256({"model": "r195"}),
    )
    adapter_bank = runtime.canonical_sha256({"adapter": "bank"})
    adapter_fit = runtime.canonical_sha256({"adapter": "fit"})
    package_manifest = runtime.canonical_sha256({"package": "r195"})
    adapter_config = {"format": runtime.R195_ADAPTER_FORMAT, "slot_capacity": 64}
    monkeypatch.setattr(
        runtime,
        "_new_seeded_environment",
        lambda: pytest.fail("plan construction must not open a native battle"),
    )
    monkeypatch.setattr(
        runtime,
        "verify_r212_artifacts",
        lambda _artifacts: (
            candidate,
            candidate.model_config_sha256,
            adapter_bank,
            adapter_fit,
            adapter_config,
            package_manifest,
        ),
    )
    artifacts = runtime.R212ArtifactIdentity(
        r195_bundle=tmp_path / "submission.tar.gz",
        r195_package_root=tmp_path / "package",
        r195_checkpoint=tmp_path / "model.pt",
        guide2vec_checkpoint=tmp_path / "guide2vec.pt",
        guide2vec_training_receipt=tmp_path / "TRAINING_RECEIPT.json",
        owner_contract=tmp_path / "r212.json",
        r195_contract=tmp_path / "r195.json",
    )
    plan = runtime.build_guide2vec_bo1000_plan(
        artifacts=artifacts,
        seed_identity_sha256=runtime.canonical_sha256({"seed": "r212"}),
    )
    _experiment, schedule = runtime.verify_guide2vec_bo1000_plan(plan)
    assert len(schedule) == 1_000
    assert len({game.pair_id for game in schedule}) == 500
    assert sum(game.guide2vec_seat == 0 for game in schedule) == 500
    assert sum(game.guide2vec_seat == 1 for game in schedule) == 500
    for pair_index in range(500):
        pair = [game for game in schedule if game.pair_index == pair_index]
        assert [game.guide2vec_seat for game in pair] == [0, 1]
        assert sum(game.guide2vec_seat == game.sealed_initial_first_actor_seat for game in pair) == 1
