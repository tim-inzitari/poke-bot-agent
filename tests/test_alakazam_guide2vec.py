"""Focused unit coverage for the r212 Alakazam Guide2Vec sidecar.

These tests intentionally exercise the sidecar in isolation.  They must never
load, train, mutate, or route through the r195 base model: the Guide2Vec
artifact is an option-conditioned bounded overlay whose only authority is to
fall back to the exact frozen direct-policy logits when it cannot safely act.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from poke_bot.guide2vec import (
    FrozenBaseIdentity,
    Guide2VecConfig,
    Guide2VecError,
    Guide2VecHead,
    assert_base_frozen,
    freeze_base_model,
    load_checkpoint_payload,
    make_checkpoint_payload,
    state_dict_sha256,
    verify_base_checkpoint,
)


R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R195_CHECKPOINT_BYTES = 127_914_385
R212_MIN_PARAMETERS = 100_000
R212_MAX_PARAMETERS = 500_000


def _identity() -> FrozenBaseIdentity:
    return FrozenBaseIdentity.alakazam_submission_55378392()


def _inputs(
    *,
    batch_size: int = 1,
    width: int = 4,
    config: Guide2VecConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic policy-visible state/option/base-logit tensors."""

    config = config or Guide2VecConfig()
    generator = torch.Generator(device="cpu").manual_seed(212)
    state = torch.randn(
        batch_size,
        config.d_model,
        generator=generator,
        dtype=torch.float32,
    )
    options = torch.randn(
        batch_size,
        width,
        config.d_model,
        generator=generator,
        dtype=torch.float32,
    )
    base_logits = torch.randn(
        batch_size,
        width,
        generator=generator,
        dtype=torch.float32,
    )
    return state, options, base_logits


def _fallback_head(*, minimum_margin: float | None = None) -> Guide2VecHead:
    """Build a head whose confidence gate necessarily abstains."""

    config = replace(
        Guide2VecConfig(),
        min_eligibility=1.0,
        **(
            {} if minimum_margin is None else {"min_score_margin": minimum_margin}
        ),
    )
    return Guide2VecHead(config).eval()


def _as_bool_rows(value: object) -> list[bool]:
    tensor = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
    return [bool(item) for item in tensor.tolist()]


def _exact_60_card_fixture_deck() -> list[int]:
    """A structurally valid deck; the archetype predicate is patched per test."""

    return list(range(60))


def test_default_head_is_tiny_and_has_a_hard_auditable_parameter_count() -> None:
    head = Guide2VecHead()
    actual = sum(parameter.numel() for parameter in head.parameters())

    # Keep the sidecar in the separately contracted tiny-head envelope.  This is
    # intentionally the complete inference parameter count, not merely the
    # trainable subset, so freezing a tensor cannot evade the cap.
    assert head.parameter_count == actual == 155_468
    assert R212_MIN_PARAMETERS <= actual <= R212_MAX_PARAMETERS


def test_head_api_has_only_public_state_legal_options_and_base_logits() -> None:
    forward = inspect.signature(Guide2VecHead.forward)
    assert tuple(forward.parameters) == (
        "self",
        "state_vec",
        "option_hidden",
        "base_logits",
        "n_options",
    )
    rerank_parameters = set(inspect.signature(Guide2VecHead.rerank).parameters)
    # These inputs would be causal violations or disallowed search authority.
    assert not {
        "selected_action",
        "target_index",
        "reward",
        "result",
        "future_state",
        "aux_labels",
        "mcts",
        "rtp",
        "planner",
        "simulator",
    }.intersection(rerank_parameters)


def test_legal_option_mask_is_strict_and_option_scores_are_permutation_equivariant() -> None:
    torch.manual_seed(212)
    head = Guide2VecHead().eval()
    state, options, base_logits = _inputs(batch_size=2, width=5, config=head.config)
    # The base policy itself has no support on batch padding.
    base_logits[0, 2:] = -torch.inf

    decision = head.rerank(
        state,
        options,
        base_logits,
        n_options=torch.tensor([2, 5]),
        expected_base_identity=_identity(),
        observed_base_identity=_identity(),
    )

    assert decision.guide_scores.shape == (2, 5)
    assert decision.adjusted_logits.shape == (2, 5)
    assert torch.isneginf(decision.guide_scores[0, 2:]).all()
    assert torch.isneginf(decision.adjusted_logits[0, 2:]).all()
    assert torch.all(decision.bonus <= head.config.max_logit_bonus)
    assert torch.all(decision.bonus >= 0)
    assert int(torch.as_tensor(decision.selected_indices)[0]) < 2
    assert int(torch.as_tensor(decision.selected_indices)[1]) < 5

    # Reordering legal candidates must only reorder their scores.  This catches
    # a positional shortcut that would make a guide score an illegal action by
    # index rather than by its current legal option representation.
    single_state, single_options, single_base = _inputs(
        batch_size=1, width=4, config=head.config
    )
    original = head.rerank(
        single_state,
        single_options,
        single_base,
        n_options=torch.tensor([4]),
        expected_base_identity=_identity(),
        observed_base_identity=_identity(),
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = head.rerank(
        single_state,
        single_options[:, permutation],
        single_base[:, permutation],
        n_options=torch.tensor([4]),
        expected_base_identity=_identity(),
        observed_base_identity=_identity(),
    )
    torch.testing.assert_close(
        permuted.guide_scores,
        original.guide_scores[:, permutation],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        permuted.adjusted_logits,
        original.adjusted_logits[:, permutation],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("failure", ("low_confidence", "nonfinite", "singleton", "tied"))
def test_abstention_is_an_exact_base_logit_fallback(failure: str) -> None:
    if failure == "tied":
        head = Guide2VecHead(
            replace(Guide2VecConfig(), min_eligibility=0.0)
        ).eval()
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
    else:
        head = _fallback_head()

    state, options, base_logits = _inputs(config=head.config)
    n_options = torch.tensor([1 if failure == "singleton" else base_logits.size(1)])
    if failure == "singleton":
        base_logits[0, 1:] = -torch.inf
    if failure == "nonfinite":
        options[0, 1, 0] = float("nan")
    base_index = torch.argmax(base_logits, dim=-1)
    decision = head.rerank(
        state,
        options,
        base_logits,
        n_options=n_options,
        expected_base_identity=_identity(),
        observed_base_identity=_identity(),
    )

    assert _as_bool_rows(decision.fallback) == [True]
    assert _as_bool_rows(decision.applied) == [False]
    torch.testing.assert_close(decision.bonus, torch.zeros_like(base_logits), rtol=0, atol=0)
    torch.testing.assert_close(
        decision.adjusted_logits,
        base_logits,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        torch.as_tensor(decision.selected_indices), base_index, rtol=0, atol=0
    )


def test_partial_batch_fallback_keeps_the_raw_base_logits_exactly() -> None:
    """A fallback row must not rewrite even malformed padding logits.

    The selected index still has to use the legal mask, but the published
    direct-policy logits are immutable fallback evidence.  This catches a
    vectorized implementation that masks all rows after adding a zero bonus.
    """

    head = Guide2VecHead().eval()
    state, options, base_logits = _inputs(batch_size=2, width=3, config=head.config)
    base_logits[0] = torch.tensor([0.25, 123.0, 456.0])
    decision = head.rerank(
        state,
        options,
        base_logits,
        n_options=torch.tensor([1, 3]),
        expected_base_identity=_identity(),
        observed_base_identity=_identity(),
    )

    assert _as_bool_rows(decision.fallback)[0] is True
    torch.testing.assert_close(
        decision.adjusted_logits[0], base_logits[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        decision.bonus[0], torch.zeros_like(base_logits[0]), rtol=0, atol=0
    )
    assert int(torch.as_tensor(decision.selected_indices)[0]) == 0


def test_nonfinite_sidecar_output_falls_back_without_leaking_nan_audit_values() -> None:
    head = Guide2VecHead().eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.fill_(float("nan"))
    state, options, base_logits = _inputs(config=head.config)
    decision = head.rerank(
        state,
        options,
        base_logits,
        n_options=torch.tensor([base_logits.size(1)]),
        expected_base_identity=_identity(),
        observed_base_identity=_identity(),
    )

    assert _as_bool_rows(decision.fallback) == [True]
    assert _as_bool_rows(decision.applied) == [False]
    assert torch.isfinite(decision.eligibility_probability).all()
    assert torch.isfinite(decision.abstain_probability).all()
    torch.testing.assert_close(decision.adjusted_logits, base_logits, rtol=0, atol=0)
    torch.testing.assert_close(decision.bonus, torch.zeros_like(base_logits), rtol=0, atol=0)


def test_nonintegral_legal_option_counts_are_rejected_before_reranking() -> None:
    head = Guide2VecHead().eval()
    state, options, base_logits = _inputs(config=head.config)

    with pytest.raises(Guide2VecError, match="n_options.*integer"):
        head.rerank(
            state,
            options,
            base_logits,
            n_options=torch.tensor([2.5]),
            expected_base_identity=_identity(),
            observed_base_identity=_identity(),
        )


def test_missing_or_unbound_base_identity_cannot_apply_the_sidecar() -> None:
    """Identity is a runtime precondition, not a best-effort audit field."""

    head = Guide2VecHead().eval()
    state, options, base_logits = _inputs(config=head.config)
    decision = head.rerank(
        state,
        options,
        base_logits,
        n_options=torch.tensor([base_logits.size(1)]),
    )

    assert _as_bool_rows(decision.fallback) == [True]
    assert _as_bool_rows(decision.applied) == [False]
    assert decision.reasons == ("base_identity_mismatch",)
    torch.testing.assert_close(decision.adjusted_logits, base_logits, rtol=0, atol=0)


def test_identity_mismatch_and_payload_mismatch_fail_closed() -> None:
    head = Guide2VecHead().eval()
    identity = _identity()
    state, options, base_logits = _inputs(config=head.config)
    mismatched = replace(identity, bundle_sha256="sha256:" + "0" * 64)

    decision = head.rerank(
        state,
        options,
        base_logits,
        n_options=torch.tensor([base_logits.size(1)]),
        expected_base_identity=identity,
        observed_base_identity=mismatched,
    )
    assert _as_bool_rows(decision.fallback) == [True]
    assert _as_bool_rows(decision.applied) == [False]
    assert "base_identity_mismatch" in tuple(decision.reasons)
    torch.testing.assert_close(decision.adjusted_logits, base_logits, rtol=0, atol=0)

    payload = make_checkpoint_payload(head, identity, metadata={"r212": True})
    restored, restored_identity, metadata = load_checkpoint_payload(
        payload, expected_base_identity=identity
    )
    assert restored_identity == identity
    assert metadata == {"r212": True}
    assert state_dict_sha256(restored.state_dict()) == state_dict_sha256(head.state_dict())
    with pytest.raises(ValueError, match="identity|base"):
        load_checkpoint_payload(payload, expected_base_identity=mismatched)


def test_exact_submission_identity_and_separate_checkpoint_bytes_are_bound(
    tmp_path: Path,
) -> None:
    identity = _identity()
    assert identity.submission_id == 55_378_392
    assert identity.checkpoint_sha256 == R195_CHECKPOINT_SHA256
    assert identity.checkpoint_bytes == R195_CHECKPOINT_BYTES
    assert identity.bundle_sha256 == R195_BUNDLE_SHA256
    assert FrozenBaseIdentity.from_mapping(identity.as_dict()) == identity

    checkpoint_path = tmp_path / "base.pt"
    checkpoint_path.write_bytes(b"frozen-direct-policy")
    local_identity = FrozenBaseIdentity(
        submission_id=55_378_392,
        checkpoint_sha256=(
            "sha256:" + hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        ),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        bundle_sha256=R195_BUNDLE_SHA256,
    )
    verify_base_checkpoint(checkpoint_path, local_identity)
    checkpoint_path.write_bytes(b"tampered-direct-policy")
    with pytest.raises(ValueError, match="digest|bytes|identity"):
        verify_base_checkpoint(checkpoint_path, local_identity)


def test_freeze_base_and_sidecar_training_never_mutate_the_base_model() -> None:
    torch.manual_seed(212)
    base = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh())
    before = {
        name: parameter.detach().clone() for name, parameter in base.named_parameters()
    }
    freeze_base_model(base)
    assert_base_frozen(base)
    assert all(not parameter.requires_grad for parameter in base.parameters())

    head = Guide2VecHead().train()
    state, options, base_logits = _inputs(config=head.config)
    guide_scores, confidence_logits = head(state, options, base_logits)
    loss = guide_scores.square().mean() + confidence_logits.square().mean()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    for name, parameter in base.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def _one_deterministic_training_step(seed: int) -> str:
    torch.manual_seed(seed)
    head = Guide2VecHead().train()
    state, options, base_logits = _inputs(config=head.config)
    guide_scores, confidence_logits = head(state, options, base_logits)
    loss = guide_scores.square().mean() + confidence_logits.square().mean()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return state_dict_sha256(head.state_dict())


def test_same_seed_produces_the_exact_same_post_training_state_digest() -> None:
    first = _one_deterministic_training_step(212)
    second = _one_deterministic_training_step(212)
    changed_seed = _one_deterministic_training_step(213)

    assert first == second
    assert first != changed_seed


def test_r212_trainer_builds_the_same_bounded_canonical_head() -> None:
    """Keep the independently-owned trainer and runtime sidecar API aligned."""

    from scripts import train_alakazam_guide2vec_r212 as trainer

    config, head = trainer._guide2vec_config_and_head(torch.device("cpu"))
    assert trainer.MAX_GUIDE_LOGIT_BONUS == pytest.approx(0.05, abs=0.0)
    assert config.max_logit_bonus == pytest.approx(0.05, abs=0.0)
    assert head.parameter_count == 155_468


def test_r212_trainer_rejects_cached_zero_confidence_guide_target(
    tmp_path: Path,
) -> None:
    """A tampered cache cannot turn a zero-confidence label into coverage data."""

    from scripts import train_alakazam_guide2vec_r212 as trainer

    path = tmp_path / "latent.pt"
    torch.save(
        {
            "state_vec": torch.zeros((1, 96), dtype=torch.float16),
            "option_hidden": torch.zeros((2, 96), dtype=torch.float16),
            "base_logits": torch.zeros(2, dtype=torch.float16),
            "option_offsets": torch.tensor([0, 2], dtype=torch.long),
            "guide_target_index": torch.tensor([0], dtype=torch.int32),
            "guide_confidence": torch.tensor([0.0], dtype=torch.float32),
        },
        path,
    )
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError, match="nonpositive-confidence"):
        trainer._validate_latent_chunk(path, digest)


def test_r212_v6_route_contract_requires_exact_64_physical_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The r195 V6 bank is fixed-width; it is never a logical-index range."""

    from poke_bot import matchup_adapter_routes, matchup_adapters_v6
    from scripts import train_alakazam_guide2vec_r212 as trainer

    def contract_with_capacity(capacity: int) -> SimpleNamespace:
        return SimpleNamespace(
            adapter_format=matchup_adapters_v6.ADAPTER_CHECKPOINT_FORMAT,
            target_ids=("alakazam", "mew"),
            physical_slots=(4, 22),
            slot_capacity=capacity,
            slot_registry_digest="sha256:" + "a" * 64,
        )

    monkeypatch.setattr(
        matchup_adapter_routes,
        "resolve_matchup_adapter_route_contract",
        lambda _config: contract_with_capacity(64),
    )
    binding = trainer._resolve_r195_v6_adapter_route_binding(
        {"adapter_config": {}}
    )
    assert binding.slot_capacity == 64
    assert binding.physical_slots == (4, 22)

    monkeypatch.setattr(
        matchup_adapter_routes,
        "resolve_matchup_adapter_route_contract",
        lambda _config: contract_with_capacity(63),
    )
    with pytest.raises(RuntimeError, match="64-slot V6"):
        trainer._resolve_r195_v6_adapter_route_binding({"adapter_config": {}})


def test_r212_tree_accepted_subset_controls_public_v6_physical_routes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only physical slots accepted by the checksum-bound public tree may route."""

    from poke_bot import matchup_adapter_routes, matchup_adapters_v6
    from poke_bot import public_matchup_router
    from scripts import train_alakazam_guide2vec_r212 as trainer

    tree_path = tmp_path / "runtime-tree.json"
    tree_path.write_text('{"runtime_contract": {}}', encoding="utf-8")
    tree_digest = trainer._sha256(tree_path)
    monkeypatch.setattr(trainer, "R195_MATCHUP_TREE_SHA256", tree_digest)

    binding = trainer.AdapterRouteBinding(
        adapter_format=matchup_adapters_v6.ADAPTER_CHECKPOINT_FORMAT,
        target_ids=("alakazam", "mew", "retired-route"),
        physical_slots=(4, 22, 63),
        slot_capacity=64,
        slot_registry_digest="sha256:" + "b" * 64,
    )
    runtime_code = trainer.RuntimeRouteCodeBinding(
        submission_bundle_path=tmp_path / "r195-submission.tar.gz",
        submission_bundle_sha256="sha256:" + "9" * 64,
        entrypoint_member="./main.py",
        entrypoint_sha256="sha256:" + "8" * 64,
        router_member="./poke_bot/public_matchup_router.py",
        router_sha256="sha256:" + "7" * 64,
    )

    class FakeTree:
        digest = tree_digest
        runtime_enabled = True
        adapter_format = binding.adapter_format
        targets = binding.target_ids
        route_physical_slots = binding.physical_slots
        slot_registry_digest = binding.slot_registry_digest
        runtime_accepted_archetype_ids = frozenset({"alakazam", "mew"})

        @classmethod
        def from_path(
            cls, _path: Path, *, require_runtime_enabled: bool = True
        ) -> "FakeTree":
            assert require_runtime_enabled is True
            return cls()

    monkeypatch.setattr(
        matchup_adapter_routes,
        "resolve_matchup_adapter_route_contract",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        matchup_adapter_routes,
        "require_runtime_route_binding",
        lambda _runtime, _contract, *, allow_legacy_v5: None,
    )
    monkeypatch.setattr(public_matchup_router, "PublicMatchupDecisionTree", FakeTree)

    state: dict[str, torch.Tensor] = {}
    for slot in (4, 22):
        state[f"matchup_adapter_bank.experts.{slot}.up.weight"] = torch.tensor([1.0])
        state[f"matchup_adapter_bank.experts.{slot}.up.bias"] = torch.tensor([0.0])
    _, actual_digest, accepted = trainer._validate_r195_runtime_matchup_tree(
        tree_path,
        adapter_identity={"adapter_config": {}},
        checkpoint_payload={"model_state_dict": state},
        route_binding=binding,
    )
    assert actual_digest == tree_digest
    assert accepted.runtime_accepted_target_ids == ("alakazam", "mew")
    assert accepted.runtime_accepted_physical_slots == (4, 22)
    assert accepted.runtime_accepted_nonzero_output_slots == (4, 22)

    shard_path = tmp_path / "source.bin"
    shard_path.write_bytes(b"r212-source")
    shard = trainer.SourceShard(
        date="2026-07-17",
        path=shard_path,
        sha256="sha256:" + "c" * 64,
        byte_count=shard_path.stat().st_size,
        records=1,
        decisions=1,
        stat_identity=trainer._stat_identity(shard_path),
    )

    def sequence() -> SimpleNamespace:
        stage = SimpleNamespace(
            options=SimpleNamespace(num_words=2),
            target_index=0,
            guide_target_index=0,
            guide_confidence=1.0,
        )
        decision = SimpleNamespace(
            # Persisted compact routes are deliberately ignored: they predate
            # the submitted r195 router.  The fake raw resolver below is the
            # only authority for this test's route.
            matchup_adapter_public_route=-1,
            policy_stages=[stage],
            env_step=0,
        )
        return SimpleNamespace(
            episode_id="episode-1",
            seat=0,
            deck=_exact_60_card_fixture_deck(),
            decisions=[decision],
            policy_targets=None,
            factorized_policy_targets=None,
        )

    class FakeRawRouteResolver:
        route = 4

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            assert kwargs["feature_shard_path"] == shard.path
            assert kwargs["feature_shard_sha256"] == shard.sha256
            assert kwargs["matchup_tree_path"] == tree_path.resolve()
            assert kwargs["matchup_tree_sha256"] == tree_digest
            assert kwargs["allowed_physical_slots"] == frozenset({4, 22})
            assert kwargs["submission_bundle_path"] == runtime_code.submission_bundle_path
            assert kwargs["submission_bundle_sha256"] == runtime_code.submission_bundle_sha256
            assert kwargs["submission_entrypoint_member"] == runtime_code.entrypoint_member
            assert kwargs["submission_entrypoint_sha256"] == runtime_code.entrypoint_sha256
            assert kwargs["public_matchup_router_member"] == runtime_code.router_member
            assert kwargs["public_matchup_router_sha256"] == runtime_code.router_sha256

        @classmethod
        def open(cls, **kwargs: object) -> "FakeRawRouteResolver":
            return cls(**kwargs)

        def __enter__(self) -> "FakeRawRouteResolver":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def resolve_sequence(self, _sequence: object) -> SimpleNamespace:
            return SimpleNamespace(
                routes=(self.route,),
                raw_member_sha256="sha256:" + "f" * 64,
                compact_alignment_sha256="sha256:" + "0" * 64,
                turn_order_short_circuits=0,
                turn_order_short_circuit_env_steps=(),
                game_resets=0,
                game_reset_env_steps=(),
            )

        def projection(self, *, expected_records: int) -> dict[str, object]:
            assert expected_records == 1
            return {
                "schema": trainer.ROUTE_RECONSTRUCTION_SCHEMA,
                "source_date": "2026-07-17",
                "source_feature_shard_sha256": shard.sha256,
                "runtime_public_tree_sha256": tree_digest,
                "runtime_code": runtime_code.as_dict(),
                "allowed_physical_slots": [4, 22],
                "records": 1,
                "decisions": 1,
                "turn_order_short_circuits": 0,
                "game_resets": 0,
                "routed_decisions": int(self.route != -1),
                "bypassed_decisions": int(self.route == -1),
                "raw_archive": {"sha256": "sha256:" + "d" * 64, "bytes": 1},
                "member_route_sha256": "sha256:" + "e" * 64,
                "compact_source_routes_ignored": True,
                "oracle_route_used": False,
            }

    monkeypatch.setattr(trainer, "is_alakazam_deck", lambda _deck: True)
    monkeypatch.setattr(trainer, "RawPublicRouteResolver", FakeRawRouteResolver)
    monkeypatch.setattr(
        trainer, "iter_feature_shard", lambda _path: iter((sequence(),))
    )
    day = trainer._VerifiedDayGames(
        shard,
        max_context=96,
        quarantine={},
        adapter_route_binding=accepted,
        raw_archive_root=tmp_path / "raw-archives",
        matchup_tree_path=tree_path,
        matchup_tree_sha256=tree_digest,
        runtime_route_code=runtime_code,
        route_sidecar=None,
    )
    assert len(list(day)) == 1
    assert day.route_tensor(device=torch.device("cpu"), expected_samples=1).tolist() == [4]
    projection = day.route_projection(expected_samples=1)
    assert projection["source"] == "r195_raw_public_matchup_route_reconstruction"
    assert projection["route_reconstruction_mode"] == "raw"
    assert projection["compact_source_routes_ignored"] is True
    assert projection["routed_samples"] == 1

    # Slot 63 is inside the 64-slot tensor bank but not accepted by this tree.
    FakeRawRouteResolver.route = 63
    monkeypatch.setattr(
        trainer, "iter_feature_shard", lambda _path: iter((sequence(),))
    )
    rejected = trainer._VerifiedDayGames(
        shard,
        max_context=96,
        quarantine={},
        adapter_route_binding=accepted,
        raw_archive_root=tmp_path / "raw-archives",
        matchup_tree_path=tree_path,
        matchup_tree_sha256=tree_digest,
        runtime_route_code=runtime_code,
        route_sidecar=None,
    )
    with pytest.raises(RuntimeError, match="runtime-accepted r195 V6 physical slot"):
        list(rejected)


def test_r212_heldout_route_reconstruction_requires_a_sealed_sidecar(
    tmp_path: Path,
) -> None:
    """Heldout raw archives cannot become an unreceipted runtime dependency."""

    from scripts import train_alakazam_guide2vec_r212 as trainer

    shard_path = tmp_path / "heldout-source.bin"
    shard_path.write_bytes(b"heldout")
    shard = trainer.SourceShard(
        date=trainer.HELDOUT_DATES[0],
        path=shard_path,
        sha256="sha256:" + "f" * 64,
        byte_count=shard_path.stat().st_size,
        records=1,
        decisions=1,
        stat_identity=trainer._stat_identity(shard_path),
    )
    binding = trainer.AdapterRouteBinding(
        adapter_format="poke-bot-matchup-adapter-bank-v6",
        target_ids=("alakazam",),
        physical_slots=(6,),
        slot_capacity=64,
        slot_registry_digest="sha256:" + "1" * 64,
        runtime_accepted_target_ids=("alakazam",),
        runtime_accepted_physical_slots=(6,),
        runtime_accepted_nonzero_output_slots=(6,),
    )
    runtime_code = trainer.RuntimeRouteCodeBinding(
        submission_bundle_path=tmp_path / "r195-submission.tar.gz",
        submission_bundle_sha256="sha256:" + "9" * 64,
        entrypoint_member="./main.py",
        entrypoint_sha256="sha256:" + "8" * 64,
        router_member="./poke_bot/public_matchup_router.py",
        router_sha256="sha256:" + "7" * 64,
    )
    with pytest.raises(RuntimeError, match="heldout source requires an immutable verified public-route sidecar"):
        trainer._VerifiedDayGames(
            shard,
            max_context=96,
            quarantine={},
            adapter_route_binding=binding,
            raw_archive_root=tmp_path / "raw-archives",
            matchup_tree_path=tmp_path / "tree.json",
            matchup_tree_sha256="sha256:" + "2" * 64,
            runtime_route_code=runtime_code,
            route_sidecar=None,
        )


def test_r212_heldout_sidecar_binds_the_shard_and_raw_archive_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Heldout sidecars cannot be read without their source/header binding."""

    from scripts import train_alakazam_guide2vec_r212 as trainer

    shard_path = tmp_path / "heldout-source.bin"
    shard_path.write_bytes(b"heldout")
    shard = trainer.SourceShard(
        date=trainer.HELDOUT_DATES[0],
        path=shard_path,
        sha256="sha256:" + "f" * 64,
        byte_count=shard_path.stat().st_size,
        records=1,
        decisions=1,
        stat_identity=trainer._stat_identity(shard_path),
    )
    raw_archive = {
        "source_date": shard.date,
        "archive_name": f"pokemon-tcg-ai-battle-episodes-{shard.date}.zip",
        "sha256": "sha256:" + "a" * 64,
        "bytes": 17,
    }
    sidecar_path = tmp_path / "heldout-sidecar.jsonl"
    sidecar_path.write_text("sealed", encoding="utf-8")
    sidecar = trainer.RouteSidecar(
        date=shard.date,
        path=sidecar_path,
        sha256="sha256:" + "b" * 64,
        raw_archive=raw_archive,
        producer_code=trainer._expected_route_sidecar_producer_code(),
    )
    binding = trainer.AdapterRouteBinding(
        adapter_format="poke-bot-matchup-adapter-bank-v6",
        target_ids=("alakazam",),
        physical_slots=(6,),
        slot_capacity=64,
        slot_registry_digest="sha256:" + "1" * 64,
        runtime_accepted_target_ids=("alakazam",),
        runtime_accepted_physical_slots=(6,),
        runtime_accepted_nonzero_output_slots=(6,),
    )
    runtime_code = trainer.RuntimeRouteCodeBinding(
        submission_bundle_path=tmp_path / "r195-submission.tar.gz",
        submission_bundle_sha256="sha256:" + "9" * 64,
        entrypoint_member="./main.py",
        entrypoint_sha256="sha256:" + "8" * 64,
        router_member="./poke_bot/public_matchup_router.py",
        router_sha256="sha256:" + "7" * 64,
    )
    stage = SimpleNamespace(
        options=SimpleNamespace(num_words=2),
        target_index=0,
        guide_target_index=0,
        guide_confidence=1.0,
    )
    sequence = SimpleNamespace(
        episode_id="heldout-episode",
        seat=0,
        deck=_exact_60_card_fixture_deck(),
        decisions=[SimpleNamespace(env_step=0, policy_stages=[stage])],
        policy_targets=None,
        factorized_policy_targets=None,
    )

    class FakeSidecarResolver:
        @classmethod
        def open(cls, **kwargs: object) -> "FakeSidecarResolver":
            assert kwargs["feature_shard_path"] == shard.path
            assert kwargs["feature_shard_sha256"] == shard.sha256
            assert kwargs["expected_raw_archive"] == raw_archive
            assert kwargs["expected_producer_code"] == sidecar.producer_code
            assert kwargs["expected_sidecar_sha256"] == sidecar.sha256
            assert kwargs["matchup_tree_sha256"] == "sha256:" + "2" * 64
            assert kwargs["allowed_physical_slots"] == frozenset({6})
            assert kwargs["submission_bundle_sha256"] == runtime_code.submission_bundle_sha256
            assert kwargs["submission_entrypoint_member"] == runtime_code.entrypoint_member
            assert kwargs["submission_entrypoint_sha256"] == runtime_code.entrypoint_sha256
            assert kwargs["public_matchup_router_member"] == runtime_code.router_member
            assert kwargs["public_matchup_router_sha256"] == runtime_code.router_sha256
            return cls()

        def __enter__(self) -> "FakeSidecarResolver":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def resolve_sequence(self, _sequence: object) -> SimpleNamespace:
            return SimpleNamespace(
                routes=(6,),
                raw_member_sha256="sha256:" + "d" * 64,
                compact_alignment_sha256="sha256:" + "e" * 64,
                turn_order_short_circuits=0,
                turn_order_short_circuit_env_steps=(),
                game_resets=0,
                game_reset_env_steps=(),
            )

        def projection(self, *, expected_records: int) -> dict[str, object]:
            assert expected_records == 1
            return {
                "schema": trainer.ROUTE_RECONSTRUCTION_SCHEMA,
                "source_date": shard.date,
                "source_feature_shard_sha256": shard.sha256,
                "runtime_public_tree_sha256": "sha256:" + "2" * 64,
                "runtime_code": runtime_code.as_dict(),
                "producer_code": dict(sidecar.producer_code),
                "allowed_physical_slots": [6],
                "records": 1,
                "decisions": 1,
                "turn_order_short_circuits": 0,
                "game_resets": 0,
                "routed_decisions": 1,
                "bypassed_decisions": 0,
                "raw_archive": raw_archive,
                "member_route_sha256": "sha256:" + "c" * 64,
                "compact_source_routes_ignored": True,
                "oracle_route_used": False,
                "sidecar": {"path": str(sidecar.path), "sha256": sidecar.sha256},
            }

    monkeypatch.setattr(trainer, "is_alakazam_deck", lambda _deck: True)
    monkeypatch.setattr(trainer, "SidecarPublicRouteResolver", FakeSidecarResolver)
    monkeypatch.setattr(trainer, "iter_feature_shard", lambda _path: iter((sequence,)))
    day = trainer._VerifiedDayGames(
        shard,
        max_context=96,
        quarantine={},
        adapter_route_binding=binding,
        raw_archive_root=tmp_path / "raw-archives",
        matchup_tree_path=tmp_path / "tree.json",
        matchup_tree_sha256="sha256:" + "2" * 64,
        runtime_route_code=runtime_code,
        route_sidecar=sidecar,
    )
    assert len(list(day)) == 1
    assert day.route_tensor(device=torch.device("cpu"), expected_samples=1).tolist() == [6]
    assert day.route_projection(expected_samples=1)["route_reconstruction_mode"] == "sidecar"


def test_r212_heldout_manifest_rejects_unsealed_route_producer_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The manifest must name the local resolver/materializer code exactly."""

    from scripts import train_alakazam_guide2vec_r212 as trainer

    day = "2026-07-22"
    expected_producer_code = {
        "schema": trainer.ROUTE_SIDECAR_PRODUCER_CODE_SCHEMA,
        "guide2vec_public_routes_sha256": "sha256:" + "a" * 64,
        "materializer_cli_sha256": "sha256:" + "b" * 64,
    }
    monkeypatch.setattr(trainer, "HELDOUT_DATES", (day,))
    monkeypatch.setattr(
        trainer,
        "_expected_route_sidecar_producer_code",
        lambda: dict(expected_producer_code),
    )

    sidecar = tmp_path / "sealed-sidecar.jsonl"
    sidecar.write_text("sealed\n", encoding="utf-8")
    manifest = tmp_path / "sidecars.json"
    payload = {
        "schema": trainer.ROUTE_SIDECAR_MANIFEST_SCHEMA,
        "producer_code": dict(expected_producer_code),
        "days": {
            day: {
                "path": sidecar.name,
                "sha256": trainer._sha256(sidecar),
                "raw_archive": {
                    "source_date": day,
                    "archive_name": f"pokemon-tcg-ai-battle-episodes-{day}.zip",
                    "sha256": "sha256:" + "c" * 64,
                    "bytes": 1,
                },
            }
        },
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    _, _, parsed = trainer._validate_route_sidecar_manifest(
        manifest, trainer._sha256(manifest)
    )
    assert parsed[day].producer_code == expected_producer_code

    payload["producer_code"] = dict(expected_producer_code)
    payload["producer_code"]["materializer_cli_sha256"] = "sha256:" + "d" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="producer code differs"):
        trainer._validate_route_sidecar_manifest(manifest, trainer._sha256(manifest))


def test_r212_trainer_excludes_isfirst_decisions_before_model_packing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An r195 IsFirst compact row must be absent from history and samples."""

    from scripts import train_alakazam_guide2vec_r212 as trainer

    shard_path = tmp_path / "source.bin"
    shard_path.write_bytes(b"r212-source")
    shard = trainer.SourceShard(
        date="2026-07-17",
        path=shard_path,
        sha256="sha256:" + "a" * 64,
        byte_count=shard_path.stat().st_size,
        records=1,
        decisions=2,
        stat_identity=trainer._stat_identity(shard_path),
    )
    binding = trainer.AdapterRouteBinding(
        adapter_format="poke-bot-matchup-adapter-bank-v6",
        target_ids=("alakazam",),
        physical_slots=(6,),
        slot_capacity=64,
        slot_registry_digest="sha256:" + "b" * 64,
        runtime_accepted_target_ids=("alakazam",),
        runtime_accepted_physical_slots=(6,),
        runtime_accepted_nonzero_output_slots=(6,),
    )
    runtime_code = trainer.RuntimeRouteCodeBinding(
        submission_bundle_path=tmp_path / "r195-submission.tar.gz",
        submission_bundle_sha256="sha256:" + "c" * 64,
        entrypoint_member="./main.py",
        entrypoint_sha256="sha256:" + "d" * 64,
        router_member="./poke_bot/public_matchup_router.py",
        router_sha256="sha256:" + "e" * 64,
    )

    @dataclass
    class MutableSequence:
        episode_id: str
        seat: int
        deck: object
        decisions: list[object]
        policy_targets: object | None = None
        factorized_policy_targets: object | None = None

    stage = SimpleNamespace(
        options=SimpleNamespace(num_words=2),
        target_index=0,
        guide_target_index=0,
        guide_confidence=1.0,
    )
    original = MutableSequence(
        episode_id="episode-isfirst",
        seat=0,
        deck=_exact_60_card_fixture_deck(),
        decisions=[
            SimpleNamespace(env_step=0, policy_stages=[stage]),
            SimpleNamespace(env_step=1, policy_stages=[stage]),
        ],
    )

    class FakeRawRouteResolver:
        game_resets = 0
        game_reset_env_steps: tuple[int, ...] = ()

        @classmethod
        def open(cls, **_kwargs: object) -> "FakeRawRouteResolver":
            return cls()

        def __enter__(self) -> "FakeRawRouteResolver":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def resolve_sequence(self, _sequence: object) -> SimpleNamespace:
            return SimpleNamespace(
                routes=(6, 6),
                raw_member_sha256="sha256:" + "f" * 64,
                compact_alignment_sha256="sha256:" + "0" * 64,
                turn_order_short_circuits=1,
                turn_order_short_circuit_env_steps=(0,),
                game_resets=self.game_resets,
                game_reset_env_steps=self.game_reset_env_steps,
            )

        def projection(self, *, expected_records: int) -> dict[str, object]:
            assert expected_records == 1
            return {
                "schema": trainer.ROUTE_RECONSTRUCTION_SCHEMA,
                "source_date": shard.date,
                "source_feature_shard_sha256": shard.sha256,
                "runtime_public_tree_sha256": "sha256:" + "1" * 64,
                "runtime_code": runtime_code.as_dict(),
                "allowed_physical_slots": [6],
                "records": 1,
                "decisions": 2,
                "turn_order_short_circuits": 1,
                "game_resets": self.game_resets,
                "routed_decisions": 2,
                "bypassed_decisions": 0,
                "raw_archive": {"sha256": "sha256:" + "2" * 64, "bytes": 1},
                "member_route_sha256": "sha256:" + "3" * 64,
                "compact_source_routes_ignored": True,
                "oracle_route_used": False,
            }

    monkeypatch.setattr(trainer, "is_alakazam_deck", lambda _deck: True)
    monkeypatch.setattr(trainer, "RawPublicRouteResolver", FakeRawRouteResolver)
    monkeypatch.setattr(trainer, "iter_feature_shard", lambda _path: iter((original,)))
    day = trainer._VerifiedDayGames(
        shard,
        # With a one-decision cap, filtering must happen before capping: if
        # the setup bypass were retained until after the cap, no real model
        # decision could survive.
        max_context=1,
        quarantine={},
        adapter_route_binding=binding,
        raw_archive_root=tmp_path / "raw-archives",
        matchup_tree_path=tmp_path / "tree.json",
        matchup_tree_sha256="sha256:" + "1" * 64,
        runtime_route_code=runtime_code,
        route_sidecar=None,
    )
    packed = list(day)

    assert len(packed) == 1
    assert [decision.env_step for decision in packed[0].decisions] == [1]
    assert day.route_tensor(device=torch.device("cpu"), expected_samples=1).tolist() == [6]
    projection = day.route_projection(expected_samples=1)
    exclusions = projection["compact_turn_order_short_circuit_exclusions"]
    assert projection["compact_turn_order_short_circuits_admitted"] is False
    assert exclusions["excluded_decisions"] == 1
    assert exclusions["excluded_policy_stages"] == 1
    assert exclusions["excluded_samples"] == 1
    assert exclusions["excluded_guide_rows"] == 1
    assert exclusions["admitted_to_model_history"] is False

    # A reset is causally valid only before the retained compact window.  The
    # raw resolver suite tests r195's ACTIVE/select-null reset behavior; this
    # trainer assertion prevents a future caller from silently accepting that
    # event after temporal history has already begun.
    FakeRawRouteResolver.game_resets = 1
    FakeRawRouteResolver.game_reset_env_steps = (1,)
    reset_spans_history = trainer._VerifiedDayGames(
        shard,
        max_context=1,
        quarantine={},
        adapter_route_binding=binding,
        raw_archive_root=tmp_path / "raw-archives",
        matchup_tree_path=tmp_path / "tree.json",
        matchup_tree_sha256="sha256:" + "1" * 64,
        runtime_route_code=runtime_code,
        route_sidecar=None,
    )
    with pytest.raises(RuntimeError, match="turn-order short-circuit proof"):
        list(reset_spans_history)


def test_r212_checkpoint_selection_is_validation_rank_nll_only() -> None:
    """Coverage BCE trains eligibility but has no model-selection authority."""

    from scripts import train_alakazam_guide2vec_r212 as trainer

    source = inspect.getsource(trainer._run_training)
    selection = source.split("best_metric = math.inf", 1)[1].split(
        "if best_state is None", 1
    )[0]
    assert 'metric = float(validation_metrics["rank_nll"])' in selection
    assert "coverage_bce" not in selection
    assert trainer.SELECTION_METRIC == (
        "validation_confidence_weighted_listwise_cross_entropy"
    )


def test_r212_calibration_counts_unlabeled_applied_stages_as_false_positives() -> None:
    """A masked stage must still hurt precision if runtime would apply to it."""

    from scripts import train_alakazam_guide2vec_r212 as trainer

    calibration = trainer.calibrate_abstention(
        {
            # Equal probabilities mean a threshold can select both rows or
            # neither.  Ignoring the unlabeled row would incorrectly pass.
            "probability": torch.tensor([0.90, 0.90]),
            "correct": torch.tensor([True, False]),
            "eligible": torch.tensor([True, False]),
            "applicable": torch.tensor([True, True]),
        }
    )
    assert calibration["status"] == "abstain_all_precision_floor_not_met"
    assert calibration["threshold"] == 1.0
    assert calibration["applied_rows"] == 0
    assert calibration["applied_labeled_rows"] == 0


def test_r212_calibrated_min_eligibility_is_bound_into_runtime_serialization() -> None:
    """The selected validation threshold must alter the serialized head config."""

    from scripts import train_alakazam_guide2vec_r212 as trainer

    runtime_config, runtime_receipt = trainer._runtime_config_from_validation(
        Guide2VecConfig(),
        {
            "status": "validation_calibrated",
            "threshold": 0.73,
            "precision": 0.75,
            "applied_rows": 4,
        },
    )
    assert runtime_config.min_eligibility == pytest.approx(0.73)
    assert runtime_receipt["eligibility_threshold"] == pytest.approx(0.73)
    assert runtime_receipt["eligibility_threshold_field"] == (
        "Guide2VecConfig.min_eligibility"
    )
    assert runtime_receipt["guide2vec_config"]["min_eligibility"] == pytest.approx(
        0.73
    )
    assert runtime_receipt["runtime_config_sha256"].startswith("sha256:")

    head = Guide2VecHead(runtime_config)
    payload = make_checkpoint_payload(
        head,
        _identity(),
        metadata={"guide2vec_runtime_config": runtime_receipt},
    )
    assert payload["config"]["min_eligibility"] == pytest.approx(0.73)
    assert "head.config = runtime_config" in inspect.getsource(trainer._run_training)


def test_r212_managed_snapshot_content_output_and_combo_guards_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Managed execution needs a source receipt, sealed output, and combo off."""

    from scripts import train_alakazam_guide2vec_r212 as trainer

    service = (
        Path(__file__).resolve().parents[1]
        / "deploy/systemd/pokebot-alakazam-guide2vec-r212.service"
    ).read_text(encoding="utf-8")
    assert "Environment=POKEBOT_GUIDE2VEC_R212_ISOLATED=1" in service
    assert "Environment=POKEBOT_GUIDE2VEC_R212_REQUIRE_CONTENT_ADDRESSED_OUTPUT=1" in service
    assert "Environment=POKEBOT_COMBO_STATE_ROUTE_ENABLED=0" in service
    assert "Environment=POKEBOT_USE_RECURSIVE_TURN_PLANNER=0" in service
    assert "stage_alakazam_guide2vec_r212_source_snapshot.py verify" in service

    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    monkeypatch.setenv("POKEBOT_GUIDE2VEC_R212_ISOLATED", "1")
    monkeypatch.delenv("POKEBOT_COMBO_STATE_ROUTE_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="must explicitly pin"):
        trainer._assert_combo_state_route_disabled_environment()
    with pytest.raises(RuntimeError, match="requires a published source snapshot"):
        trainer._validated_source_snapshot()

    monkeypatch.setenv("POKEBOT_COMBO_STATE_ROUTE_ENABLED", "0")
    trainer._assert_combo_state_route_disabled_environment()

    monkeypatch.setenv(
        "POKEBOT_GUIDE2VEC_R212_REQUIRE_CONTENT_ADDRESSED_OUTPUT", "1"
    )
    args = SimpleNamespace(output_root=tmp_path / "output", output_dir=None)
    run_dir = trainer._resolve_run_dir(args, {"schema": "test"})
    assert run_dir.parent == args.output_root
    assert run_dir.name.startswith("r212-")

    args.output_dir = tmp_path / "manual-output"
    with pytest.raises(RuntimeError, match="content-addressed output rejects"):
        trainer._resolve_run_dir(args, {"schema": "test"})

    monkeypatch.setenv("POKEBOT_COMBO_STATE_ROUTE_ENABLED", "1")
    with pytest.raises(RuntimeError, match="requires POKEBOT_COMBO_STATE_ROUTE_ENABLED=0"):
        trainer._assert_combo_state_route_disabled_environment()
