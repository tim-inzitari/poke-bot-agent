from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch

from poke_bot import checkpoint
from poke_bot.agent import PolicyAgent
from poke_bot.dataset import DecisionSample, GameSequence
from poke_bot.matchup_adapter_activation import (
    ADAPTER_REHEARSAL_AUTHORIZATION_SCHEMA,
    ACTIVATION_RECEIPT_SCHEMA,
    CORPUS_MANIFEST_SCHEMA,
    FIRST_ELIGIBLE_ITERATION,
    MatchupAdapterGameRouter,
    PublicMatchupRecognizer,
    ShadowMatchupAdapterRouter,
    UNACCEPTED_RUNTIME_ROUTES,
    ZERO_DORMANT_CHECKPOINT_SCHEMA,
    build_activation_receipt,
    build_adapter_rehearsal_authorization,
    gate_exclusions,
    merge_dormant_adapter_checkpoint,
    materialize_zero_dormant_adapter_checkpoint,
    parse_corpus_manifest,
    prepare_adapter_corpus_records,
    runtime_model_route,
    training_route_for_decision,
    validate_adapter_training_authorization,
)
from poke_bot.matchup_adapters import (
    EXPERT_IDS,
    HIDDEN_DIM,
    UNKNOWN_ROUTE,
    MatchupAdapterBank,
    route_for_archetype,
)
from poke_bot.train import (
    build_matchup_adapter_training_contract,
    matchup_adapter_split_contract,
    split_matchup_adapter_sequences,
)


def _obs(*public_ids: int, hidden_ids: tuple[int, ...] = ()) -> dict:
    opponent = {
        "active": [{"id": value} for value in public_ids],
        "bench": [],
        "discard": [],
        # These are deliberate decoys. The recognizer must never walk them.
        "hand": [{"id": value} for value in hidden_ids],
        "deck": [{"id": value} for value in hidden_ids],
        "prizes": [{"id": value} for value in hidden_ids],
    }
    return {
        "opp_archetype": "marnie-s-grimmsnarl-ex",
        "opponent_id": "metadata-must-not-route",
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": [], "discard": []},
                opponent,
            ],
        },
    }


def _routes_for_first_active(first_active: int | None, length: int) -> list[int]:
    recognizer = PublicMatchupRecognizer()
    if first_active == 0:
        # Setup/public history can contain the first evidence-bearing state;
        # the first recorded decision then becomes the second confirmation.
        assert recognizer.observe(_obs(646)).route == UNKNOWN_ROUTE
    rows: list[int] = []
    for index in range(length):
        evidence = bool(
            first_active is not None
            and (
                index >= first_active - 1
                if first_active > 0
                else index >= 0
            )
        )
        rows.append(recognizer.observe(_obs(646) if evidence else _obs()).route)
    return rows


@pytest.mark.parametrize(
    ("first_active", "length"),
    [
        (0, 3),
        (1, 3),
        (20, 24),
        (300, 304),
        (319, 323),
        (320, 324),
        (321, 325),
        (39, 40),  # final position
        (None, 325),
    ],
)
def test_arbitrary_trigger_position_exact_noop_and_selected_output(
    first_active: int | None,
    length: int,
) -> None:
    routes = _routes_for_first_active(first_active, length)
    expected_route = route_for_archetype("marnie-s-grimmsnarl-ex")
    if first_active is None:
        assert routes == [UNKNOWN_ROUTE] * length
    else:
        assert routes[:first_active] == [UNKNOWN_ROUTE] * first_active
        assert routes[first_active:] == [expected_route] * (length - first_active)

    torch.manual_seed(22)
    state = torch.randn(length, HIDDEN_DIM)
    bank = MatchupAdapterBank(enabled=True)
    with torch.no_grad():
        bank.experts[expected_route].up.bias.copy_(
            torch.linspace(-0.3, 0.3, HIDDEN_DIM)
        )
    actual = bank(state, torch.tensor(routes))
    inactive = torch.tensor(routes) == UNKNOWN_ROUTE
    assert torch.equal(actual[inactive], state[inactive])
    if first_active is not None:
        assert not torch.equal(actual[~inactive], state[~inactive])


def test_prefix_causality_chunk_resume_rollover_reset_and_branch_isolation() -> None:
    length = 325
    observations = [_obs() for _ in range(length)]
    observations[299:] = [_obs(646) for _ in range(length - 299)]

    one_pass = PublicMatchupRecognizer()
    expected = [one_pass.observe(row).route for row in observations]
    prefix_only = PublicMatchupRecognizer()
    prefix = [prefix_only.observe(row).route for row in observations[:250]]
    assert prefix == expected[:250]  # future evidence cannot rewrite the prefix

    chunked = PublicMatchupRecognizer()
    actual = [chunked.observe(row).route for row in observations[:200]]
    saved = copy.deepcopy(chunked.state_dict())
    resumed = PublicMatchupRecognizer()
    resumed.load_state_dict(saved)
    actual.extend(resumed.observe(row).route for row in observations[200:])
    assert actual == expected
    assert expected[300] == route_for_archetype("marnie-s-grimmsnarl-ex")

    branch = resumed.fork()
    resumed.reset()
    assert resumed.observe(_obs()).route == UNKNOWN_ROUTE
    assert branch.state_dict() != resumed.state_dict()
    clean_game = MatchupAdapterGameRouter(first_eligible_iteration=16)
    assert clean_game.observe(_obs()).route == UNKNOWN_ROUTE


def test_public_recognizer_ignores_private_metadata_and_fails_on_conflicts() -> None:
    recognizer = PublicMatchupRecognizer()
    for _ in range(3):
        assert recognizer.observe(_obs(hidden_ids=(646, 647, 648))).route == UNKNOWN_ROUTE

    recognizer.reset()
    assert recognizer.observe(_obs(646)).route == UNKNOWN_ROUTE
    conflict = recognizer.observe(_obs(646, 117))
    assert conflict.route == UNKNOWN_ROUTE
    assert conflict.status == "conflict"

    recognizer.reset()
    assert recognizer.observe(_obs(646, 431)).route == UNKNOWN_ROUTE
    ambiguous = recognizer.observe(_obs(646, 431))
    assert ambiguous.route == UNKNOWN_ROUTE
    assert ambiguous.status == "ambiguous"


def test_global_boundary_is_fixed_per_game_and_pre_activation_is_exact_unknown() -> None:
    pre = MatchupAdapterGameRouter(first_eligible_iteration=15)
    assert [pre.observe(_obs(646)).route for _ in range(4)] == [UNKNOWN_ROUTE] * 4
    assert all(pre.observe(_obs(646)).status == "pre_activation" for _ in range(2))

    live = MatchupAdapterGameRouter(first_eligible_iteration=FIRST_ELIGIBLE_ITERATION)
    assert live.observe(_obs(646)).route == UNKNOWN_ROUTE
    assert live.observe(_obs(646)).route == route_for_archetype(
        "marnie-s-grimmsnarl-ex"
    )
    assert live.observe(_obs()).route == UNKNOWN_ROUTE  # immediate deactivate


def test_shadow_router_traces_arbitrary_midgame_activation_but_model_stays_unknown() -> None:
    expected = route_for_archetype("marnie-s-grimmsnarl-ex")
    for first_active in (20, 300, 320):
        shadow = ShadowMatchupAdapterRouter()
        for index in range(first_active + 3):
            evidence = index >= first_active - 1
            decision = shadow.observe(
                _obs(646) if evidence else _obs(),
                scope="game_root",
                depth=index,
            )
            assert shadow.model_route == UNKNOWN_ROUTE
            if index < first_active:
                assert decision.route == UNKNOWN_ROUTE
            else:
                assert decision.route == expected
        snapshot = shadow.audit.snapshot()
        assert snapshot["runtime_enabled"] is False
        assert snapshot["model_route"] == UNKNOWN_ROUTE
        assert snapshot["recognized_observations"] == 3
        assert all(row["model_route"] == UNKNOWN_ROUTE for row in snapshot["events"])


def test_shadow_reset_fork_isolation_and_unaccepted_runtime_routes_fail_closed() -> None:
    shadow = ShadowMatchupAdapterRouter()
    for _ in range(3):
        assert shadow.observe(_obs(hidden_ids=(646, 647, 648))).route == UNKNOWN_ROUTE
        assert shadow.model_route == UNKNOWN_ROUTE
    shadow.reset_for_new_game()
    assert shadow.observe(_obs(646)).route == UNKNOWN_ROUTE
    branch = shadow.fork()
    expected = route_for_archetype("marnie-s-grimmsnarl-ex")
    assert branch.observe(_obs(646), scope="belief_search_branch", depth=1).route == expected
    # The owning game still has only one evidence state; a branch cannot latch it.
    assert shadow.observe(_obs()).route == UNKNOWN_ROUTE
    shadow.reset_for_new_game()
    assert shadow.audit.snapshot()["observations"] == 0
    assert shadow.observe(_obs(646)).route == UNKNOWN_ROUTE

    for route in sorted(UNACCEPTED_RUNTIME_ROUTES):
        assert runtime_model_route(route, enabled=False) == UNKNOWN_ROUTE
        with pytest.raises(RuntimeError, match="no accepted runtime recognizer"):
            runtime_model_route(route, enabled=True)
    with pytest.raises(TypeError, match="exact boolean"):
        runtime_model_route(UNKNOWN_ROUTE, enabled="false")
    poisoned_state = PublicMatchupRecognizer().state_dict()
    poisoned_state["last_decision"].update(
        route=route_for_archetype("crustle"),
        archetype_id="crustle",
        status="recognized",
    )
    with pytest.raises(ValueError, match="not runtime-authorized"):
        PublicMatchupRecognizer().load_state_dict(poisoned_state)

    agent = PolicyAgent(model=None, deck=[1] * 60)
    agent._matchup_adapter_shadow_router.observe(_obs())
    assert agent.matchup_adapter_shadow_snapshot()["observations"] == 1
    agent.reset_game()
    assert agent.matchup_adapter_shadow_snapshot()["observations"] == 0


def test_mixed_routes_update_only_exact_selected_experts() -> None:
    torch.manual_seed(23)
    bank = MatchupAdapterBank(enabled=True)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=0.02, weight_decay=0.9)
    states = torch.randn(9, HIDDEN_DIM)
    routes = torch.tensor([1, 1, UNKNOWN_ROUTE, 2, 2, UNKNOWN_ROUTE, 4, 4, -1])
    before = {name: value.detach().clone() for name, value in bank.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    bank(states, routes).square().sum().backward()
    for route, expert in enumerate(bank.experts):
        gradients = [parameter.grad for parameter in expert.parameters()]
        if route in {1, 2, 4}:
            assert all(value is not None for value in gradients)
        else:
            assert all(value is None for value in gradients)
    optimizer.step()
    after = bank.state_dict()
    for route in range(len(EXPERT_IDS)):
        changed = any(
            not torch.equal(value, after[name])
            for name, value in before.items()
            if name.startswith(f"experts.{route}.")
        )
        assert changed is (route in {1, 2, 4})


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _boundary(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent = tmp_path / "learner.pt"
    torch.save(
        {
            "model_config": {"matchup_adapters_enabled": False},
            "model_state_dict": {"base.weight": torch.arange(6).reshape(2, 3)},
            "optimizer_state_dict": {"state": {"sentinel": 7}},
            "scaler_state_dict": {"scale": 9},
            "rng_state": {"sentinel": 11},
            "step": 123,
            "epoch": 14,
            "rl_iteration": 15,
        },
        parent,
    )
    learner = {"path": str(parent), "digest": _sha(parent)}
    state = {
        "version": 1,
        "last_completed_iteration": 15,
        "next_iteration": 16,
        "learner": learner,
    }
    run = tmp_path / "run"
    (run / "commits").mkdir(parents=True)
    (run / "commits" / "iter_00015.json").write_text(json.dumps(state) + "\n")
    (run / "loop_state.json").write_text(json.dumps(state) + "\n")
    receipt = tmp_path / "activation.json"
    return run, parent, receipt


def test_activation_receipt_requires_exact_clean_15_to_16_boundary(tmp_path: Path) -> None:
    run, parent, receipt = _boundary(tmp_path)
    proof = build_activation_receipt(run_dir=run, output_path=receipt)
    assert proof.parent_checkpoint == parent.resolve()
    assert json.loads(receipt.read_text())["schema"] == ACTIVATION_RECEIPT_SCHEMA
    with pytest.raises(FileExistsError):
        build_activation_receipt(run_dir=run, output_path=receipt)

    run2, _parent2, receipt2 = _boundary(tmp_path / "stale")
    stale = json.loads((run2 / "loop_state.json").read_text())
    stale.update(last_completed_iteration=16, next_iteration=17)
    (run2 / "loop_state.json").write_text(json.dumps(stale) + "\n")
    with pytest.raises(RuntimeError, match="exact committed"):
        build_activation_receipt(run_dir=run2, output_path=receipt2)

    run3, _parent3, receipt3 = _boundary(tmp_path / "started")
    (run3 / "shards").mkdir()
    (run3 / "shards" / "iter_00016.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="already started"):
        build_activation_receipt(run_dir=run3, output_path=receipt3)


def test_later_rehearsal_authorization_is_exact_and_closes_when_next_iter_starts(
    tmp_path: Path,
) -> None:
    run, parent, _initial_receipt = _boundary(tmp_path)
    state = json.loads((run / "loop_state.json").read_text())
    state.update(last_completed_iteration=25, next_iteration=26)
    serialized = json.dumps(state) + "\n"
    (run / "commits" / "iter_00015.json").unlink()
    (run / "commits" / "iter_00025.json").write_text(serialized)
    (run / "loop_state.json").write_text(serialized)
    receipt = tmp_path / "rehearsal.json"

    proof = build_adapter_rehearsal_authorization(
        run_dir=run,
        completed_iteration=25,
        output_path=receipt,
    )
    assert proof.completed_iteration == 25
    assert proof.first_eligible_iteration == 26
    assert json.loads(receipt.read_text())["schema"] == (
        ADAPTER_REHEARSAL_AUTHORIZATION_SCHEMA
    )
    assert validate_adapter_training_authorization(
        receipt, parent_checkpoint=parent
    ) == proof

    (run / "shards").mkdir()
    (run / "shards" / "iter_00026.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="no longer clean"):
        validate_adapter_training_authorization(
            receipt, parent_checkpoint=parent
        )
    # The isolated adapter optimizer consumes the same immutable issuance
    # proof after this exact iteration has produced artifacts.  That explicit
    # mode must not weaken the default pre-start validator above.
    assert validate_adapter_training_authorization(
        receipt,
        parent_checkpoint=parent,
        permit_post_boundary_use=True,
    ) == proof


def test_specialist_authorization_accepts_checksum_bound_v6_inherited_training(
    tmp_path: Path,
) -> None:
    source_family = tmp_path / "source"
    source_family.mkdir()
    source_checkpoint = source_family / "model.pt"
    source_checkpoint.write_bytes(b"specialist-bootstrap")
    source_checkpoint_digest = (
        "sha256:" + hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    )
    required_coverage = [
        "temporal_action_rows",
        "opponent_hand_rows",
        "opponent_remainder_rows",
        "opponent_private_prize_rows",
        "lethal_threat_rows",
        "prize_race_rows",
    ]
    source_manifest = source_family / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "checkpoint_digest": source_checkpoint_digest,
                "model_path": str(source_checkpoint),
                "provenance": {
                    "acting_seat_archetype": "teal-mask-ogerpon-ex",
                    "epochs_max": 25,
                    "trained_target_coverage": required_coverage,
                },
                "evidence": {"epochs_completed": 25},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_manifest_digest = (
        "sha256:" + hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    )

    derivative_family = tmp_path / "derivative"
    derivative_family.mkdir()
    parent = derivative_family / "model.pt"
    parent.write_bytes(b"router-format-6-derivative")
    parent_digest = (
        "sha256:" + hashlib.sha256(parent.read_bytes()).hexdigest()
    )
    derivative_manifest = derivative_family / "manifest.json"
    derivative_manifest.write_text(
        json.dumps(
            {
                "checkpoint_digest": parent_digest,
                "model_path": str(parent),
                "provenance": {
                    "kind": "matchup_adapter_v6_runtime_derivative",
                    "source_family": str(source_family),
                    "source_family_immutable": True,
                    "source_family_manifest_sha256": source_manifest_digest,
                    "source_checkpoint_digest": source_checkpoint_digest,
                },
                "evidence": {
                    "training_evidence_inherited_from_source": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    derivative_manifest_digest = (
        "sha256:"
        + hashlib.sha256(derivative_manifest.read_bytes()).hexdigest()
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.matchup_adapter_specialist_bootstrap_"
                    "authorization/v1"
                ),
                "specialist_id": "teal-mask-ogerpon-ex",
                "runtime_enabled": False,
                "parent_untouched": True,
                "optimizer_scope": "matchup_adapter_bank_only",
                "first_eligible_iteration": 0,
                "completed_iteration": -1,
                "parent_checkpoint": str(parent),
                "parent_checkpoint_digest": parent_digest,
                "protected_manifest": str(derivative_manifest),
                "protected_manifest_digest": derivative_manifest_digest,
                "required_target_coverage": required_coverage,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    proof = validate_adapter_training_authorization(
        authorization,
        parent_checkpoint=parent,
        permit_post_boundary_use=True,
    )
    assert proof.parent_checkpoint == parent.resolve()
    assert proof.commit_path == derivative_manifest.resolve()

    source_manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inherited training evidence"):
        validate_adapter_training_authorization(
            authorization,
            parent_checkpoint=parent,
            permit_post_boundary_use=True,
        )


def test_activation_receipt_rejects_partial_ledger_drift_and_early_start_markers(
    tmp_path: Path,
) -> None:
    run, _parent, receipt = _boundary(tmp_path / "ledger-drift")
    loop = json.loads((run / "loop_state.json").read_text())
    loop["opponent_pool"] = [{"digest": "stale-but-same-learner"}]
    (run / "loop_state.json").write_text(json.dumps(loop) + "\n")
    with pytest.raises(RuntimeError, match="exact committed"):
        build_activation_receipt(run_dir=run, output_path=receipt)

    runtime_run, _runtime_parent, runtime_receipt = _boundary(
        tmp_path / "runtime-started"
    )
    (runtime_run / "iteration_runtime.json").write_text(
        json.dumps({"iteration": 16, "phase": "collect"}) + "\n"
    )
    with pytest.raises(RuntimeError, match="already started"):
        build_activation_receipt(
            run_dir=runtime_run,
            output_path=runtime_receipt,
        )

    temp_run, _temp_parent, temp_receipt = _boundary(tmp_path / "temp-started")
    (temp_run / "checkpoints").mkdir()
    (temp_run / "checkpoints" / "iter_00016.pt.tmp.7").touch()
    with pytest.raises(RuntimeError, match="already started"):
        build_activation_receipt(run_dir=temp_run, output_path=temp_receipt)


def test_activation_receipt_rejects_ambiguous_parent_runtime_flag(
    tmp_path: Path,
) -> None:
    run, parent, receipt = _boundary(tmp_path)
    payload = checkpoint.load_checkpoint(parent)
    payload["extra"] = {"matchup_adapters_runtime_enabled": True}
    torch.save(payload, parent)
    state = json.loads((run / "commits" / "iter_00015.json").read_text())
    state["learner"]["digest"] = _sha(parent)
    serialized = json.dumps(state) + "\n"
    (run / "commits" / "iter_00015.json").write_text(serialized)
    (run / "loop_state.json").write_text(serialized)
    with pytest.raises(RuntimeError, match="runtime-active"):
        build_activation_receipt(run_dir=run, output_path=receipt)


def test_zero_bank_materializer_is_immutable_and_preserves_training_state(
    tmp_path: Path,
) -> None:
    run, parent, receipt = _boundary(tmp_path)
    parent_payload = checkpoint.load_checkpoint(parent)
    parent_payload["model_config"].pop("matchup_adapters_enabled")
    parent_payload["scheduler_state_dict"] = {"last_epoch": 15}
    parent_payload["python_rng_state"] = (3, (1, 2, 3), None)
    torch.save(parent_payload, parent)
    state = json.loads((run / "commits" / "iter_00015.json").read_text())
    state["learner"]["digest"] = _sha(parent)
    serialized = json.dumps(state) + "\n"
    (run / "commits" / "iter_00015.json").write_text(serialized)
    (run / "loop_state.json").write_text(serialized)
    parent_bytes = parent.read_bytes()
    build_activation_receipt(run_dir=run, output_path=receipt)

    output = tmp_path / "iter15-zero-dormant.pt"
    materialize_zero_dormant_adapter_checkpoint(
        parent_checkpoint=parent,
        activation_receipt=receipt,
        output_path=output,
    )

    assert parent.read_bytes() == parent_bytes
    saved = checkpoint.load_checkpoint(output)
    assert saved["optimizer_state_dict"] == parent_payload["optimizer_state_dict"]
    assert saved["scaler_state_dict"] == parent_payload["scaler_state_dict"]
    assert saved["scheduler_state_dict"] == parent_payload["scheduler_state_dict"]
    assert saved["rng_state"] == parent_payload["rng_state"]
    assert saved["python_rng_state"] == parent_payload["python_rng_state"]
    assert saved["step"] == parent_payload["step"]
    assert saved["epoch"] == parent_payload["epoch"]
    assert saved["rl_iteration"] == parent_payload["rl_iteration"]
    assert saved["model_config"]["matchup_adapters_enabled"] is False
    dormant = saved["extra"]["dormant_matchup_adapter_bank"]
    assert dormant["schema"] == ZERO_DORMANT_CHECKPOINT_SCHEMA
    assert dormant["runtime_enabled"] is False
    assert dormant["training_enabled"] is False
    assert dormant["optimizer_imported"] is False
    assert dormant["frozen"] is True
    assert dormant["zero_output"] is True
    assert torch.equal(
        saved["model_state_dict"]["base.weight"],
        parent_payload["model_state_dict"]["base.weight"],
    )
    adapter = {
        name: value
        for name, value in saved["model_state_dict"].items()
        if name.startswith("matchup_adapter_bank.")
    }
    assert adapter
    assert all(
        int(value.count_nonzero().item()) == 0
        for name, value in adapter.items()
        if name.endswith(".up.weight") or name.endswith(".up.bias")
    )
    with pytest.raises(FileExistsError):
        materialize_zero_dormant_adapter_checkpoint(
            parent_checkpoint=parent,
            activation_receipt=receipt,
            output_path=output,
        )


def test_merge_preserves_parent_training_state_and_discards_adapter_optimizer(
    tmp_path: Path,
) -> None:
    run, parent, receipt = _boundary(tmp_path)
    build_activation_receipt(run_dir=run, output_path=receipt)
    parent_payload = checkpoint.load_checkpoint(parent)
    bank = MatchupAdapterBank()
    with torch.no_grad():
        for route, expert in enumerate(bank.experts):
            expert.up.bias.fill_(0.01 * (route + 1))
    adapter_state = {
        **parent_payload["model_state_dict"],
        **{
            f"matchup_adapter_bank.{name}": value
            for name, value in bank.state_dict().items()
        },
    }
    adapter_path = tmp_path / "adapter-only.pt"
    corpus_digest = "sha256:" + "b" * 64
    gate_contract_digest = "sha256:" + "c" * 64
    per_route = {
        archetype_id: {
            "train_sequences": 2,
            "train_decisions": 10,
            "val_sequences": 1,
            "val_decisions": 5,
        }
        for archetype_id in EXPERT_IDS
    }
    split_contract = {
        "schema": "poke_bot.matchup_adapter_training_split/v1",
        "routing": "offline-oracle-package-and-full-deck-audited",
        "runtime_router_separate": True,
        "corpus_manifest_digest": corpus_digest,
        "active_gate_contract_digest": gate_contract_digest,
        "membership_digest": "sha256:" + "d" * 64,
        "per_route": per_route,
    }
    training_contract = {
        "schema": "poke_bot.matchup_adapter_training_contract/v1",
        "routing": "offline-oracle-package-and-full-deck-audited",
        "runtime_router_separate": True,
        "runtime_enabled": False,
        "optimizer_scope": "matchup_adapter_bank_only",
        "loss_scope": ["policy", "value"],
        "expert_ids": list(EXPERT_IDS),
        "adapter_config": bank.config_dict(),
        "corpus_manifest_digest": corpus_digest,
        "active_gate_contract_digest": gate_contract_digest,
        "split": split_contract,
        "inputs": {
            "schema": "poke_bot.matchup_adapter_input_provenance/v1",
            "source_jsonl_digest": "sha256:" + "1" * 64,
            "corpus_manifest_file_digest": "sha256:" + "2" * 64,
            "active_gate_contract_file_digest": "sha256:" + "3" * 64,
            "implementation_digest": "sha256:" + "4" * 64,
        },
    }
    validation = {
        archetype_id: {
            "route": route,
            "n_games": 1,
            "n_decisions": 5,
            "total_loss": 1.0,
            "policy_loss": 0.5,
            "value_loss": 0.5,
            "policy_acc": 0.25,
        }
        for route, archetype_id in enumerate(EXPERT_IDS)
    }
    torch.save(
        {
            **parent_payload,
            "model_state_dict": adapter_state,
            "optimizer_state_dict": {"adapter_only": True},
            "extra": {
                "matchup_adapter_config": bank.config_dict(),
                "matchup_adapters_runtime_enabled": False,
                "matchup_adapter_parent_checkpoint": str(parent.resolve()),
                "matchup_adapter_parent_checkpoint_digest": _sha(parent),
                "matchup_adapter_activation_receipt": str(receipt.resolve()),
                "matchup_adapter_activation_receipt_digest": _sha(receipt),
                "matchup_adapter_training_contract": training_contract,
                "matchup_adapter_per_route_validation": validation,
            },
        },
        adapter_path,
    )
    output = tmp_path / "merged.pt"
    merge_dormant_adapter_checkpoint(
        parent_checkpoint=parent,
        adapter_checkpoint=adapter_path,
        activation_receipt=receipt,
        output_path=output,
    )
    merged = checkpoint.load_checkpoint(output)
    for key in (
        "optimizer_state_dict",
        "scaler_state_dict",
        "rng_state",
        "step",
        "epoch",
        "rl_iteration",
    ):
        assert merged[key] == parent_payload[key]
    assert merged["extra"]["dormant_matchup_adapter_bank"]["optimizer_imported"] is False
    assert merged["model_config"]["matchup_adapters_enabled"] is False
    assert merged["extra"]["matchup_adapter_training_contract"] == training_contract

    zero_allowed = copy.deepcopy(checkpoint.load_checkpoint(adapter_path))
    zero_id = EXPERT_IDS[6]
    zero_allowed["extra"]["matchup_adapter_training_contract"][
        "zero_example_routes_remain_dormant"
    ] = True
    zero_allowed["extra"]["matchup_adapter_training_contract"]["split"][
        "per_route"
    ][zero_id] = {
        "train_sequences": 0,
        "train_decisions": 0,
        "val_sequences": 0,
        "val_decisions": 0,
    }
    zero_allowed["extra"]["matchup_adapter_per_route_validation"][zero_id] = {
        "route": 6,
        "status": "dormant_no_validation_examples",
        "n_games": 0,
        "n_decisions": 0,
    }
    for suffix in ("weight", "bias"):
        zero_allowed["model_state_dict"][
            f"matchup_adapter_bank.experts.6.up.{suffix}"
        ].zero_()
    zero_allowed_path = tmp_path / "adapter-with-zero-example-route.pt"
    torch.save(zero_allowed, zero_allowed_path)
    zero_merged_path = tmp_path / "merged-with-zero-example-route.pt"
    merge_dormant_adapter_checkpoint(
        parent_checkpoint=parent,
        adapter_checkpoint=zero_allowed_path,
        activation_receipt=receipt,
        output_path=zero_merged_path,
    )
    zero_fit = checkpoint.load_checkpoint(zero_merged_path)["extra"][
        "dormant_matchup_adapter_fit"
    ]
    assert zero_fit["route_decisions"][zero_id] == 0
    assert zero_id in zero_fit["dormant_no_example_archetype_ids"]
    assert zero_id not in zero_fit["trained_archetype_ids"]

    partial = copy.deepcopy(checkpoint.load_checkpoint(adapter_path))
    partial["model_state_dict"].pop("matchup_adapter_bank.experts.6.up.bias")
    partial_path = tmp_path / "partial.pt"
    torch.save(partial, partial_path)
    with pytest.raises(RuntimeError, match="incomplete"):
        merge_dormant_adapter_checkpoint(
            parent_checkpoint=parent,
            adapter_checkpoint=partial_path,
            activation_receipt=receipt,
            output_path=tmp_path / "partial-merged.pt",
        )

    untrained = copy.deepcopy(checkpoint.load_checkpoint(adapter_path))
    for suffix in ("weight", "bias"):
        untrained["model_state_dict"][
            f"matchup_adapter_bank.experts.6.up.{suffix}"
        ].zero_()
    untrained_path = tmp_path / "untrained.pt"
    torch.save(untrained, untrained_path)
    with pytest.raises(RuntimeError, match=EXPERT_IDS[6]):
        merge_dormant_adapter_checkpoint(
            parent_checkpoint=parent,
            adapter_checkpoint=untrained_path,
            activation_receipt=receipt,
            output_path=tmp_path / "untrained-merged.pt",
        )

    wrong_validation = copy.deepcopy(checkpoint.load_checkpoint(adapter_path))
    wrong_validation["extra"]["matchup_adapter_per_route_validation"][
        EXPERT_IDS[0]
    ]["route"] = 1
    wrong_validation_path = tmp_path / "wrong-validation.pt"
    torch.save(wrong_validation, wrong_validation_path)
    with pytest.raises(RuntimeError, match="invalid per-route validation"):
        merge_dormant_adapter_checkpoint(
            parent_checkpoint=parent,
            adapter_checkpoint=wrong_validation_path,
            activation_receipt=receipt,
            output_path=tmp_path / "wrong-validation-merged.pt",
        )

    weakened_contract = copy.deepcopy(checkpoint.load_checkpoint(adapter_path))
    weakened_contract["extra"]["matchup_adapter_training_contract"][
        "optimizer_scope"
    ] = "all_model_parameters"
    weakened_contract_path = tmp_path / "weakened-contract.pt"
    torch.save(weakened_contract, weakened_contract_path)
    with pytest.raises(RuntimeError, match="training contract"):
        merge_dormant_adapter_checkpoint(
            parent_checkpoint=parent,
            adapter_checkpoint=weakened_contract_path,
            activation_receipt=receipt,
            output_path=tmp_path / "weakened-contract-merged.pt",
        )

    wrong_parent_path = copy.deepcopy(checkpoint.load_checkpoint(adapter_path))
    wrong_parent_path["extra"]["matchup_adapter_parent_checkpoint"] = str(
        tmp_path / "same-digest-different-parent.pt"
    )
    wrong_parent_path_file = tmp_path / "wrong-parent-path.pt"
    torch.save(wrong_parent_path, wrong_parent_path_file)
    with pytest.raises(RuntimeError, match="dormant child"):
        merge_dormant_adapter_checkpoint(
            parent_checkpoint=parent,
            adapter_checkpoint=wrong_parent_path_file,
            activation_receipt=receipt,
            output_path=tmp_path / "wrong-parent-path-merged.pt",
        )

    # Later exact-boundary rehearsals use the dedicated authorization schema,
    # not the one-time iteration-15 activation receipt.  Their trained child
    # must remain mergeable without weakening any parent/optimizer checks.
    later_state = json.loads((run / "loop_state.json").read_text())
    later_state.update(last_completed_iteration=26, next_iteration=27)
    serialized = json.dumps(later_state) + "\n"
    (run / "commits" / "iter_00015.json").unlink()
    (run / "commits" / "iter_00026.json").write_text(serialized)
    (run / "loop_state.json").write_text(serialized)
    rehearsal_receipt = tmp_path / "iter26-rehearsal-authorization.json"
    build_adapter_rehearsal_authorization(
        run_dir=run,
        completed_iteration=26,
        output_path=rehearsal_receipt,
    )
    rehearsal_child = copy.deepcopy(checkpoint.load_checkpoint(adapter_path))
    rehearsal_child["extra"]["matchup_adapter_activation_receipt"] = str(
        rehearsal_receipt.resolve()
    )
    rehearsal_child["extra"]["matchup_adapter_activation_receipt_digest"] = _sha(
        rehearsal_receipt
    )
    rehearsal_child_path = tmp_path / "iter26-rehearsal-child.pt"
    torch.save(rehearsal_child, rehearsal_child_path)
    rehearsal_merged_path = tmp_path / "iter26-rehearsal-merged.pt"
    merge_dormant_adapter_checkpoint(
        parent_checkpoint=parent,
        adapter_checkpoint=rehearsal_child_path,
        activation_receipt=rehearsal_receipt,
        output_path=rehearsal_merged_path,
    )
    rehearsal_merged = checkpoint.load_checkpoint(rehearsal_merged_path)
    assert rehearsal_merged["extra"]["dormant_matchup_adapter_fit"]["epochs"] == 14
    assert (
        rehearsal_merged["extra"]["dormant_matchup_adapter_bank"][
            "activation_receipt"
        ]
        == str(rehearsal_receipt.resolve())
    )


def _gate_contract(gate_id: str, gate_digest: str, alias: str = "gate-alias") -> dict:
    return {
        "active_gate_id": "gate-v1",
        "next_gate": {
            "id": "gate-v1",
            "evaluation": {
                "formal_eval_disjoint_from_training": True,
                "package_digest_deduplicated": True,
            },
            "roster": [
                {"opponent_id": gate_id, "content_digest": gate_digest}
            ],
            "excluded_aliases": [
                {
                    "opponent_id": alias,
                    "canonical_opponent_id": gate_id,
                }
            ],
        },
    }


def _manifest(packages: list[dict]) -> dict:
    return {"schema": CORPUS_MANIFEST_SCHEMA, "packages": packages}


@pytest.mark.parametrize("aliases", [[""], ["alias", " alias "]])
def test_corpus_manifest_rejects_empty_or_duplicate_normalized_aliases(
    aliases: list[str],
) -> None:
    with pytest.raises(ValueError, match="alias"):
        parse_corpus_manifest(
            _manifest(
                [
                    {
                        "opponent_id": "safe-marnie",
                        "content_digest": "sha256:" + "a" * 64,
                        "archetype_id": "marnie-s-grimmsnarl-ex",
                        "aliases": aliases,
                    }
                ]
            )
        )


def _fake_sequence(record: dict, max_context: int) -> GameSequence:
    decisions = [
        DecisionSample(
            board=None,  # conversion behavior is outside this contract test
            options=None,
            action=[0],
            action_combo_index=0,
            action_combos=[[0]],
            env_step=index,
        )
        for index, _step in enumerate(record["steps"][-max_context:])
    ]
    return GameSequence(
        episode_id=str(record["episode_id"]),
        seat=int(record["seat"]),
        archetype=str(record["archetype"]),
        opp_archetype=str(record["opp_archetype"]),
        deck=[1] * 60,
        value=1.0,
        decisions=decisions,
    )


def test_corpus_partition_gate_exclusion_alias_collision_and_oracle_tickets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import poke_bot.dataset as dataset_mod
    import poke_bot.matchup_adapter_activation as activation_mod

    monkeypatch.setattr(
        dataset_mod,
        "convert_record",
        lambda record, max_context, verify_info_set: (
            _fake_sequence(record, max_context),
            None,
            {},
        ),
    )
    monkeypatch.setattr(
        activation_mod,
        "visible_opponent_card_ids",
        lambda obs: frozenset(obs.get("visible", [])),
    )
    gate_digest = "sha256:" + "f" * 64
    safe_digest = "sha256:" + "1" * 64
    manifest = _manifest(
        [
            {
                "opponent_id": "safe-marnie",
                "content_digest": safe_digest,
                "archetype_id": "marnie-s-grimmsnarl-ex",
                "aliases": ["safe-marnie-alias"],
            },
            {
                "opponent_id": "gate-agent",
                "content_digest": gate_digest,
                "archetype_id": "marnie-s-grimmsnarl-ex",
            },
        ]
    )
    gate = _gate_contract("gate-agent", gate_digest)

    def record(episode: str, opponent_id: str, digest: str) -> dict:
        return {
            "episode_id": episode,
            "seat": 0,
            "archetype": "alakazam",
            "opp_archetype": opponent_id,
            "steps": [
                {"observation": {"visible": [646]}},
                {"observation": {"visible": [646]}},
            ],
            "target_provenance": {
                "opponent_id": opponent_id,
                "opponent_content_digest": digest,
            },
        }

    prepared = prepare_adapter_corpus_records(
        [
            record("safe", "safe-marnie", safe_digest),
            record("heldout", "gate-agent", gate_digest),
        ],
        corpus_manifest=manifest,
        gate_contract=gate,
        max_context=320,
    )
    assert prepared.excluded_gate_records == 1
    assert len(prepared.sequences) == 1
    sequence = prepared.sequences[0]
    expected = route_for_archetype("marnie-s-grimmsnarl-ex")
    assert sequence.opp_archetype == "marnie-s-grimmsnarl-ex"
    assert [training_route_for_decision(sequence, row) for row in sequence.decisions] == [expected, expected]
    assert sequence.decisions[0].matchup_adapter_public_route == UNKNOWN_ROUTE
    assert sequence.decisions[1].matchup_adapter_public_route == expected
    restored = pickle.loads(pickle.dumps(sequence))
    assert training_route_for_decision(restored, restored.decisions[0]) == expected
    assert restored.matchup_adapter_training_ticket == (
        sequence.matchup_adapter_training_ticket
    )
    restored.decisions[0].matchup_adapter_public_route = route_for_archetype(
        "garchomp"
    )
    with pytest.raises(RuntimeError, match="public-prefix audit route"):
        training_route_for_decision(restored, restored.decisions[0])

    collision = copy.deepcopy(manifest)
    collision["packages"].append(
        {
            "opponent_id": "other",
            "content_digest": "sha256:" + "2" * 64,
            "archetype_id": "garchomp",
            "aliases": ["safe-marnie-alias"],
        }
    )
    with pytest.raises(ValueError, match="alias collision"):
        parse_corpus_manifest(collision)
    exclusions = gate_exclusions(gate)
    assert "gate-alias" in exclusions.opponent_ids


def _ticketed_split_sequence(archetype_id: str, episode_id: str) -> GameSequence:
    route = route_for_archetype(archetype_id)
    sequence = GameSequence(
        episode_id=episode_id,
        seat=0,
        archetype="alakazam",
        opp_archetype=archetype_id,
        deck=[1] * 60,
        value=1.0,
        decisions=[
            DecisionSample(
                board=None,
                options=None,
                action=[0],
                action_combo_index=0,
                action_combos=[[0]],
                env_step=0,
                matchup_adapter_oracle_route=route,
            )
        ],
        matchup_adapter_training_ticket={
            "schema": "poke_bot.matchup_adapter_training_ticket/v1",
            "opponent_id": f"package-{archetype_id}",
            "package_digest": "sha256:"
            + hashlib.sha256(archetype_id.encode()).hexdigest(),
            "archetype_id": archetype_id,
            "route": route,
            "corpus_manifest_digest": "sha256:" + "b" * 64,
            "gate_contract_digest": "sha256:" + "c" * 64,
            "episode_id": episode_id,
            "seat": 0,
        },
    )
    return sequence


def test_training_route_rejects_coercible_or_tampered_route_and_seat_ids() -> None:
    archetype_id = EXPERT_IDS[1]

    ticket_bool_route = _ticketed_split_sequence(archetype_id, "ticket-bool-route")
    ticket_bool_route.matchup_adapter_training_ticket["route"] = True
    with pytest.raises(RuntimeError, match="malformed adapter training ticket"):
        training_route_for_decision(
            ticket_bool_route,
            ticket_bool_route.decisions[0],
        )

    oracle_bool_route = _ticketed_split_sequence(archetype_id, "oracle-bool-route")
    oracle_bool_route.decisions[0].matchup_adapter_oracle_route = True
    with pytest.raises(RuntimeError, match="oracle training route.*exact integer"):
        training_route_for_decision(
            oracle_bool_route,
            oracle_bool_route.decisions[0],
        )

    public_bool_route = _ticketed_split_sequence(archetype_id, "public-bool-route")
    public_bool_route.decisions[0].matchup_adapter_public_route = True
    with pytest.raises(RuntimeError, match="public-prefix audit route.*exact integer"):
        training_route_for_decision(
            public_bool_route,
            public_bool_route.decisions[0],
        )

    bool_sequence_seat = _ticketed_split_sequence(archetype_id, "bool-seat")
    bool_sequence_seat.seat = True
    bool_sequence_seat.matchup_adapter_training_ticket["seat"] = 1
    with pytest.raises(RuntimeError, match="no longer matches sequence"):
        training_route_for_decision(
            bool_sequence_seat,
            bool_sequence_seat.decisions[0],
        )

    out_of_range_seat = _ticketed_split_sequence(archetype_id, "seat-two")
    out_of_range_seat.seat = 2
    out_of_range_seat.matchup_adapter_training_ticket["seat"] = 2
    with pytest.raises(RuntimeError, match="invalid adapter training ticket identity"):
        training_route_for_decision(
            out_of_range_seat,
            out_of_range_seat.decisions[0],
        )


def test_route_stratified_split_contract_is_complete_disjoint_and_stable() -> None:
    sequences = [
        _ticketed_split_sequence(archetype_id, f"{route}-{episode}")
        for route, archetype_id in enumerate(EXPERT_IDS)
        for episode in range(4)
    ]
    train_rows, val_rows = split_matchup_adapter_sequences(
        sequences,
        val_frac=0.25,
        seed=77,
    )
    train_ids = {row.episode_id for row in train_rows}
    val_ids = {row.episode_id for row in val_rows}
    assert train_ids.isdisjoint(val_ids)
    split = matchup_adapter_split_contract(train_rows, val_rows)
    assert all(
        all(int(count) > 0 for count in route_row.values())
        for route_row in split["per_route"].values()
    )
    assert split == matchup_adapter_split_contract(train_rows, val_rows)

    inputs = {
        "schema": "poke_bot.matchup_adapter_input_provenance/v1",
        "source_jsonl_digest": "sha256:" + "1" * 64,
        "corpus_manifest_file_digest": "sha256:" + "2" * 64,
        "active_gate_contract_file_digest": "sha256:" + "3" * 64,
        "implementation_digest": "sha256:" + "4" * 64,
    }
    contract = build_matchup_adapter_training_contract(
        train_rows,
        val_rows,
        input_provenance=inputs,
    )
    assert contract["runtime_enabled"] is False
    assert contract["expert_ids"] == list(EXPERT_IDS)
    assert contract["split"] == split

    changed = copy.deepcopy(train_rows)
    changed[0].episode_id += "-tampered"
    changed[0].matchup_adapter_training_ticket["episode_id"] = changed[0].episode_id
    assert (
        matchup_adapter_split_contract(changed, val_rows)["membership_digest"]
        != split["membership_digest"]
    )


def test_route_stratified_split_rejects_cross_matchup_episode_collision() -> None:
    rows = [
        _ticketed_split_sequence(EXPERT_IDS[0], "collision"),
        _ticketed_split_sequence(EXPERT_IDS[1], "collision"),
    ]
    with pytest.raises(RuntimeError, match="multiple adapter routes"):
        split_matchup_adapter_sequences(rows, val_frac=0.5, seed=0)
