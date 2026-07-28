import hashlib
import json

import torch
import pytest

from poke_bot.batched_infer import (
    LeafPacket,
    apply_runtime_matchup_adapter_contract,
)
from poke_bot.mcts import LeafEvaluator
from scripts import run_remote_worker


def test_leaf_evaluator_uses_remote_backend_without_local_model() -> None:
    calls: list[list[LeafPacket]] = []

    def remote(packets):
        calls.append(list(packets))
        return [
            LeafPacket(
                obs=packet.obs,
                your_deck=packet.your_deck,
                root_seat=packet.root_seat,
                value=0.25,
                priors=[0.75, 0.25],
                combos=[[0], [1]],
            )
            for packet in packets
        ]

    evaluator = LeafEvaluator(
        None,
        [10, 20],
        [30, 40],
        root_seat=1,
        device=torch.device("cpu"),
        leaf_backend=remote,
    )

    value, priors, combos = evaluator.evaluate_one(object())

    assert len(calls) == 1
    assert value == 0.25
    assert priors == [0.75, 0.25]
    assert combos == [[0], [1]]


def test_every_fresh_leaf_model_reapplies_matchup_runtime(monkeypatch) -> None:
    loaded = []

    class Bank:
        enabled = False

        def requires_grad_(self, value):
            assert value is False
            return self

    class Model:
        matchup_adapter_bank = Bank()

    monkeypatch.setenv("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "1")
    monkeypatch.setenv("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", "/runtime/tree.json")
    monkeypatch.setattr(
        "poke_bot.public_matchup_router.load_runtime_public_matchup_tree",
        lambda path: loaded.append(path),
    )
    first = Model()
    second = Model()
    assert apply_runtime_matchup_adapter_contract(first) is True
    assert apply_runtime_matchup_adapter_contract(second) is True
    assert first.matchup_adapter_bank.enabled is True
    assert second.matchup_adapter_bank.enabled is True
    assert loaded == ["/runtime/tree.json", "/runtime/tree.json"]

    monkeypatch.delenv("POKEBOT_PUBLIC_MATCHUP_TREE_PATH")
    with pytest.raises(ValueError, match="require POKEBOT_PUBLIC_MATCHUP_TREE_PATH"):
        apply_runtime_matchup_adapter_contract(Model())


def test_remote_runtime_health_proves_the_exact_activation_contract(
    tmp_path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    tree_path = tmp_path / "public-matchup-tree-runtime-v31.json"
    tree_path.write_bytes(b"tree")
    tree_digest = "sha256:" + hashlib.sha256(tree_path.read_bytes()).hexdigest()
    marker_path = tmp_path / run_remote_worker.MATCHUP_RUNTIME_MARKER
    marker_path.write_text(
        json.dumps(
            {
                "schema": run_remote_worker.MATCHUP_RUNTIME_MARKER_SCHEMA,
                "runtime_enabled": True,
                "tree_file": tree_path.name,
                "tree_digest": tree_digest,
                "accepted_archetype_ids": ["crustle"],
                "continuous_reevaluation": True,
                "one_route_per_decision": True,
            }
        ),
        encoding="utf-8",
    )

    class Tree:
        runtime_accepted_archetype_ids = frozenset({"crustle"})
        runtime_consecutive_required = 2
        digest = tree_digest

    monkeypatch.setattr(
        "poke_bot.public_matchup_router.PublicMatchupDecisionTree.from_path",
        lambda *_args, **_kwargs: Tree(),
    )
    monkeypatch.setattr(
        "poke_bot.checkpoint.load_checkpoint",
        lambda *_args, **_kwargs: {
            "extra": {
                "dormant_matchup_adapter_fit": {
                    "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
                    "route_decisions": {"crustle": 123},
                }
            }
        },
    )
    checkpoint_digest = "sha256:" + "c" * 64
    monkeypatch.setattr(
        "poke_bot.checkpoint.checkpoint_digest",
        lambda *_args, **_kwargs: checkpoint_digest,
    )
    # _activate_matchup_runtime_from_marker intentionally exports these for
    # subsequently spawned workers. Register their pre-call values with the
    # pytest monkeypatch ledger so this test cannot leak runtime activation
    # into later collection/recovery tests in the same interpreter.
    monkeypatch.setenv("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "test-pending")
    monkeypatch.setenv("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", "test-pending")
    monkeypatch.setenv("POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE", "test-pending")

    runtime = run_remote_worker._activate_matchup_runtime_from_marker(
        checkpoint_path
    )

    assert runtime == {
        "marker": str(marker_path),
        "marker_digest": (
            "sha256:" + hashlib.sha256(marker_path.read_bytes()).hexdigest()
        ),
        "tree": str(tree_path),
        "tree_digest": tree_digest,
        "checkpoint_digest": checkpoint_digest,
        "accepted_archetype_ids": ["crustle"],
        "continuous_reevaluation": True,
        "one_route_per_decision": True,
        "unknown_route_exact_bypass": True,
        "consecutive_required": 2,
        "zero_materialized_adapters_allowed": False,
    }


def test_remote_runtime_accepts_trained_routes_plus_proven_zero_dormant_routes(
    tmp_path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    tree_path = tmp_path / "tree.json"
    tree_path.write_bytes(b"tree")
    tree_digest = "sha256:" + hashlib.sha256(tree_path.read_bytes()).hexdigest()
    marker_path = tmp_path / run_remote_worker.MATCHUP_RUNTIME_MARKER
    marker_path.write_text(
        json.dumps(
            {
                "schema": run_remote_worker.MATCHUP_RUNTIME_MARKER_SCHEMA,
                "runtime_enabled": True,
                "tree_file": tree_path.name,
                "tree_digest": tree_digest,
                "accepted_archetype_ids": ["crustle", "walrein"],
                "continuous_reevaluation": True,
                "one_route_per_decision": True,
                "zero_materialized_adapters_allowed": True,
            }
        ),
        encoding="utf-8",
    )

    class Tree:
        runtime_accepted_archetype_ids = frozenset({"crustle", "walrein"})
        runtime_consecutive_required = 2
        digest = tree_digest

    monkeypatch.setattr(
        "poke_bot.public_matchup_router.PublicMatchupDecisionTree.from_path",
        lambda *_args, **_kwargs: Tree(),
    )
    monkeypatch.setattr(
        "poke_bot.checkpoint.load_checkpoint",
        lambda *_args, **_kwargs: {
            "model_state_dict": {
                "matchup_adapter_bank.experts.1.up.weight": torch.zeros(3, 2),
                "matchup_adapter_bank.experts.1.up.bias": torch.zeros(3),
            },
            "extra": {
                "matchup_adapter_config": {
                    "expert_ids": ["crustle", "walrein"],
                },
                "dormant_matchup_adapter_bank": {
                    "schema": "poke_bot.trained_dormant_matchup_adapter/v1",
                    "zero_output": False,
                },
                "dormant_matchup_adapter_fit": {
                    "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
                    "route_decisions": {"crustle": 123, "walrein": 0},
                    "dormant_no_example_archetype_ids": ["walrein"],
                    "zero_example_routes_remain_dormant": True,
                },
            },
        },
    )
    monkeypatch.setattr(
        "poke_bot.checkpoint.checkpoint_digest",
        lambda *_args, **_kwargs: "sha256:" + "c" * 64,
    )

    runtime = run_remote_worker._activate_matchup_runtime_from_marker(
        checkpoint_path, apply_environment=False
    )

    assert runtime["zero_materialized_adapters_allowed"] is True
    assert runtime["accepted_archetype_ids"] == ["crustle", "walrein"]


def test_remote_runtime_rejects_nonzero_untrained_dormant_route(
    tmp_path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    tree_path = tmp_path / "tree.json"
    tree_path.write_bytes(b"tree")
    tree_digest = "sha256:" + hashlib.sha256(tree_path.read_bytes()).hexdigest()
    (tmp_path / run_remote_worker.MATCHUP_RUNTIME_MARKER).write_text(
        json.dumps(
            {
                "schema": run_remote_worker.MATCHUP_RUNTIME_MARKER_SCHEMA,
                "runtime_enabled": True,
                "tree_file": tree_path.name,
                "tree_digest": tree_digest,
                "accepted_archetype_ids": ["walrein"],
                "continuous_reevaluation": True,
                "one_route_per_decision": True,
                "zero_materialized_adapters_allowed": True,
            }
        ),
        encoding="utf-8",
    )

    class Tree:
        runtime_accepted_archetype_ids = frozenset({"walrein"})
        runtime_consecutive_required = 2
        digest = tree_digest

    monkeypatch.setattr(
        "poke_bot.public_matchup_router.PublicMatchupDecisionTree.from_path",
        lambda *_args, **_kwargs: Tree(),
    )
    monkeypatch.setattr(
        "poke_bot.checkpoint.load_checkpoint",
        lambda *_args, **_kwargs: {
            "model_state_dict": {
                "matchup_adapter_bank.experts.0.up.weight": torch.ones(3, 2),
                "matchup_adapter_bank.experts.0.up.bias": torch.zeros(3),
            },
            "extra": {
                "matchup_adapter_config": {"expert_ids": ["walrein"]},
                "dormant_matchup_adapter_fit": {
                    "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
                    "route_decisions": {"walrein": 0},
                    "dormant_no_example_archetype_ids": ["walrein"],
                    "zero_example_routes_remain_dormant": True,
                },
            },
        },
    )

    with pytest.raises(
        ValueError, match="does not contain every accepted adapter"
    ):
        run_remote_worker._activate_matchup_runtime_from_marker(
            checkpoint_path, apply_environment=False
        )


def test_reload_runtime_revalidates_the_new_checkpoint(
    tmp_path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "candidate.pt"
    checkpoint_path.write_bytes(b"candidate")
    expected = {
        "checkpoint_digest": "sha256:" + "d" * 64,
        "accepted_archetype_ids": ["alakazam"],
    }
    calls = []

    def activate(path, *, apply_environment=True):
        calls.append((path, apply_environment))
        return expected

    monkeypatch.setattr(
        run_remote_worker,
        "_activate_matchup_runtime_from_marker",
        activate,
    )
    assert (
        run_remote_worker._reload_matchup_runtime_contract(
            checkpoint_path,
            {"checkpoint_digest": "sha256:" + "c" * 64},
        )
        is expected
    )
    assert calls == [(checkpoint_path, False)]
    calls.clear()
    assert (
        run_remote_worker._reload_matchup_runtime_contract(
            checkpoint_path,
            None,
        )
        is None
    )
    assert calls == []


def test_reload_runtime_rejects_missing_activation_marker(
    tmp_path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "candidate.pt"
    checkpoint_path.write_bytes(b"candidate")
    monkeypatch.setattr(
        run_remote_worker,
        "_activate_matchup_runtime_from_marker",
        lambda _path, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="lost its explicit activation marker"):
        run_remote_worker._reload_matchup_runtime_contract(
            checkpoint_path,
            {"checkpoint_digest": "sha256:" + "c" * 64},
        )


def test_runtime_identity_change_includes_tree_and_route_roster() -> None:
    base = {
        "tree_digest": "sha256:" + "a" * 64,
        "accepted_archetype_ids": ["hops-trevenant", "alakazam"],
    }
    same_reordered = {
        "tree_digest": base["tree_digest"],
        "accepted_archetype_ids": ["alakazam", "hops-trevenant"],
    }
    changed_tree = {**base, "tree_digest": "sha256:" + "b" * 64}
    changed_roster = {
        **base,
        "accepted_archetype_ids": [*base["accepted_archetype_ids"], "walrein"],
    }

    identity = run_remote_worker._matchup_runtime_worker_identity
    assert identity(base) == identity(same_reordered)
    assert identity(base) != identity(changed_tree)
    assert identity(base) != identity(changed_roster)


def test_worker_runtime_probe_reads_the_child_environment(
    tmp_path, monkeypatch
) -> None:
    from poke_bot.remote_sim_jobs import remote_matchup_runtime_probe

    tree = tmp_path / "tree.json"
    tree.write_text(
        json.dumps(
            {
                "runtime_contract": {
                    "accepted_archetype_ids": [
                        "hops-trevenant",
                        "alakazam",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "1")
    monkeypatch.setenv("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", str(tree))

    result = remote_matchup_runtime_probe({})
    probe = result["runtime_probe"]
    assert probe["runtime_enabled"] is True
    assert probe["tree_digest"] == (
        "sha256:" + hashlib.sha256(tree.read_bytes()).hexdigest()
    )
    assert probe["accepted_archetype_ids"] == [
        "alakazam",
        "hops-trevenant",
    ]


def test_controller_contract_rebinds_a_stale_worker_before_the_job(
    tmp_path, monkeypatch
) -> None:
    from poke_bot.remote_sim_jobs import remote_matchup_runtime_probe

    canonical = tmp_path / "canonical-tree.json"
    canonical.write_text(
        json.dumps(
            {
                "runtime_contract": {
                    "accepted_archetype_ids": [
                        "alakazam",
                        "hops-trevenant",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    digest = "sha256:" + hashlib.sha256(canonical.read_bytes()).hexdigest()
    monkeypatch.setenv("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "1")
    monkeypatch.setenv(
        "POKEBOT_PUBLIC_MATCHUP_TREE_PATH",
        str(tmp_path / "stale-tree.json"),
    )

    result = remote_matchup_runtime_probe(
        {
            "_controller_matchup_runtime": {
                "tree": str(canonical),
                "tree_digest": digest,
                "accepted_archetype_ids": [
                    "hops-trevenant",
                    "alakazam",
                ],
            }
        }
    )

    assert result["runtime_probe"]["tree_digest"] == digest
    assert result["runtime_probe"]["accepted_archetype_ids"] == [
        "alakazam",
        "hops-trevenant",
    ]


def test_controller_contract_rejects_tree_byte_drift(
    tmp_path,
) -> None:
    from poke_bot.remote_sim_jobs import remote_matchup_runtime_probe

    tree = tmp_path / "tree.json"
    tree.write_text(
        json.dumps(
            {"runtime_contract": {"accepted_archetype_ids": ["hops-trevenant"]}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tree digest changed"):
        remote_matchup_runtime_probe(
            {
                "_controller_matchup_runtime": {
                    "tree": str(tree),
                    "tree_digest": "sha256:" + "0" * 64,
                    "accepted_archetype_ids": ["hops-trevenant"],
                }
            }
        )
