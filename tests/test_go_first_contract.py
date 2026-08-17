from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from poke_bot.agent import PolicyAgent, forced_go_first_action


def _prompt(*, context=41, yes_type=1, no_type=2, yes_index=0) -> dict:
    options = [{"type": yes_type}, {"type": no_type}]
    if yes_index == 1:
        options.reverse()
    return {
        "logs": [],
        "current": None,
        "select": {
            "context": context,
            "type": 9,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        },
    }


@pytest.mark.parametrize(
    ("context", "yes_type", "no_type"),
    [(41, 1, 2), ("IsFirst", "Yes", "No"), ("IS_FIRST", "YES", "NO")],
)
def test_turn_order_contract_always_selects_yes(context, yes_type, no_type) -> None:
    assert forced_go_first_action(
        _prompt(context=context, yes_type=yes_type, no_type=no_type)
    ) == [0]


def test_turn_order_contract_finds_yes_regardless_of_option_order() -> None:
    assert forced_go_first_action(_prompt(yes_index=1)) == [1]


def test_non_turn_order_prompt_is_not_overridden() -> None:
    assert forced_go_first_action(_prompt(context=43)) is None


def test_ambiguous_turn_order_prompt_fails_closed() -> None:
    prompt = _prompt()
    prompt["select"]["option"] = [{"type": 2}, {"type": 2}]
    with pytest.raises(RuntimeError, match="one legal Yes"):
        forced_go_first_action(prompt)


def test_submission_policy_bypasses_missing_model_and_still_goes_first() -> None:
    policy = PolicyAgent(model=None, deck=[1] * 60)
    assert policy.greedy_select(_prompt()) == [0]


def test_submission_entrypoint_resolves_turn_order_before_runtime_imports() -> None:
    main_path = Path(__file__).resolve().parents[1] / "submission" / "main.py"
    source = main_path.read_text(encoding="utf-8")
    assert not any(
        isinstance(node, ast.Name) and node.id == "__file__"
        for node in ast.walk(ast.parse(source))
    )
    spec = importlib.util.spec_from_file_location("isolated_submission_main", main_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.agent(_prompt()) == [0]
    assert module.agent(_prompt(yes_index=1)) == [1]
    assert module._MODEL is None
    assert module._POLICY is None


def test_submission_entrypoint_can_choose_second_from_packaged_profile(
    tmp_path: Path,
) -> None:
    source = Path(__file__).resolve().parents[1] / "submission" / "main.py"
    main_path = tmp_path / "main.py"
    main_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "deck.csv").write_text("1\n" * 60, encoding="utf-8")
    (tmp_path / "turn_order_profile.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.submission_turn_order_profile/v1",
                "turn_order_preference": "second_if_allowed",
            }
        ),
        encoding="utf-8",
    )
    prior = Path.cwd()
    os.chdir(tmp_path)
    try:
        spec = importlib.util.spec_from_file_location(
            "isolated_submission_second",
            main_path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.agent(_prompt()) == [1]
        assert module.agent(_prompt(yes_index=1)) == [0]
        assert module._MODEL is None
    finally:
        os.chdir(prior)


def test_submission_entrypoint_enables_only_a_shipped_runtime_matchup_tree() -> None:
    main_path = Path(__file__).resolve().parents[1] / "submission" / "main.py"
    source = main_path.read_text(encoding="utf-8")

    assert 'matchup_tree = _agent_dir() / "matchup_tree.json"' in source
    assert 'os.environ["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"' in source
    assert (
        'os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(matchup_tree)'
        in source
    )
    assert "trained matchup adapter checkpoint requires matchup_tree.json" in source
    assert "validate_zero_dormant_checkpoint(checkpoint, allow_trained=True)" in source
    assert "_assert_trained_matchup_tree_binding" in source
    assert "_assert_trained_matchup_runtime" in source
    assert source.index("if matchup_tree.is_file():") < source.index(
        "PolicyAgent(model=model"
    )


def test_submission_entrypoint_requires_tree_only_for_a_trained_adapter_bank() -> None:
    main_path = Path(__file__).resolve().parents[1] / "submission" / "main.py"
    spec = importlib.util.spec_from_file_location(
        "submission_adapter_checkpoint_contract", main_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._checkpoint_has_trained_matchup_adapter_bank(
        {
            "extra": {
                "dormant_matchup_adapter_bank": {
                    "schema": "poke_bot.trained_dormant_matchup_adapter/v1",
                    "zero_output": False,
                }
            }
        }
    )
    assert not module._checkpoint_has_trained_matchup_adapter_bank(
        {
            "extra": {
                "dormant_matchup_adapter_bank": {
                    "schema": "poke_bot.trained_dormant_matchup_adapter/v1",
                    "zero_output": True,
                }
            }
        }
    )
    assert not module._checkpoint_has_trained_matchup_adapter_bank({})


def test_submission_entrypoint_checks_enabled_frozen_exact_trained_runtime(
    tmp_path: Path,
) -> None:
    main_path = Path(__file__).resolve().parents[1] / "submission" / "main.py"
    spec = importlib.util.spec_from_file_location(
        "submission_adapter_runtime_contract", main_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tree_path = tmp_path / "matchup_tree.json"
    tree_path.write_text("exact submitted tree\n", encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(tree_path.read_bytes()).hexdigest()

    class Parameter:
        requires_grad = False

    class Bank:
        enabled = True

        @staticmethod
        def parameters() -> list[Parameter]:
            return [Parameter()]

    class Model:
        matchup_adapter_bank = Bank()

    class Tree:
        runtime_enabled = True

    Tree.digest = digest

    class Router:
        tree = Tree()

    class Policy:
        matchup_adapter_runtime = True
        _matchup_adapter_shadow_router = Router()

    module._assert_trained_matchup_runtime(
        model=Model(), policy=Policy(), matchup_tree=tree_path
    )

    Policy.matchup_adapter_runtime = False
    with pytest.raises(RuntimeError, match="runtime was not enabled"):
        module._assert_trained_matchup_runtime(
            model=Model(), policy=Policy(), matchup_tree=tree_path
        )


def test_submission_entrypoint_rejects_zero_output_accepted_adapter_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_path = Path(__file__).resolve().parents[1] / "submission" / "main.py"
    spec = importlib.util.spec_from_file_location(
        "submission_adapter_tree_binding_contract", main_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from poke_bot import matchup_adapter_routes, public_matchup_router

    class Contract:
        target_ids = ("alakazam",)
        physical_slots = (7,)
        adapter_format = "v6"
        slot_registry_digest = "sha256:" + "a" * 64

        @property
        def physical_slot_by_target(self) -> dict[str, int]:
            return {"alakazam": 7}

    class Tree:
        targets = ("alakazam",)
        route_physical_slots = (7,)
        adapter_format = "v6"
        slot_registry_digest = "sha256:" + "a" * 64
        runtime_accepted_archetype_ids = frozenset({"alakazam"})
        runtime_enabled = True

    contract = Contract()
    tree = Tree()
    monkeypatch.setattr(
        matchup_adapter_routes,
        "resolve_matchup_adapter_route_contract",
        lambda _config: contract,
    )
    monkeypatch.setattr(
        matchup_adapter_routes,
        "require_runtime_route_binding",
        lambda runtime, received_contract: (
            None
            if runtime.get("bound") is True and received_contract is contract
            else (_ for _ in ()).throw(ValueError("runtime binding changed"))
        ),
    )
    monkeypatch.setattr(
        public_matchup_router.PublicMatchupDecisionTree,
        "from_path",
        classmethod(
            lambda _cls, _path, *, require_runtime_enabled: (
                tree
                if require_runtime_enabled
                else (_ for _ in ()).throw(AssertionError("runtime required"))
            )
        ),
    )

    class Count:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    class Tensor:
        def __init__(self, value: int) -> None:
            self.value = value

        def detach(self) -> Tensor:
            return self

        def count_nonzero(self) -> Count:
            return Count(self.value)

    tree_path = tmp_path / "matchup_tree.json"
    tree_path.write_text('{"runtime_contract":{"bound":true}}\n', encoding="utf-8")
    payload = {
        "archetype_id": "alakazam",
        "extra": {"matchup_adapter_config": {}},
        "model_state_dict": {
            "matchup_adapter_bank.experts.7.up.weight": Tensor(1),
            "matchup_adapter_bank.experts.7.up.bias": Tensor(0),
        },
    }
    module._assert_trained_matchup_tree_binding(
        checkpoint_payload=payload,
        matchup_tree=tree_path,
    )

    payload["model_state_dict"]["matchup_adapter_bank.experts.7.up.weight"] = (
        Tensor(0)
    )
    with pytest.raises(RuntimeError, match="zero-output"):
        module._assert_trained_matchup_tree_binding(
            checkpoint_payload=payload,
            matchup_tree=tree_path,
        )


def test_submission_builder_gates_trained_adapter_routes_and_runtime_smoke() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_submission.sh"
    ).read_text(encoding="utf-8")

    assert "POKEBOT_SUBMISSION_MATCHUP_TREE" in source
    assert "require_runtime_route_binding(runtime, route_contract)" in source
    assert "tree.route_physical_slots" in source
    assert 'prefix + "weight"' in source
    assert 'prefix + "bias"' in source
    assert "zero-output" in source
    assert "trained adapter package enabled + frozen + exact tree" in source
    assert "-u POKEBOT_MATCHUP_ADAPTER_RUNTIME" in source
