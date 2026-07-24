from __future__ import annotations

import importlib.util
import ast
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


def test_submission_entrypoint_enables_only_a_shipped_runtime_matchup_tree() -> None:
    main_path = Path(__file__).resolve().parents[1] / "submission" / "main.py"
    source = main_path.read_text(encoding="utf-8")

    assert 'matchup_tree = _agent_dir() / "matchup_tree.json"' in source
    assert 'os.environ["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"' in source
    assert (
        'os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(matchup_tree)'
        in source
    )
    assert source.index("if matchup_tree.is_file():") < source.index(
        "PolicyAgent(model=model"
    )
