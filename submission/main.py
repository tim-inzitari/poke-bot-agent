"""Competition submission agent: history-conditioned policy, fail-closed actions.

Hard constraints:
  - No ``__file__`` at import time (isolated tarball / Kaggle).
  - Deck from ``deck.csv`` next to ``main.py`` or ``/kaggle_simulations/agent/``.
  - Deterministically choose first before importing cg or loading the model.
  - Info-set only (features.assert_info_set inside the policy runtime).
  - Fail-closed: illegal selects -> legal random fallback.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path


_AGENT_DIR_CANDIDATES = (
    Path.cwd(),
    Path("/kaggle_simulations/agent"),
)


def _agent_dir() -> Path:
    for directory in _AGENT_DIR_CANDIDATES:
        if (directory / "deck.csv").is_file():
            return directory
    return Path.cwd()


def _read_deck() -> list[int]:
    path = _agent_dir() / "deck.csv"
    deck: list[int] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        deck.append(int(line.split(",")[0]))
        if len(deck) >= 60:
            break
    if len(deck) != 60:
        raise ValueError(f"deck.csv must have 60 cards, got {len(deck)}")
    return deck


_DECK: list[int] | None = None
_MODEL = None
_CLOCK = None
_POLICY = None
_RNG = random.Random(0)


def _go_first_choice(obs_dict: dict) -> list[int] | None:
    """Resolve IsFirst directly from the wire enum without runtime imports."""

    selection = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not isinstance(selection, dict):
        return None
    context = selection.get("context")
    normalized_context = "".join(
        character for character in str(context).lower() if character.isalnum()
    )
    if context != 41 and normalized_context != "isfirst":
        return None
    options = list(selection.get("option") or [])
    yes = [
        index
        for index, option in enumerate(options)
        if isinstance(option, dict)
        and (
            option.get("type") == 1
            or str(option.get("type") or "").strip().lower() == "yes"
        )
    ]
    return yes if len(yes) == 1 else []


def _ensure_agent_path() -> None:
    agent_dir = str(_agent_dir())
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)


def _ensure_runtime():
    global _DECK, _MODEL, _CLOCK, _POLICY
    if _DECK is None:
        _DECK = _read_deck()
    if _MODEL is None:
        _ensure_agent_path()
        # Vendored ``cg/`` sits directly beside this entry point. The shared
        # runtime path resolver otherwise looks only for repository/Kaggle
        # development layouts that do not exist inside the submitted tarball.
        os.environ.setdefault("CG_LIB_PATH", str(_agent_dir()))
        import torch
        from poke_bot.agent import PolicyAgent
        from poke_bot.checkpoint import assert_trusted_policy_checkpoint
        from poke_bot.train import load_model_from_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = _agent_dir() / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError("model.pt is required")
        matchup_tree = _agent_dir() / "matchup_tree.json"
        if matchup_tree.is_file():
            # The shipped tree is itself runtime-gated and consumes only
            # cumulative public opponent cards. PolicyAgent validates the
            # artifact before enabling the frozen trained adapter bank.
            os.environ["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"
            os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(matchup_tree)
        assert_trusted_policy_checkpoint(checkpoint)
        model = load_model_from_checkpoint(checkpoint, device=device)
        model.eval()
        _MODEL = model
        _POLICY = PolicyAgent(model=model, deck=_DECK, use_mcts=False)
        _CLOCK = None
    return _DECK, _MODEL, _POLICY


def _fail_closed(obs_dict: dict, preferred: list[int]) -> list[int]:
    selection = obs_dict.get("select") if obs_dict else None
    if selection is None:
        return preferred
    option_count = len(selection.get("option") or [])
    if option_count <= 0:
        return []
    minimum = int(selection.get("minCount", 0) or 0)
    maximum = min(int(selection.get("maxCount", 0) or 0), option_count)
    minimum = max(0, min(minimum, maximum))
    clean: list[int] = []
    for raw in preferred:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= index < option_count and index not in clean:
            clean.append(index)
    if minimum <= len(clean) <= maximum and clean:
        return clean[:maximum]
    if maximum <= 0:
        return []
    count = _RNG.randint(minimum, maximum) if maximum >= minimum else maximum
    return _RNG.sample(range(option_count), count) if count > 0 else []


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point."""

    go_first = _go_first_choice(obs_dict)
    if go_first is not None:
        return _fail_closed(obs_dict, go_first)

    deck, _model, policy = _ensure_runtime()
    _ensure_agent_path()
    from cg.api import to_observation_class

    observation = to_observation_class(obs_dict)
    if observation.select is None:
        if policy is not None:
            policy.reset_game()
        return list(deck)

    try:
        action = policy.greedy_select(obs_dict)
    except Exception:
        action = []
    return _fail_closed(obs_dict, action)
