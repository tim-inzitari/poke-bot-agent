import os
import sys

# Kaggle loads main.py via exec(); __file__ is not defined there.
for _candidate in ("/kaggle_simulations/agent", os.getcwd(), "."):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from cg.api import Observation, to_observation_class

from policy_runtime import get_policy_agent

_DECK: list[int] | None = None


def read_deck_csv() -> list[int]:
    """Read deck.csv.

    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    global _DECK
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        policy = get_policy_agent()
        policy.reset()
        _DECK = read_deck_csv()
        return _DECK

    policy = get_policy_agent()
    return policy.choose_action(obs_dict, our_deck=_DECK)
