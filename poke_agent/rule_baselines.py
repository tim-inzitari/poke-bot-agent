from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from poke_agent.actions import legal_actions
from poke_agent.attack_plan import CardId, choose_attack_plan_action, get_card, score_actions_with_attack_plan


LUCARIO_DECK = [
    673, 673, 674, 674, 675, 675, 676, 676,
    676, 677, 677, 677, 678, 678, 678, 678,
    1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141,
    1141, 1141, 1142, 1142, 1142, 1142, 1152, 1152,
    6, 1159, 1182, 1182, 1192, 1192, 1192, 1192,
    1227, 1227, 1227, 1227, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 1182, 677, 1252,
]


CRUSTLE_DECK = [
    344, 344, 344, 344, 345, 345, 345, 345,
    1147, 1147, 1147, 1147, 1159, 1264, 1264, 1264,
    1264, 1212, 1212, 1212, 1212, 1224, 1224, 1224,
    1224, 18, 18, 18, 18, 11, 11, 11,
    11, 1086, 1086, 1086, 1086, 14, 14, 14,
    14, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1,
]


ALAKAZAM_DECK = [
    741, 741, 741, 741, 742, 742, 742, 742,
    743, 743, 743, 305, 305, 305, 66, 66,
    140, 142, 858, 343, 1152, 1152, 1152, 1152,
    1086, 1086, 1086, 1086, 1079, 1079, 1079, 1097,
    1129, 1156, 1156, 1156, 1081, 1081, 1081, 1182,
    1182, 1231, 1231, 1231, 1231, 1225, 1225, 1225,
    1225, 1264, 1264, 1264, 1264, 5, 5, 19,
    19, 19, 19, 13,
]


DRAGAPULT_DECK = [
    119, 119, 119, 119, 120, 120, 120, 120,
    121, 121, 121, 235, 235, 140, 184, 1071,
    1086, 1086, 1086, 1086, 1121, 1121, 1121, 1121,
    1198, 1198, 1198, 1198, 1120, 1120, 1120, 1120,
    1227, 1227, 1227, 1227, 1182, 1182, 1182, 1152,
    1152, 1152, 1210, 1210, 1097, 1097, 1079, 1079,
    1256, 1256, 1156, 1080, 5, 5, 5, 5,
    2, 2, 2, 2,
]


DEFAULT_BASELINE_NAMES = [
    "penguin_public_scores_915",
    "yaroslav_lucario_v2_crustle_aware",
    "biohack_day2_new",
    "dashimaki_day1_crustle",
    "kojimar_simple_lucario",
    "ryota_alakazam_best5",
    "skarin_dragapult",
    "alyce_lucario_v2_bot",
    "biohack_crustle",
]

FALLBACK_BASELINE_NAMES = [
    "kokinn_lucario",
    "roman_lucario",
]


@dataclass(frozen=True)
class BaselineOpponent:
    name: str
    family: str
    deck: list[int]
    style: str
    module_dir: Path | None = None
    source: str = "public-style"

    def make_agent(self) -> Callable[[dict[str, Any]], list[int]]:
        if self.module_dir is not None:
            return make_external_agent(self.module_dir)

        deck = list(self.deck)
        style = self.style

        def agent(obs_dict: dict[str, Any]) -> list[int]:
            select = obs_dict.get("select")
            if select is None:
                return deck
            options = select.get("option") or []
            min_count = int(select.get("minCount", 1))
            max_count = int(select.get("maxCount", 1))
            actions = legal_actions(len(options), min_count, max_count)
            if not actions:
                return []
            return choose_public_style_action(obs_dict, actions, style=style)

        return agent


def _read_deck_csv(path: Path) -> list[int]:
    deck = [int(token) for token in path.read_text(encoding="utf-8").replace(",", "\n").split()]
    if len(deck) != 60:
        raise ValueError(f"{path} must contain exactly 60 card IDs, found {len(deck)}")
    return deck


def make_external_agent(module_dir: Path) -> Callable[[dict[str, Any]], list[int]]:
    module_dir = Path(module_dir)
    main_path = module_dir / "main.py"
    deck_path = module_dir / "deck.csv"
    if not main_path.exists() or not deck_path.exists():
        raise FileNotFoundError(f"baseline needs main.py and deck.csv: {module_dir}")

    old_cwd = Path.cwd()
    module_name = f"kaggle_public_baseline_{module_dir.name}_{uuid.uuid4().hex}"
    sys.path.insert(0, str(module_dir))
    try:
        os.chdir(module_dir)
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import {main_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(module_dir))
        except ValueError:
            pass

    if not hasattr(module, "agent"):
        raise AttributeError(f"{main_path} does not expose agent(obs_dict)")
    agent_fn = module.agent

    def agent(obs_dict: dict[str, Any]) -> list[int]:
        old_call_cwd = Path.cwd()
        try:
            os.chdir(module_dir)
            return agent_fn(obs_dict)
        finally:
            os.chdir(old_call_cwd)

    return agent


def _option_card_id(obs: Any, option: Any, my_index: int) -> int | None:
    area = getattr(option, "area", None)
    if area is None:
        area = "HAND"
    index = int(getattr(option, "index", 0))
    player_index = int(getattr(option, "playerIndex", my_index))
    card = get_card(obs, area, index, player_index)
    if card is None:
        return None
    raw = getattr(card, "id", getattr(card, "cardId", None))
    return int(raw) if raw is not None else None


def _enumish_name(value: Any, enum_cls: Any | None = None) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    if enum_cls is not None:
        try:
            return str(enum_cls(value).name)
        except Exception:
            pass
    return str(value)


def _style_bonus(obs_dict: dict[str, Any], action: list[int], *, style: str) -> float:
    try:
        from cg.api import OptionType, SelectContext, to_observation_class

        obs = to_observation_class(obs_dict)
    except Exception:
        return 0.0

    select = obs.select
    if select is None or _enumish_name(select.context, SelectContext) != "MAIN":
        return 0.0

    my_index = obs.current.yourIndex
    bonus = 0.0
    for option_index in action:
        option = select.option[option_index]
        option_type = _enumish_name(option.type, OptionType)
        card_id = _option_card_id(obs, option, my_index)
        if style.startswith("lucario"):
            if card_id in {CardId.RIOLU, CardId.MEGA_LUCARIO_EX, CardId.MAKUHITA, CardId.HARIYAMA}:
                bonus += 500.0
            if option_type == "ATTACH" and card_id == CardId.BASIC_FIGHTING:
                bonus += 400.0
            if option_type == "ATTACK":
                bonus += 250.0
        elif style == "crustle":
            if card_id in {CardId.DWEBBLE, CardId.CRUSTLE, CardId.HERO_CAPE}:
                bonus += 700.0
            if option_type == "ATTACK":
                bonus -= 50.0
        elif style == "alakazam":
            if card_id in {CardId.ABRA, CardId.KADABRA, CardId.ALAKAZAM, CardId.DUNSPARCE, CardId.DUDUNSPARCE}:
                bonus += 650.0
            if card_id in {CardId.BUDDY_BUDDY_POFFIN, CardId.POKE_PAD, CardId.RARE_CANDY}:
                bonus += 450.0
        elif style == "dragapult":
            if card_id in {CardId.DREEPY, CardId.DRAKLOAK, CardId.DRAGAPULT_EX, CardId.RARE_CANDY}:
                bonus += 700.0
            if card_id in {CardId.CRISPIN, CardId.BOSS_ORDERS, CardId.CRUSHING_HAMMER}:
                bonus += 350.0
    return bonus


def choose_public_style_action(
    obs_dict: dict[str, Any],
    actions: list[list[int]],
    *,
    style: str,
) -> list[int]:
    plan_scores = score_actions_with_attack_plan(obs_dict, actions)
    max_abs = max((abs(score) for score in plan_scores), default=0.0)
    best_action = actions[0]
    best_score = float("-inf")
    for index, action in enumerate(actions):
        plan_score = plan_scores[index] / max_abs if max_abs > 0 else 0.0
        score = plan_score + 0.15 * _style_bonus(obs_dict, action, style=style)
        if score > best_score:
            best_score = score
            best_action = action
    if best_score == float("-inf"):
        return choose_attack_plan_action(obs_dict, actions)
    return best_action


def _external_baseline_dirs(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    base = root / "baselines" / "kaggle_public"
    if not base.exists():
        return {}
    return {
        path.name: path
        for path in base.iterdir()
        if path.is_dir() and (path / "main.py").exists() and (path / "deck.csv").exists()
    }


def all_baseline_opponents(root: Path | None = None) -> dict[str, BaselineOpponent]:
    registry = {
        "kokinn_lucario": BaselineOpponent("kokinn_lucario", "lucario", LUCARIO_DECK, "lucario_search"),
        "yaroslav_lucario": BaselineOpponent("yaroslav_lucario", "lucario", LUCARIO_DECK, "lucario_anti_wall"),
        "penguin_lucario": BaselineOpponent("penguin_lucario", "lucario", LUCARIO_DECK, "lucario_beatdown"),
        "kacchan_lucario": BaselineOpponent("kacchan_lucario", "lucario", LUCARIO_DECK, "lucario_midrange"),
        "roman_lucario": BaselineOpponent("roman_lucario", "lucario", LUCARIO_DECK, "lucario_thick"),
        "biohack_crustle": BaselineOpponent("biohack_crustle", "crustle", CRUSTLE_DECK, "crustle"),
        "crustle": BaselineOpponent("crustle", "crustle", CRUSTLE_DECK, "crustle"),
        "alakazam": BaselineOpponent("alakazam", "alakazam", ALAKAZAM_DECK, "alakazam"),
        "dragapult": BaselineOpponent("dragapult", "dragapult", DRAGAPULT_DECK, "dragapult"),
    }

    for name, path in _external_baseline_dirs(root).items():
        try:
            deck = _read_deck_csv(path / "deck.csv")
        except ValueError:
            continue
        family = "external"
        if "lucario" in name or "penguin" in name or "yaroslav" in name or "kojimar" in name:
            family = "lucario"
        elif "crustle" in name or "biohack" in name:
            family = "crustle"
        elif "alakazam" in name or "ryota" in name:
            family = "alakazam"
        elif "dragapult" in name or "skarin" in name:
            family = "dragapult"
        registry[name] = BaselineOpponent(
            name=name,
            family=family,
            deck=deck,
            style=family,
            module_dir=path,
            source="full-public-main.py",
        )

    aliases = {
        "penguin_lucario": "penguin_public_scores_915",
        "yaroslav_lucario": "yaroslav_lucario_v2_crustle_aware",
        "biohack_crustle": "biohack_day2_new",
        "crustle": "dashimaki_day1_crustle",
        "alakazam": "ryota_alakazam_best5",
        "dragapult": "skarin_dragapult",
    }
    for alias, target in aliases.items():
        if target in registry:
            original = registry[target]
            registry[alias] = BaselineOpponent(
                name=alias,
                family=original.family,
                deck=original.deck,
                style=original.style,
                module_dir=original.module_dir,
                source=f"alias:{target}",
            )
    return registry


def baseline_opponents_from_names(
    names: list[str] | str | None,
    *,
    root: Path | None = None,
) -> list[BaselineOpponent]:
    registry = all_baseline_opponents(root)
    if names is None:
        requested = list(DEFAULT_BASELINE_NAMES)
    elif isinstance(names, str):
        requested = [name.strip() for name in names.split(",") if name.strip()]
    else:
        requested = list(names)
    if requested in (["public"], ["public_full"]):
        requested = list(DEFAULT_BASELINE_NAMES)
    elif requested == ["fallbacks"]:
        requested = list(FALLBACK_BASELINE_NAMES)
    elif requested == ["public_plus_fallbacks"]:
        requested = [*DEFAULT_BASELINE_NAMES, *FALLBACK_BASELINE_NAMES]

    missing = [name for name in requested if name not in registry]
    if missing:
        raise ValueError(f"unknown rule baselines: {missing}; available={sorted(registry)}")
    return [registry[name] for name in requested]
