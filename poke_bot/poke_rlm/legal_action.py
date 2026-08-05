"""Structured legal actions and selector resolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .reasons import ReasonCode


@dataclass(frozen=True)
class LegalAction:
    """One currently legal CABT option / ordered combo."""

    option_index_path: tuple[int, ...]
    option_type: Optional[int] = None
    card_id: Optional[int] = None
    attack_id: Optional[int] = None
    source: str = ""
    destination: str = ""
    fingerprint: str = ""
    ends_turn: bool = False
    once_per_turn: bool = False
    metadata: tuple[tuple[str, str], ...] = ()

    def as_action(self) -> list[int]:
        return list(self.option_index_path)


@dataclass(frozen=True)
class ActionSelector:
    """Stable selector that is re-resolved against the current legal list."""

    fingerprint: str = ""
    option_type: Optional[int] = None
    card_id: Optional[int] = None
    attack_id: Optional[int] = None
    source: str = ""
    destination: str = ""
    planned_option_index_path: tuple[int, ...] = ()


def canonical_action_fingerprint(
    *,
    option_index_path: Sequence[int],
    option_type: Optional[int] = None,
    card_id: Optional[int] = None,
    attack_id: Optional[int] = None,
    source: str = "",
    destination: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> str:
    payload = {
        "path": [int(x) for x in option_index_path],
        "type": option_type,
        "card_id": card_id,
        "attack_id": attack_id,
        "source": source,
        "destination": destination,
        "extra": extra or {},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"act:{digest[:24]}"


def legal_actions_from_select(
    obs_dict: dict[str, Any],
    *,
    complete_combos: Optional[Sequence[Sequence[int]]] = None,
) -> tuple[LegalAction, ...]:
    """Build structured legal actions from obs select and/or enumerated combos."""
    select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    options = []
    if isinstance(select, dict):
        options = list(select.get("option") or [])
    actions: list[LegalAction] = []
    if complete_combos is not None:
        paths = [tuple(int(x) for x in c) for c in complete_combos]
    elif options:
        # Single-index forced/simple path when combos not supplied.
        lo = int(select.get("minCount", 0) or 0) if isinstance(select, dict) else 0
        hi = int(select.get("maxCount", 0) or 0) if isinstance(select, dict) else 0
        if lo == 1 and hi == 1:
            paths = [(i,) for i in range(len(options))]
        else:
            paths = [(i,) for i in range(len(options))]
    else:
        return ()
    for path in paths:
        opt0 = None
        if path and 0 <= path[0] < len(options) and isinstance(options[path[0]], dict):
            opt0 = options[path[0]]
        otype = None
        card_id = None
        attack_id = None
        if isinstance(opt0, dict):
            try:
                otype = int(opt0.get("type")) if opt0.get("type") is not None else None
            except (TypeError, ValueError):
                otype = None
            try:
                card_id = int(opt0["cardId"]) if "cardId" in opt0 else None
            except (TypeError, ValueError, KeyError):
                card_id = None
            try:
                attack_id = int(opt0["attackId"]) if "attackId" in opt0 else None
            except (TypeError, ValueError, KeyError):
                attack_id = None
        fp = canonical_action_fingerprint(
            option_index_path=path,
            option_type=otype,
            card_id=card_id,
            attack_id=attack_id,
        )
        actions.append(
            LegalAction(
                option_index_path=path,
                option_type=otype,
                card_id=card_id,
                attack_id=attack_id,
                fingerprint=fp,
            )
        )
    return tuple(actions)


def resolve_selector(
    selector: ActionSelector,
    legal: Sequence[LegalAction],
) -> tuple[Optional[LegalAction], ReasonCode]:
    if not legal:
        return None, ReasonCode.EMPTY_LEGAL
    if selector.fingerprint:
        matches = [a for a in legal if a.fingerprint == selector.fingerprint]
        if len(matches) == 1:
            return matches[0], ReasonCode.OK
        if len(matches) > 1:
            return None, ReasonCode.SELECTOR_AMBIGUOUS
    typed = [
        a
        for a in legal
        if (selector.option_type is None or a.option_type == selector.option_type)
        and (selector.card_id is None or a.card_id == selector.card_id)
        and (selector.attack_id is None or a.attack_id == selector.attack_id)
        and (not selector.source or a.source == selector.source)
        and (not selector.destination or a.destination == selector.destination)
    ]
    if len(typed) == 1:
        return typed[0], ReasonCode.OK
    if len(typed) > 1 and selector.planned_option_index_path:
        for a in typed:
            if a.option_index_path == selector.planned_option_index_path:
                return a, ReasonCode.OK
        return None, ReasonCode.SELECTOR_AMBIGUOUS
    if selector.planned_option_index_path:
        for a in legal:
            if a.option_index_path == selector.planned_option_index_path:
                # Index matched but fingerprint/type may have drifted.
                return a, ReasonCode.STALE_OPTION_INDEX
    return None, ReasonCode.SELECTOR_UNRESOLVED
