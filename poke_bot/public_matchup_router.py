"""Conservative matchup routing from public observations only.

The router is deliberately separate from the neural policy and from training
labels.  It may inspect only the opponent's public board and discard.  Hidden
hand, deck, prizes, orchestration metadata, and the recorded opponent label are
never inputs.  Unknown and conflicting evidence always abstains.

This module does not enable any adapter by itself.  Runtime integration and
adapter weights remain separately gated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any


UNKNOWN_PUBLIC_MATCHUP = None
PUBLIC_TREE_SCHEMA = "poke_bot.public_matchup_decision_tree/v1"

# Exact Pokemon signatures from the pinned official Iono deck.  Card 266 is
# Iono's Electrode and is included for family completeness even though the
# current official list does not run it.  Card 267 is N's Joltik, not Iono's,
# and must not activate this route.
IONO_PUBLIC_POKEMON_IDS = frozenset({265, 266, 268, 269, 270, 271})

PUBLIC_MATCHUP_SIGNATURES: Mapping[str, frozenset[int]] = {
    "iono": IONO_PUBLIC_POKEMON_IDS,
}

_PUBLIC_OPPONENT_ZONES = ("active", "bench", "discard")
_PUBLIC_CARD_CHILDREN = ("tools", "energyCards", "preEvolution")


def _card_id(value: Any) -> int | None:
    if not isinstance(value, Mapping) or value.get("id") is None:
        return None
    try:
        return int(value["id"])
    except (TypeError, ValueError):
        return None


def _public_card_ids(values: Any) -> set[int]:
    """Collect IDs from an explicitly public card/Pokemon container."""

    stack = list(values) if isinstance(values, (list, tuple)) else []
    result: set[int] = set()
    while stack:
        value = stack.pop()
        if not isinstance(value, Mapping):
            continue
        card_id = _card_id(value)
        if card_id is not None:
            result.add(card_id)
        for key in _PUBLIC_CARD_CHILDREN:
            children = value.get(key)
            if isinstance(children, (list, tuple)):
                stack.extend(children)
    return result


def visible_opponent_card_ids(obs: Any) -> frozenset[int]:
    """Return card IDs visible on the opponent's board or in their discard.

    Malformed/setup observations fail closed to an empty set.  In particular,
    this function intentionally never walks the observation recursively: doing
    so could accidentally consume the opponent hand, deck, prizes, or external
    job metadata if a future schema adds them.
    """

    if not isinstance(obs, Mapping):
        return frozenset()
    current = obs.get("current")
    if not isinstance(current, Mapping):
        return frozenset()
    players = current.get("players")
    if not isinstance(players, (list, tuple)) or len(players) != 2:
        return frozenset()
    try:
        your_index = int(current.get("yourIndex"))
    except (TypeError, ValueError):
        return frozenset()
    if your_index not in (0, 1):
        return frozenset()
    opponent = players[1 - your_index]
    if not isinstance(opponent, Mapping):
        return frozenset()

    result: set[int] = set()
    for zone in _PUBLIC_OPPONENT_ZONES:
        result.update(_public_card_ids(opponent.get(zone)))
    return frozenset(result)


def public_matchup_from_observation(obs: Any) -> str | None:
    """Return one exact public-evidence matchup, otherwise abstain."""

    visible = visible_opponent_card_ids(obs)
    matches = [
        matchup_id
        for matchup_id, signature in PUBLIC_MATCHUP_SIGNATURES.items()
        if visible & signature
    ]
    return matches[0] if len(matches) == 1 else UNKNOWN_PUBLIC_MATCHUP


@dataclass(frozen=True)
class PublicTreePrediction:
    """One causal prediction from the exported public-card decision tree."""

    archetype_id: str | None
    route: int
    confidence: float
    leaf: int


class PublicMatchupDecisionTree:
    """Dependency-free evaluator for a checksummed exported sklearn tree.

    The evaluator consumes only cumulative public opponent card IDs.  Loading
    is fail-closed: the artifact's canonical route positions must exactly match the
    append-only adapter bank.  ``unknown`` remains a separate abstention class.
    """

    def __init__(self, payload: Mapping[str, Any], *, digest: str) -> None:
        from .matchup_adapters import EXPERT_IDS, UNKNOWN_ROUTE

        if payload.get("schema") != PUBLIC_TREE_SCHEMA:
            raise ValueError("invalid public matchup decision-tree schema")
        prediction = dict(payload.get("prediction_contract") or {})
        targets = tuple(str(value) for value in payload.get("targets") or ())
        if (
            targets != tuple(EXPERT_IDS)
            or tuple(prediction.get("route_class_names") or ()) != targets
            or int(prediction.get("route_output_width", -1)) != len(targets)
            or int(prediction.get("adapter_count", -1)) != len(targets)
            or prediction.get("unknown_is_separate_abstention") is not True
            or int(prediction.get("unknown_class_index", -1)) != len(targets)
        ):
            raise ValueError("public tree does not match the 22-position adapter contract")
        tree = dict(payload.get("tree") or {})
        class_names = tuple(str(value) for value in tree.get("class_names") or ())
        if class_names != targets + ("unknown",):
            raise ValueError("public tree class order is not route-safe")
        arrays = {
            key: tuple(tree.get(key) or ())
            for key in (
                "children_left",
                "children_right",
                "feature_card_id",
                "threshold",
                "weighted_class_counts",
            )
        }
        lengths = {len(value) for value in arrays.values()}
        if lengths != {int(tree.get("node_count", -1))} or not lengths:
            raise ValueError("public tree arrays have inconsistent lengths")
        self.targets = targets
        self.unknown_index = len(targets)
        self.unknown_route = UNKNOWN_ROUTE
        self.digest = str(digest)
        self.runtime_enabled = payload.get("runtime_enabled") is True
        runtime = dict(payload.get("runtime_contract") or {})
        self.runtime_accepted_archetype_ids = frozenset(
            str(value) for value in runtime.get("accepted_archetype_ids") or ()
        )
        self.runtime_min_leaf_confidence = float(
            runtime.get("min_leaf_confidence", 1.0)
        )
        self.runtime_per_archetype_min_leaf_confidence = {
            str(key): float(value)
            for key, value in (
                runtime.get("per_archetype_min_leaf_confidence") or {}
            ).items()
        }
        self.runtime_consecutive_required = int(
            runtime.get("consecutive_required", 2)
        )
        if self.runtime_enabled:
            if (
                not self.runtime_accepted_archetype_ids
                or not self.runtime_accepted_archetype_ids.issubset(targets)
                or set(self.runtime_per_archetype_min_leaf_confidence)
                != set(self.runtime_accepted_archetype_ids)
                or any(
                    not 0.0 < value <= 1.0
                    for value in self.runtime_per_archetype_min_leaf_confidence.values()
                )
                or not 0.0 < self.runtime_min_leaf_confidence <= 1.0
                or self.runtime_consecutive_required < 1
                or runtime.get("unknown_route_exact_bypass") is not True
                or runtime.get("one_route_per_decision") is not True
            ):
                raise ValueError("public tree runtime activation contract is unsafe")
        self._left = tuple(int(value) for value in arrays["children_left"])
        self._right = tuple(int(value) for value in arrays["children_right"])
        self._feature = tuple(int(value) for value in arrays["feature_card_id"])
        self._threshold = tuple(float(value) for value in arrays["threshold"])
        self._counts = tuple(
            tuple(float(item) for item in row)
            for row in arrays["weighted_class_counts"]
        )
        if any(len(row) != len(class_names) for row in self._counts):
            raise ValueError("public tree leaf class width is inconsistent")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        require_runtime_enabled: bool = True,
    ) -> "PublicMatchupDecisionTree":
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("public tree artifact root must be an object")
        result = cls(payload, digest="sha256:" + hashlib.sha256(raw).hexdigest())
        if require_runtime_enabled and not result.runtime_enabled:
            raise ValueError("public tree has not passed its runtime activation gate")
        return result

    def predict_card_ids(self, card_ids: Sequence[int]) -> PublicTreePrediction:
        present = {int(value) for value in card_ids if int(value) >= 0}
        node = 0
        visited = 0
        while self._left[node] >= 0 or self._right[node] >= 0:
            visited += 1
            if visited > len(self._left):
                raise ValueError("public tree contains a cycle")
            feature = self._feature[node]
            value = 1.0 if feature in present else 0.0
            node = self._right[node] if value > self._threshold[node] else self._left[node]
            if node < 0 or node >= len(self._left):
                raise ValueError("public tree contains an invalid child index")
        counts = self._counts[node]
        total = sum(counts)
        if total <= 0.0:
            return PublicTreePrediction(None, self.unknown_route, 0.0, node)
        class_index = max(range(len(counts)), key=counts.__getitem__)
        confidence = counts[class_index] / total
        if class_index == self.unknown_index:
            return PublicTreePrediction(None, self.unknown_route, confidence, node)
        return PublicTreePrediction(
            self.targets[class_index], class_index, confidence, node
        )

    def runtime_prediction(self, card_ids: Sequence[int]) -> PublicTreePrediction:
        """Return an activation-qualified prediction or exact abstention."""

        prediction = self.predict_card_ids(card_ids)
        threshold = self.runtime_per_archetype_min_leaf_confidence.get(
            str(prediction.archetype_id), self.runtime_min_leaf_confidence
        )
        if (
            not self.runtime_enabled
            or prediction.archetype_id not in self.runtime_accepted_archetype_ids
            or float(prediction.confidence) < threshold
        ):
            return PublicTreePrediction(
                None, self.unknown_route, float(prediction.confidence), prediction.leaf
            )
        return prediction


@lru_cache(maxsize=8)
def load_runtime_public_matchup_tree(path: str) -> PublicMatchupDecisionTree:
    """Load one immutable activated tree per worker process."""

    return PublicMatchupDecisionTree.from_path(
        Path(path).expanduser().resolve(), require_runtime_enabled=True
    )


@dataclass
class RuntimePublicMatchupRouter:
    """Causal per-game router backed by the activated public-card tree."""

    tree: PublicMatchupDecisionTree

    def __post_init__(self) -> None:
        self._public_card_ids: set[int] = set()
        self._pending_route = self.tree.unknown_route
        self._pending_count = 0
        self._model_route = self.tree.unknown_route
        self._observations = 0
        self._recognized_observations = 0
        self._per_route: dict[int, int] = {}
        self._route_transitions: list[dict[str, int]] = []

    @classmethod
    def from_path(cls, path: str | Path) -> "RuntimePublicMatchupRouter":
        return cls(load_runtime_public_matchup_tree(str(Path(path).expanduser().resolve())))

    @property
    def candidate_model_route(self) -> int:
        return int(self._model_route)

    @property
    def audit(self) -> "RuntimePublicMatchupRouter":
        """Expose the snapshot interface shared with the shadow router.

        Search diagnostics historically call ``router.audit.snapshot(...)``.
        The activated runtime router owns its audit counters directly, so it
        safely serves as its own read-only audit view.
        """

        return self

    def observe(self, observation: Any, **_: Any) -> PublicTreePrediction:
        self._observations += 1
        self._public_card_ids.update(visible_opponent_card_ids(observation))
        prediction = self.tree.runtime_prediction(tuple(self._public_card_ids))
        route = int(prediction.route)
        if route == self._pending_route:
            self._pending_count += 1
        else:
            self._pending_route = route
            self._pending_count = 1
        if self._pending_count >= self.tree.runtime_consecutive_required:
            # Re-evaluate throughout the game.  A qualified replacement route
            # switches the one active adapter; repeated abstention returns the
            # bank to exact bypass.  Until the new verdict is confirmed, retain
            # the previous stable route instead of flapping on one observation.
            previous_route = self._model_route
            self._model_route = route
            if route != previous_route and len(self._route_transitions) < 32:
                self._route_transitions.append(
                    {
                        "observation": int(self._observations),
                        "from_route": int(previous_route),
                        "to_route": int(route),
                    }
                )
            if route != self.tree.unknown_route:
                self._recognized_observations += 1
                self._per_route[route] = self._per_route.get(route, 0) + 1
        return prediction

    def reset_for_new_game(self) -> None:
        self.__post_init__()

    def fork(self) -> "RuntimePublicMatchupRouter":
        clone = type(self)(self.tree)
        clone._public_card_ids = set(self._public_card_ids)
        clone._pending_route = self._pending_route
        clone._pending_count = self._pending_count
        clone._model_route = self._model_route
        clone._route_transitions = [dict(row) for row in self._route_transitions]
        # Branch telemetry is intentionally independent; route state is copied.
        return clone

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        accepted_routes = {
            str(archetype_id): int(self.tree.targets.index(archetype_id))
            for archetype_id in sorted(
                self.tree.runtime_accepted_archetype_ids
            )
        }
        active_archetype_id = (
            self.tree.targets[self._model_route]
            if 0 <= self._model_route < len(self.tree.targets)
            else None
        )
        return {
            "schema": "poke_bot.matchup_adapter_runtime_audit/v1",
            "mode": "causal_public_tree",
            "runtime_enabled": True,
            "tree_digest": self.tree.digest,
            "model_route": int(self._model_route),
            "initial_model_route": int(self.tree.unknown_route),
            "active_archetype_id": active_archetype_id,
            "observations": int(self._observations),
            "recognized_observations": int(self._recognized_observations),
            "route_transition_count": len(self._route_transitions),
            "route_transitions": (
                [dict(row) for row in self._route_transitions]
                if include_events
                else []
            ),
            "route_transitions_truncated": len(self._route_transitions) >= 32,
            "accepted_archetype_ids": sorted(
                self.tree.runtime_accepted_archetype_ids
            ),
            "accepted_routes": accepted_routes,
            "per_route": {
                str(route): count for route, count in sorted(self._per_route.items())
            },
        }


__all__ = [
    "IONO_PUBLIC_POKEMON_IDS",
    "PUBLIC_MATCHUP_SIGNATURES",
    "UNKNOWN_PUBLIC_MATCHUP",
    "PUBLIC_TREE_SCHEMA",
    "PublicMatchupDecisionTree",
    "PublicTreePrediction",
    "RuntimePublicMatchupRouter",
    "load_runtime_public_matchup_tree",
    "public_matchup_from_observation",
    "visible_opponent_card_ids",
]
