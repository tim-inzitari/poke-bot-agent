"""Materialize exact, information-set-safe expert shards from Kaggle traces.

Official episode JSON embeds the engine visualizer trace at
``steps[0][0]["visualize"]``.  Each trace row contains a masked pre-action
``obs`` and an authoritative full post-action ``current``.  Consequently, for
decision ``i >= 1``, row ``i - 1`` is the exact full pre-state aligned with the
masked input in row ``i``.  This module validates that contract before placing
private cards in auxiliary *targets only*.

The Kaggle ``configuration.seed`` is not the native libcg shuffle seed.  This
module intentionally consumes the recorded authoritative trace instead of
attempting a non-reproducible simulator rerun.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import time
import zipfile
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from . import archetypes, deck_guides, features
from .dataset import (
    DATASET_CACHE_SCHEMA_VERSION,
    GameSequence,
    featurize_step,
)
from .feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    _target_coverage,
    compact_temporal_expert_sequence,
    iter_feature_shard,
)
from .replay_import import (
    _agent_names,
    _final_winner,
    _is_option_index_action,
    _strip_opp_private,
    episode_id_of,
    extract_setup_decks,
    seat_value,
)
from .strategic_heads import (
    EXPANDED_STRATEGIC_SCHEMA,
    TARGET_SCHEMA_DIGEST,
    StrategicTargetContractError,
    attach_expanded_strategic_labels,
    expanded_strategic_sequence_coverage,
    masked_expanded_strategic_coverage,
    merge_expanded_strategic_coverages,
)
from .slowking_combo_targets import (
    attach_slowking_combo_state_labels,
    is_exact_slowking_deck,
)


VISUAL_TRACE_SCHEMA = "pokebot-authoritative-visual-trace/v1"
RECEIPT_FORMAT = "pokebot-authoritative-visual-day-receipt"
RECEIPT_FORMAT_VERSION = 1
REQUIRED_ARCHETYPE = "alakazam"
ALL_RECOGNIZED_ARCHETYPES = "*"
HIDDEN_TARGET_SOURCE = "official_authoritative_visual_trace_v1"
TARGET_CONSUMER_CONTRACT = {
    "loss_wired": {
        "opp_archetype": "aux_head cross-entropy",
        "opp_hand": "opp_hand_head masked BCE",
        "opp_hidden_remainder": "opp_remainder_head masked BCE",
        "lethal_threat": "lethal_threat_head masked BCE",
        "prize_race": "prize_race_head masked smooth-L1",
        "current_deck_guide": (
            "observed causal learned-head strategic curriculum; "
            "direct policy cross-entropy forbidden"
        ),
        "combo_state": "combo_state_head masked typed selected-option loss",
    },
    # These exact private fields are retained for provenance/future audited
    # tasks. They are not inputs and the current trainer has no matching loss.
    "stored_without_loss": ["acting_archetype", "own_prizes"],
    "expanded_strategic_targets": {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "digest": TARGET_SCHEMA_DIGEST,
        "location": "DecisionSample.aux_labels.expanded_strategic",
        "policy_observation_eligible": False,
    },
}


class VisualTraceError(ValueError):
    """The raw episode cannot be proven safe and aligned."""


@dataclass(frozen=True)
class VisualEpisodeResult:
    records: list[dict[str, Any]]
    stats: dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(raw)


def _without_search_token(observation: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(observation)
    result.pop("search_begin_input", None)
    return result


def _without_names(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_names(item)
            for key, item in value.items()
            if key != "name"
        }
    if isinstance(value, list):
        return [_without_names(item) for item in value]
    return value


def _visible_cards_match(full: Any, masked: Any) -> bool:
    """Compare card lists while allowing an intentionally facedown ``None``."""
    if not isinstance(full, list) or not isinstance(masked, list):
        return False
    if len(full) != len(masked):
        return False
    for exact, visible in zip(full, masked):
        if visible is None:
            continue
        if _without_names(exact) != visible:
            return False
    return True


def _card_ids(zone: Any, *, player: int, field: str) -> list[int]:
    if not isinstance(zone, list):
        raise VisualTraceError(f"full player {player} {field} is not a list")
    ids: list[int] = []
    serials: set[int] = set()
    for card in zone:
        if not isinstance(card, dict):
            raise VisualTraceError(
                f"full player {player} {field} contains a masked/non-card entry"
            )
        try:
            card_id = int(card["id"])
            serial = int(card["serial"])
            owner = int(card["playerIndex"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VisualTraceError(
                f"full player {player} {field} has malformed card"
            ) from exc
        if owner != player:
            raise VisualTraceError(
                f"full player {player} {field} contains owner {owner}"
            )
        if serial in serials:
            raise VisualTraceError(
                f"full player {player} {field} repeats serial {serial}"
            )
        serials.add(serial)
        ids.append(card_id)
    return ids


def _collect_serial_cards(value: Any) -> list[dict[int, int]]:
    found: list[dict[int, int]] = [{}, {}]

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if (
                isinstance(item.get("id"), int)
                and isinstance(item.get("serial"), int)
                and int(item.get("playerIndex", -1)) in (0, 1)
            ):
                owner = int(item["playerIndex"])
                serial = int(item["serial"])
                card_id = int(item["id"])
                previous = found[owner].get(serial)
                if previous is not None and previous != card_id:
                    raise VisualTraceError(
                        f"serial {serial} changes card id {previous}->{card_id}"
                    )
                found[owner][serial] = card_id
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return found


def _validate_full_zones(current: dict[str, Any]) -> list[dict[str, list[int]]]:
    players = current.get("players") or []
    if not isinstance(players, list) or len(players) != 2:
        raise VisualTraceError("authoritative current is missing two players")
    result: list[dict[str, list[int]]] = []
    for player, state in enumerate(players):
        if not isinstance(state, dict):
            raise VisualTraceError(f"authoritative player {player} is malformed")
        zones = {
            "hand": _card_ids(state.get("hand"), player=player, field="hand"),
            "deck": _card_ids(state.get("deck"), player=player, field="deck"),
            "prize": _card_ids(state.get("prize"), player=player, field="prize"),
        }
        if int(state.get("handCount", -1)) != len(zones["hand"]):
            raise VisualTraceError(f"player {player} hand count mismatch")
        if int(state.get("deckCount", -1)) != len(zones["deck"]):
            raise VisualTraceError(f"player {player} deck count mismatch")
        serials = [
            int(card["serial"])
            for field in ("hand", "deck", "prize")
            for card in state[field]
        ]
        if len(serials) != len(set(serials)):
            raise VisualTraceError(
                f"player {player} hidden zones share a physical card"
            )
        result.append(zones)
    return result


def _validate_public_projection(
    full: dict[str, Any], masked: dict[str, Any], actor: int
) -> None:
    if int(full.get("yourIndex", -1)) != actor:
        raise VisualTraceError("full pre-state actor does not match masked actor")
    if int(masked.get("yourIndex", -1)) != actor:
        raise VisualTraceError("masked observation actor is inconsistent")
    for key in (
        "energyAttached",
        "firstPlayer",
        "result",
        "retreated",
        "stadiumPlayed",
        "supporterPlayed",
        "turn",
        "turnActionCount",
    ):
        if full.get(key) != masked.get(key):
            raise VisualTraceError(f"public state mismatch at current.{key}")
    for key in ("stadium", "looking"):
        if _without_names(full.get(key)) != masked.get(key):
            raise VisualTraceError(f"public state mismatch at current.{key}")

    exact_players = full.get("players") or []
    visible_players = masked.get("players") or []
    if len(exact_players) != 2 or len(visible_players) != 2:
        raise VisualTraceError("public projection is missing two players")
    for player in (0, 1):
        exact = exact_players[player]
        visible = visible_players[player]
        if not isinstance(exact, dict) or not isinstance(visible, dict):
            raise VisualTraceError("public player state is malformed")
        if visible.get("deck") is not None:
            raise VisualTraceError("masked observation exposes a deck order")
        for key in (
            "deckCount",
            "handCount",
            "benchMax",
            "asleep",
            "burned",
            "confused",
            "paralyzed",
            "poisoned",
        ):
            if exact.get(key) != visible.get(key):
                raise VisualTraceError(
                    f"public state mismatch at player {player}.{key}"
                )
        if not _visible_cards_match(exact.get("active"), visible.get("active")):
            raise VisualTraceError(f"active projection mismatch for player {player}")
        for key in ("bench", "discard"):
            if _without_names(exact.get(key)) != visible.get(key):
                raise VisualTraceError(
                    f"public state mismatch at player {player}.{key}"
                )
        if player == actor:
            if _without_names(exact.get("hand")) != visible.get("hand"):
                raise VisualTraceError("acting player's own hand is not exact")
        elif visible.get("hand") is not None:
            raise VisualTraceError("masked observation exposes opponent hand")
        if not _visible_cards_match(exact.get("prize"), visible.get("prize")):
            raise VisualTraceError(f"prize projection mismatch for player {player}")


def _locate_trace(payload: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise VisualTraceError("episode has no complete step history")
    holders: list[tuple[int, int, list[Any]]] = []
    for step_index, step in enumerate(steps):
        if not isinstance(step, list) or len(step) != 2:
            raise VisualTraceError(f"episode step {step_index} is not two-seat")
        for seat, entry in enumerate(step):
            if not isinstance(entry, dict):
                raise VisualTraceError(
                    f"episode step {step_index} seat {seat} is malformed"
                )
            trace = entry.get("visualize")
            if isinstance(trace, list):
                holders.append((step_index, seat, trace))
    if len(holders) != 1 or holders[0][:2] != (0, 0):
        raise VisualTraceError(
            "episode must have one authoritative visualize trace at steps[0][0]"
        )
    trace = holders[0][2]
    if len(trace) != len(steps) - 1:
        raise VisualTraceError(
            f"visual trace length {len(trace)} != steps-1 {len(steps)-1}"
        )
    return steps, trace


def _prize_counts(observation: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    current = observation.get("current") or {}
    players = current.get("players") or []
    if len(players) != 2:
        return None, None
    actor = int(current.get("yourIndex", -1))
    if actor not in (0, 1):
        return None, None
    try:
        return len(players[actor]["prize"]), len(players[1 - actor]["prize"])
    except (KeyError, TypeError):
        return None, None


def _attach_strategy_labels(steps: list[dict[str, Any]], horizon: int = 8) -> None:
    """Pure-Python equivalent of the public prize/lethal label attachment.

    Elmo's bounded feature host intentionally has no torch installation, so
    importing the training-loss module merely to construct these scalar labels
    would make raw materialization needlessly GPU-framework dependent.
    """
    counts = [_prize_counts(step.get("observation") or {}) for step in steps]
    for index, step in enumerate(steps):
        own, opponent = counts[index]
        aux = dict(step.get("aux_labels") or {})
        if own is not None and opponent is not None:
            aux["prize_race"] = [float(own) / 6.0, float(opponent) / 6.0]
            end = min(len(counts), index + 1 + max(0, int(horizon)))
            aux["lethal_threat"] = float(
                any(
                    later_own is not None and int(later_own) < int(own)
                    for later_own, _later_opponent in counts[index + 1 : end]
                )
            )
        step["aux_labels"] = aux


def _record_to_temporal_sequence(
    record: dict[str, Any], *, max_context: int
) -> tuple[GameSequence, dict[str, int]]:
    raw_steps = list(record.get("steps") or [])
    if not raw_steps:
        raise VisualTraceError("validated expert record has no decisions")
    start = max(0, len(raw_steps) - int(max_context))
    steps = raw_steps[start:]
    deck = [int(value) for value in record.get("deck") or []]
    if len(deck) != 60:
        raise VisualTraceError("validated expert record lost its 60-card deck")
    try:
        decisions = [
            featurize_step(step, deck, verify_info_set=True) for step in steps
        ]
    except Exception as exc:
        raise VisualTraceError("training featurization rejected validated trace") from exc
    if not decisions:
        raise VisualTraceError("training featurization produced no decisions")
    return (
        GameSequence(
            episode_id=str(record.get("episode_id") or ""),
            seat=int(record.get("seat", 0)),
            archetype=str(record.get("archetype") or ""),
            opp_archetype=str(record.get("opp_archetype") or ""),
            deck=deck,
            value=float(record.get("value", 0.0)),
            decisions=decisions,
            info_set_ok=True,
            source=str(record.get("source") or ""),
            target_provenance=dict(record.get("target_provenance") or {}),
        ),
        {
            "decisions_truncated": start,
            "policy_targets_padded": 0,
            "policy_targets_truncated": 0,
        },
    )


def convert_visual_episode(
    payload: dict[str, Any],
    classifier: Any,
    *,
    source: str,
    required_archetype: str = REQUIRED_ARCHETYPE,
) -> VisualEpisodeResult:
    """Validate one episode and emit only the requested acting-seat records.

    ``"*"`` emits every classifier-recognized archetype.  The default remains
    Alakazam so existing sealed-corpus identities and callers are unchanged.
    """
    requested = str(required_archetype).strip().casefold()
    if not requested:
        raise VisualTraceError("required archetype cannot be empty")

    def selected(label: Any) -> bool:
        normalized = str(label or "").strip().casefold()
        if requested == ALL_RECOGNIZED_ARCHETYPES:
            return bool(normalized and normalized != archetypes.UNKNOWN)
        return normalized == requested
    steps, trace = _locate_trace(payload)
    setup_decks = extract_setup_decks(payload)
    classified_decks, labels = classifier.classify_episode(payload)
    if len(classified_decks) != 2 or len(labels) != 2:
        raise VisualTraceError("classifier did not return two seats")
    if classified_decks != setup_decks:
        raise VisualTraceError("classifier decks disagree with episode setup")
    if any(deck is None or len(deck) != 60 for deck in setup_decks):
        raise VisualTraceError("episode setup is missing a 60-card deck")

    first_action = trace[0].get("action") if isinstance(trace[0], dict) else None
    if first_action != setup_decks:
        raise VisualTraceError("visual trace setup decks disagree with raw actions")
    initial_cards = _collect_serial_cards(trace[0])
    for player in (0, 1):
        if len(initial_cards[player]) != 60:
            raise VisualTraceError(
                f"initial visual state has {len(initial_cards[player])} cards for player {player}"
            )
        if Counter(initial_cards[player].values()) != Counter(setup_decks[player]):
            raise VisualTraceError(
                f"initial visual cards disagree with setup deck for player {player}"
            )

    seat_steps: list[list[dict[str, Any]]] = [[], []]
    exact_target_rows = [0, 0]
    transitions = 0
    decisions = 0
    for index, row in enumerate(trace):
        if not isinstance(row, dict):
            raise VisualTraceError(f"visual transition {index} is malformed")
        actions = row.get("action")
        expected_actions = [entry.get("action") for entry in steps[index + 1]]
        if actions != expected_actions:
            raise VisualTraceError(f"visual action misalignment at transition {index}")
        if _collect_serial_cards(row) != initial_cards:
            raise VisualTraceError(
                f"physical-card conservation failed at transition {index}"
            )
        observation = row.get("obs")
        if not isinstance(observation, dict):
            raise VisualTraceError(f"visual transition {index} has no masked obs")
        current = observation.get("current")
        actor = 0 if index == 0 else int((current or {}).get("yourIndex", -1))
        if actor not in (0, 1):
            raise VisualTraceError(f"visual transition {index} has invalid actor")
        outer_observation = steps[index][actor].get("observation") or {}
        if _without_search_token(outer_observation) != _without_search_token(
            observation
        ):
            raise VisualTraceError(f"masked observation misalignment at {index}")
        transitions += 1
        if index == 0 or not isinstance(current, dict) or observation.get("select") is None:
            continue

        pre_current = trace[index - 1].get("current")
        if not isinstance(pre_current, dict):
            raise VisualTraceError(f"visual transition {index-1} has no full current")
        zones = _validate_full_zones(pre_current)
        _validate_public_projection(pre_current, current, actor)
        action = actions[actor]
        select = observation.get("select") or {}
        options = select.get("option") or []
        if not _is_option_index_action(
            action,
            len(options),
            min_count=int(select.get("minCount", 0)),
            max_count=int(select.get("maxCount", len(options))),
        ):
            raise VisualTraceError(f"illegal/ambiguous recorded action at {index}")

        masked_observation, leaked_aux, report = _strip_opp_private(observation)
        if report.remasked or any(value is not None for value in leaked_aux.values()):
            raise VisualTraceError(
                f"visual masked input leaked opponent-private state at {index}"
            )
        opponent = 1 - actor
        hidden_remainder = (
            list(zones[opponent]["hand"])
            + list(zones[opponent]["deck"])
            + list(zones[opponent]["prize"])
        )
        aux = {
            "opp_hand": list(zones[opponent]["hand"]),
            "opp_deck_order": list(zones[opponent]["deck"]),
            "opp_prizes": list(zones[opponent]["prize"]),
            "own_prizes": list(zones[actor]["prize"]),
            "opp_hidden_remainder": hidden_remainder,
            "opp_hidden_remainder_source": HIDDEN_TARGET_SOURCE,
            "opp_private_zone_source": HIDDEN_TARGET_SOURCE,
            "own_private_prize_source": HIDDEN_TARGET_SOURCE,
            "acting_archetype": str(labels[actor].deck_id),
            "opp_archetype": str(labels[opponent].deck_id),
            "opp_agent": _agent_names(payload)[opponent],
        }
        seat_steps[actor].append(
            {
                "env_step": index,
                "observation": masked_observation,
                "action": [int(value) for value in action],
                "select_min_count": int(select.get("minCount", 0)),
                "select_max_count": int(select.get("maxCount", 0)),
                "legal_action_count": len(options),
                "aux_labels": aux,
                # The visual trace's current state is the authoritative full
                # post-action state for this decision. The strategic target
                # builder reduces it to a bounded public snapshot and removes
                # this temporary field before serialization.
                "transition_after": {
                    "current": copy.deepcopy(row.get("current")),
                },
            }
        )
        exact_target_rows[actor] += 1
        decisions += 1

    winner = _final_winner(payload)
    if winner not in (0, 1, 2):
        raise VisualTraceError("episode has no final winner/draw")
    agent_names = _agent_names(payload)
    episode_id = episode_id_of(payload)
    records: list[dict[str, Any]] = []
    for seat in (0, 1):
        if not selected(labels[seat].deck_id):
            continue
        if not seat_steps[seat]:
            raise VisualTraceError(
                f"{requested} seat {seat} has no validated acting decisions"
            )
        _attach_strategy_labels(seat_steps[seat])
        combo_coverage = None
        if is_exact_slowking_deck(list(setup_decks[seat] or [])):
            combo_coverage = attach_slowking_combo_state_labels(
                seat_steps[seat],
                deck=list(setup_decks[seat] or []),
            )
        try:
            strategic_contract = attach_expanded_strategic_labels(
                seat_steps[seat],
                game_value=seat_value(winner, seat),
                terminal_complete=True,
            )
        except StrategicTargetContractError as exc:
            raise VisualTraceError(
                "expanded strategic target construction rejected validated trace"
            ) from exc
        opponent = 1 - seat
        records.append(
            {
                "episode_id": episode_id,
                "source": source,
                "seat": seat,
                "agent": agent_names[seat],
                "opp_agent": agent_names[opponent],
                "archetype": str(labels[seat].deck_id),
                "opp_archetype": str(labels[opponent].deck_id),
                "deck": list(setup_decks[seat] or []),
                "opp_deck": list(setup_decks[opponent] or []),
                "value": seat_value(winner, seat),
                "winner": winner,
                "steps": seat_steps[seat],
                "info_set_ok": True,
                "info_set_flags": [],
                "info_set_remasked": False,
                "n_decisions": len(seat_steps[seat]),
                "target_provenance": {
                    "schema": VISUAL_TRACE_SCHEMA,
                    "hidden_targets": HIDDEN_TARGET_SOURCE,
                    "alignment": "pre_full=v[i-1].current;masked_input=v[i].obs",
                    "expanded_strategic_targets": strategic_contract,
                    **(
                        {"slowking_combo_state_targets": combo_coverage}
                        if combo_coverage is not None
                        else {}
                    ),
                },
            }
        )
    return VisualEpisodeResult(
        records=records,
        stats={
            "transitions_validated": transitions,
            "decisions_validated": decisions,
            "exact_target_rows": sum(exact_target_rows),
            "selected_records": len(records),
            # Backward-compatible accounting consumed by the original sealed
            # Alakazam materializer/tests.
            "alakazam_records": len(records) if requested == REQUIRED_ARCHETYPE else 0,
            "required_archetype": requested,
            "seat_labels": [str(label.deck_id) for label in labels],
            "label_methods": [str(label.method) for label in labels],
        },
    )


@contextmanager
def _guide_targets_enabled(guide_id: Optional[str]) -> Iterator[None]:
    keys = (
        "POKEBOT_CURRENT_DECK_GUIDE",
        "POKEBOT_CURRENT_DECK_GUIDE_TARGETS",
        "POKEBOT_ALAKAZAM_GUIDE_TARGETS",
    )
    previous = {key: os.environ.get(key) for key in keys}
    selected = str(guide_id or "").strip().casefold()
    if selected:
        if selected not in deck_guides.supported_ids():
            raise VisualTraceError(f"unsupported current-deck guide: {selected!r}")
        os.environ["POKEBOT_CURRENT_DECK_GUIDE"] = selected
        os.environ["POKEBOT_CURRENT_DECK_GUIDE_TARGETS"] = "1"
        if selected == "alakazam":
            os.environ["POKEBOT_ALAKAZAM_GUIDE_TARGETS"] = "1"
        else:
            os.environ.pop("POKEBOT_ALAKAZAM_GUIDE_TARGETS", None)
    else:
        for key in keys:
            os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


_ZIP_HANDLES: dict[str, zipfile.ZipFile] = {}
_WORKER_CLASSIFIER: Any = None


def _init_worker(classifier: Any) -> None:
    global _WORKER_CLASSIFIER
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _WORKER_CLASSIFIER = classifier


def _read_zip_member(archive_path: str, member: str) -> bytes:
    archive = _ZIP_HANDLES.get(archive_path)
    if archive is None:
        archive = zipfile.ZipFile(archive_path, "r")
        _ZIP_HANDLES[archive_path] = archive
    return archive.read(member)


def _sequence_coverage(
    sequence: GameSequence, *, required_archetype: str
) -> dict[str, int]:
    coverage = _target_coverage(sequence)
    coverage.update(
        {
            "opponent_deck_order_rows": 0,
            "own_private_prize_rows": 0,
            "acting_archetype_rows": 0,
        }
    )
    for decision in sequence.decisions:
        aux = dict(decision.aux_labels or {})
        coverage["opponent_deck_order_rows"] += int(
            aux.get("opp_deck_order") is not None
        )
        coverage["own_private_prize_rows"] += int(
            aux.get("own_prizes") is not None
        )
        acting = str(aux.get("acting_archetype") or "").strip().casefold()
        expected = str(sequence.archetype or "").strip().casefold()
        coverage["acting_archetype_rows"] += int(
            acting == expected
            and (
                required_archetype == ALL_RECOGNIZED_ARCHETYPES
                or acting == required_archetype
            )
        )
    return coverage


def _materialize_member_job(
    archive_path: str,
    member: str,
    source: str,
    max_context: int,
    required_archetype: str,
    guide_id: Optional[str],
) -> dict[str, Any]:
    classifier = _WORKER_CLASSIFIER
    if classifier is None:
        raise RuntimeError("visual-trace worker has no classifier")
    raw = _read_zip_member(archive_path, member)
    payload: dict[str, Any] = {}
    try:
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VisualTraceError(f"invalid episode JSON: {member}") from exc
        result = convert_visual_episode(
            payload,
            classifier,
            source=source,
            required_archetype=required_archetype,
        )
        sequences: list[GameSequence] = []
        details_total = Counter()
        coverage = Counter()
        with _guide_targets_enabled(guide_id):
            for record in result.records:
                sequence, details = _record_to_temporal_sequence(
                    record, max_context=max_context
                )
                details_total.update(details)
                compact_temporal_expert_sequence(sequence)
                sequences.append(sequence)
                coverage.update(
                    _sequence_coverage(
                        sequence, required_archetype=required_archetype
                    )
                )
    except VisualTraceError as exc:
        message = str(exc)
        reason = "validation_error"
        for fragment, label in (
            ("visual trace length", "visual_trace_length_mismatch"),
            ("authoritative visualize trace", "visual_trace_missing"),
            ("visual action misalignment", "action_alignment_mismatch"),
            ("masked observation misalignment", "observation_alignment_mismatch"),
            ("physical-card conservation", "physical_card_conservation"),
            ("public state mismatch", "public_projection_mismatch"),
            ("projection mismatch", "public_projection_mismatch"),
            ("exposes opponent", "private_information_leak"),
            ("leaked opponent-private", "private_information_leak"),
            ("illegal/ambiguous recorded action", "illegal_recorded_action"),
            ("training featurization", "training_featurization_rejected"),
            ("invalid episode JSON", "invalid_json"),
        ):
            if fragment in message:
                reason = label
                break
        return {
            "member": member,
            "member_bytes": len(raw),
            "member_sha256": _sha256_bytes(raw),
            "module_version": str(payload.get("module_version") or ""),
            "schema_version": int(payload.get("schema_version", -1)),
            "rejected": True,
            "drop_reason": reason,
            "drop_message": message[:500],
            "sequences": [],
            "conversion": {},
            "coverage": {},
            "expanded_strategic_targets": (
                masked_expanded_strategic_coverage(0)
            ),
        }
    return {
        "member": member,
        "member_bytes": len(raw),
        "member_sha256": _sha256_bytes(raw),
        "module_version": str(payload.get("module_version") or ""),
        "schema_version": int(payload.get("schema_version", -1)),
        "rejected": False,
        "episode": result.stats,
        "sequences": sequences,
        "conversion": dict(details_total),
        "coverage": dict(coverage),
        "expanded_strategic_targets": merge_expanded_strategic_coverages(
            tuple(
                expanded_strategic_sequence_coverage(sequence.decisions)
                for sequence in sequences
            )
        ),
    }


def _available_memory_bytes() -> Optional[int]:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _receipt_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".receipt.json")


def _metadata_path(output_path: Path) -> Path:
    # Compatible with scripts/assemble_feature_manifest.py (*.features.json).
    return output_path.with_suffix(output_path.suffix + ".json")


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_resume(
    archive_path: Path,
    output_path: Path,
    receipt_path: Path,
    metadata_path: Path,
    *,
    source_date: str,
    classifier_digest: str,
    max_context: int,
    max_episodes: int,
    required_archetype: str,
    guide_id: Optional[str],
    guide_version: Optional[str],
) -> Optional[dict[str, Any]]:
    if (
        not output_path.exists()
        and not receipt_path.exists()
        and not metadata_path.exists()
    ):
        return None
    if (
        not output_path.is_file()
        or not receipt_path.is_file()
        or not metadata_path.is_file()
    ):
        raise VisualTraceError("partial prior materialization exists; refusing reuse")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualTraceError("existing materialization receipt is invalid") from exc
    expected = (
        receipt.get("format") == RECEIPT_FORMAT
        and int(receipt.get("format_version", -1)) == RECEIPT_FORMAT_VERSION
        and receipt.get("source_date") == source_date
        and (receipt.get("classifier") or {}).get("sha256") == classifier_digest
        and int((receipt.get("schemas") or {}).get("dataset", -1))
        == DATASET_CACHE_SCHEMA_VERSION
        and int((receipt.get("schemas") or {}).get("feature", -1))
        == features.FEATURE_SCHEMA_VERSION
        and (receipt.get("schemas") or {}).get("compact_mode")
        == COMPACT_MODE_TEMPORAL_EXPERT
        and int((receipt.get("schemas") or {}).get("max_context", -1))
        == int(max_context)
        and (receipt.get("schemas") or {}).get(
            "expanded_strategic_targets"
        )
        == {
            "schema": EXPANDED_STRATEGIC_SCHEMA,
            "digest": TARGET_SCHEMA_DIGEST,
        }
        and receipt.get("target_consumer_contract") == TARGET_CONSUMER_CONTRACT
        and int((receipt.get("selection") or {}).get("max_episodes", -1))
        == int(max_episodes)
        and str(
            (receipt.get("selection") or {}).get("acting_seat_archetype") or ""
        ).strip().casefold()
        == required_archetype
        and (receipt.get("selection") or {}).get("current_deck_guide")
        == guide_id
        and (receipt.get("schemas") or {}).get("guide") == guide_version
    )
    if not expected:
        raise VisualTraceError("existing receipt contract does not match this run")
    if _sha256_file(archive_path) != (receipt.get("source_archive") or {}).get(
        "sha256"
    ):
        raise VisualTraceError("source archive digest changed")
    if _sha256_file(output_path) != (receipt.get("output") or {}).get("sha256"):
        raise VisualTraceError("existing feature shard digest is invalid")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualTraceError("existing feature metadata is invalid") from exc
    if (
        metadata.get("sha256") != (receipt.get("output") or {}).get("sha256")
        or metadata.get("classifier_sha256") != classifier_digest
        or metadata.get("visual_trace_schema") != VISUAL_TRACE_SCHEMA
        or metadata.get("target_consumer_contract") != TARGET_CONSUMER_CONTRACT
        or metadata.get("expanded_strategic_targets")
        != {
            "schema": EXPANDED_STRATEGIC_SCHEMA,
            "digest": TARGET_SCHEMA_DIGEST,
        }
        or (metadata.get("stats") or {}).get(
            "expanded_strategic_targets"
        )
        != (receipt.get("stats") or {}).get("expanded_strategic_targets")
    ):
        raise VisualTraceError("existing feature metadata contract is invalid")
    expanded = (receipt.get("stats") or {}).get(
        "expanded_strategic_targets"
    )
    try:
        validated_expanded = merge_expanded_strategic_coverages((expanded,))
    except StrategicTargetContractError as exc:
        raise VisualTraceError(
            "existing expanded strategic target coverage is invalid"
        ) from exc
    if int(validated_expanded["decisions"]) != int(
        (receipt.get("stats") or {}).get("decisions_kept", -1)
    ):
        raise VisualTraceError(
            "existing expanded strategic target coverage count is invalid"
        )
    expected_records = int((receipt.get("stats") or {}).get("records_kept", -1))
    if sum(1 for _ in iter_feature_shard(output_path)) != expected_records:
        raise VisualTraceError("existing feature shard count is invalid")
    return receipt


def materialize_day(
    archive_path: Path,
    output_path: Path,
    *,
    classifier: Any,
    source_date: str,
    workers: int = 6,
    max_in_flight: int = 0,
    max_episodes: int = 0,
    max_context: int = 320,
    resume: bool = True,
    min_available_bytes: int = 8 * 1024**3,
    min_records: int = 1,
    required_archetype: str = REQUIRED_ARCHETYPE,
    current_deck_guide: Optional[str] = "alakazam",
) -> dict[str, Any]:
    """Build one immutable, checksummed, bounded-memory feature shard."""
    archive_path = Path(archive_path).resolve()
    output_path = Path(output_path).resolve()
    receipt_path = _receipt_path(output_path)
    metadata_path = _metadata_path(output_path)
    requested = str(required_archetype).strip().casefold()
    guide_id = str(current_deck_guide or "").strip().casefold() or None
    if guide_id is not None and guide_id not in deck_guides.supported_ids():
        raise ValueError(f"unsupported current-deck guide: {guide_id!r}")
    previous_guide = os.environ.get("POKEBOT_CURRENT_DECK_GUIDE")
    try:
        if guide_id is None:
            os.environ.pop("POKEBOT_CURRENT_DECK_GUIDE", None)
        else:
            os.environ["POKEBOT_CURRENT_DECK_GUIDE"] = guide_id
        guide_version = deck_guides.guide_version()
    finally:
        if previous_guide is None:
            os.environ.pop("POKEBOT_CURRENT_DECK_GUIDE", None)
        else:
            os.environ["POKEBOT_CURRENT_DECK_GUIDE"] = previous_guide
    if not requested:
        raise ValueError("required_archetype cannot be empty")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if workers < 1 or max_context < 1 or max_episodes < 0:
        raise ValueError("workers/max_context must be positive; max_episodes nonnegative")
    contract = getattr(classifier, "contract", None)
    if not isinstance(contract, dict):
        raise VisualTraceError("classifier has no immutable contract")
    classifier_digest = _json_digest(contract)
    if resume:
        reused = _validate_resume(
            archive_path,
            output_path,
            receipt_path,
            metadata_path,
            source_date=source_date,
            classifier_digest=classifier_digest,
            max_context=max_context,
            max_episodes=max_episodes,
            required_archetype=requested,
            guide_id=guide_id,
            guide_version=guide_version,
        )
        if reused is not None:
            return {**reused, "resumed": True}
    elif output_path.exists() or receipt_path.exists() or metadata_path.exists():
        raise FileExistsError(output_path)

    available = _available_memory_bytes()
    if available is not None and available < int(min_available_bytes):
        raise VisualTraceError(
            f"memory floor failed: available={available} required={min_available_bytes}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.partial.{os.getpid()}")
    receipt_partial = receipt_path.with_name(
        f".{receipt_path.name}.partial.{os.getpid()}"
    )
    metadata_partial = metadata_path.with_name(
        f".{metadata_path.name}.partial.{os.getpid()}"
    )
    if partial.exists() or receipt_partial.exists() or metadata_partial.exists():
        raise VisualTraceError("stale same-process partial materialization exists")

    archive_digest = _sha256_file(archive_path)
    archive_bytes = archive_path.stat().st_size
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = sorted(
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.endswith(".json")
        )
    if max_episodes > 0:
        members = members[:max_episodes]
    if not members:
        raise VisualTraceError("daily archive contains no episode JSON")
    source = f"pokemon-tcg-ai-battle-episodes-{source_date}"
    inflight = max(workers, max_in_flight or workers * 2)
    started = time.time()
    stats: dict[str, Any] = {
        "episodes_total": len(members),
        "episodes_validated": 0,
        "episodes_rejected": 0,
        "transitions_validated": 0,
        "decisions_validated": 0,
        "records_total": 0,
        "records_kept": 0,
        "records_dropped": 0,
        "decisions_kept": 0,
        "decisions_truncated": 0,
        "policy_targets_padded": 0,
        "policy_targets_truncated": 0,
        "drop_reasons": {},
        "drop_reason_examples": {},
        "target_coverage": {},
        "expanded_strategic_targets": masked_expanded_strategic_coverage(0),
        "seat_labels": {},
        "label_methods": {},
    }
    member_manifest = hashlib.sha256()
    module_versions: Counter[str] = Counter()
    schema_versions: Counter[str] = Counter()
    header = {
        "format": SHARD_FORMAT,
        "format_version": SHARD_FORMAT_VERSION,
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": features.FEATURE_SCHEMA_VERSION,
        "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
        "required_archetype": requested,
        "source_dates": [source_date],
        "source_archive": archive_path.name,
        "source_archive_sha256": archive_digest,
        "visual_trace_schema": VISUAL_TRACE_SCHEMA,
        "classifier_sha256": classifier_digest,
        "max_context": int(max_context),
        "guide_id": guide_id,
        "guide_version": guide_version,
        "target_consumer_contract": TARGET_CONSUMER_CONTRACT,
        "expanded_strategic_targets": {
            "schema": EXPANDED_STRATEGIC_SCHEMA,
            "digest": TARGET_SCHEMA_DIGEST,
        },
    }

    def account(result: dict[str, Any], stream: Any) -> None:
        module_versions[result["module_version"]] += 1
        schema_versions[str(result["schema_version"])] += 1
        member_row = {
            "member": result["member"],
            "bytes": result["member_bytes"],
            "sha256": result["member_sha256"],
        }
        member_manifest.update(
            json.dumps(member_row, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
        )
        if bool(result.get("rejected")):
            reason = str(result.get("drop_reason") or "validation_error")
            stats["episodes_rejected"] += 1
            stats["records_dropped"] += 1
            stats["drop_reasons"][reason] = int(
                stats["drop_reasons"].get(reason, 0)
            ) + 1
            stats["drop_reason_examples"].setdefault(
                reason,
                {
                    "member": str(result.get("member") or ""),
                    "message": str(result.get("drop_message") or "")[:500],
                },
            )
            return

        stats["episodes_validated"] += 1
        episode = result["episode"]
        stats["transitions_validated"] += int(episode["transitions_validated"])
        stats["decisions_validated"] += int(episode["decisions_validated"])
        for label in episode["seat_labels"]:
            stats["seat_labels"][label] = int(stats["seat_labels"].get(label, 0)) + 1
        for method in episode["label_methods"]:
            stats["label_methods"][method] = int(
                stats["label_methods"].get(method, 0)
            ) + 1
        sequences = result["sequences"]
        stats["records_total"] += len(sequences)
        for sequence in sequences:
            pickle.dump(sequence, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stats["records_kept"] += 1
            stats["decisions_kept"] += len(sequence)
        for key, value in result["conversion"].items():
            stats[key] = int(stats.get(key, 0)) + int(value)
        for key, value in result["coverage"].items():
            coverage = stats["target_coverage"]
            coverage[key] = int(coverage.get(key, 0)) + int(value)
        stats["expanded_strategic_targets"] = (
            merge_expanded_strategic_coverages(
                (
                    stats["expanded_strategic_targets"],
                    result["expanded_strategic_targets"],
                )
            )
        )

    try:
        with partial.open("xb") as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            if workers == 1:
                _init_worker(classifier)
                for member in members:
                    account(
                        _materialize_member_job(
                            str(archive_path),
                            member,
                            source,
                            max_context,
                            requested,
                            guide_id,
                        ),
                        stream,
                    )
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_worker,
                    initargs=(classifier,),
                ) as pool:
                    pending: deque[Future[dict[str, Any]]] = deque()
                    next_member = 0

                    def submit_one() -> bool:
                        nonlocal next_member
                        if next_member >= len(members):
                            return False
                        pending.append(
                            pool.submit(
                                _materialize_member_job,
                                str(archive_path),
                                members[next_member],
                                source,
                                max_context,
                                requested,
                                guide_id,
                            )
                        )
                        next_member += 1
                        return True

                    while len(pending) < inflight and submit_one():
                        pass
                    while pending:
                        account(pending.popleft().result(), stream)
                        submit_one()
            if int(stats["records_kept"]) < int(min_records):
                raise VisualTraceError(
                    f"{requested} records {stats['records_kept']} < minimum {min_records}"
                )
            footer = {
                "format": SHARD_FORMAT + "-footer",
                "format_version": SHARD_FORMAT_VERSION,
                "stats": stats,
            }
            pickle.dump(footer, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        if sum(1 for _ in iter_feature_shard(partial)) != int(stats["records_kept"]):
            raise VisualTraceError("new feature shard failed count validation")
        output_digest = _sha256_file(partial)
        os.replace(partial, output_path)
        _fsync_directory(output_path.parent)
        metadata = {
            **header,
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": output_digest,
            "stats": stats,
            "workers": int(workers),
            "max_in_flight": int(inflight),
            "elapsed_seconds": time.time() - started,
        }
        with metadata_partial.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(metadata_partial, metadata_path)
        receipt = {
            "format": RECEIPT_FORMAT,
            "format_version": RECEIPT_FORMAT_VERSION,
            "source_date": source_date,
            "source_archive": {
                "path": str(archive_path),
                "bytes": archive_bytes,
                "sha256": archive_digest,
                "episode_members": len(members),
                "member_manifest_sha256": "sha256:" + member_manifest.hexdigest(),
                "module_versions": dict(sorted(module_versions.items())),
                "schema_versions": dict(sorted(schema_versions.items())),
            },
            "classifier": {"contract": contract, "sha256": classifier_digest},
            "schemas": {
                "visual_trace": VISUAL_TRACE_SCHEMA,
                "dataset": DATASET_CACHE_SCHEMA_VERSION,
                "feature": features.FEATURE_SCHEMA_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                "max_context": int(max_context),
                "guide": guide_version,
                "expanded_strategic_targets": {
                    "schema": EXPANDED_STRATEGIC_SCHEMA,
                    "digest": TARGET_SCHEMA_DIGEST,
                },
            },
            "selection": {
                "acting_seat_archetype": requested,
                "max_episodes": int(max_episodes),
                "current_deck_guide": guide_id,
            },
            "target_consumer_contract": TARGET_CONSUMER_CONTRACT,
            "output": {
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": output_digest,
                "metadata_path": str(metadata_path),
                "metadata_sha256": _sha256_file(metadata_path),
            },
            "stats": stats,
            "workers": int(workers),
            "max_in_flight": int(inflight),
            "elapsed_seconds": time.time() - started,
            "resumed": False,
        }
        with receipt_partial.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(receipt_partial, receipt_path)
        os.chmod(output_path, 0o444)
        os.chmod(metadata_path, 0o444)
        os.chmod(receipt_path, 0o444)
        _fsync_directory(output_path.parent)
        return receipt
    except BaseException:
        partial.unlink(missing_ok=True)
        metadata_partial.unlink(missing_ok=True)
        receipt_partial.unlink(missing_ok=True)
        raise


__all__ = [
    "HIDDEN_TARGET_SOURCE",
    "ALL_RECOGNIZED_ARCHETYPES",
    "RECEIPT_FORMAT",
    "RECEIPT_FORMAT_VERSION",
    "TARGET_CONSUMER_CONTRACT",
    "VISUAL_TRACE_SCHEMA",
    "VisualEpisodeResult",
    "VisualTraceError",
    "convert_visual_episode",
    "materialize_day",
]
