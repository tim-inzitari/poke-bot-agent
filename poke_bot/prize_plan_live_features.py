"""Reconstruct the frozen Prize-plan-v2 action ABI from public replay rows.

This is an offline cache-materialization helper, not a serving path.  It uses
the same public-rule representation and factorized teacher-forcing programs as
the sealed recent-20 pack, while discarding hidden replay fields before either
operation.  No terminal result, future transition, Prize identity, critic
target, or unchosen-action label enters the returned features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from . import features
from .alakazam_public_rule_adapter_r298 import (
    build_public_rule_representation,
    sanitize_public_observation,
)
from .alakazam_rule_derivative_model_r298 import (
    R298_SEMANTIC_PROJECTION_FEATURE_DIM,
    _option_semantic_features,
    semantic_option_feature_rows,
)
from .prize_plan_v2_sidecar import PrizePlanV2SidecarConfig


class PrizePlanLiveFeatureError(ValueError):
    """A public replay decision cannot satisfy the frozen sidecar ABI."""


@dataclass(frozen=True)
class LiveCompleteActionFeatures:
    """One chosen complete action in the sidecar's fixed structural ABI."""

    episode_id: str
    seat: int
    env_step: int
    stage_count: int
    first_stage_menu: tuple[tuple[float, ...], ...]
    selected_stage_features: tuple[tuple[float, ...], ...]
    selected_option_indices: tuple[int, ...]
    selected_legal_counts: tuple[int, ...]
    selected_action_programs: tuple[tuple[int, ...], ...]


def _public_prize_counts(
    observation: Mapping[str, Any], *, acting_seat: int
) -> tuple[int, int] | None:
    public = sanitize_public_observation(observation)
    current = public.get("current")
    if not isinstance(current, Mapping) or current.get("yourIndex") != acting_seat:
        return None
    players = current.get("players")
    if not isinstance(players, list) or len(players) != 2:
        return None
    counts: list[int] = []
    for seat in (acting_seat, 1 - acting_seat):
        player = players[seat]
        prizes = player.get("prize") if isinstance(player, Mapping) else None
        if not isinstance(prizes, list) or not 1 <= len(prizes) <= 6:
            return None
        counts.append(len(prizes))
    return counts[0], counts[1]


def h3_causal_interval_availability(
    decisions: list[Mapping[str, Any]],
    *,
    acting_seat: int,
    start_index: int,
) -> tuple[bool, str | None]:
    """Prove three next-complete-same-seat public segments for live replay.

    A completed per-seat compact trajectory already includes all intervening
    opponent activity between its strictly increasing environment steps.  The
    fourth pre-action frame is therefore the exact H3 endpoint.  Setup counts,
    a missing endpoint, ambiguous order, or any increasing remaining-Prize
    count masks the actor term rather than assigning a fabricated zero target.
    """

    if acting_seat not in (0, 1) or start_index < 0:
        raise PrizePlanLiveFeatureError("H3 interval identity is malformed")
    window = decisions[start_index : start_index + 4]
    if len(window) != 4:
        return False, "terminal_censored_before_three_complete_same_seat_segments"
    steps: list[int] = []
    counts: list[tuple[int, int]] = []
    for decision in window:
        raw_step = decision.get("env_step")
        observation = decision.get("observation")
        if (
            isinstance(raw_step, bool)
            or not isinstance(raw_step, int)
            or raw_step < 0
            or not isinstance(observation, Mapping)
        ):
            return False, "missing_or_ambiguous_public_segment_evidence"
        steps.append(raw_step)
        pair = _public_prize_counts(observation, acting_seat=acting_seat)
        if pair is None:
            return False, "pre_action_public_prize_count_outside_valid_1_to_6_range"
        counts.append(pair)
    if any(right <= left for left, right in zip(steps, steps[1:])):
        return False, "non_increasing_same_seat_environment_steps"
    for before, after in zip(counts, counts[1:]):
        if after[0] > before[0] or after[1] > before[1]:
            return False, "non_monotone_public_prize_count"
    return True, None


def _feature_row(values: list[float], *, label: str) -> tuple[float, ...]:
    if len(values) != R298_SEMANTIC_PROJECTION_FEATURE_DIM or any(
        not math.isfinite(float(value)) for value in values
    ):
        raise PrizePlanLiveFeatureError(f"{label} semantic feature row is malformed")
    return tuple(float(value) for value in values)


def _stop_feature(representation: Mapping[str, Any]) -> tuple[float, ...]:
    return _feature_row(
        _option_semantic_features(
            representation,
            {
                "semantic": {"option": {"option_type": "pass"}},
                "referenced_card_ids": [],
            },
        ),
        label="STOP",
    )


def complete_action_features_from_public_replay(
    *,
    episode_id: str,
    seat: int,
    env_step: int,
    observation: Mapping[str, Any],
    action: list[int],
    config: PrizePlanV2SidecarConfig | None = None,
) -> LiveCompleteActionFeatures:
    """Build exact public 40-D menus and selected factorized action structure."""

    sidecar = config or PrizePlanV2SidecarConfig()
    if not episode_id or seat not in (0, 1) or env_step < 0:
        raise PrizePlanLiveFeatureError("replay action identity is malformed")
    if not isinstance(observation, Mapping) or not isinstance(action, list) or not action:
        raise PrizePlanLiveFeatureError("replay observation/action is absent")
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token > sidecar.max_action_token_value
        for token in action
    ):
        raise PrizePlanLiveFeatureError("selected action is outside the v2 token ABI")

    # The adapter's sanitizer is the single information boundary for both the
    # semantic rows and the factorized program reconstruction.  Hidden hands,
    # deck order, unrevealed Prize identities, targets, and future transitions
    # therefore cannot perturb the cache input.
    public = sanitize_public_observation(observation)
    representation_obj = build_public_rule_representation(public)
    representation = representation_obj.to_dict()
    raw_options = tuple(
        _feature_row(row, label="public option")
        for row in semantic_option_feature_rows(representation)
    )
    try:
        stages = features.factorized_teacher_forcing_stages(public, action)
    except Exception as exc:
        raise PrizePlanLiveFeatureError(
            "public action cannot be reconstructed as a legal factorized program"
        ) from exc
    if not stages or len(stages) > sidecar.max_action_stages:
        raise PrizePlanLiveFeatureError("factorized action stage count exceeds v2 ABI")

    prefix: list[int] = []
    selected_features: list[tuple[float, ...]] = []
    selected_indices: list[int] = []
    selected_counts: list[int] = []
    selected_programs: list[tuple[int, ...]] = []
    first_menu: tuple[tuple[float, ...], ...] | None = None
    stop = _stop_feature(representation)
    for stage_index, (programs_raw, selected_raw) in enumerate(stages):
        programs = [tuple(int(token) for token in program) for program in programs_raw]
        selected = int(selected_raw)
        if (
            not programs
            or len(programs) > sidecar.max_legal_options
            or not 0 <= selected < len(programs)
            or len(set(programs)) != len(programs)
        ):
            raise PrizePlanLiveFeatureError("factorized legal program menu is malformed")
        rows: list[tuple[float, ...]] = []
        for program in programs:
            if len(program) > sidecar.max_action_program_tokens:
                raise PrizePlanLiveFeatureError("factorized program exceeds v2 ABI")
            if program == tuple(prefix):
                rows.append(stop)
                continue
            if not program or program[:-1] != tuple(prefix):
                raise PrizePlanLiveFeatureError("factorized program does not extend its prefix")
            option_index = program[-1]
            if not 0 <= option_index < len(raw_options):
                raise PrizePlanLiveFeatureError("factorized program references an absent option")
            rows.append(raw_options[option_index])
        if stage_index == 0:
            first_menu = tuple(rows)
        chosen_program = programs[selected]
        selected_features.append(rows[selected])
        selected_indices.append(selected)
        selected_counts.append(len(programs))
        selected_programs.append(chosen_program)
        prefix = list(chosen_program)

    if tuple(prefix) != tuple(action) or first_menu is None:
        raise PrizePlanLiveFeatureError("factorized selected program does not equal replay action")
    return LiveCompleteActionFeatures(
        episode_id=episode_id,
        seat=seat,
        env_step=env_step,
        stage_count=len(stages),
        first_stage_menu=first_menu,
        selected_stage_features=tuple(selected_features),
        selected_option_indices=tuple(selected_indices),
        selected_legal_counts=tuple(selected_counts),
        selected_action_programs=tuple(selected_programs),
    )


__all__ = [
    "LiveCompleteActionFeatures",
    "PrizePlanLiveFeatureError",
    "complete_action_features_from_public_replay",
    "h3_causal_interval_availability",
]
