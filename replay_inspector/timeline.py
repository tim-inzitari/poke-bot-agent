"""Causal, acting-seat decision addresses for archived Kaggle replays."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


class ReplayTimelineError(ValueError):
    """The replay cannot be addressed without guessing."""


def _int_or(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default


def context_name(observation: dict[str, Any]) -> str:
    raw = (observation.get("select") or {}).get("context")
    return "unknown" if raw is None else str(raw)


def is_forced_turn_order(observation: dict[str, Any]) -> bool:
    raw = (observation.get("select") or {}).get("context")
    if isinstance(raw, str):
        return "".join(ch for ch in raw if ch.isalnum()).casefold() == "isfirst"
    return _int_or(raw, -1) == 41


@dataclass(frozen=True)
class DecisionStage:
    step_index: int
    factorized_stage: int
    turn: int
    context: str
    candidates: tuple[tuple[int, ...], ...]
    target_index: int
    recorded_action: tuple[int, ...]
    model_forward_expected: bool

    def to_dict(self, *, include_candidates: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {
            "step_index": self.step_index,
            "factorized_stage": self.factorized_stage,
            "turn": self.turn,
            "context": self.context,
            "candidate_count": len(self.candidates),
            "target_index": self.target_index,
            "recorded_action": list(self.recorded_action),
            "recorded_stage_choice": (
                list(self.candidates[self.target_index])
                if 0 <= self.target_index < len(self.candidates)
                else None
            ),
            "model_forward_expected": self.model_forward_expected,
            "status": (
                "decision"
                if self.model_forward_expected
                else "forced_turn_order_or_runtime_short_circuit"
            ),
        }
        if include_candidates:
            row["candidates"] = [list(candidate) for candidate in self.candidates]
        return row


def _factorized_stages(
    observation: dict[str, Any], action: list[int]
) -> Iterable[tuple[list[list[int]], int]]:
    # Lazy import keeps archive browsing available when the model runtime is not
    # installed on the machine serving replay metadata.
    from poke_bot import features

    return features.factorized_teacher_forcing_stages(observation, action)


def assert_active_select_actor(
    observation: Mapping[str, Any], *, seat: int, step_index: int
) -> None:
    """Require an active select row to corroborate its exact acting seat.

    This is shared with the model-forward reconstruction path so a direct
    backend caller cannot bypass the selector's archive-integrity predicate.
    ``yourIndex`` comes from parsed JSON and must therefore be an actual
    integer, not a bool, numeric string, or coercible value.
    """

    select = observation.get("select")
    if not isinstance(select, Mapping):
        raise ReplayTimelineError(
            f"active selectable row has malformed select at step {step_index}"
        )
    current = observation.get("current")
    actor = current.get("yourIndex") if isinstance(current, Mapping) else None
    if type(actor) is not int or actor != seat:
        raise ReplayTimelineError(
            "active selectable row has missing, malformed, or mismatched "
            f"current.yourIndex at step {step_index}"
        )


def decision_timeline(replay: dict[str, Any], seat: int) -> list[DecisionStage]:
    """List every factorized acting-seat decision in engine order.

    The action for ``steps[i]`` is stored in ``steps[i + 1]`` by Kaggle.  We
    never read the omniscient ``visualize.current`` state here.  A stale
    ``select`` field on an inactive cache row is not a decision; for a row
    that is ``ACTIVE`` and has a select, its masked observation must also
    explicitly identify this exact seat through JSON-integer ``yourIndex``.
    """

    steps = replay.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ReplayTimelineError("replay steps are missing or truncated")
    if seat < 0:
        raise ReplayTimelineError("seat must be non-negative")

    output: list[DecisionStage] = []
    for step_index in range(len(steps) - 1):
        current_rows = steps[step_index]
        next_rows = steps[step_index + 1]
        if not isinstance(current_rows, list) or not isinstance(next_rows, list):
            raise ReplayTimelineError(f"step {step_index} is not seat-indexed")
        if seat >= len(current_rows) or seat >= len(next_rows):
            raise ReplayTimelineError(f"seat {seat} absent at step {step_index}")
        row = current_rows[seat]
        next_row = next_rows[seat]
        if not isinstance(row, dict) or not isinstance(next_row, dict):
            raise ReplayTimelineError(f"seat row is malformed at step {step_index}")
        observation = row.get("observation") or {}
        select = observation.get("select") if isinstance(observation, dict) else None
        if (
            row.get("status") != "ACTIVE"
            or not isinstance(observation, dict)
            or select is None
        ):
            continue
        assert_active_select_actor(observation, seat=seat, step_index=step_index)
        raw_action = next_row.get("action") or []
        if not isinstance(raw_action, list):
            raise ReplayTimelineError(f"action is malformed at step {step_index}")
        try:
            action = [int(value) for value in raw_action]
            stages = list(_factorized_stages(observation, action))
        except (TypeError, ValueError) as exc:
            raise ReplayTimelineError(
                f"illegal recorded action at step {step_index}: {exc}"
            ) from exc
        current = observation.get("current")
        turn = _int_or(current.get("turn") if isinstance(current, Mapping) else None, 0)
        forward_expected = not is_forced_turn_order(observation)
        for stage_index, (candidates, target_index) in enumerate(stages):
            normalized = tuple(
                tuple(int(value) for value in candidate) for candidate in candidates
            )
            target = int(target_index)
            if not (0 <= target < len(normalized)):
                raise ReplayTimelineError(
                    f"target index out of range at step {step_index} stage {stage_index}"
                )
            output.append(
                DecisionStage(
                    step_index=step_index,
                    factorized_stage=stage_index,
                    turn=turn,
                    context=context_name(observation),
                    candidates=normalized,
                    target_index=target,
                    recorded_action=tuple(action),
                    model_forward_expected=forward_expected,
                )
            )
    return output


def find_stage(
    replay: dict[str, Any],
    *,
    seat: int,
    step_index: int,
    factorized_stage: int,
) -> DecisionStage:
    for stage in decision_timeline(replay, seat):
        if (
            stage.step_index == step_index
            and stage.factorized_stage == factorized_stage
        ):
            return stage
    raise KeyError(
        f"decision stage not found: seat={seat} step={step_index} "
        f"stage={factorized_stage}"
    )


def causal_observation(
    replay: dict[str, Any], *, seat: int, step_index: int
) -> dict[str, Any]:
    """Return only the masked observation supplied to the acting agent."""

    steps = replay.get("steps")
    if not isinstance(steps, list) or not (0 <= step_index < len(steps)):
        raise KeyError(f"step not found: {step_index}")
    rows = steps[step_index]
    if not isinstance(rows, list) or not (0 <= seat < len(rows)):
        raise KeyError(f"seat not found: {seat}")
    row = rows[seat]
    if not isinstance(row, dict) or not isinstance(row.get("observation"), dict):
        raise KeyError(f"observation not found: seat={seat} step={step_index}")
    return row["observation"]
