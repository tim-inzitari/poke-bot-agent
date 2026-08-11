"""Observed-label promotion telemetry for the own-deck successor.

This module deliberately reports only what occurred in an expert trajectory.
It never scores an unselected legal action, does not generate a target, and
does not alter a policy or a model route.  The trainer calls it in an explicit
offline/shadow collection mode after the normal causal forward pass.

The fixed output is mergeable across batches: every rate carries its factual
numerator and denominator, Brier scores retain their unnormalised sum, and
multiclass ECE retains fixed-bin sufficient statistics.  That makes an epoch
or checkpoint metric an exact aggregate rather than an average of batch
averages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .own_deck_supervision import (
    OWN_DECK_SUPERVISION_SCHEMA,
    OWN_DECK_SUPERVISION_VERSION,
    TERMINAL_CONVERSION_CLASS_COUNT,
    TERMINAL_CONVERSION_CLASS_SLICE,
    TERMINAL_CONVERSION_OPPONENT_KNOCKOUT_INDEX,
    TERMINAL_CONVERSION_OUTPUT_DIM,
    TERMINAL_CONVERSION_PRIZE_CLOSEOUT_INDEX,
    VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM,
    VISIBLE_TUTOR_COMPLETION_SELECTED_FROM_VISIBLE_DECK_INDEX,
    terminal_conversion_target_mask,
    terminal_conversion_target_vector,
    visible_tutor_completion_target_mask,
    visible_tutor_completion_target_vector,
)

OWN_DECK_PROMOTION_METRICS_SCHEMA = "poke_bot.own_deck_promotion_metrics/v1"
DEFAULT_CLOSEOUT_THRESHOLD = 0.5
DEFAULT_TERMINAL_ECE_BINS = 10
_RECALL_KINDS: tuple[str, ...] = (
    "own_win",
    "prize_closeout",
    "opponent_knockout",
)
_MISSED_CLOSEOUT_BASIS = "policy_top1_vs_observed_expert_selected_option"


class OwnDeckPromotionMetricsError(ValueError):
    """A purported observed-label telemetry batch is malformed."""


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator > 0 else None


def _brier_record(numerator: float, denominator: int) -> dict[str, float | int | None]:
    return {
        "brier_numerator": float(numerator),
        "brier_denominator": int(denominator),
        "brier": (float(numerator) / float(denominator) if denominator > 0 else None),
    }


def _count_record(
    numerator: int, denominator: int, *, value_name: str
) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        value_name: _ratio(int(numerator), int(denominator)),
    }


def _validate_threshold_and_bins(
    *, threshold: float, ece_bins: int
) -> tuple[float, int]:
    if isinstance(threshold, bool):
        raise OwnDeckPromotionMetricsError("closeout threshold must be numeric")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise OwnDeckPromotionMetricsError(
            "closeout threshold must be numeric"
        ) from exc
    if not math.isfinite(threshold_value) or not 0.0 < threshold_value <= 1.0:
        raise OwnDeckPromotionMetricsError(
            "closeout threshold must be finite in (0, 1]"
        )
    if isinstance(ece_bins, bool) or not isinstance(ece_bins, int):
        raise OwnDeckPromotionMetricsError("terminal ECE bin count must be int")
    if not 2 <= int(ece_bins) <= 100:
        raise OwnDeckPromotionMetricsError("terminal ECE bin count must be in [2, 100]")
    return threshold_value, int(ece_bins)


def _family_labels_for_row(
    row: object,
    *,
    family: str,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Return one stage-aligned target family, masking a missing stage row."""

    if family == "terminal_conversion":
        output_dim = TERMINAL_CONVERSION_OUTPUT_DIM
        vector_builder = terminal_conversion_target_vector
        mask_builder = terminal_conversion_target_mask
    elif family == "visible_tutor_completion":
        output_dim = VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM
        vector_builder = visible_tutor_completion_target_vector
        mask_builder = visible_tutor_completion_target_mask
    else:  # pragma: no cover - private callers use only the two fixed families.
        raise OwnDeckPromotionMetricsError(f"unknown supervision family {family!r}")
    if row is None:
        return (0.0,) * output_dim, (False,) * output_dim
    if not isinstance(row, Mapping):
        raise OwnDeckPromotionMetricsError("supervision row must be a mapping")
    if (
        row.get("schema") != OWN_DECK_SUPERVISION_SCHEMA
        or row.get("version") != OWN_DECK_SUPERVISION_VERSION
    ):
        raise OwnDeckPromotionMetricsError("supervision row schema mismatch")
    labels = row.get(family)
    if not isinstance(labels, Mapping):
        raise OwnDeckPromotionMetricsError(
            f"supervision row lacks {family} target family"
        )
    try:
        target = tuple(float(value) for value in vector_builder(labels))
        mask = tuple(bool(value) for value in mask_builder(labels))
    except (TypeError, ValueError, KeyError) as exc:
        raise OwnDeckPromotionMetricsError("supervision targets are malformed") from exc
    if (
        len(target) != output_dim
        or len(mask) != output_dim
        or not all(math.isfinite(value) for value in target)
    ):
        raise OwnDeckPromotionMetricsError("supervision target shape is invalid")
    if family == "visible_tutor_completion":
        return target, mask
    class_start = int(TERMINAL_CONVERSION_CLASS_SLICE.start or 0)
    class_stop = int(TERMINAL_CONVERSION_CLASS_SLICE.stop or 0)
    categorical_mask = mask[class_start:class_stop]
    if any(categorical_mask) and not all(categorical_mask):
        raise OwnDeckPromotionMetricsError(
            "terminal class target must be wholly masked or wholly present"
        )
    if all(categorical_mask):
        categorical_target = target[class_start:class_stop]
        if any(
            value < 0.0 or value > 1.0 for value in categorical_target
        ) or not math.isclose(sum(categorical_target), 1.0, abs_tol=1e-6):
            raise OwnDeckPromotionMetricsError("terminal class target is not one-hot")
    for index, masked in enumerate(mask):
        if index < class_stop or not masked:
            continue
        if not 0.0 <= target[index] <= 1.0:
            raise OwnDeckPromotionMetricsError(
                "terminal scalar target is outside [0, 1]"
            )
    return target, mask


def _terminal_ece(
    *,
    bin_rows: Sequence[int],
    bin_confidence_sum: Sequence[float],
    bin_correct_sum: Sequence[int],
) -> float | None:
    total = sum(int(value) for value in bin_rows)
    if total <= 0:
        return None
    error = 0.0
    for rows, confidence_sum, correct_sum in zip(
        bin_rows, bin_confidence_sum, bin_correct_sum
    ):
        if int(rows) <= 0:
            continue
        confidence = float(confidence_sum) / float(rows)
        accuracy = float(correct_sum) / float(rows)
        error += abs(accuracy - confidence) * float(rows)
    return error / float(total)


def collect_own_deck_promotion_metrics(
    *,
    policy_logits: torch.Tensor,
    selected_indices: torch.Tensor,
    terminal_logits: torch.Tensor,
    visible_tutor_supervision_rows: Sequence[object] | None = None,
    terminal_conversion_supervision_rows: Sequence[object] | None = None,
    supervision_rows: Sequence[object] | None = None,
    threshold: float = DEFAULT_CLOSEOUT_THRESHOLD,
    ece_bins: int = DEFAULT_TERMINAL_ECE_BINS,
) -> dict[str, Any]:
    """Collect factual successor telemetry for selected expert actions.

    ``policy_logits`` is used only for agreement with the action actually
    selected from an observed tutor menu. ``terminal_logits`` is sampled only
    at the selected expert option. The two supervision arrays are deliberately
    separate: the factorized tutor label is attached to its actual card stage,
    while a terminal conversion label is attached to the completed-action
    stage. Thus every denominator below names an observed, masked target and
    never assumes an outcome for another legal option.

    The multiclass Brier numerator is the sum, over labeled selected actions,
    of the four-class squared-error vector.  ECE uses fixed confidence bins and
    exposes the sufficient statistics needed for exact cross-batch merging.
    Selected-option recall applies ``threshold`` only to the terminal head.
    ``missed_expert_closeout`` separately counts direct policy top-1 choices
    that disagree with the observed expert option on a factual closeout row.
    """

    threshold_value, ece_bin_count = _validate_threshold_and_bins(
        threshold=threshold, ece_bins=ece_bins
    )
    if policy_logits.ndim != 2:
        raise OwnDeckPromotionMetricsError("policy logits must be [rows, options]")
    if terminal_logits.ndim != 3:
        raise OwnDeckPromotionMetricsError(
            "terminal logits must be [rows, options, outputs]"
        )
    rows, options = policy_logits.shape
    if (
        terminal_logits.shape[0] != rows
        or terminal_logits.shape[1] != options
        or terminal_logits.shape[2] != TERMINAL_CONVERSION_OUTPUT_DIM
    ):
        raise OwnDeckPromotionMetricsError(
            "terminal logits do not align with policy rows"
        )
    if selected_indices.ndim != 1 or selected_indices.numel() != rows:
        raise OwnDeckPromotionMetricsError("selected indices do not align with rows")
    if supervision_rows is not None:
        if (
            visible_tutor_supervision_rows is not None
            or terminal_conversion_supervision_rows is not None
        ):
            raise OwnDeckPromotionMetricsError(
                "legacy combined supervision cannot mix with stage-aligned rows"
            )
        visible_tutor_supervision_rows = supervision_rows
        terminal_conversion_supervision_rows = supervision_rows
    if (
        visible_tutor_supervision_rows is None
        or terminal_conversion_supervision_rows is None
    ):
        raise OwnDeckPromotionMetricsError(
            "stage-aligned tutor and terminal supervision rows are required"
        )
    if (
        len(visible_tutor_supervision_rows) != rows
        or len(terminal_conversion_supervision_rows) != rows
    ):
        raise OwnDeckPromotionMetricsError("supervision rows do not align with logits")
    if rows <= 0 or options <= 0:
        raise OwnDeckPromotionMetricsError(
            "promotion metrics require nonempty option rows"
        )
    selected = selected_indices.detach().to(
        device=policy_logits.device, dtype=torch.long
    )
    if bool(((selected < 0) | (selected >= options)).any().item()):
        raise OwnDeckPromotionMetricsError("selected option index is out of range")

    terminal_targets: list[tuple[float, ...]] = []
    terminal_masks: list[tuple[bool, ...]] = []
    tutor_targets: list[tuple[float, ...]] = []
    tutor_masks: list[tuple[bool, ...]] = []
    for row in terminal_conversion_supervision_rows:
        terminal_target, terminal_mask = _family_labels_for_row(
            row, family="terminal_conversion"
        )
        terminal_targets.append(terminal_target)
        terminal_masks.append(terminal_mask)
    for row in visible_tutor_supervision_rows:
        tutor_target, tutor_mask = _family_labels_for_row(
            row, family="visible_tutor_completion"
        )
        tutor_targets.append(tutor_target)
        tutor_masks.append(tutor_mask)

    # Collection is always detached. It adds neither a loss term nor a path
    # from the metrics back into the optimizer, and does not inspect any route.
    with torch.no_grad():
        policy = policy_logits.detach().float()
        terminal = terminal_logits.detach().float()
        if not bool(
            torch.isfinite(policy).logical_or(torch.isneginf(policy)).all().item()
        ):
            raise OwnDeckPromotionMetricsError(
                "policy logits contain non-finite values"
            )
        if not bool(torch.isfinite(terminal).all().item()):
            raise OwnDeckPromotionMetricsError(
                "terminal logits contain non-finite values"
            )
        row_index = torch.arange(rows, device=policy.device)
        policy_top1 = policy.argmax(dim=1)
        selected_terminal = terminal[row_index, selected]
        class_start = int(TERMINAL_CONVERSION_CLASS_SLICE.start or 0)
        class_stop = int(TERMINAL_CONVERSION_CLASS_SLICE.stop or 0)
        class_probabilities = torch.softmax(
            selected_terminal[:, class_start:class_stop], dim=-1
        )
        scalar_probabilities = torch.sigmoid(
            selected_terminal[
                :,
                [
                    TERMINAL_CONVERSION_PRIZE_CLOSEOUT_INDEX,
                    TERMINAL_CONVERSION_OPPONENT_KNOCKOUT_INDEX,
                ],
            ]
        )

    tutor_denominator = 0
    tutor_numerator = 0
    terminal_brier_numerator = 0.0
    terminal_brier_denominator = 0
    ece_rows = [0] * ece_bin_count
    ece_confidence_sum = [0.0] * ece_bin_count
    ece_correct_sum = [0] * ece_bin_count
    scalar_brier_numerator = {
        "prize_closeout": 0.0,
        "opponent_knockout": 0.0,
    }
    scalar_brier_denominator = {
        "prize_closeout": 0,
        "opponent_knockout": 0,
    }
    recall_numerator = {kind: 0 for kind in _RECALL_KINDS}
    recall_denominator = {kind: 0 for kind in _RECALL_KINDS}
    closeout_denominator = 0
    closeout_numerator = 0
    closeout_by_kind_numerator = {kind: 0 for kind in _RECALL_KINDS}
    closeout_by_kind_denominator = {kind: 0 for kind in _RECALL_KINDS}

    class_targets = [
        terminal_target[class_start:class_stop] for terminal_target in terminal_targets
    ]
    for index in range(rows):
        tutor_target = tutor_targets[index]
        tutor_mask = tutor_masks[index]
        if (
            len(tutor_target)
            > VISIBLE_TUTOR_COMPLETION_SELECTED_FROM_VISIBLE_DECK_INDEX
            and len(tutor_mask)
            > VISIBLE_TUTOR_COMPLETION_SELECTED_FROM_VISIBLE_DECK_INDEX
            and tutor_mask[VISIBLE_TUTOR_COMPLETION_SELECTED_FROM_VISIBLE_DECK_INDEX]
            and tutor_target[VISIBLE_TUTOR_COMPLETION_SELECTED_FROM_VISIBLE_DECK_INDEX]
            >= 0.5
        ):
            tutor_denominator += 1
            tutor_numerator += int(
                int(policy_top1[index].item()) == int(selected[index].item())
            )

        terminal_target = terminal_targets[index]
        terminal_mask = terminal_masks[index]
        class_mask = terminal_mask[class_start:class_stop]
        class_target = class_targets[index]
        has_class = bool(class_mask) and all(class_mask)
        if any(class_mask) and not has_class:
            raise OwnDeckPromotionMetricsError(
                "terminal class target must be wholly masked or wholly present"
            )
        if has_class:
            actual_class = max(
                range(TERMINAL_CONVERSION_CLASS_COUNT), key=class_target.__getitem__
            )
            probabilities = class_probabilities[index]
            target_tensor = torch.tensor(
                class_target,
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            terminal_brier_numerator += float(
                (probabilities - target_tensor).square().sum().item()
            )
            terminal_brier_denominator += 1
            confidence, predicted_class = probabilities.max(dim=0)
            bin_index = min(
                int(float(confidence.item()) * ece_bin_count),
                ece_bin_count - 1,
            )
            ece_rows[bin_index] += 1
            ece_confidence_sum[bin_index] += float(confidence.item())
            ece_correct_sum[bin_index] += int(
                int(predicted_class.item()) == int(actual_class)
            )
            if actual_class == 1:  # TERMINAL_CONVERSION_CLASSES[1] == own_win
                recall_denominator["own_win"] += 1
                recall_numerator["own_win"] += int(
                    float(probabilities[1].item()) >= threshold_value
                )

        scalar_specs = (
            (
                "prize_closeout",
                TERMINAL_CONVERSION_PRIZE_CLOSEOUT_INDEX,
                0,
            ),
            (
                "opponent_knockout",
                TERMINAL_CONVERSION_OPPONENT_KNOCKOUT_INDEX,
                1,
            ),
        )
        positive_kinds: list[str] = []
        if (
            has_class
            and max(
                range(TERMINAL_CONVERSION_CLASS_COUNT), key=class_target.__getitem__
            )
            == 1
        ):
            positive_kinds.append("own_win")
        for kind, target_index, probability_index in scalar_specs:
            if not terminal_mask[target_index]:
                continue
            target = float(terminal_target[target_index])
            probability = float(scalar_probabilities[index, probability_index].item())
            scalar_brier_numerator[kind] += (probability - target) ** 2
            scalar_brier_denominator[kind] += 1
            if target >= 0.5:
                recall_denominator[kind] += 1
                hit = probability >= threshold_value
                recall_numerator[kind] += int(hit)
                positive_kinds.append(kind)
        if positive_kinds:
            closeout_denominator += 1
            policy_missed_expert = int(
                int(policy_top1[index].item()) != int(selected[index].item())
            )
            # This asks whether the policy would have failed to choose the
            # observed expert closeout.  It is intentionally independent of
            # the terminal-head score threshold used by selected-option recall.
            closeout_numerator += policy_missed_expert
            for kind in positive_kinds:
                closeout_by_kind_denominator[kind] += 1
                closeout_by_kind_numerator[kind] += policy_missed_expert

    recall = {
        kind: _count_record(
            recall_numerator[kind], recall_denominator[kind], value_name="recall"
        )
        for kind in _RECALL_KINDS
    }
    missed_by_kind = {
        kind: _count_record(
            closeout_by_kind_numerator[kind],
            closeout_by_kind_denominator[kind],
            value_name="miss_rate",
        )
        for kind in _RECALL_KINDS
    }
    return {
        "schema": OWN_DECK_PROMOTION_METRICS_SCHEMA,
        "collection": {
            "observed_expert_labels_only": True,
            "gradient_contribution": False,
            "runtime_route_authority": False,
        },
        "closeout_threshold": threshold_value,
        "terminal_ece_bins": ece_bin_count,
        "visible_tutor_observed_menu_expert_top1": _count_record(
            tutor_numerator, tutor_denominator, value_name="agreement"
        ),
        "selected_option_factual_recall": recall,
        "terminal_multiclass": {
            **_brier_record(terminal_brier_numerator, terminal_brier_denominator),
            "ece": _terminal_ece(
                bin_rows=ece_rows,
                bin_confidence_sum=ece_confidence_sum,
                bin_correct_sum=ece_correct_sum,
            ),
            "ece_bin_rows": ece_rows,
            "ece_bin_confidence_sum": ece_confidence_sum,
            "ece_bin_correct_sum": ece_correct_sum,
        },
        "terminal_scalar_brier": {
            kind: _brier_record(
                scalar_brier_numerator[kind], scalar_brier_denominator[kind]
            )
            for kind in ("prize_closeout", "opponent_knockout")
        },
        "missed_expert_closeout": {
            "basis": _MISSED_CLOSEOUT_BASIS,
            **_count_record(
                closeout_numerator,
                closeout_denominator,
                value_name="miss_rate",
            ),
            "by_kind": missed_by_kind,
        },
    }


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OwnDeckPromotionMetricsError(f"{label} must be a mapping")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise OwnDeckPromotionMetricsError(f"{label} must be a nonnegative int")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OwnDeckPromotionMetricsError(
            f"{label} must be a nonnegative int"
        ) from exc
    if parsed < 0 or parsed != value:
        raise OwnDeckPromotionMetricsError(f"{label} must be a nonnegative int")
    return parsed


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise OwnDeckPromotionMetricsError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OwnDeckPromotionMetricsError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise OwnDeckPromotionMetricsError(f"{label} must be finite")
    return parsed


def _metric_count(
    record: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, int]:
    numerator = _nonnegative_int(record.get("numerator"), label=f"{label}.numerator")
    denominator = _nonnegative_int(
        record.get("denominator"), label=f"{label}.denominator"
    )
    if numerator > denominator:
        raise OwnDeckPromotionMetricsError(f"{label}.numerator exceeds denominator")
    return numerator, denominator


def _metric_brier(
    record: Mapping[str, Any],
    *,
    label: str,
) -> tuple[float, int]:
    numerator = _finite_float(
        record.get("brier_numerator"), label=f"{label}.brier_numerator"
    )
    denominator = _nonnegative_int(
        record.get("brier_denominator"), label=f"{label}.brier_denominator"
    )
    if numerator < 0.0:
        raise OwnDeckPromotionMetricsError(
            f"{label}.brier_numerator must be nonnegative"
        )
    return numerator, denominator


def merge_own_deck_promotion_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge exact sufficient statistics from compatible metric batches.

    An empty collection remains absent rather than fabricating a zero-valued
    evaluation result.  Every populated record must use the same threshold and
    ECE binning, so a merged epoch/checkpoint remains auditable.
    """

    if not records:
        return {}
    first = _mapping(records[0], label="promotion metric")
    if first.get("schema") != OWN_DECK_PROMOTION_METRICS_SCHEMA:
        raise OwnDeckPromotionMetricsError("promotion metric schema mismatch")
    threshold, ece_bins = _validate_threshold_and_bins(
        threshold=first.get("closeout_threshold"),
        ece_bins=first.get("terminal_ece_bins"),
    )
    tutor_numerator = 0
    tutor_denominator = 0
    recall_numerator = {kind: 0 for kind in _RECALL_KINDS}
    recall_denominator = {kind: 0 for kind in _RECALL_KINDS}
    terminal_brier_numerator = 0.0
    terminal_brier_denominator = 0
    ece_rows = [0] * ece_bins
    ece_confidence_sum = [0.0] * ece_bins
    ece_correct_sum = [0] * ece_bins
    scalar_brier_numerator = {
        "prize_closeout": 0.0,
        "opponent_knockout": 0.0,
    }
    scalar_brier_denominator = {
        "prize_closeout": 0,
        "opponent_knockout": 0,
    }
    closeout_numerator = 0
    closeout_denominator = 0
    closeout_by_kind_numerator = {kind: 0 for kind in _RECALL_KINDS}
    closeout_by_kind_denominator = {kind: 0 for kind in _RECALL_KINDS}

    for raw in records:
        record = _mapping(raw, label="promotion metric")
        if record.get("schema") != OWN_DECK_PROMOTION_METRICS_SCHEMA:
            raise OwnDeckPromotionMetricsError("promotion metric schema mismatch")
        actual_threshold, actual_bins = _validate_threshold_and_bins(
            threshold=record.get("closeout_threshold"),
            ece_bins=record.get("terminal_ece_bins"),
        )
        if actual_threshold != threshold or actual_bins != ece_bins:
            raise OwnDeckPromotionMetricsError(
                "promotion metric threshold or ECE bins changed within merge"
            )
        collection = _mapping(record.get("collection"), label="collection")
        if collection != {
            "observed_expert_labels_only": True,
            "gradient_contribution": False,
            "runtime_route_authority": False,
        }:
            raise OwnDeckPromotionMetricsError(
                "promotion metric collection contract changed"
            )
        numerator, denominator = _metric_count(
            _mapping(
                record.get("visible_tutor_observed_menu_expert_top1"),
                label="visible tutor metric",
            ),
            label="visible tutor metric",
        )
        tutor_numerator += numerator
        tutor_denominator += denominator
        recall = _mapping(
            record.get("selected_option_factual_recall"), label="selected recall"
        )
        for kind in _RECALL_KINDS:
            numerator, denominator = _metric_count(
                _mapping(recall.get(kind), label=f"selected recall.{kind}"),
                label=f"selected recall.{kind}",
            )
            recall_numerator[kind] += numerator
            recall_denominator[kind] += denominator
        terminal = _mapping(
            record.get("terminal_multiclass"), label="terminal multiclass"
        )
        brier_numerator, brier_denominator = _metric_brier(
            terminal, label="terminal multiclass"
        )
        terminal_brier_numerator += brier_numerator
        terminal_brier_denominator += brier_denominator
        row_values = list(terminal.get("ece_bin_rows") or ())
        confidence_values = list(terminal.get("ece_bin_confidence_sum") or ())
        correct_values = list(terminal.get("ece_bin_correct_sum") or ())
        if not (
            len(row_values) == len(confidence_values) == len(correct_values) == ece_bins
        ):
            raise OwnDeckPromotionMetricsError("terminal ECE bins are malformed")
        for index in range(ece_bins):
            rows = _nonnegative_int(row_values[index], label="terminal ECE rows")
            confidence = _finite_float(
                confidence_values[index], label="terminal ECE confidence"
            )
            correct = _nonnegative_int(
                correct_values[index], label="terminal ECE correct"
            )
            if confidence < 0.0 or confidence > float(rows) or correct > rows:
                raise OwnDeckPromotionMetricsError("terminal ECE bin is invalid")
            ece_rows[index] += rows
            ece_confidence_sum[index] += confidence
            ece_correct_sum[index] += correct
        scalar = _mapping(record.get("terminal_scalar_brier"), label="terminal scalar")
        for kind in ("prize_closeout", "opponent_knockout"):
            numerator, denominator = _metric_brier(
                _mapping(scalar.get(kind), label=f"terminal scalar.{kind}"),
                label=f"terminal scalar.{kind}",
            )
            scalar_brier_numerator[kind] += numerator
            scalar_brier_denominator[kind] += denominator
        missed = _mapping(record.get("missed_expert_closeout"), label="missed closeout")
        if missed.get("basis") != _MISSED_CLOSEOUT_BASIS:
            raise OwnDeckPromotionMetricsError("missed-closeout basis mismatch")
        numerator, denominator = _metric_count(missed, label="missed closeout")
        closeout_numerator += numerator
        closeout_denominator += denominator
        missed_by_kind = _mapping(
            missed.get("by_kind"), label="missed closeout by kind"
        )
        for kind in _RECALL_KINDS:
            kind_numerator, kind_denominator = _metric_count(
                _mapping(missed_by_kind.get(kind), label=f"missed closeout.{kind}"),
                label=f"missed closeout.{kind}",
            )
            # The two metric families share factual positive-label support,
            # but differ intentionally in their numerators: selected-option
            # recall thresholds the terminal head, while this gate observes
            # whether direct policy top-1 matched the expert choice.
            selected_kind = _mapping(recall.get(kind), label=f"selected recall.{kind}")
            _selected_numerator, selected_denominator = _metric_count(
                selected_kind, label=f"selected recall.{kind}"
            )
            if kind_denominator != selected_denominator:
                raise OwnDeckPromotionMetricsError(
                    "missed-closeout per-kind support does not match selected recall"
                )
            closeout_by_kind_numerator[kind] += kind_numerator
            closeout_by_kind_denominator[kind] += kind_denominator

    if sum(ece_rows) != terminal_brier_denominator:
        raise OwnDeckPromotionMetricsError(
            "terminal ECE support does not match terminal Brier denominator"
        )

    recall = {
        kind: _count_record(
            recall_numerator[kind], recall_denominator[kind], value_name="recall"
        )
        for kind in _RECALL_KINDS
    }
    missed_by_kind = {
        kind: _count_record(
            closeout_by_kind_numerator[kind],
            closeout_by_kind_denominator[kind],
            value_name="miss_rate",
        )
        for kind in _RECALL_KINDS
    }
    return {
        "schema": OWN_DECK_PROMOTION_METRICS_SCHEMA,
        "collection": {
            "observed_expert_labels_only": True,
            "gradient_contribution": False,
            "runtime_route_authority": False,
        },
        "closeout_threshold": threshold,
        "terminal_ece_bins": ece_bins,
        "visible_tutor_observed_menu_expert_top1": _count_record(
            tutor_numerator, tutor_denominator, value_name="agreement"
        ),
        "selected_option_factual_recall": recall,
        "terminal_multiclass": {
            **_brier_record(terminal_brier_numerator, terminal_brier_denominator),
            "ece": _terminal_ece(
                bin_rows=ece_rows,
                bin_confidence_sum=ece_confidence_sum,
                bin_correct_sum=ece_correct_sum,
            ),
            "ece_bin_rows": ece_rows,
            "ece_bin_confidence_sum": ece_confidence_sum,
            "ece_bin_correct_sum": ece_correct_sum,
        },
        "terminal_scalar_brier": {
            kind: _brier_record(
                scalar_brier_numerator[kind], scalar_brier_denominator[kind]
            )
            for kind in ("prize_closeout", "opponent_knockout")
        },
        "missed_expert_closeout": {
            "basis": _MISSED_CLOSEOUT_BASIS,
            **_count_record(
                closeout_numerator,
                closeout_denominator,
                value_name="miss_rate",
            ),
            "by_kind": missed_by_kind,
        },
    }


__all__ = [
    "DEFAULT_CLOSEOUT_THRESHOLD",
    "DEFAULT_TERMINAL_ECE_BINS",
    "OWN_DECK_PROMOTION_METRICS_SCHEMA",
    "OwnDeckPromotionMetricsError",
    "collect_own_deck_promotion_metrics",
    "merge_own_deck_promotion_metrics",
]
