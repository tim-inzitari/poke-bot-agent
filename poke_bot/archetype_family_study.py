"""Deterministic, training-ineligible Marnie family shadow-study contracts."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .archetype_loss_contract import canonical_residual_weights
from .specialist_archetype_family import singular_package_variant, validate_manifest


SCHEMA = "poke_bot.marnie_archetype_family_shadow_study/v1"
PLAN_SCHEMA = "poke_bot.marnie_archetype_family_shadow_plan/v1"
SELECTED_VECTOR_SCHEMA = "poke_bot.archetype_loss_vector/v1"
MONITOR_SCHEMA = "poke_bot.marnie_archetype_family_post_activation_monitor/v1"


class FamilyStudyError(ValueError):
    pass


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def antithetic_spsa(
    weights: Mapping[str, float], *, round_index: int, seed_book: str
) -> dict[str, Any]:
    """Return the locked +/- log-space candidates for one of two rounds."""
    if round_index not in (1, 2):
        raise FamilyStudyError("Marnie permits exactly two SPSA rounds")
    magnitude = 0.20 if round_index == 1 else 0.10
    rng = random.Random(int(hashlib.sha256(f"{seed_book}:{round_index}".encode()).hexdigest(), 16))
    direction = {name: rng.choice((-1, 1)) for name in sorted(weights)}
    candidates: dict[str, dict[str, float]] = {"plus": {}, "minus": {}}
    for name, raw in sorted(weights.items()):
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise FamilyStudyError("SPSA weights must be finite and positive")
        candidates["plus"][name] = value * math.exp(direction[name] * math.log1p(magnitude))
        candidates["minus"][name] = value * math.exp(-direction[name] * math.log1p(magnitude))
    return {
        "round": round_index,
        "magnitude": magnitude,
        "seed_book": seed_book,
        "direction": direction,
        "candidates": candidates,
        "digest": _digest(candidates),
    }


def paired_panel_schedule(
    cluster_ids: Sequence[str], opponent_ids: Sequence[str], *, pairs_per_cell: int, seed: str
) -> list[dict[str, Any]]:
    """Materialize deterministic paired units, balanced by seat."""
    if pairs_per_cell <= 0 or not cluster_ids or not opponent_ids:
        raise FamilyStudyError("paired panel dimensions must be positive")
    rows = [
        {"cluster_id": cluster, "opponent_id": opponent, "pair": pair, "treatment_seat": pair % 2}
        for cluster in sorted(cluster_ids)
        for opponent in sorted(opponent_ids)
        for pair in range(int(pairs_per_cell))
    ]
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    rng.shuffle(rows)
    return rows


def _one_sided_hoeffding_lower(values: Sequence[float], confidence: float) -> float:
    if not values or not 0.0 < float(confidence) < 1.0:
        raise FamilyStudyError("invalid paired confidence inputs")
    if any(not math.isfinite(float(value)) or abs(float(value)) > 1.0 for value in values):
        raise FamilyStudyError("paired score deltas must be finite in [-1,1]")
    mean = sum(float(value) for value in values) / len(values)
    radius = math.sqrt(2.0 * math.log(1.0 / (1.0 - confidence)) / len(values))
    return max(-1.0, mean - radius)


def _cluster_variants(
    manifest: Mapping[str, Any], split: str
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in manifest.get("variants") or ():
        if str(row.get("split")) == split:
            grouped[str(row["cluster_id"])].append(str(row["variant_id"]))
    return {
        cluster_id: tuple(sorted(variant_ids))
        for cluster_id, variant_ids in sorted(grouped.items())
    }


def _panel_rows(
    *,
    phase: str,
    cluster_variants: Mapping[str, Sequence[str]],
    opponent_ids: Sequence[str],
    pairs_per_cell: int,
    seed: str,
) -> list[dict[str, Any]]:
    base = paired_panel_schedule(
        tuple(cluster_variants),
        tuple(opponent_ids),
        pairs_per_cell=int(pairs_per_cell),
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    for row in base:
        variants = tuple(cluster_variants[str(row["cluster_id"])])
        if not variants:
            raise FamilyStudyError("panel cluster has no exact observed variants")
        variant_id = variants[int(row["pair"]) % len(variants)]
        schedule_id = (
            f"{phase}:{row['cluster_id']}:{row['opponent_id']}:"
            f"{int(row['pair']):03d}"
        )
        rows.append(
            {
                **row,
                "phase": phase,
                "variant_id": variant_id,
                "schedule_id": schedule_id,
                "requested_seed": int(
                    hashlib.sha256(f"{seed}:{schedule_id}".encode()).hexdigest()[:15],
                    16,
                ),
                "training_eligible": False,
                "replay_eligible": False,
                "formal_gate": False,
            }
        )
    return rows


def family_shadow_plan(
    manifest: Mapping[str, Any], opponent_ids: Sequence[str], *, seed_book: str
) -> dict[str, Any]:
    """Freeze exact 1,020/4,284/1,020 paired family evaluation schedules."""
    validated = validate_manifest(manifest, require_activation_ready=True)
    opponents = tuple(sorted(str(value) for value in opponent_ids))
    if len(opponents) != 17 or len(set(opponents)) != 17 or not all(opponents):
        raise FamilyStudyError("family study requires the exact 17-opponent roster")
    development = _cluster_variants(validated, "dev")
    locked = _cluster_variants(validated, "locked")
    if len(development) < 3 or len(locked) < 3:
        raise FamilyStudyError(
            "family study requires at least three development and three locked clusters"
        )
    def _select_three(
        rows: Mapping[str, Sequence[str]], label: str
    ) -> dict[str, Sequence[str]]:
        selected = sorted(
            rows,
            key=lambda cluster_id: hashlib.sha256(
                f"{seed_book}:{label}:{cluster_id}".encode()
            ).hexdigest(),
        )[:3]
        return {cluster_id: rows[cluster_id] for cluster_id in sorted(selected)}

    development_panel = _select_three(development, "development")
    locked_panel = _select_three(locked, "locked")
    package = singular_package_variant(validated)
    package_cluster = {str(package["cluster_id"]): (str(package["variant_id"]),)}
    panels = {
        "development": _panel_rows(
            phase="development",
            cluster_variants=development_panel,
            opponent_ids=opponents,
            pairs_per_cell=20,
            seed=f"{seed_book}:development",
        ),
        "locked": _panel_rows(
            phase="locked",
            cluster_variants=locked_panel,
            opponent_ids=opponents,
            pairs_per_cell=84,
            seed=f"{seed_book}:locked",
        ),
        "package": _panel_rows(
            phase="package",
            cluster_variants=package_cluster,
            opponent_ids=opponents,
            pairs_per_cell=60,
            seed=f"{seed_book}:package",
        ),
    }
    expected = {"development": 1020, "locked": 4284, "package": 1020}
    if {name: len(rows) for name, rows in panels.items()} != expected:
        raise FamilyStudyError("family study panel cardinality changed")
    payload = {
        "schema": PLAN_SCHEMA,
        "family_id": str(validated["family_id"]),
        "manifest_sha256": str(validated["artifact_sha256"]),
        "seed_book": str(seed_book),
        "opponent_ids": list(opponents),
        "selected_cluster_ids": {
            "development": sorted(development_panel),
            "locked": sorted(locked_panel),
        },
        "available_cluster_counts": {
            "development": len(development),
            "locked": len(locked),
        },
        "panels": panels,
        "panel_units": expected,
        "engine_seedable": False,
        "paired_contract": "same_requested_seed_opponent_list_and_seat_v1",
        "training_eligible": False,
        "replay_eligible": False,
    }
    payload["artifact_sha256"] = _digest(payload)
    return payload


def _score(row: Mapping[str, Any]) -> float:
    value = float(row.get("score", math.nan))
    if value not in {0.0, 0.5, 1.0}:
        raise FamilyStudyError("study score must be draw-aware 0/0.5/1")
    return value


def _exact_pairs(
    rows: Iterable[Mapping[str, Any]],
    *,
    left: str,
    right: str,
    expected_units: int,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        schedule_id = str(row.get("schedule_id") or "")
        treatment = str(row.get("treatment") or "")
        if not schedule_id or treatment not in {left, right}:
            raise FamilyStudyError("unexpected study treatment or schedule identity")
        if treatment in grouped[schedule_id]:
            raise FamilyStudyError("duplicate study treatment row")
        grouped[schedule_id][treatment] = row
    if len(grouped) != int(expected_units) or any(
        set(cell) != {left, right} for cell in grouped.values()
    ):
        raise FamilyStudyError("paired study rows are incomplete")
    pairs = []
    for schedule_id in sorted(grouped):
        left_row, right_row = grouped[schedule_id][left], grouped[schedule_id][right]
        identity_fields = (
            "phase", "cluster_id", "variant_id", "opponent_id",
            "requested_seed", "treatment_seat",
        )
        if any(left_row.get(key) != right_row.get(key) for key in identity_fields):
            raise FamilyStudyError("paired study schedule identity drifted")
        for row in (left_row, right_row):
            if (
                row.get("training_eligible") is not False
                or row.get("replay_eligible") is not False
                or row.get("formal_gate") is not False
            ):
                raise FamilyStudyError("evaluation row escaped training isolation")
        pairs.append((left_row, right_row))
    return pairs


def compile_development_round(
    rows: Sequence[Mapping[str, Any]], *, round_index: int
) -> dict[str, Any]:
    pairs = _exact_pairs(rows, left="plus", right="minus", expected_units=1020)
    valid_pairs = [
        (left, right)
        for left, right in pairs
        if not bool(left.get("invalid")) and not bool(right.get("invalid"))
    ]
    if len(valid_pairs) != len(pairs):
        return {
            "round": int(round_index),
            "status": "failed",
            "reason": "invalid_or_crashed_development_unit",
            "paired_units": len(pairs),
        }
    plus_minus = [_score(left) - _score(right) for left, right in valid_pairs]
    minus_plus = [-value for value in plus_minus]
    plus_lower = _one_sided_hoeffding_lower(plus_minus, 0.95)
    minus_lower = _one_sided_hoeffding_lower(minus_plus, 0.95)
    selected = "plus" if plus_lower > 0.0 else "minus" if minus_lower > 0.0 else None
    return {
        "round": int(round_index),
        "status": "conclusive" if selected else "inconclusive",
        "selected_direction": selected,
        "paired_units": len(valid_pairs),
        "plus_minus_delta": sum(plus_minus) / len(plus_minus),
        "plus_minus_lb95": plus_lower,
        "minus_plus_lb95": minus_lower,
        "confidence_method": "one_sided_hoeffding_matched_draw_aware_delta",
    }


def compile_activation_metrics(
    *,
    locked_rows: Sequence[Mapping[str, Any]],
    package_rows: Sequence[Mapping[str, Any]],
    gradient_diagnostics: Mapping[str, Any],
    policy_drift: Mapping[str, Any],
) -> dict[str, Any]:
    locked = _exact_pairs(
        locked_rows, left="candidate", right="parent", expected_units=4284
    )
    package = _exact_pairs(
        package_rows, left="candidate", right="parent", expected_units=1020
    )
    locked_valid = [
        (candidate, parent)
        for candidate, parent in locked
        if not bool(candidate.get("invalid")) and not bool(parent.get("invalid"))
    ]
    package_valid = [
        (candidate, parent)
        for candidate, parent in package
        if not bool(candidate.get("invalid")) and not bool(parent.get("invalid"))
    ]
    locked_delta = [_score(candidate) - _score(parent) for candidate, parent in locked_valid]
    package_delta = [_score(candidate) - _score(parent) for candidate, parent in package_valid]
    per_list_values: dict[str, list[float]] = defaultdict(list)
    for candidate, parent in locked_valid:
        per_list_values[str(candidate["variant_id"])].append(
            _score(candidate) - _score(parent)
        )
    per_list = {
        variant_id: {
            "pairs": len(values),
            "delta": sum(values) / len(values),
            "delta_lb90": _one_sided_hoeffding_lower(values, 0.90),
        }
        for variant_id, values in sorted(per_list_values.items())
    }
    list_means = sorted(
        ((row["delta"], variant_id) for variant_id, row in per_list.items())
    )
    tail_count = max(1, math.ceil(0.20 * len(list_means)))
    tail_ids = {variant_id for _delta, variant_id in list_means[:tail_count]}
    tail_values = [
        _score(candidate) - _score(parent)
        for candidate, parent in locked_valid
        if str(candidate["variant_id"]) in tail_ids
    ]
    invalid_candidate = sum(bool(candidate.get("invalid")) for candidate, _parent in [*locked, *package])
    invalid_parent = sum(bool(parent.get("invalid")) for _candidate, parent in [*locked, *package])
    invalid_total = len(locked) + len(package)
    metrics = {
        "macro_win_rate_improvement": (
            sum(row["delta"] for row in per_list.values()) / len(per_list)
            if per_list else -1.0
        ),
        "macro_win_rate_improvement_lb95": (
            _one_sided_hoeffding_lower(locked_delta, 0.95)
            if locked_delta else -1.0
        ),
        "current_package_delta": (
            sum(package_delta) / len(package_delta) if package_delta else -1.0
        ),
        "current_package_delta_lb95": (
            _one_sided_hoeffding_lower(package_delta, 0.95)
            if package_delta else -1.0
        ),
        "cvar20_variant_ids": sorted(tail_ids),
        "cvar20_delta_lb95": (
            _one_sided_hoeffding_lower(tail_values, 0.95)
            if tail_values else -1.0
        ),
        "per_list": per_list,
        "all_list_delta_lb90_ge_minus_003": bool(per_list) and all(
            float(row["delta_lb90"]) >= -0.03 for row in per_list.values()
        ),
        "invalid_crash_increase": (
            (invalid_candidate - invalid_parent) / invalid_total
            if invalid_total else math.inf
        ),
        "complete_required_label_coverage": bool(
            gradient_diagnostics.get("complete_required_label_coverage")
        ),
        "finite_gradients": bool(gradient_diagnostics.get("finite_gradients")),
        "auxiliary_to_core_gradient_norm": float(
            gradient_diagnostics.get("auxiliary_to_core_gradient_norm", math.inf)
        ),
        "total_core_gradient_cosine": float(
            gradient_diagnostics.get("total_core_gradient_cosine", -math.inf)
        ),
        "mean_policy_kl": float(policy_drift.get("mean_policy_kl", math.inf)),
        "p99_policy_kl": float(policy_drift.get("p99_policy_kl", math.inf)),
        "greedy_action_flip_rate": float(
            policy_drift.get("greedy_action_flip_rate", math.inf)
        ),
        "development_paired_units": 1020,
        "locked_paired_units": len(locked),
        "package_guard_pairs": len(package),
        "replay_eligible": False,
    }
    metrics["activation_gate"] = activation_gate(metrics)
    return metrics


def selected_loss_vector(
    *,
    weights: Mapping[str, float],
    manifest_sha256: str,
    loss_contract_sha256: str,
    study_sha256: str,
    selected_round: int,
    selected_direction: str,
) -> dict[str, Any]:
    residuals = canonical_residual_weights(weights)
    if selected_direction not in {"plus", "minus"} or int(selected_round) not in {1, 2}:
        raise FamilyStudyError("selected loss vector lacks a conclusive SPSA identity")
    payload = {
        "schema": SELECTED_VECTOR_SCHEMA,
        "status": "selected_by_passing_shadow_study",
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "residual_objectives": residuals,
        "manifest_sha256": str(manifest_sha256),
        "loss_contract_sha256": str(loss_contract_sha256),
        "study_sha256": str(study_sha256),
        "selected_round": int(selected_round),
        "selected_direction": selected_direction,
        "serving_authority": False,
        "activates_only_with_family_sampler": True,
    }
    payload["artifact_sha256"] = _digest(payload)
    return payload


def owner_ceiling_loss_vector(
    *,
    weights: Mapping[str, float],
    manifest_sha256: str,
    loss_contract_sha256: str,
    study_sha256: str,
    activation_authority_sha256: str,
) -> dict[str, Any]:
    """Seal exact tested round-1-plus weights under separate owner authority."""
    residuals = canonical_residual_weights(weights)
    if not str(activation_authority_sha256).startswith("sha256:"):
        raise FamilyStudyError("owner-ceiling vector lacks bound authority")
    payload = {
        "schema": SELECTED_VECTOR_SCHEMA,
        "status": "selected_by_owner_ceiling_after_inconclusive_study",
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "residual_objectives": residuals,
        "manifest_sha256": str(manifest_sha256),
        "loss_contract_sha256": str(loss_contract_sha256),
        "study_sha256": str(study_sha256),
        "selected_round": 1,
        "selected_direction": "plus",
        "activation_authority_sha256": str(activation_authority_sha256),
        "measured_study_passed": False,
        "serving_authority": False,
        "activates_only_with_family_sampler": True,
    }
    payload["artifact_sha256"] = _digest(payload)
    return payload


def validate_study_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA or payload.get("passed") is not True:
        raise FamilyStudyError("family shadow study did not pass")
    if payload.get("training_eligible") is not False or payload.get("replay_eligible") is not False:
        raise FamilyStudyError("family shadow study escaped isolation")
    selection = payload.get("selection") or {}
    if not selection.get("activate") or selection.get("direction") not in {"plus", "minus"}:
        raise FamilyStudyError("family shadow study lacks a conclusive selection")
    gate = payload.get("activation_gate") or {}
    if gate.get("passed") is not True or not all(dict(gate.get("checks") or {}).values()):
        raise FamilyStudyError("family shadow study gate is incomplete")
    same_parent = payload.get("same_parent_validation") or {}
    if same_parent.get("valid") is not True:
        raise FamilyStudyError("family shadow study lacks same-parent validation")
    return dict(payload)


def validate_same_parent_shadow(payload: Mapping[str, Any]) -> dict[str, Any]:
    required_equal = (
        "parent_checkpoint_sha256",
        "sealed_rows_sha256",
        "split_sha256",
        "batch_order_sha256",
        "optimizer_settings_sha256",
        "seed_book_sha256",
        "update_count",
    )
    left, right = payload.get("plus") or {}, payload.get("minus") or {}
    for key in required_equal:
        if not left.get(key) or left.get(key) != right.get(key):
            raise FamilyStudyError(f"same-parent shadow invariant changed: {key}")
    if left.get("loss_vector_sha256") == right.get("loss_vector_sha256"):
        raise FamilyStudyError("shadow candidates do not differ by loss vector")
    for row in (left, right):
        if row.get("served") or row.get("promoted") or row.get("replay_eligible") is not False:
            raise FamilyStudyError("shadow checkpoint escaped isolation")
    return {"valid": True, "parent_checkpoint_sha256": left["parent_checkpoint_sha256"]}


def activation_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply every locked gameplay, integrity, gradient and policy gate."""
    checks = {
        "macro_win_lb": float(metrics.get("macro_win_rate_improvement_lb95", -math.inf)) >= 0.01,
        "package_guard": float(metrics.get("current_package_delta_lb95", -math.inf)) >= -0.01,
        "cvar20": float(metrics.get("cvar20_delta_lb95", -math.inf)) >= -0.01,
        "per_list": bool(metrics.get("all_list_delta_lb90_ge_minus_003", False)),
        "invalid_crash": float(metrics.get("invalid_crash_increase", math.inf)) <= 0.001,
        "labels": bool(metrics.get("complete_required_label_coverage", False)),
        "finite_gradients": bool(metrics.get("finite_gradients", False)),
        "aux_gradient": float(metrics.get("auxiliary_to_core_gradient_norm", math.inf)) <= 0.50,
        "gradient_cosine": float(metrics.get("total_core_gradient_cosine", -math.inf)) >= 0.80,
        "mean_kl": float(metrics.get("mean_policy_kl", math.inf)) <= 0.02,
        "p99_kl": float(metrics.get("p99_policy_kl", math.inf)) <= 0.10,
        "greedy_flip": float(metrics.get("greedy_action_flip_rate", math.inf)) <= 0.05,
        "development_units": int(metrics.get("development_paired_units", -1)) == 1020,
        "locked_units": int(metrics.get("locked_paired_units", -1)) == 4284,
        "package_units": int(metrics.get("package_guard_pairs", -1)) == 1020,
        "training_ineligible": metrics.get("replay_eligible") is False,
    }
    status = "passed" if all(checks.values()) else "failed_closed"
    return {"schema": SCHEMA, "status": status, "passed": status == "passed", "checks": checks}


def select_after_rounds(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Allow one retry only; incomplete/inconclusive evidence activates nothing."""
    if not rounds or len(rounds) > 2:
        raise FamilyStudyError("study must contain one or two SPSA rounds")
    for index, row in enumerate(rounds, 1):
        if int(row.get("round", -1)) != index:
            raise FamilyStudyError("SPSA rounds are not contiguous")
        if row.get("status") == "conclusive" and row.get("selected_direction") in {"plus", "minus"}:
            return {"activate": True, "round": index, "direction": row["selected_direction"]}
        if row.get("status") not in {"inconclusive", "failed"}:
            raise FamilyStudyError("invalid SPSA round status")
    return {"activate": False, "reason": "failed_or_inconclusive_after_allowed_rounds"}


def rollback_required(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics.get("probability_regression_worse_than_002", 0.0)) >= 0.99
        or float(metrics.get("current_package_delta_lb95", math.inf)) < -0.01
        or not bool(metrics.get("invalid_game_check", False))
        or not bool(metrics.get("causal_integrity_check", False))
        or not bool(metrics.get("latency_check", False))
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.inf
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _probability_mean_below(values: Sequence[float], threshold: float) -> float:
    """Normal posterior approximation for a bounded paired mean.

    This is deliberately conservative at zero variance: an observed constant
    effect is assigned probability one or zero instead of inventing sampling
    noise. The immutable receipt records this exact method.
    """
    samples = [float(value) for value in values]
    if not samples or any(not math.isfinite(value) for value in samples):
        return 1.0
    mean = statistics.fmean(samples)
    if len(samples) < 2:
        return float(mean < float(threshold))
    variance = statistics.variance(samples)
    if variance <= 0.0:
        return float(mean < float(threshold))
    standard_error = math.sqrt(variance / len(samples))
    return statistics.NormalDist().cdf((float(threshold) - mean) / standard_error)


def compile_post_activation_monitor(
    *,
    locked_rows: Sequence[Mapping[str, Any]],
    package_rows: Sequence[Mapping[str, Any]],
    latency_relative_cap: float = 1.10,
    latency_absolute_slack_seconds: float = 0.010,
) -> dict[str, Any]:
    """Compile the fresh paired monitor that controls boundary rollback."""
    locked = _exact_pairs(
        locked_rows, left="candidate", right="parent", expected_units=4284
    )
    package = _exact_pairs(
        package_rows, left="candidate", right="parent", expected_units=1020
    )
    all_pairs = [*locked, *package]
    valid_locked = [
        (candidate, parent)
        for candidate, parent in locked
        if not bool(candidate.get("invalid")) and not bool(parent.get("invalid"))
    ]
    valid_package = [
        (candidate, parent)
        for candidate, parent in package
        if not bool(candidate.get("invalid")) and not bool(parent.get("invalid"))
    ]
    locked_delta = [
        _score(candidate) - _score(parent) for candidate, parent in valid_locked
    ]
    package_delta = [
        _score(candidate) - _score(parent) for candidate, parent in valid_package
    ]
    candidate_invalid = sum(bool(candidate.get("invalid")) for candidate, _ in all_pairs)
    parent_invalid = sum(bool(parent.get("invalid")) for _, parent in all_pairs)
    candidate_latency = [
        float(candidate.get("decision_latency_seconds", math.inf))
        for candidate, _ in all_pairs
        if not bool(candidate.get("invalid"))
    ]
    parent_latency = [
        float(parent.get("decision_latency_seconds", math.inf))
        for _, parent in all_pairs
        if not bool(parent.get("invalid"))
    ]
    candidate_p99 = _quantile(candidate_latency, 0.99)
    parent_p99 = _quantile(parent_latency, 0.99)
    latency_limit = max(
        parent_p99 * float(latency_relative_cap),
        parent_p99 + float(latency_absolute_slack_seconds),
    )
    causal_integrity = all(
        row.get("causal_integrity") is True
        and row.get("training_eligible") is False
        and row.get("replay_eligible") is False
        for pair in all_pairs
        for row in pair
    )
    metrics = {
        "probability_regression_worse_than_002": _probability_mean_below(
            locked_delta, -0.02
        ),
        "probability_method": "paired_mean_normal_posterior_v1",
        "locked_paired_units": len(locked),
        "locked_valid_pairs": len(valid_locked),
        "current_package_pairs": len(package),
        "current_package_valid_pairs": len(valid_package),
        "current_package_delta_lb95": (
            _one_sided_hoeffding_lower(package_delta, 0.95)
            if package_delta
            else -1.0
        ),
        "candidate_invalid_games": candidate_invalid,
        "parent_invalid_games": parent_invalid,
        "invalid_game_check": candidate_invalid == 0 and parent_invalid == 0,
        "causal_integrity_check": causal_integrity,
        "candidate_decision_latency_p99_seconds": candidate_p99,
        "parent_decision_latency_p99_seconds": parent_p99,
        "decision_latency_limit_seconds": latency_limit,
        "latency_check": bool(
            math.isfinite(candidate_p99)
            and math.isfinite(parent_p99)
            and candidate_p99 <= latency_limit
        ),
        "training_eligible": False,
        "replay_eligible": False,
    }
    metrics["rollback_required"] = rollback_required(metrics)
    return metrics


def validate_post_activation_monitor(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != MONITOR_SCHEMA:
        raise FamilyStudyError("wrong post-activation monitor schema")
    if payload.get("training_eligible") is not False or payload.get("replay_eligible") is not False:
        raise FamilyStudyError("post-activation monitor escaped evaluation isolation")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("locked_paired_units", -1)) != 4284 or int(
        metrics.get("current_package_pairs", -1)
    ) != 1020:
        raise FamilyStudyError("post-activation monitor is incomplete")
    expected = rollback_required(metrics)
    if bool(payload.get("rollback_required")) != expected:
        raise FamilyStudyError("post-activation rollback decision changed")
    return dict(payload)


__all__ = [
    "FamilyStudyError", "SCHEMA", "activation_gate", "antithetic_spsa",
    "MONITOR_SCHEMA", "PLAN_SCHEMA", "SELECTED_VECTOR_SCHEMA", "compile_activation_metrics",
    "compile_post_activation_monitor",
    "compile_development_round", "family_shadow_plan", "paired_panel_schedule",
    "rollback_required", "select_after_rounds", "selected_loss_vector",
    "validate_post_activation_monitor", "validate_same_parent_shadow", "validate_study_receipt",
]
