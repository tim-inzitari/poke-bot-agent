"""Torch-free semantic validator for r198 RTP evaluation authority.

The three-arm evaluator deliberately produces a *review* receipt, rather than
changing a selector.  Promotion consumers must therefore inspect its semantic
claims instead of treating a matching pathname or a checksum of arbitrary
bytes as authority.  This module is intentionally independent of the planner,
checkpoint loader, and submission package so every consumer can use the same
fail-closed checks without creating an import cycle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from poke_bot.rtp_evaluation_immutable_io import (
    ImmutableEvidenceIOError,
    ImmutableFileBytes,
    lexical_absolute_path,
    read_immutable_file_bytes,
)


EVALUATION_RECEIPT_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_evaluation_receipt/v2"
)
EXECUTION_RECEIPT_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_execution_receipt/v1"
)
DIRECT_BRIDGE_ARM = "direct_bridge_recursive_disabled"
ARMS = ("no_rtp", DIRECT_BRIDGE_ARM, "recursive_rtp")
R198_CANDIDATE_CONTRACT_SHA256 = (
    "sha256:bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
R198_PARENT_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R198_SIDECAR_SHA256 = (
    "sha256:23eb09cbfa5e9e8d3aec3b8af4dc03a71db811ce9b7c32c6c5ece65bc3f3dc31"
)
R198_SIDECAR_CONFIG_SHA256 = (
    "sha256:7fb0658f0358c93636524a40ddd52f9f76199de261963a85dbf5946901a9f676"
)
R198_DECK_FILE_SHA256 = (
    "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
)
R198_DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
R198_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
CANDIDATE_EVALUATION_BINDING_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_candidate_evaluation_binding/v1"
)
R198_OFFICIAL_CONTROL_OPPONENTS = {
    "iono": "sha256:6ba8e818b698774b6e437364e9457600eda950fbefb663d8e4ad39cdaf0371e2",
    "dragapult-ex": "sha256:835dcbcc26366faa04d902db727620d4b12618b6a66d000dccb9c9b86e9d62a0",
    "mega-abomasnow-ex": "sha256:57a9499b2bee493a830abaf5a3e19b8a73faea200faee87aeeb2864bab25c2fb",
    "mega-lucario-ex": "sha256:98f20936d430c6cc60f3eb1da8230392bf6dce8ecacf97773bda4db63f56376a",
}
R198_RESEARCH_CONTROL_REGISTRY_SHA256 = (
    "sha256:78fd8e52df1464db94e74a49247a67ced41b5d164dc86fafec3229f2c1e47edc"
)
R198_PAIRED_CELLS = 1_000
R198_ROWS = R198_PAIRED_CELLS * len(ARMS)
R198_MINIMUM_RECURSIVE_DECISIONS = 100
R198_MINIMUM_RECURSIVE_SHARE = 0.05
R198_MAXIMUM_UNEXPECTED_FALLBACK_RATE = 0.01
R198_MAX_ACTION_COMBOS = 1024
INTENDED_COMPLEX_DECISION_SCOPE = "new_turn_complexity_gate_only"
FORCED_TURN_ORDER_CONTROL = "forced_go_first_contract"
_SUCCESSFUL_RECURSIVE_MODES = frozenset(
    {"recursive_plan", "continue_plan", "replan_with_program"}
)
_RECURSIVE_FALLBACK_MODES = frozenset({"direct_policy_fallback", "replan_direct"})
_NONRECURSIVE_DIRECT_POLICY_MODE = "direct_policy"
_OVER_CAP_FACTORIZED_FALLBACK_MODE = "over_cap_factorized_fallback"
_OVER_CAP_FACTORIZED_FALLBACK_REASON = "complete_ordered_action_space_over_cap"
_OVER_CAP_ACTION_SPACE_FIELDS = frozenset(
    {
        "n_options",
        "min_count",
        "max_count",
        "counts",
        "complete_ordered_action_cardinality",
        "complete_ordered_action_cap",
        "over_cap",
        "complete_ordered_actions_materialized",
        "complete_ordered_action_truncated",
    }
)
_OVER_CAP_TRACE_FIELDS = frozenset(
    {
        "decision_index",
        "arm",
        "mode",
        "classification",
        "action_space",
        "action_space_sha256",
        "observation_sha256",
        "candidate_policy_input_sha256",
        "logical_pre_action_sha256",
        "returned_action",
        "factorized_teacher_forcing_legal",
        "factorized_teacher_forcing_stage_count",
        "complexity_probe_not_invoked",
        "neural_passes",
        "required_neural_passes",
        "neural_budget_failure",
        "rtp_diagnostic",
        "included_in_candidate_decisions",
        "included_in_candidate_latency",
        "excluded_from_planner_eligible_candidate_decisions",
        "excluded_from_intended_complex_denominator",
        "excluded_from_direct_bridge_metrics",
        "excluded_from_recursive_metrics",
        "excluded_from_fallback_metrics",
        "excluded_from_neural_pass_metrics",
        "excluded_from_recursive_latency",
    }
)
_REQUIRED_RECURSIVE_MODE_COUNTS = tuple(
    sorted(_SUCCESSFUL_RECURSIVE_MODES | _RECURSIVE_FALLBACK_MODES)
)

_REQUIRED_GATES = frozenset(
    {
        "frozen_artifact_identity",
        "true_rng_tape_or_snapshot_pairing",
        "immutable_file_backed_execution_evidence",
        "separate_source_excluded_evaluation_only_cohort",
        "trusted_counterfactual_candidate_targets",
        "complete_three_arm_schedule",
        "opponent_seat_support",
        "opponent_seat_stratified_recursive_efficacy",
        "direct_bridge_path_exercised",
        "recursive_path_exercised",
        "recursive_share_of_intended_complex_decisions",
        "deterministic_normal_recursive_six_pass_preflight",
        "deterministic_forced_replan_five_pass_preflight",
        "no_neural_pass_budget_failures",
        "unexpected_recursive_fallback_rate",
        "no_illegal_action_regression",
        "no_new_forfeit_regression",
        "recursive_effect_lower_bound",
        "recursive_p95_decision_latency_slo",
    }
)


class RTPPromotionEvidenceError(ValueError):
    """An evaluation receipt cannot truthfully grant a promotion review."""


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RTPPromotionEvidenceError(f"{label} must be an object")
    return dict(value)


def _nonempty_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RTPPromotionEvidenceError(f"{label} is required")
    return text


def _sha256(value: Any, label: str) -> str:
    digest = _nonempty_text(value, label).lower()
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise RTPPromotionEvidenceError(f"{label} is not a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise RTPPromotionEvidenceError(f"{label} is not a SHA-256 digest") from exc
    return digest


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise RTPPromotionEvidenceError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RTPPromotionEvidenceError(f"{label} must be an integer") from exc
    if parsed < minimum:
        raise RTPPromotionEvidenceError(f"{label} must be at least {minimum}")
    return parsed


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise RTPPromotionEvidenceError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RTPPromotionEvidenceError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise RTPPromotionEvidenceError(f"{label} is out of range")
    return parsed


_EvidenceReadCache = dict[str, ImmutableFileBytes]


@dataclass(frozen=True)
class R198PackagedEvaluationCapability:
    """Process-local authority for one sealed submission evaluation copy.

    This is intentionally not an environment value.  The submitted entrypoint
    arms it only after it has checked the package profile and its physical
    ``0444`` promotion/evaluation receipts.  A generic caller can set
    ``POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT`` but cannot thereby cause a
    checkpoint consumer to skip the source-evidence archive.
    """

    package_root: str
    promotion_receipt_path: str
    promotion_receipt_sha256: str
    evaluation_receipt_path: str
    evaluation_receipt_sha256: str


_PACKAGED_EVALUATION_CAPABILITIES: dict[str, R198PackagedEvaluationCapability] = {}


def _read_immutable_evidence(
    path: str | Path,
    label: str,
    *,
    evidence_cache: _EvidenceReadCache | None = None,
) -> ImmutableFileBytes:
    """Read exact immutable bytes once, without a pathname re-open race."""

    try:
        lexical = lexical_absolute_path(path, label)
    except ImmutableEvidenceIOError as exc:
        raise RTPPromotionEvidenceError(str(exc)) from exc
    cache = evidence_cache if evidence_cache is not None else {}
    material = cache.get(str(lexical))
    if material is not None:
        return material
    try:
        material = read_immutable_file_bytes(lexical, label, exact_mode=0o444)
    except ImmutableEvidenceIOError as exc:
        raise RTPPromotionEvidenceError(str(exc)) from exc
    cache[str(lexical)] = material
    return material


def _json_from_immutable_evidence(
    identity: Mapping[str, Any],
    label: str,
    *,
    evidence_cache: _EvidenceReadCache,
) -> dict[str, Any]:
    """Parse the exact bytes already hash-verified for an identity."""

    material = _read_immutable_evidence(
        _nonempty_text(identity.get("path"), f"{label}.path"),
        label,
        evidence_cache=evidence_cache,
    )
    if (
        material.sha256 != _sha256(identity.get("sha256"), f"{label}.sha256")
        or material.bytes
        != _integer(identity.get("bytes"), f"{label}.bytes", minimum=1)
    ):
        raise RTPPromotionEvidenceError(f"{label} identity changed")
    try:
        return _mapping(json.loads(material.payload.decode("utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RTPPromotionEvidenceError(f"{label} is not valid JSON") from exc


def read_r198_immutable_json_object(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read an exact-mode physical JSON artifact once for a promotion consumer.

    The returned identity describes the bytes that were parsed.  It is used for
    promotion receipts as well as evaluation source inputs, so callers never
    hash a pathname and then parse a later re-open of that pathname.
    """

    material = _read_immutable_evidence(path, label)
    if expected_sha256 is not None and material.sha256 != _sha256(
        expected_sha256, f"{label} SHA-256"
    ):
        raise RTPPromotionEvidenceError(f"{label} digest mismatch")
    try:
        payload = _mapping(json.loads(material.payload.decode("utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RTPPromotionEvidenceError(f"{label} is not valid JSON") from exc
    return material.identity(), payload


def arm_r198_packaged_evaluation_capability(
    *,
    package_root: str | Path,
    runtime_profile: Mapping[str, Any],
) -> R198PackagedEvaluationCapability:
    """Arm the narrow package-only exception to archive dereferencing.

    Only ``submission/main.py`` calls this after enforcing the recursive
    profile.  The capability is stored in process memory rather than encoded in
    an environment variable, and is cross-bound to the profile's two receipt
    digests and the promotion receipt's declared evaluation identity.
    """

    profile = _mapping(runtime_profile, "recursive RTP runtime profile")
    promotion_filename = profile.get("rtp_promotion_receipt_file")
    evaluation_filename = profile.get("rtp_evaluation_receipt_file")
    if (
        profile.get("rtp_mode") != "recursive"
        or promotion_filename != "rtp_promotion_receipt.json"
        or evaluation_filename != "rtp_evaluation_receipt.json"
    ):
        raise RTPPromotionEvidenceError(
            "recursive RTP runtime profile cannot arm packaged evaluation evidence"
        )
    root = lexical_absolute_path(package_root, "recursive RTP package root")
    promotion_path = root / "rtp_promotion_receipt.json"
    evaluation_path = root / "rtp_evaluation_receipt.json"
    profile_promotion_digest = _sha256(
        profile.get("rtp_promotion_receipt_sha256"),
        "runtime profile rtp_promotion_receipt_sha256",
    )
    profile_evaluation_digest = _sha256(
        profile.get("rtp_evaluation_receipt_sha256"),
        "runtime profile rtp_evaluation_receipt_sha256",
    )
    promotion_identity, promotion = read_r198_immutable_json_object(
        promotion_path,
        label="packaged RTP promotion receipt",
        expected_sha256=profile_promotion_digest,
    )
    evaluation_identity, _ = read_r198_immutable_json_object(
        evaluation_path,
        label="packaged RTP evaluation receipt",
        expected_sha256=profile_evaluation_digest,
    )
    if _sha256(
        promotion.get("evaluation_receipt_sha256"),
        "packaged promotion evaluation_receipt_sha256",
    ) != evaluation_identity["sha256"]:
        raise RTPPromotionEvidenceError(
            "packaged promotion receipt is not bound to its evaluation receipt"
        )
    capability = R198PackagedEvaluationCapability(
        package_root=str(root),
        promotion_receipt_path=str(promotion_identity["path"]),
        promotion_receipt_sha256=str(promotion_identity["sha256"]),
        evaluation_receipt_path=str(evaluation_identity["path"]),
        evaluation_receipt_sha256=str(evaluation_identity["sha256"]),
    )
    _PACKAGED_EVALUATION_CAPABILITIES[capability.evaluation_receipt_path] = capability
    return capability


def resolve_r198_packaged_evaluation_capability(
    *, environment: Mapping[str, str] | None = None
) -> R198PackagedEvaluationCapability | None:
    """Resolve a package exception only if this process previously armed it.

    An environment value is a non-authoritative routing hint.  Unarmed,
    malformed, or mismatched hints deliberately resolve to ``None`` so an
    ordinary source consumer continues to validate its local immutable
    archive.  The sealed entrypoint separately requires a non-``None`` result
    before it permits package-local evidence mode.
    """

    source = os.environ if environment is None else environment
    claimed = str(source.get("POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT") or "").strip()
    if not claimed:
        return None
    try:
        path = lexical_absolute_path(claimed, "POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT")
    except ImmutableEvidenceIOError:
        return None
    capability = _PACKAGED_EVALUATION_CAPABILITIES.get(str(path))
    if capability is None:
        return None
    promotion_path = str(source.get("POKEBOT_RTP_PROMOTION_RECEIPT") or "").strip()
    promotion_digest = str(
        source.get("POKEBOT_RTP_PROMOTION_RECEIPT_SHA256") or ""
    ).strip()
    try:
        normalized_promotion = str(
            lexical_absolute_path(promotion_path, "POKEBOT_RTP_PROMOTION_RECEIPT")
        )
    except ImmutableEvidenceIOError:
        return None
    try:
        normalized_promotion_digest = _sha256(
            promotion_digest, "POKEBOT_RTP_PROMOTION_RECEIPT_SHA256"
        )
    except RTPPromotionEvidenceError:
        return None
    if (
        normalized_promotion != capability.promotion_receipt_path
        or normalized_promotion_digest != capability.promotion_receipt_sha256
    ):
        return None
    return capability


def _immutable_identity(
    raw: Any,
    label: str,
    *,
    require_local: bool,
    evidence_cache: _EvidenceReadCache | None = None,
) -> dict[str, Any]:
    value = _mapping(raw, label)
    identity = {
        "path": _nonempty_text(value.get("path"), f"{label}.path"),
        "sha256": _sha256(value.get("sha256"), f"{label}.sha256"),
        "bytes": _integer(value.get("bytes"), f"{label}.bytes", minimum=1),
    }
    if require_local:
        material = _read_immutable_evidence(
            identity["path"], label, evidence_cache=evidence_cache
        )
        if material.sha256 != identity["sha256"] or material.bytes != identity["bytes"]:
            raise RTPPromotionEvidenceError(f"{label} identity changed")
        identity["path"] = str(material.path)
    return identity


def _validate_gate_table(
    raw: Any, *, require_passed: bool = True
) -> dict[str, dict[str, Any]]:
    gates = _mapping(raw, "evaluation promotion_gates")
    missing = sorted(_REQUIRED_GATES.difference(gates))
    if missing:
        raise RTPPromotionEvidenceError(
            "evaluation receipt lacks required promotion gates: " + ", ".join(missing)
        )
    for name, raw_gate in gates.items():
        gate = _mapping(raw_gate, f"evaluation promotion_gates.{name}")
        if not isinstance(gate.get("passed"), bool):
            raise RTPPromotionEvidenceError(
                f"evaluation promotion gate has a non-boolean passed flag: {name}"
            )
        if require_passed and gate.get("passed") is not True:
            raise RTPPromotionEvidenceError(f"evaluation promotion gate did not pass: {name}")
    return {name: _mapping(value, f"evaluation promotion_gates.{name}") for name, value in gates.items()}


def _canonical_arm(value: Any, label: str) -> str:
    """Receipt rows are canonical; only archived result input may use v1's alias."""

    arm = _nonempty_text(value, label)
    return DIRECT_BRIDGE_ARM if arm == "direct_bridge" else arm


def _score(value: Any, label: str) -> float:
    score = _number(value, label)
    if score not in {0.0, 0.5, 1.0}:
        raise RTPPromotionEvidenceError(f"{label} must be one of 0, 0.5, 1")
    return score


def _same_number(actual: Any, expected: float, label: str) -> None:
    value = _number(actual, label)
    # The compiler writes normal JSON doubles.  This allows a harmless JSON
    # round trip but not a changed statistic or a strategically rounded LCB.
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RTPPromotionEvidenceError(f"{label} differs from recomputed evidence")


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _one_sided_hoeffding_lower(deltas: Sequence[float], confidence: float) -> float:
    if not deltas:
        raise RTPPromotionEvidenceError("cannot compute an efficacy bound without pairs")
    return max(
        -1.0,
        sum(deltas) / len(deltas)
        - math.sqrt(2.0 * math.log(1.0 / (1.0 - confidence)) / len(deltas)),
    )


def _validated_forced_turn_order_control_trace(
    raw: Any, *, label: str
) -> list[dict[str, Any]]:
    """Independently revalidate non-planner turn-order controls.

    The compiler's receipt is not promotion authority by itself.  Recheck the
    external control's exact context/action proof here so neither an omitted
    trace nor a hand-authored summary can change recursive policy metrics.
    """

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RTPPromotionEvidenceError("telemetry lacks forced turn-order control trace")
    required_fields = {
        "control_index",
        "control",
        "prompt_context",
        "prompt_context_encoding",
        "expected_action",
        "returned_action",
        "verified_observation_action_contract",
        "rtp_diagnostics_absent",
        "complexity_probe_not_invoked",
        "excluded_from_candidate_decisions",
        "excluded_from_intended_complex_denominator",
        "excluded_from_latency",
    }
    normalized: list[dict[str, Any]] = []
    for index, raw_trace in enumerate(raw):
        trace = _mapping(raw_trace, f"{label}.forced_turn_order_control_trace[{index}]")
        if set(trace) != required_fields:
            raise RTPPromotionEvidenceError(
                "forced turn-order trace does not have the exact canonical field set"
            )
        if type(trace.get("control_index")) is not int or trace["control_index"] != index:
            raise RTPPromotionEvidenceError("forced turn-order control indexes are not monotonic")
        if trace.get("control") != FORCED_TURN_ORDER_CONTROL:
            raise RTPPromotionEvidenceError("forced turn-order trace has an unknown control")
        encoding = _nonempty_text(
            trace.get("prompt_context_encoding"), f"{label}.forced prompt encoding"
        )
        context = trace.get("prompt_context")
        if encoding == "numeric_41":
            if (
                type(context) is not int
                or context != 41
            ):
                raise RTPPromotionEvidenceError("forced turn-order trace has an invalid prompt context")
            context = 41
        elif (
            encoding == "enum_is_first"
            and type(context) is str
            and context == "IsFirst"
        ):
            context = "IsFirst"
        else:
            raise RTPPromotionEvidenceError("forced turn-order trace has an invalid prompt context")

        def _action(field: str) -> list[int]:
            value = trace.get(field)
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != 1
                or type(value[0]) is not int
                or value[0] < 0
            ):
                raise RTPPromotionEvidenceError(
                    f"forced turn-order {field} must contain exactly one action index"
                )
            return [value[0]]

        expected = _action("expected_action")
        returned = _action("returned_action")
        if expected != returned:
            raise RTPPromotionEvidenceError(
                "forced turn-order returned action differs from expected Yes"
            )
        for field in (
            "verified_observation_action_contract",
            "rtp_diagnostics_absent",
            "complexity_probe_not_invoked",
            "excluded_from_candidate_decisions",
            "excluded_from_intended_complex_denominator",
            "excluded_from_latency",
        ):
            if trace.get(field) is not True:
                raise RTPPromotionEvidenceError(
                    f"forced turn-order trace lacks {field} attestation"
                )
        normalized.append(
            {
                "control_index": index,
                "control": FORCED_TURN_ORDER_CONTROL,
                "prompt_context": context,
                "prompt_context_encoding": encoding,
                "expected_action": expected,
                "returned_action": returned,
                "verified_observation_action_contract": True,
                "rtp_diagnostics_absent": True,
                "complexity_probe_not_invoked": True,
                "excluded_from_candidate_decisions": True,
                "excluded_from_intended_complex_denominator": True,
                "excluded_from_latency": True,
            }
        )
    if len(normalized) > 1:
        raise RTPPromotionEvidenceError("a game has more than one forced turn-order control")
    return normalized


def _validate_r198_forced_turn_order_contract(
    telemetry: Mapping[str, Any], *, candidate_seat: int, label: str
) -> None:
    """Independently bind external-control evidence to the sealed snapshot ABI."""

    if candidate_seat not in {0, 1}:
        raise RTPPromotionEvidenceError(f"{label} has an invalid candidate seat")
    expected_count = 1 if candidate_seat == 0 else 0
    observed_count = _integer(
        telemetry.get("forced_turn_order_controls"),
        f"{label}.forced_turn_order_controls",
    )
    trace = telemetry.get("forced_turn_order_control_trace")
    if observed_count != expected_count:
        raise RTPPromotionEvidenceError(
            "forced turn-order control count does not match the frozen seat ABI"
        )
    if candidate_seat == 1:
        if trace != []:
            raise RTPPromotionEvidenceError(
                "candidate seat 1 cannot attest the physical-seat-0 IsFirst control"
            )
        return
    expected_trace = [
        {
            "control_index": 0,
            "control": FORCED_TURN_ORDER_CONTROL,
            "prompt_context": 41,
            "prompt_context_encoding": "numeric_41",
            "expected_action": [0],
            "returned_action": [0],
            "verified_observation_action_contract": True,
            "rtp_diagnostics_absent": True,
            "complexity_probe_not_invoked": True,
            "excluded_from_candidate_decisions": True,
            "excluded_from_intended_complex_denominator": True,
            "excluded_from_latency": True,
        }
    ]
    if trace != expected_trace:
        raise RTPPromotionEvidenceError(
            "candidate seat 0 lacks the exact frozen IsFirst/Yes control"
        )


def _exact_nonbool_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RTPPromotionEvidenceError(f"{label} must be an exact non-bool integer")
    return value


def _over_cap_action_space(raw: Any, *, label: str) -> dict[str, Any]:
    value = _mapping(raw, label)
    if set(value) != _OVER_CAP_ACTION_SPACE_FIELDS:
        raise RTPPromotionEvidenceError(f"{label} has an invalid exact field set")
    n_options = _exact_nonbool_integer(value.get("n_options"), f"{label}.n_options")
    min_count = _exact_nonbool_integer(value.get("min_count"), f"{label}.min_count")
    max_count = _exact_nonbool_integer(value.get("max_count"), f"{label}.max_count")
    cap = _exact_nonbool_integer(value.get("complete_ordered_action_cap"), f"{label}.cap")
    cardinality = _exact_nonbool_integer(
        value.get("complete_ordered_action_cardinality"), f"{label}.cardinality"
    )
    if cap != R198_MAX_ACTION_COMBOS:
        raise RTPPromotionEvidenceError(f"{label} cap differs from exact r198 1024")
    if not (0 <= min_count <= max_count <= n_options):
        raise RTPPromotionEvidenceError(f"{label} has invalid exact selection bounds")
    counts = value.get("counts")
    expected_counts = list(range(min_count, max_count + 1))
    if (
        not isinstance(counts, Sequence)
        or isinstance(counts, (str, bytes))
        or any(type(item) is not int for item in counts)
        or list(counts) != expected_counts
    ):
        raise RTPPromotionEvidenceError(f"{label} counts are not exact")
    if cardinality != sum(math.perm(n_options, count) for count in expected_counts):
        raise RTPPromotionEvidenceError(f"{label} cardinality does not recompute")
    if value.get("over_cap") is not True or cardinality <= cap:
        raise RTPPromotionEvidenceError(f"{label} does not attest a true over-cap space")
    if value.get("complete_ordered_actions_materialized") is not False:
        raise RTPPromotionEvidenceError(f"{label} attests complete action materialization")
    if value.get("complete_ordered_action_truncated") is not False:
        raise RTPPromotionEvidenceError(f"{label} attests truncation")
    return {
        "n_options": n_options,
        "min_count": min_count,
        "max_count": max_count,
        "counts": expected_counts,
        "complete_ordered_action_cardinality": cardinality,
        "complete_ordered_action_cap": cap,
        "over_cap": True,
        "complete_ordered_actions_materialized": False,
        "complete_ordered_action_truncated": False,
    }


def _over_cap_factorized_action(
    raw: Any, action_space: Mapping[str, Any], *, label: str
) -> tuple[list[int], int]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RTPPromotionEvidenceError(f"{label} returned_action must be a sequence")
    action = list(raw)
    if any(type(item) is not int for item in action):
        raise RTPPromotionEvidenceError(f"{label} returned_action has a non-exact index")
    n_options = int(action_space["n_options"])
    min_count = int(action_space["min_count"])
    max_count = int(action_space["max_count"])
    if (
        len(action) < min_count
        or len(action) > max_count
        or len(action) != len(set(action))
        or any(item < 0 or item >= n_options for item in action)
    ):
        raise RTPPromotionEvidenceError(f"{label} returned_action is not factorized-legal")
    return action, max(len(action) + int(len(action) < max_count), 1)


def _over_cap_diagnostic(raw: Any, *, arm: str, label: str) -> dict[str, Any] | None:
    if arm == "no_rtp":
        if raw is not None:
            raise RTPPromotionEvidenceError(f"{label} no-RTP over-cap trace has diagnostics")
        return None
    value = _mapping(raw, f"{label}.rtp_diagnostic")
    expected = {
        "mode": "fallback",
        "fallback_code": "action_space_too_large",
        "neural_passes": 0,
        "required_neural_passes": 0,
        "legal_count": 0,
        "decision_mode": "",
    }
    if set(value) != set(expected):
        raise RTPPromotionEvidenceError(f"{label} RTP diagnostic has an invalid field set")
    for key, expected_value in expected.items():
        actual = value.get(key)
        if isinstance(expected_value, int):
            if type(actual) is not int or actual != expected_value:
                raise RTPPromotionEvidenceError(f"{label} RTP diagnostic differs at {key}")
        elif actual != expected_value:
            raise RTPPromotionEvidenceError(f"{label} RTP diagnostic differs at {key}")
    return expected


def _validated_over_cap_trace(
    raw_trace: Any,
    declared_digest: Any,
    *,
    arm: str,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_trace, Sequence) or isinstance(raw_trace, (str, bytes)):
        raise RTPPromotionEvidenceError("telemetry lacks an over-cap factorized trace")
    supplied = list(raw_trace)
    if _sha256(declared_digest, f"{label}.over_cap_factorized_fallback_trace_sha256") != canonical_digest(supplied):
        raise RTPPromotionEvidenceError("over-cap factorized trace digest differs")
    result: list[dict[str, Any]] = []
    indexes: set[int] = set()
    true_fields = (
        "factorized_teacher_forcing_legal",
        "complexity_probe_not_invoked",
        "included_in_candidate_decisions",
        "included_in_candidate_latency",
        "excluded_from_planner_eligible_candidate_decisions",
        "excluded_from_intended_complex_denominator",
        "excluded_from_direct_bridge_metrics",
        "excluded_from_recursive_metrics",
        "excluded_from_fallback_metrics",
        "excluded_from_neural_pass_metrics",
        "excluded_from_recursive_latency",
    )
    for position, raw in enumerate(supplied):
        value = _mapping(raw, f"{label}.over_cap_factorized_fallback_trace[{position}]")
        if set(value) != _OVER_CAP_TRACE_FIELDS:
            raise RTPPromotionEvidenceError("over-cap factorized trace has an invalid exact field set")
        decision_index = _exact_nonbool_integer(value.get("decision_index"), f"{label}.over-cap decision index")
        if decision_index in indexes:
            raise RTPPromotionEvidenceError("over-cap factorized trace repeats a decision index")
        indexes.add(decision_index)
        if value.get("arm") != arm or value.get("mode") != _OVER_CAP_FACTORIZED_FALLBACK_MODE:
            raise RTPPromotionEvidenceError("over-cap factorized trace arm/mode is invalid")
        if value.get("classification") != _OVER_CAP_FACTORIZED_FALLBACK_REASON:
            raise RTPPromotionEvidenceError("over-cap factorized trace classification is invalid")
        action_space = _over_cap_action_space(value.get("action_space"), label=f"{label}.over-cap action_space")
        action_space_sha256 = _sha256(value.get("action_space_sha256"), f"{label}.over-cap action_space_sha256")
        if action_space_sha256 != canonical_digest(action_space):
            raise RTPPromotionEvidenceError("over-cap action-space digest does not recompute")
        observation_sha256 = _sha256(value.get("observation_sha256"), f"{label}.over-cap observation_sha256")
        policy_input_sha256 = _sha256(value.get("candidate_policy_input_sha256"), f"{label}.over-cap policy_input_sha256")
        logical_sha256 = _sha256(value.get("logical_pre_action_sha256"), f"{label}.over-cap logical_sha256")
        if logical_sha256 != canonical_digest(
            {
                "observation_sha256": observation_sha256,
                "action_space_sha256": action_space_sha256,
                "candidate_policy_input_sha256": policy_input_sha256,
            }
        ):
            raise RTPPromotionEvidenceError("over-cap logical pre-action digest does not recompute")
        action, stage_count = _over_cap_factorized_action(value.get("returned_action"), action_space, label=label)
        if value.get("factorized_teacher_forcing_legal") is not True or _exact_nonbool_integer(
            value.get("factorized_teacher_forcing_stage_count"), f"{label}.over-cap stage count"
        ) != stage_count:
            raise RTPPromotionEvidenceError("over-cap factorized legality attestation differs")
        for field in true_fields:
            if value.get(field) is not True:
                raise RTPPromotionEvidenceError(f"over-cap trace lacks {field}")
        if value.get("neural_budget_failure") is not False:
            raise RTPPromotionEvidenceError("over-cap trace records a neural budget failure")
        if any(
            _exact_nonbool_integer(value.get(field), f"{label}.over-cap {field}") != 0
            for field in ("neural_passes", "required_neural_passes")
        ):
            raise RTPPromotionEvidenceError("over-cap trace spent neural passes")
        diagnostic = _over_cap_diagnostic(value.get("rtp_diagnostic"), arm=arm, label=label)
        result.append(
            {
                "decision_index": decision_index,
                "arm": arm,
                "mode": _OVER_CAP_FACTORIZED_FALLBACK_MODE,
                "classification": _OVER_CAP_FACTORIZED_FALLBACK_REASON,
                "action_space": action_space,
                "action_space_sha256": action_space_sha256,
                "observation_sha256": observation_sha256,
                "candidate_policy_input_sha256": policy_input_sha256,
                "logical_pre_action_sha256": logical_sha256,
                "returned_action": action,
                "factorized_teacher_forcing_legal": True,
                "factorized_teacher_forcing_stage_count": stage_count,
                "complexity_probe_not_invoked": True,
                "neural_passes": 0,
                "required_neural_passes": 0,
                "neural_budget_failure": False,
                "rtp_diagnostic": diagnostic,
                "included_in_candidate_decisions": True,
                "included_in_candidate_latency": True,
                "excluded_from_planner_eligible_candidate_decisions": True,
                "excluded_from_intended_complex_denominator": True,
                "excluded_from_direct_bridge_metrics": True,
                "excluded_from_recursive_metrics": True,
                "excluded_from_fallback_metrics": True,
                "excluded_from_neural_pass_metrics": True,
                "excluded_from_recursive_latency": True,
            }
        )
    return result


def _validated_row_telemetry(raw: Any, *, arm: str, label: str) -> dict[str, Any]:
    """Recheck the compiler's trace-derived telemetry without importing it.

    A promotion receipt must not be able to name arbitrary immutable blobs and
    then assert hand-authored summary counters.  The trace is the primitive
    evidence for recursion, fallback classification, and recursive latency.
    """

    value = _mapping(raw, f"{label}.telemetry")
    metric_names = (
        "candidate_decisions",
        "planner_eligible_candidate_decisions",
        "over_cap_factorized_fallback_decisions",
        "forced_turn_order_controls",
        "intended_complex_decisions",
        "recursive_intended_complex_decisions",
        "successful_recursive_intended_complex_decisions",
        "direct_bridge_decisions",
        "recursive_decisions",
        "fallback_decisions",
        "unexpected_recursive_fallback_decisions",
        "expected_recursive_fallback_decisions",
        "neural_budget_exceeded",
        "neural_budget_failures",
        "illegal_action_count",
        "candidate_forfeit_count",
    )
    metrics = {name: _integer(value.get(name), f"{label}.telemetry.{name}") for name in metric_names}
    for name in (
        "candidate_decisions",
        "planner_eligible_candidate_decisions",
        "over_cap_factorized_fallback_decisions",
    ):
        metrics[name] = _exact_nonbool_integer(value.get(name), f"{label}.telemetry.{name}")
    if metrics["candidate_decisions"] != (
        metrics["planner_eligible_candidate_decisions"]
        + metrics["over_cap_factorized_fallback_decisions"]
    ):
        raise RTPPromotionEvidenceError(
            "candidate decisions must equal planner-eligible plus over-cap decisions"
        )
    over_cap_trace = _validated_over_cap_trace(
        value.get("over_cap_factorized_fallback_trace"),
        value.get("over_cap_factorized_fallback_trace_sha256"),
        arm=arm,
        label=label,
    )
    if len(over_cap_trace) != metrics["over_cap_factorized_fallback_decisions"]:
        raise RTPPromotionEvidenceError("over-cap counter does not match its dedicated trace")
    forced_turn_order_trace = _validated_forced_turn_order_control_trace(
        value.get("forced_turn_order_control_trace"), label=label
    )
    if metrics["forced_turn_order_controls"] != len(forced_turn_order_trace):
        raise RTPPromotionEvidenceError(
            "forced turn-order control count does not match its trace"
        )
    if value.get("intended_complex_decision_scope") != INTENDED_COMPLEX_DECISION_SCOPE:
        raise RTPPromotionEvidenceError("evaluation telemetry has an ambiguous complexity denominator")
    if metrics["successful_recursive_intended_complex_decisions"] != metrics[
        "recursive_intended_complex_decisions"
    ]:
        raise RTPPromotionEvidenceError("successful recursive-use counters disagree")
    if metrics["neural_budget_exceeded"] != metrics["neural_budget_failures"]:
        raise RTPPromotionEvidenceError("neural budget counters disagree")
    if metrics["candidate_forfeit_count"] > 1:
        raise RTPPromotionEvidenceError("candidate_forfeit_count exceeds one game forfeit")
    if any(
        metrics[name] > metrics["candidate_decisions"]
        for name in (
            "intended_complex_decisions",
            "direct_bridge_decisions",
            "recursive_decisions",
            "fallback_decisions",
            "illegal_action_count",
        )
    ):
        raise RTPPromotionEvidenceError("telemetry count exceeds candidate decisions")
    if (
        metrics["recursive_intended_complex_decisions"] > metrics["intended_complex_decisions"]
        or metrics["recursive_intended_complex_decisions"] > metrics["recursive_decisions"]
        or metrics["unexpected_recursive_fallback_decisions"]
        > metrics["intended_complex_decisions"]
        or metrics["unexpected_recursive_fallback_decisions"]
        + metrics["expected_recursive_fallback_decisions"]
        > metrics["fallback_decisions"]
    ):
        raise RTPPromotionEvidenceError("telemetry recursion/fallback counters are inconsistent")
    declared_latency_seconds = _number(
        value.get("latency_seconds"), f"{label}.telemetry.latency_seconds", minimum=0.0
    )
    raw_counts = _mapping(value.get("recursive_mode_counts"), f"{label}.recursive_mode_counts")
    if set(raw_counts) != set(_REQUIRED_RECURSIVE_MODE_COUNTS):
        raise RTPPromotionEvidenceError("telemetry has noncanonical recursive mode counts")
    declared_counts = {
        mode: _integer(raw_counts.get(mode), f"{label}.recursive_mode_counts.{mode}")
        for mode in _REQUIRED_RECURSIVE_MODE_COUNTS
    }
    trace_raw = value.get("decision_latency_trace")
    if not isinstance(trace_raw, Sequence) or isinstance(trace_raw, (str, bytes)):
        raise RTPPromotionEvidenceError("telemetry lacks a decision latency trace")
    if len(trace_raw) != metrics["candidate_decisions"]:
        raise RTPPromotionEvidenceError("telemetry trace does not cover every candidate decision")
    coarse_mode = {
        "no_rtp": "no_rtp",
        "direct_bridge": DIRECT_BRIDGE_ARM,
        _NONRECURSIVE_DIRECT_POLICY_MODE: DIRECT_BRIDGE_ARM,
        "recursive_plan": "recursive_rtp",
        "continue_plan": "recursive_rtp",
        "replan_with_program": "recursive_rtp",
        "direct_policy_fallback": "fallback",
        "replan_direct": "fallback",
        _OVER_CAP_FACTORIZED_FALLBACK_MODE: _OVER_CAP_FACTORIZED_FALLBACK_MODE,
    }
    allowed_by_arm = {
        "no_rtp": {"no_rtp", _OVER_CAP_FACTORIZED_FALLBACK_MODE},
        DIRECT_BRIDGE_ARM: {"direct_bridge", _OVER_CAP_FACTORIZED_FALLBACK_MODE},
        "recursive_rtp": (
            _SUCCESSFUL_RECURSIVE_MODES
            | _RECURSIVE_FALLBACK_MODES
            | {_NONRECURSIVE_DIRECT_POLICY_MODE, _OVER_CAP_FACTORIZED_FALLBACK_MODE}
        ),
    }
    observed_counts = {mode: 0 for mode in _REQUIRED_RECURSIVE_MODE_COUNTS}
    observed_intended = 0
    observed_recursive_intended = 0
    observed_direct_bridge = 0
    observed_fallback = {"expected": 0, "unexpected": 0}
    recursive_latencies: list[float] = []
    observed_latency_seconds = 0.0
    over_cap_by_decision = {item["decision_index"]: item for item in over_cap_trace}
    observed_over_cap_indexes: set[int] = set()
    for index, raw_trace in enumerate(trace_raw):
        trace = _mapping(raw_trace, f"{label}.trace[{index}]")
        if _integer(trace.get("decision_index"), f"{label}.trace[{index}].decision_index") != index:
            raise RTPPromotionEvidenceError("telemetry decision indexes are not monotonic")
        planner_mode = _nonempty_text(trace.get("planner_mode"), f"{label}.trace[{index}].planner_mode")
        if planner_mode not in coarse_mode or planner_mode not in allowed_by_arm[arm]:
            raise RTPPromotionEvidenceError("telemetry planner mode is incompatible with its arm")
        if trace.get("mode") != coarse_mode[planner_mode]:
            raise RTPPromotionEvidenceError("telemetry coarse mode disagrees with planner mode")
        _nonempty_text(trace.get("planner_reason"), f"{label}.trace[{index}].planner_reason")
        intended = trace.get("intended_complex")
        fallback = trace.get("fallback_classification")
        latency = _number(trace.get("latency_seconds"), f"{label}.trace[{index}].latency_seconds", minimum=0.0)
        observed_latency_seconds += latency
        if planner_mode == _OVER_CAP_FACTORIZED_FALLBACK_MODE:
            if set(trace) != {
                "decision_index",
                "mode",
                "planner_mode",
                "planner_reason",
                "intended_complex",
                "fallback_classification",
                "latency_seconds",
                "over_cap_trace_index",
            }:
                raise RTPPromotionEvidenceError("over-cap latency trace has an invalid exact field set")
            trace_index = _exact_nonbool_integer(
                trace.get("over_cap_trace_index"), f"{label}.over-cap trace index"
            )
            if trace_index >= len(over_cap_trace):
                raise RTPPromotionEvidenceError("over-cap latency trace index is out of range")
            special = over_cap_trace[trace_index]
            if special["decision_index"] != index:
                raise RTPPromotionEvidenceError("over-cap latency trace does not bind its special trace")
            if (
                trace.get("planner_reason") != _OVER_CAP_FACTORIZED_FALLBACK_REASON
                or intended is not None
                or fallback is not None
            ):
                raise RTPPromotionEvidenceError("over-cap latency trace entered planner/fallback accounting")
            observed_over_cap_indexes.add(index)
        else:
            if not isinstance(intended, bool):
                raise RTPPromotionEvidenceError("telemetry intended_complex must be boolean")
            if "over_cap_trace_index" in trace:
                raise RTPPromotionEvidenceError("ordinary trace carries an over-cap trace index")
        if planner_mode in _RECURSIVE_FALLBACK_MODES:
            if not intended or fallback not in observed_fallback:
                raise RTPPromotionEvidenceError("recursive fallback lacks intended-complex classification")
            observed_counts[planner_mode] += 1
            observed_fallback[str(fallback)] += 1
        else:
            if fallback not in {None, ""}:
                raise RTPPromotionEvidenceError("non-fallback trace declares a fallback classification")
            if planner_mode in _SUCCESSFUL_RECURSIVE_MODES:
                observed_counts[planner_mode] += 1
                recursive_latencies.append(latency)
                if intended:
                    observed_recursive_intended += 1
            elif planner_mode in {"direct_bridge", _NONRECURSIVE_DIRECT_POLICY_MODE}:
                observed_direct_bridge += 1
        if intended:
            observed_intended += 1
    if (
        observed_counts != declared_counts
        or len(recursive_latencies) != metrics["recursive_decisions"]
        or observed_intended != metrics["intended_complex_decisions"]
        or observed_recursive_intended != metrics["recursive_intended_complex_decisions"]
        or observed_direct_bridge != metrics["direct_bridge_decisions"]
        or sum(observed_counts[mode] for mode in _RECURSIVE_FALLBACK_MODES)
        != metrics["fallback_decisions"]
        or observed_fallback["expected"] != metrics["expected_recursive_fallback_decisions"]
        or observed_fallback["unexpected"] != metrics["unexpected_recursive_fallback_decisions"]
    ):
        raise RTPPromotionEvidenceError("telemetry counters do not match their decision trace")
    if observed_over_cap_indexes != set(over_cap_by_decision):
        raise RTPPromotionEvidenceError(
            "over-cap dedicated trace does not match its latency entries"
        )
    if metrics["planner_eligible_candidate_decisions"] != (
        metrics["candidate_decisions"] - len(observed_over_cap_indexes)
    ):
        raise RTPPromotionEvidenceError(
            "planner-eligible counter does not exclude exactly the over-cap decisions"
        )
    if not math.isclose(
        declared_latency_seconds,
        observed_latency_seconds,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RTPPromotionEvidenceError(
            "telemetry latency_seconds does not equal its candidate decision trace total"
        )
    recursive_latencies_raw = value.get("recursive_decision_latency_seconds")
    if not isinstance(recursive_latencies_raw, Sequence) or isinstance(
        recursive_latencies_raw, (str, bytes)
    ):
        raise RTPPromotionEvidenceError("telemetry lacks recursive-decision latency samples")
    observed_latencies = [
        _number(sample, f"{label}.recursive_decision_latency_seconds")
        for sample in recursive_latencies_raw
    ]
    if observed_latencies != recursive_latencies:
        raise RTPPromotionEvidenceError("recursive latency samples do not match planner-mode trace")
    normal_raw = value.get("normal_recursive_plan_passes")
    forced_raw = value.get("forced_replan_passes")
    if not isinstance(normal_raw, Sequence) or isinstance(normal_raw, (str, bytes)) or not isinstance(
        forced_raw, Sequence
    ) or isinstance(forced_raw, (str, bytes)):
        raise RTPPromotionEvidenceError("telemetry lacks planner-pass observations")
    normal = [_integer(item, f"{label}.normal_recursive_plan_passes") for item in normal_raw]
    forced = [_integer(item, f"{label}.forced_replan_passes") for item in forced_raw]
    if arm == "recursive_rtp":
        if any(item != 6 for item in normal) or any(item != 5 for item in forced):
            raise RTPPromotionEvidenceError("recursive planner pass observations violate the 6/5 contract")
        if metrics["fallback_decisions"] != sum(observed_fallback.values()):
            raise RTPPromotionEvidenceError("recursive fallbacks are not fully classified")
    elif normal or forced:
        raise RTPPromotionEvidenceError("nonrecursive arm recorded planner passes")
    if arm == "no_rtp" and any(
        metrics[name]
        for name in (
            "direct_bridge_decisions",
            "recursive_decisions",
            "recursive_intended_complex_decisions",
            "neural_budget_exceeded",
            "neural_budget_failures",
        )
    ):
        raise RTPPromotionEvidenceError("no_rtp row records RTP activity")
    if arm == DIRECT_BRIDGE_ARM and any(
        metrics[name]
        for name in (
            "recursive_decisions",
            "recursive_intended_complex_decisions",
            "neural_budget_exceeded",
            "neural_budget_failures",
        )
    ):
        raise RTPPromotionEvidenceError("direct bridge row records recursive activity")
    if arm != "recursive_rtp" and (
        metrics["expected_recursive_fallback_decisions"]
        or metrics["unexpected_recursive_fallback_decisions"]
    ):
        raise RTPPromotionEvidenceError("nonrecursive arm records recursive fallback classification")
    return {
        **metrics,
        "intended_complex_decision_scope": INTENDED_COMPLEX_DECISION_SCOPE,
        "recursive_mode_counts": declared_counts,
        "decision_latency_trace": [dict(item) for item in trace_raw],
        "over_cap_factorized_fallback_trace": over_cap_trace,
        "over_cap_factorized_fallback_trace_sha256": canonical_digest(over_cap_trace),
        "forced_turn_order_control_trace": forced_turn_order_trace,
        "recursive_decision_latency_seconds": recursive_latencies,
        "normal_recursive_plans": len(normal),
        "forced_replans": len(forced),
    }


def _validate_terminal_score(row: Mapping[str, Any], label: str) -> float:
    candidate_seat = _integer(row.get("candidate_seat"), f"{label}.candidate_seat")
    if candidate_seat not in {0, 1}:
        raise RTPPromotionEvidenceError("candidate seat is invalid")
    terminal = _mapping(row.get("terminal_outcome"), f"{label}.terminal_outcome")
    winner = _nonempty_text(terminal.get("winner"), f"{label}.terminal_outcome.winner")
    code = _integer(terminal.get("engine_result_code"), f"{label}.terminal_outcome.engine_result_code")
    expected = {
        "candidate": (candidate_seat, 1.0),
        "opponent": (1 - candidate_seat, 0.0),
        "draw": (2, 0.5),
    }
    if winner not in expected or expected[winner][0] != code:
        raise RTPPromotionEvidenceError("terminal outcome is not scored from the candidate seat")
    if terminal.get("termination") != "completed" or terminal.get("failed_seat") is not None:
        raise RTPPromotionEvidenceError("terminal outcome is not a completed non-failed game")
    if terminal.get("candidate_seat") != candidate_seat or not isinstance(
        terminal.get("candidate_forfeit"), bool
    ):
        raise RTPPromotionEvidenceError("terminal outcome has invalid candidate metadata")
    return expected[winner][1]


def _validate_row_runtime_identity(
    row: Mapping[str, Any], frozen: Mapping[str, Any], label: str
) -> None:
    """Bind each row's runtime description back to frozen candidate inputs."""

    arm = _nonempty_text(row.get("arm"), f"{label}.arm")
    arms = _mapping(frozen.get("arms"), "evaluation frozen arms")
    arm_spec = _mapping(arms.get(arm), f"evaluation frozen arm {arm}")
    profile = _mapping(arm_spec.get("profile"), f"evaluation frozen {arm} profile")
    runtime = _mapping(row.get("runtime_identity"), f"{label}.runtime_identity")
    sidecar = arm_spec.get("rtp_sidecar")
    direct_sidecar = _mapping(
        _mapping(arms.get(DIRECT_BRIDGE_ARM), f"evaluation frozen arm {DIRECT_BRIDGE_ARM}").get(
            "rtp_sidecar"
        ),
        "evaluation frozen complexity probe sidecar",
    )
    expected: dict[str, Any] = {
        "runtime_artifact_sha256": _sha256(
            _mapping(arm_spec.get("runtime_artifact"), f"evaluation frozen {arm} runtime").get("sha256"),
            f"evaluation frozen {arm} runtime sha",
        ),
        "runtime_profile_sha256": _sha256(
            _mapping(arm_spec.get("runtime_profile"), f"evaluation frozen {arm} profile identity").get("sha256"),
            f"evaluation frozen {arm} profile sha",
        ),
        "action_attached_rtp_sidecar_sha256": (
            None
            if sidecar is None
            else _sha256(
                _mapping(sidecar, f"evaluation frozen {arm} sidecar").get("sha256"),
                f"evaluation frozen {arm} sidecar sha",
            )
        ),
        "complexity_probe_sidecar_sha256": _sha256(
            direct_sidecar.get("sha256"), "evaluation frozen complexity probe sidecar sha"
        ),
        "complexity_probe_sidecar_instrumentation_only": True,
        # Instrumentation must never count against the recursive decision
        # latency SLO.  This belongs in the canonical runtime identity (rather
        # than merely in a child receipt) so a receipt cannot silently
        # substitute a timed complexity probe after execution.
        "complexity_probe_latency_excluded": True,
        "rtp_action_attachment_enabled": arm != "no_rtp",
        "rtp_action_authority_enabled": False,
        "arm": arm,
    }
    for name, identity in _mapping(
        frozen.get("shared_artifacts"), "evaluation shared artifacts"
    ).items():
        expected[f"{name}_sha256"] = _sha256(
            _mapping(identity, f"evaluation shared artifact {name}").get("sha256"),
            f"evaluation shared artifact {name} sha",
        )
    for field in (
        "recursive_turn_planner_enabled",
        "direct_bridge_enabled",
        "force_direct_bridge_only",
        "max_neural_passes",
        "max_action_combos",
    ):
        expected[field] = profile.get(field)
    if dict(runtime) != expected:
        raise RTPPromotionEvidenceError("validated row runtime identity is not the frozen arm contract")


def _validate_row_rng_identity(row: Mapping[str, Any], receipt: Mapping[str, Any], label: str) -> dict[str, Any]:
    rng = _mapping(row.get("rng_identity"), f"{label}.rng_identity")
    paired = _mapping(receipt.get("paired_rng_contract"), "paired_rng_contract")
    capability = _mapping(paired.get("capability"), "paired_rng_contract.capability")
    abi = _mapping(capability.get("abi"), "paired_rng_contract.capability.abi")
    expected = {
        "id": _nonempty_text(rng.get("id"), f"{label}.rng_identity.id"),
        "kind": "snapshot",
        "sha256": _sha256(rng.get("sha256"), f"{label}.rng_identity.sha256"),
        "bytes": _integer(rng.get("bytes"), f"{label}.rng_identity.bytes", minimum=1),
        "seal_sha256": _sha256(rng.get("seal_sha256"), f"{label}.rng_identity.seal_sha256"),
        "capture_boundary": _nonempty_text(
            rng.get("capture_boundary"), f"{label}.rng_identity.capture_boundary"
        ),
        "boundary_tag": _integer(rng.get("boundary_tag"), f"{label}.rng_identity.boundary_tag"),
    }
    if rng.get("kind") != "snapshot" or rng.get("restored_or_replayed") is not True:
        raise RTPPromotionEvidenceError("validated row lacks a restored native snapshot")
    if (
        expected["capture_boundary"] != abi.get("capture_boundary")
        or expected["boundary_tag"] != abi.get("boundary_tag")
    ):
        raise RTPPromotionEvidenceError("validated row snapshot differs from the pairing ABI boundary")
    if paired.get("requested_seed_is_pairing_proof") is not False or paired.get(
        "all_rows_attested_restored_or_replayed"
    ) is not True:
        raise RTPPromotionEvidenceError("receipt makes an invalid true-RNG pairing claim")
    return {**expected, "restored_or_replayed": True}


def _validate_local_execution_payload(
    *,
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    raw_result: Mapping[str, Any],
    execution: Mapping[str, Any],
    transcript: Mapping[str, Any],
    label: str,
    evidence_cache: _EvidenceReadCache,
) -> None:
    """Tie a normalized receipt row to both source result and child receipt."""

    inputs = _mapping(row.get("execution_input_sha256"), f"{label}.execution_input_sha256")
    if set(inputs) != {"runtime_identity", "rng_identity", "telemetry", "terminal_outcome"}:
        raise RTPPromotionEvidenceError("validated row lacks exact execution-input hashes")
    for name in inputs:
        digest = _sha256(inputs.get(name), f"{label}.execution_input_sha256.{name}")
        if canonical_digest(raw_result.get(name)) != digest:
            raise RTPPromotionEvidenceError(f"archived result differs at {name}")
    raw_execution = _immutable_identity(
        raw_result.get("execution_receipt"),
        f"{label}.source execution receipt",
        require_local=True,
        evidence_cache=evidence_cache,
    )
    raw_transcript = _immutable_identity(
        raw_result.get("transcript"),
        f"{label}.source transcript",
        require_local=True,
        evidence_cache=evidence_cache,
    )
    if raw_execution["sha256"] != execution["sha256"] or raw_transcript["sha256"] != transcript["sha256"]:
        raise RTPPromotionEvidenceError("archived result cites different execution evidence")
    if _score(raw_result.get("candidate_score"), f"{label}.source candidate_score") != _score(
        row.get("candidate_score"), f"{label}.candidate_score"
    ):
        raise RTPPromotionEvidenceError("archived result candidate score differs from validated row")
    transcript_payload = _json_from_immutable_evidence(
        transcript,
        f"{label}.transcript",
        evidence_cache=evidence_cache,
    )
    if transcript_payload.get("schema") != (
        "poke_bot.recursive_turn_planner.three_arm_transcript/v1"
    ):
        raise RTPPromotionEvidenceError("transcript is not an r198 child transcript")
    transcript_material = {
        key: value
        for key, value in transcript_payload.items()
        if key not in {"created_at_utc", "transcript_input_sha256"}
    }
    if _sha256(
        transcript_payload.get("transcript_input_sha256"),
        f"{label}.transcript_input_sha256",
    ) != canonical_digest(transcript_material):
        raise RTPPromotionEvidenceError("transcript canonical digest mismatch")
    if (
        transcript_payload.get("cell_id") != row.get("cell_id")
        or transcript_payload.get("arm") != row.get("arm")
        or canonical_digest(transcript_payload.get("decision_latency_trace"))
        != canonical_digest(telemetry["decision_latency_trace"])
        or canonical_digest(transcript_payload.get("over_cap_factorized_fallback_trace"))
        != canonical_digest(telemetry["over_cap_factorized_fallback_trace"])
    ):
        raise RTPPromotionEvidenceError(
            "transcript does not reconcile decision or over-cap telemetry"
        )
    payload = _json_from_immutable_evidence(
        execution,
        f"{label}.execution receipt",
        evidence_cache=evidence_cache,
    )
    if payload.get("schema") != EXECUTION_RECEIPT_SCHEMA or payload.get("status") != "completed":
        raise RTPPromotionEvidenceError("execution receipt is not a completed r198 child receipt")
    if payload.get("termination") != "completed" or payload.get("failed_seat") is not None:
        raise RTPPromotionEvidenceError("execution receipt is not a completed non-failed game")
    for field in ("engine_error", "candidate_error", "opponent_error"):
        if payload.get(field) not in {None, ""}:
            raise RTPPromotionEvidenceError(f"execution receipt contains {field}")
    if (
        payload.get("cell_id") != row.get("cell_id")
        or payload.get("arm") != row.get("arm")
        or payload.get("opponent_id") != row.get("opponent_id")
        or payload.get("candidate_seat") != row.get("candidate_seat")
        or payload.get("evaluation_case_id") != row.get("evaluation_case_id")
    ):
        raise RTPPromotionEvidenceError("execution receipt is bound to a different cell/arm")
    expected_payload_digests = {
        "transcript_sha256": transcript["sha256"],
        "runtime_identity_sha256": inputs["runtime_identity"],
        "rng_identity_sha256": inputs["rng_identity"],
        "telemetry_sha256": inputs["telemetry"],
        "terminal_outcome_sha256": inputs["terminal_outcome"],
        "evaluation_corpus_sha256": row.get("evaluation_corpus_sha256"),
        "evaluation_case_bindings_sha256": row.get("evaluation_case_bindings_sha256"),
    }
    for field, expected in expected_payload_digests.items():
        if _sha256(payload.get(field), f"{label}.execution receipt {field}") != expected:
            raise RTPPromotionEvidenceError(f"execution receipt differs at {field}")
    if _score(payload.get("candidate_score"), f"{label}.execution receipt candidate_score") != _score(
        row.get("candidate_score"), f"{label}.candidate_score"
    ):
        raise RTPPromotionEvidenceError("execution receipt candidate score differs")
    for field in (
        "fresh_process_per_arm",
        "process_model_load",
        "fresh_candidate_agent",
        "fresh_opponent_module",
        "candidate_reset_called",
        "engine_restore_before_first_select",
        "no_remote_leaf_sampling_mcts",
        "package_snapshot_verified_before_import",
        "complexity_probe_latency_excluded",
    ):
        if payload.get(field) is not True:
            raise RTPPromotionEvidenceError(f"execution receipt does not attest {field}")
    if payload.get("launch_mode") != "subprocess_exec":
        raise RTPPromotionEvidenceError("execution receipt does not use a fresh exec child")


def _recompute_arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RTPPromotionEvidenceError("cannot summarize an empty arm")
    scores = [_score(row.get("candidate_score"), "candidate_score") for row in rows]
    telemetry_rows = [
        _validated_row_telemetry(row.get("telemetry"), arm=str(row["arm"]), label="validated row")
        for row in rows
    ]
    total_names = (
        "candidate_decisions",
        "planner_eligible_candidate_decisions",
        "over_cap_factorized_fallback_decisions",
        "forced_turn_order_controls",
        "intended_complex_decisions",
        "recursive_intended_complex_decisions",
        "direct_bridge_decisions",
        "recursive_decisions",
        "fallback_decisions",
        "unexpected_recursive_fallback_decisions",
        "expected_recursive_fallback_decisions",
        "neural_budget_exceeded",
        "neural_budget_failures",
        "illegal_action_count",
        "candidate_forfeit_count",
        "normal_recursive_plans",
        "forced_replans",
    )
    totals = {name: sum(int(item[name]) for item in telemetry_rows) for name in total_names}
    trace = [entry for telemetry in telemetry_rows for entry in telemetry["decision_latency_trace"]]
    over_cap_trace = [
        entry
        for telemetry in telemetry_rows
        for entry in telemetry["over_cap_factorized_fallback_trace"]
    ]
    forced_turn_order_trace = [
        entry
        for telemetry in telemetry_rows
        for entry in telemetry["forced_turn_order_control_trace"]
    ]
    recursive_latencies = [
        latency
        for telemetry in telemetry_rows
        for latency in telemetry["recursive_decision_latency_seconds"]
    ]
    if len(trace) != totals["candidate_decisions"]:
        raise RTPPromotionEvidenceError("summary trace support does not match decision count")
    if len(over_cap_trace) != totals["over_cap_factorized_fallback_decisions"]:
        raise RTPPromotionEvidenceError(
            "summary over-cap trace support does not match its decision count"
        )
    if len(forced_turn_order_trace) != totals["forced_turn_order_controls"]:
        raise RTPPromotionEvidenceError(
            "summary forced turn-order trace support does not match control count"
        )
    if totals["forced_turn_order_controls"] != R198_PAIRED_CELLS // 2:
        raise RTPPromotionEvidenceError(
            "arm forced turn-order control count does not match the frozen seat ABI"
        )
    mode_totals = {
        mode: sum(int(telemetry["recursive_mode_counts"][mode]) for telemetry in telemetry_rows)
        for mode in _REQUIRED_RECURSIVE_MODE_COUNTS
    }
    by_opponent: dict[str, int] = defaultdict(int)
    seats = {"0": 0, "1": 0}
    for row in rows:
        by_opponent[_nonempty_text(row.get("opponent_id"), "opponent_id")] += 1
        seats[str(_integer(row.get("candidate_seat"), "candidate_seat"))] += 1
    intended = totals["intended_complex_decisions"]
    return {
        "games": len(rows),
        "mean_candidate_score": sum(scores) / len(scores),
        "wins": sum(score == 1.0 for score in scores),
        "draws": sum(score == 0.5 for score in scores),
        "losses": sum(score == 0.0 for score in scores),
        "candidate_seat_counts": seats,
        "by_opponent": dict(sorted(by_opponent.items())),
        "telemetry": {
            **totals,
            "intended_complex_decision_scope": INTENDED_COMPLEX_DECISION_SCOPE,
            "recursive_mode_counts": mode_totals,
            "fallback_rate": None
            if totals["planner_eligible_candidate_decisions"] == 0
            else totals["fallback_decisions"]
            / totals["planner_eligible_candidate_decisions"],
            "over_cap_factorized_fallback_rate": None
            if totals["candidate_decisions"] == 0
            else totals["over_cap_factorized_fallback_decisions"]
            / totals["candidate_decisions"],
            "recursive_share_of_intended_complex_decisions": None
            if intended == 0
            else totals["recursive_intended_complex_decisions"] / intended,
            "unexpected_recursive_fallback_rate": None
            if intended == 0
            else totals["unexpected_recursive_fallback_decisions"] / intended,
            "decision_latency_trace": {"count": len(trace), "sha256": canonical_digest(trace)},
            "forced_turn_order_control_trace": {
                "count": len(forced_turn_order_trace),
                "sha256": canonical_digest(forced_turn_order_trace),
            },
            "over_cap_factorized_fallback_trace": {
                "count": len(over_cap_trace),
                "sha256": canonical_digest(over_cap_trace),
            },
            "recursive_decision_latency_distribution": {
                "count": len(recursive_latencies),
                "sha256": canonical_digest(recursive_latencies),
                "p50_seconds": _percentile(recursive_latencies, 0.50),
                "p95_seconds": _percentile(recursive_latencies, 0.95),
                "max_seconds": max(recursive_latencies, default=None),
            },
        },
    }


def _verify_forced_turn_order_pairing(
    per_cell_controls: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> None:
    """Require byte-identical external-control evidence in every paired cell."""

    for cell_id, controls_by_arm in per_cell_controls.items():
        if set(controls_by_arm) != set(ARMS):
            raise RTPPromotionEvidenceError(
                "forced turn-order controls lack complete three-arm pairing"
            )
        reference = controls_by_arm["no_rtp"]
        for arm in ARMS:
            observed = controls_by_arm[arm]
            if observed["count"] != reference["count"]:
                raise RTPPromotionEvidenceError(
                    "forced turn-order control counts differ across a paired cell"
                )
            if canonical_digest(observed["trace"]) != canonical_digest(reference["trace"]):
                raise RTPPromotionEvidenceError(
                    "forced turn-order control traces differ across a paired cell"
                )


def _validate_conditional_over_cap_action_parity(
    per_cell_traces: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, int]:
    """Recompute the narrow cross-arm parity claim for over-cap selections.

    B/C bridge state is intentionally not comparable to A.  The dedicated
    trace instead provides a causal factorized-policy input digest.  Only
    rows in the same paired cell with that *same* digest are comparable; no
    count or action equality is inferred for divergent policy histories.
    """

    groups: dict[tuple[str, str], list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    total = 0
    for cell_id, traces_by_arm in per_cell_traces.items():
        if set(traces_by_arm) != set(ARMS):
            raise RTPPromotionEvidenceError(
                "over-cap traces lack complete three-arm pairing"
            )
        for arm, traces in traces_by_arm.items():
            for trace in traces:
                value = _mapping(trace, "over-cap factorized trace")
                fingerprint = _sha256(
                    value.get("logical_pre_action_sha256"),
                    "over-cap logical pre-action digest",
                )
                groups[(cell_id, fingerprint)].append((arm, value))
                total += 1
    comparable_groups = 0
    comparable_arm_rows = 0
    for entries in groups.values():
        if len({arm for arm, _trace in entries}) < 2:
            continue
        comparable_groups += 1
        comparable_arm_rows += len(entries)
        actions = {
            tuple(
                _over_cap_factorized_action(
                    trace.get("returned_action"),
                    _over_cap_action_space(trace.get("action_space"), label="over-cap parity action_space"),
                    label="over-cap parity trace",
                )[0]
            )
            for _arm, trace in entries
        }
        if len(actions) != 1:
            raise RTPPromotionEvidenceError(
                "over-cap factorized actions differ despite identical logical pre-action inputs"
            )
    return {
        "over_cap_trace_rows": total,
        "logical_pre_action_groups": len(groups),
        "cross_arm_comparable_groups": comparable_groups,
        "cross_arm_comparable_arm_rows": comparable_arm_rows,
    }


def _recompute_paired_summary(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    confidence: float,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise RTPPromotionEvidenceError("paired comparison lacks exact cell support")
    deltas = [
        _score(right[cell].get("candidate_score"), "right candidate_score")
        - _score(left[cell].get("candidate_score"), "left candidate_score")
        for cell in sorted(left)
    ]
    return {
        "endpoint": "paired_expected_score_delta",
        "pairs": len(deltas),
        "mean_score_delta": sum(deltas) / len(deltas),
        "one_sided_lower_confidence_bound": _one_sided_hoeffding_lower(deltas, confidence),
        "confidence_level": confidence,
        "confidence_method": "one_sided_hoeffding_for_true_rng_paired_score_deltas",
    }


def _validate_panel_and_rows(
    receipt: Mapping[str, Any],
    *,
    require_local_evidence: bool,
    evidence_cache: _EvidenceReadCache,
) -> list[dict[str, Any]]:
    frozen = _mapping(receipt.get("frozen_artifacts"), "evaluation frozen_artifacts")
    opponents_raw = frozen.get("opponents")
    if not isinstance(opponents_raw, Sequence) or isinstance(opponents_raw, (str, bytes)):
        raise RTPPromotionEvidenceError("evaluation frozen opponents must be a list")
    opponents: dict[str, str] = {}
    for index, raw_opponent in enumerate(opponents_raw):
        opponent = _mapping(raw_opponent, f"evaluation opponent[{index}]")
        name = _nonempty_text(opponent.get("id"), f"evaluation opponent[{index}].id")
        if name in opponents:
            raise RTPPromotionEvidenceError("evaluation receipt repeats an opponent")
        opponents[name] = _sha256(
            opponent.get("content_digest"), f"evaluation opponent[{index}].content_digest"
        )
    if opponents != R198_OFFICIAL_CONTROL_OPPONENTS:
        raise RTPPromotionEvidenceError("evaluation receipt does not bind the official r198 panel")
    official_panel = _mapping(receipt.get("official_control_panel"), "official_control_panel")
    registry = _immutable_identity(
        official_panel.get("registry"),
        "official_control_panel.registry",
        require_local=require_local_evidence,
        evidence_cache=evidence_cache,
    )
    if registry["sha256"] != R198_RESEARCH_CONTROL_REGISTRY_SHA256:
        raise RTPPromotionEvidenceError("evaluation receipt has the wrong control registry")
    if _mapping(official_panel.get("opponents"), "official_control_panel.opponents") != R198_OFFICIAL_CONTROL_OPPONENTS:
        raise RTPPromotionEvidenceError("evaluation receipt control-panel rows are wrong")

    binding = _mapping(
        receipt.get("r197_source_exclusion_binding"), "r197_source_exclusion_binding"
    )
    if _sha256(
        binding.get("candidate_contract_sha256"), "candidate_contract_sha256"
    ) != R198_CANDIDATE_CONTRACT_SHA256:
        raise RTPPromotionEvidenceError("evaluation receipt is not bound to the exact r198 candidate")
    if (
        binding.get("r197_source_disjoint") is not True
        or binding.get("evaluation_only") is not True
        or _integer(binding.get("source_identity_overlap_count"), "source_identity_overlap_count") != 0
    ):
        raise RTPPromotionEvidenceError("evaluation receipt lacks source-exclusion evidence")
    # This is a property of the immutable completed r197 candidate, not an
    # evaluation runner claim.  A hand-authored receipt may not flip it to
    # ``true`` and thereby manufacture promotion authority.
    if (
        binding.get("candidate_target_status") != "masked_absent_no_fabrication"
        or binding.get("trusted_counterfactual_candidate_targets_available") is not False
    ):
        raise RTPPromotionEvidenceError(
            "evaluation receipt tampers with the fixed r197 masked-target provenance"
        )
    cohort = _immutable_identity(
        binding.get("evaluation_only_cohort"),
        "evaluation_only_cohort",
        require_local=require_local_evidence,
        evidence_cache=evidence_cache,
    )
    case_bindings = binding.get("evaluation_case_bindings")
    if not isinstance(case_bindings, Sequence) or isinstance(case_bindings, (str, bytes)):
        raise RTPPromotionEvidenceError("evaluation receipt lacks frozen evaluation case bindings")
    normalized_cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    case_keys: set[tuple[str, int, int]] = set()
    expected_case_keys = {
        (opponent, seat, replicate)
        for opponent in R198_OFFICIAL_CONTROL_OPPONENTS
        for seat in (0, 1)
        for replicate in range(125)
    }
    for index, raw_case in enumerate(case_bindings):
        case = _mapping(raw_case, f"evaluation case binding[{index}]")
        case_id = _nonempty_text(case.get("case_id"), f"evaluation case binding[{index}].case_id")
        if case_id in case_ids:
            raise RTPPromotionEvidenceError("evaluation receipt repeats a case id")
        case_ids.add(case_id)
        opponent = _nonempty_text(case.get("opponent_id"), "evaluation case opponent_id")
        digest = _sha256(case.get("content_digest"), "evaluation case content_digest")
        seat = _integer(case.get("candidate_seat"), "evaluation case candidate_seat")
        replicate = _integer(case.get("replicate"), "evaluation case replicate")
        if (opponent, seat, replicate) not in expected_case_keys or (
            R198_OFFICIAL_CONTROL_OPPONENTS[opponent] != digest
        ):
            raise RTPPromotionEvidenceError("evaluation receipt has an invalid official case")
        case_key = (opponent, seat, replicate)
        if case_key in case_keys:
            raise RTPPromotionEvidenceError("evaluation receipt repeats an official case tuple")
        case_keys.add(case_key)
        normalized_cases.append(
            {
                "case_id": case_id,
                "opponent_id": opponent,
                "content_digest": digest,
                "candidate_seat": seat,
                "replicate": replicate,
            }
        )
    if len(normalized_cases) != R198_PAIRED_CELLS or case_keys != expected_case_keys:
        raise RTPPromotionEvidenceError("evaluation receipt lacks 1,000 frozen official cases")
    if _sha256(
        binding.get("evaluation_case_bindings_sha256"),
        "evaluation_case_bindings_sha256",
    ) != canonical_digest(sorted(normalized_cases, key=lambda item: item["case_id"])):
        raise RTPPromotionEvidenceError("evaluation case binding digest mismatch")
    case_by_id = {case["case_id"]: case for case in normalized_cases}
    case_binding_digest = _sha256(
        binding.get("evaluation_case_bindings_sha256"),
        "evaluation_case_bindings_sha256",
    )

    rows_raw = receipt.get("validated_rows")
    if not isinstance(rows_raw, Sequence) or isinstance(rows_raw, (str, bytes)):
        raise RTPPromotionEvidenceError("evaluation receipt lacks validated rows")
    if len(rows_raw) != R198_ROWS:
        raise RTPPromotionEvidenceError("evaluation receipt does not contain 3,000 arm rows")
    if _sha256(receipt.get("result_rows_sha256"), "result_rows_sha256") != canonical_digest(
        list(rows_raw)
    ):
        raise RTPPromotionEvidenceError("validated rows do not match result_rows_sha256")

    source_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if require_local_evidence:
        results_identity = _immutable_identity(
            receipt.get("results"),
            "evaluation results",
            require_local=True,
            evidence_cache=evidence_cache,
        )
        results_payload = _json_from_immutable_evidence(
            results_identity,
            "evaluation results",
            evidence_cache=evidence_cache,
        )
        raw_source_rows = results_payload.get("rows")
        if not isinstance(raw_source_rows, Sequence) or isinstance(raw_source_rows, (str, bytes)):
            raise RTPPromotionEvidenceError("evaluation results lack source rows")
        if len(raw_source_rows) != R198_ROWS:
            raise RTPPromotionEvidenceError("evaluation results have the wrong row count")
        for index, raw_source in enumerate(raw_source_rows):
            source = _mapping(raw_source, f"evaluation results.rows[{index}]")
            key = (
                _nonempty_text(source.get("cell_id"), "evaluation result cell_id"),
                _canonical_arm(source.get("arm"), "evaluation result arm"),
            )
            if key[1] not in ARMS or key in source_rows:
                raise RTPPromotionEvidenceError("evaluation results have duplicate/noncanonical arm rows")
            source_rows[key] = source

    expected_cohort_digest = cohort["sha256"]
    by_cell: dict[str, set[str]] = defaultdict(set)
    cell_case_ids: dict[str, set[str]] = defaultdict(set)
    case_rows: dict[str, int] = defaultdict(int)
    execution_digests: set[str] = set()
    transcript_digests: set[str] = set()
    process_ids: set[str] = set()
    launch_nonces: set[str] = set()
    per_cell_rng: dict[str, set[str]] = defaultdict(set)
    per_cell_candidate_rng: dict[str, set[str]] = defaultdict(set)
    per_cell_environment: dict[str, set[str]] = defaultdict(set)
    per_cell_forced_turn_order: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    per_cell_over_cap_traces: dict[
        str, dict[str, Sequence[Mapping[str, Any]]]
    ] = defaultdict(dict)
    normalized_rows: list[dict[str, Any]] = []
    frozen = _mapping(receipt.get("frozen_artifacts"), "evaluation frozen_artifacts")
    for index, raw_row in enumerate(rows_raw):
        row = _mapping(raw_row, f"validated_rows[{index}]")
        cell_id = _nonempty_text(row.get("cell_id"), f"validated_rows[{index}].cell_id")
        arm = _nonempty_text(row.get("arm"), f"validated_rows[{index}].arm")
        if arm not in ARMS or arm in by_cell[cell_id]:
            raise RTPPromotionEvidenceError("evaluation rows have a noncanonical or duplicate arm")
        by_cell[cell_id].add(arm)
        case_id = _nonempty_text(row.get("evaluation_case_id"), "validated row evaluation_case_id")
        case_rows[case_id] += 1
        cell_case_ids[cell_id].add(case_id)
        case = case_by_id.get(case_id)
        if case is None or (
            row.get("opponent_id") != case["opponent_id"]
            or _integer(row.get("candidate_seat"), "validated row candidate_seat")
            != case["candidate_seat"]
        ):
            raise RTPPromotionEvidenceError("validated row does not bind its frozen evaluation case")
        if _sha256(
            row.get("evaluation_case_bindings_sha256"),
            "validated row evaluation_case_bindings_sha256",
        ) != case_binding_digest:
            raise RTPPromotionEvidenceError("validated row has the wrong evaluation-case binding")
        if _sha256(
            row.get("evaluation_corpus_sha256"), "validated row evaluation_corpus_sha256"
        ) != expected_cohort_digest:
            raise RTPPromotionEvidenceError("validated row is not bound to the frozen cohort")
        terminal_score = _validate_terminal_score(row, f"validated_rows[{index}]")
        if _score(row.get("candidate_score"), "validated row candidate_score") != terminal_score:
            raise RTPPromotionEvidenceError("validated row score disagrees with terminal result")
        telemetry = _validated_row_telemetry(
            row.get("telemetry"), arm=arm, label=f"validated_rows[{index}]"
        )
        _validate_r198_forced_turn_order_contract(
            telemetry,
            candidate_seat=_integer(row.get("candidate_seat"), "validated row candidate_seat"),
            label=f"validated_rows[{index}]",
        )
        per_cell_forced_turn_order[cell_id][arm] = {
            "count": telemetry["forced_turn_order_controls"],
            "trace": telemetry["forced_turn_order_control_trace"],
        }
        per_cell_over_cap_traces[cell_id][arm] = telemetry[
            "over_cap_factorized_fallback_trace"
        ]
        if telemetry["candidate_forfeit_count"] != int(
            _mapping(row.get("terminal_outcome"), "validated row terminal_outcome").get(
                "candidate_forfeit"
            )
        ):
            raise RTPPromotionEvidenceError("validated row forfeit telemetry disagrees with terminal result")
        _validate_row_runtime_identity(row, frozen, f"validated_rows[{index}]")
        rng = _validate_row_rng_identity(row, receipt, f"validated_rows[{index}]")
        per_cell_rng[cell_id].add(canonical_digest(rng))
        evidence = _mapping(row.get("execution_evidence"), "validated row execution_evidence")
        execution = _immutable_identity(
            evidence.get("execution_receipt"),
            "validated row execution receipt",
            require_local=require_local_evidence,
            evidence_cache=evidence_cache,
        )
        transcript = _immutable_identity(
            evidence.get("transcript"),
            "validated row transcript",
            require_local=require_local_evidence,
            evidence_cache=evidence_cache,
        )
        if execution["sha256"] in execution_digests or transcript["sha256"] in transcript_digests:
            raise RTPPromotionEvidenceError("evaluation receipt reuses execution evidence")
        execution_digests.add(execution["sha256"])
        transcript_digests.add(transcript["sha256"])
        process_id = _nonempty_text(evidence.get("process_id"), "validated row process_id")
        launch_nonce = _nonempty_text(evidence.get("launch_nonce"), "validated row launch_nonce")
        if process_id in process_ids or launch_nonce in launch_nonces:
            raise RTPPromotionEvidenceError("evaluation receipt reuses a child process or launch nonce")
        process_ids.add(process_id)
        launch_nonces.add(launch_nonce)
        candidate_rng = _sha256(
            evidence.get("candidate_rng_initial_state_sha256"), "validated row candidate RNG"
        )
        environment = _sha256(
            evidence.get("common_sanitized_environment_sha256"),
            "validated row common sanitized environment",
        )
        per_cell_candidate_rng[cell_id].add(candidate_rng)
        per_cell_environment[cell_id].add(environment)
        if require_local_evidence:
            source = source_rows.get((cell_id, arm))
            if source is None:
                raise RTPPromotionEvidenceError("validated row is absent from immutable results")
            if (
                source.get("completed") is not True
                or source.get("invalid") not in {None, False}
                or source.get("error") not in {None, ""}
                or source.get("opponent_id") != row.get("opponent_id")
                or source.get("candidate_seat") != row.get("candidate_seat")
                or source.get("evaluation_case_id") != row.get("evaluation_case_id")
                or _sha256(
                    source.get("evaluation_case_bindings_sha256"),
                    "evaluation source case binding",
                )
                != case_binding_digest
            ):
                raise RTPPromotionEvidenceError("immutable source result differs from its validated row")
            _validate_local_execution_payload(
                row=row,
                telemetry=telemetry,
                raw_result=source,
                execution=execution,
                transcript=transcript,
                label=f"validated_rows[{index}]",
                evidence_cache=evidence_cache,
            )
        normalized_rows.append(row)
    if len(by_cell) != R198_PAIRED_CELLS or any(arms != set(ARMS) for arms in by_cell.values()):
        raise RTPPromotionEvidenceError("evaluation receipt has incomplete three-arm pairing")
    _verify_forced_turn_order_pairing(per_cell_forced_turn_order)
    over_cap_parity = _validate_conditional_over_cap_action_parity(
        per_cell_over_cap_traces
    )
    expected_over_cap = {
        "classification": _OVER_CAP_FACTORIZED_FALLBACK_REASON,
        "complete_ordered_action_cap": R198_MAX_ACTION_COMBOS,
        "conditional_cross_arm_action_parity": over_cap_parity,
    }
    if canonical_digest(
        _mapping(
            receipt.get("over_cap_factorized_fallback"),
            "over_cap_factorized_fallback",
        )
    ) != canonical_digest(expected_over_cap):
        raise RTPPromotionEvidenceError(
            "evaluation receipt over-cap factorized summary does not derive from validated rows"
        )
    if any(len(case_ids_for_cell) != 1 for case_ids_for_cell in cell_case_ids.values()):
        raise RTPPromotionEvidenceError("a paired cell mixes frozen evaluation cases")
    if set(case_rows) != case_ids or any(count != len(ARMS) for count in case_rows.values()):
        raise RTPPromotionEvidenceError("evaluation receipt does not execute each frozen case in all arms")
    if require_local_evidence and set(source_rows) != {
        (str(row["cell_id"]), str(row["arm"])) for row in normalized_rows
    }:
        raise RTPPromotionEvidenceError("immutable results have unvalidated or missing rows")
    if any(
        len(per_cell_rng[cell_id]) != 1
        or len(per_cell_candidate_rng[cell_id]) != 1
        or len(per_cell_environment[cell_id]) != 1
        for cell_id in by_cell
    ):
        raise RTPPromotionEvidenceError("three-arm rows are not paired to one RNG/common environment")
    return normalized_rows


def _validate_efficacy_semantics(
    receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    require_local_evidence: bool,
    evidence_cache: _EvidenceReadCache,
) -> None:
    """Re-derive scores, paired LCBs, strata, and reliability from row evidence."""

    by_arm: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in ARMS}
    for row in rows:
        arm = _nonempty_text(row.get("arm"), "validated row arm")
        if arm not in by_arm:
            raise RTPPromotionEvidenceError("validated row has a noncanonical arm")
        by_arm[arm].append(row)
    summaries = _mapping(receipt.get("arm_summaries"), "arm_summaries")
    if set(summaries) != set(ARMS):
        raise RTPPromotionEvidenceError("evaluation receipt arm summaries are noncanonical")
    recomputed_summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        recomputed = _recompute_arm_summary(by_arm[arm])
        claimed = _mapping(summaries.get(arm), f"arm_summaries.{arm}")
        if canonical_digest(claimed) != canonical_digest(recomputed):
            raise RTPPromotionEvidenceError(f"{arm} summary does not derive from validated rows")
        recomputed_summaries[arm] = recomputed
        if recomputed["telemetry"]["neural_budget_failures"] != 0:
            raise RTPPromotionEvidenceError("evaluation receipt records a neural budget failure")

    direct = _mapping(recomputed_summaries[DIRECT_BRIDGE_ARM]["telemetry"], "direct telemetry")
    recursive = _mapping(recomputed_summaries["recursive_rtp"]["telemetry"], "recursive telemetry")
    if _integer(direct.get("direct_bridge_decisions"), "direct bridge decisions") < 1:
        raise RTPPromotionEvidenceError("evaluation receipt did not exercise the direct bridge")
    if _integer(recursive.get("recursive_decisions"), "recursive decisions") < R198_MINIMUM_RECURSIVE_DECISIONS:
        raise RTPPromotionEvidenceError("evaluation receipt did not exercise recursion sufficiently")
    intended = _integer(recursive.get("intended_complex_decisions"), "intended_complex_decisions")
    successful = _integer(
        recursive.get("recursive_intended_complex_decisions"),
        "successful_recursive_intended_complex_decisions",
    )
    if intended < 1 or successful / intended < R198_MINIMUM_RECURSIVE_SHARE:
        raise RTPPromotionEvidenceError("evaluation receipt misses the 5% successful recursive-use floor")
    unexpected = _integer(
        recursive.get("unexpected_recursive_fallback_decisions"),
        "unexpected_recursive_fallback_decisions",
    )
    if unexpected > intended or unexpected / intended > R198_MAXIMUM_UNEXPECTED_FALLBACK_RATE:
        raise RTPPromotionEvidenceError("evaluation receipt exceeds the unexpected fallback ceiling")
    latency = _mapping(
        recursive.get("recursive_decision_latency_distribution"),
        "recursive decision latency distribution",
    )
    if _integer(latency.get("count"), "recursive latency samples") < R198_MINIMUM_RECURSIVE_DECISIONS:
        raise RTPPromotionEvidenceError("evaluation receipt lacks recursive-only latency evidence")
    p95 = _number(latency.get("p95_seconds"), "recursive latency p95_seconds", minimum=0.0)
    slo = _mapping(receipt.get("latency_slo"), "latency_slo")
    if (
        slo.get("status") != "approved"
        or slo.get("owner_authorized") is not True
        or slo.get("metric") != "recursive_decision_p95_seconds"
    ):
        raise RTPPromotionEvidenceError("evaluation receipt lacks an owner-authorized latency SLO")
    maximum_latency = _number(
        slo.get("maximum_p95_latency_seconds"), "latency_slo maximum", minimum=0.0
    )
    if p95 > maximum_latency:
        raise RTPPromotionEvidenceError("recursive latency evidence exceeds its owner SLO")
    slo_identity = _immutable_identity(
        slo.get("receipt"),
        "latency_slo.receipt",
        require_local=require_local_evidence,
        evidence_cache=evidence_cache,
    )
    if require_local_evidence:
        slo_payload = _json_from_immutable_evidence(
            slo_identity,
            "latency_slo.receipt",
            evidence_cache=evidence_cache,
        )
        if (
            slo_payload.get("schema")
            != "poke_bot.recursive_turn_planner.r198_recursive_latency_slo/v1"
            or slo_payload.get("status") != "approved"
            or slo_payload.get("owner_authorized") is not True
            or slo_payload.get("metric") != "recursive_decision_p95_seconds"
            or _sha256(
                slo_payload.get("candidate_contract_sha256"), "latency SLO candidate contract"
            )
            != R198_CANDIDATE_CONTRACT_SHA256
            or _number(
                slo_payload.get("maximum_p95_latency_seconds"), "latency SLO receipt maximum", minimum=0.0
            )
            != maximum_latency
        ):
            raise RTPPromotionEvidenceError("latency SLO receipt does not bind this candidate threshold")

    row_maps = {
        arm: {str(row["cell_id"]): row for row in by_arm[arm]}
        for arm in ARMS
    }
    comparison_names = {
        f"{DIRECT_BRIDGE_ARM}_minus_no_rtp": ("no_rtp", DIRECT_BRIDGE_ARM),
        f"recursive_rtp_minus_{DIRECT_BRIDGE_ARM}": (DIRECT_BRIDGE_ARM, "recursive_rtp"),
        "recursive_rtp_minus_no_rtp": ("no_rtp", "recursive_rtp"),
    }
    comparisons = _mapping(receipt.get("comparisons"), "comparisons")
    if set(comparisons) != set(comparison_names):
        raise RTPPromotionEvidenceError("evaluation receipt comparison set is noncanonical")
    primary_name = f"recursive_rtp_minus_{DIRECT_BRIDGE_ARM}"
    for name, (left_arm, right_arm) in comparison_names.items():
        claimed = _mapping(comparisons.get(name), name)
        confidence = _number(claimed.get("confidence_level"), f"{name} confidence")
        if not 0.90 <= confidence < 1.0:
            raise RTPPromotionEvidenceError("evaluation comparison weakens the r198 confidence floor")
        expected = _recompute_paired_summary(
            row_maps[left_arm], row_maps[right_arm], confidence=confidence
        )
        if name == primary_name:
            groups: dict[str, tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]] = {}
            for cell_id, left in row_maps[left_arm].items():
                right = row_maps[right_arm][cell_id]
                stratum = f"{left['opponent_id']}|seat{left['candidate_seat']}"
                if stratum not in groups:
                    groups[stratum] = ({}, {})
                groups[stratum][0][cell_id] = left
                groups[stratum][1][cell_id] = right
            expected["opponent_seat_stratified"] = {
                stratum: _recompute_paired_summary(group_left, group_right, confidence=confidence)
                for stratum, (group_left, group_right) in sorted(groups.items())
            }
        if canonical_digest(claimed) != canonical_digest(expected):
            raise RTPPromotionEvidenceError(f"{name} does not derive from validated scores")
    primary = _mapping(comparisons.get(primary_name), primary_name)
    if _integer(primary.get("pairs"), "primary pairs") != R198_PAIRED_CELLS:
        raise RTPPromotionEvidenceError("evaluation receipt has the wrong paired support")
    if _number(primary.get("one_sided_lower_confidence_bound"), "primary LCB") <= 0.0:
        raise RTPPromotionEvidenceError("evaluation receipt has no strictly positive recursive efficacy LCB")
    strata = _mapping(primary.get("opponent_seat_stratified"), "opponent_seat_stratified")
    expected_strata = {
        f"{opponent}|seat{seat}"
        for opponent in R198_OFFICIAL_CONTROL_OPPONENTS
        for seat in (0, 1)
    }
    if set(strata) != expected_strata or any(
        _integer(_mapping(summary, "stratum").get("pairs"), "stratum pairs") != 125
        for summary in strata.values()
    ):
        raise RTPPromotionEvidenceError("evaluation receipt lacks exact opponent×seat support")

    regressions = _mapping(receipt.get("reliability_regressions"), "reliability_regressions")
    expected_regressions: dict[str, dict[str, dict[str, int]]] = {}
    for label, metric in (("illegal_action", "illegal_action_count"), ("new_forfeit", "candidate_forfeit_count")):
        expected_regressions[label] = {}
        for baseline_arm in (DIRECT_BRIDGE_ARM, "no_rtp"):
            key = f"recursive_rtp_vs_{baseline_arm}"
            increases = [
                max(
                    0,
                    int(_mapping(row_maps["recursive_rtp"][cell]["telemetry"], "telemetry")[metric])
                    - int(_mapping(row_maps[baseline_arm][cell]["telemetry"], "telemetry")[metric]),
                )
                for cell in sorted(row_maps[baseline_arm])
            ]
            expected_regressions[label][key] = {
                "new_count": sum(increases),
                "cells_with_regression": sum(value > 0 for value in increases),
                "maximum_per_cell_increase": max(increases, default=0),
            }
    if canonical_digest(regressions) != canonical_digest(expected_regressions):
        raise RTPPromotionEvidenceError("reliability regressions do not derive from validated rows")
    if any(
        summary["new_count"] != 0
        for category in expected_regressions.values()
        for summary in category.values()
    ):
        raise RTPPromotionEvidenceError("evaluation has a new illegal-action or forfeit regression")


def _candidate_evaluation_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate raw-artifact and semantic serving identities separately."""

    binding = _mapping(receipt.get("candidate_evaluation_binding"), "candidate_evaluation_binding")
    expected: dict[str, Any] = {
        "schema": CANDIDATE_EVALUATION_BINDING_SCHEMA,
        "status": "bound",
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "parent_checkpoint_sha256": R198_PARENT_CHECKPOINT_SHA256,
        "sidecar_sha256": R198_SIDECAR_SHA256,
        "sidecar_config_sha256": R198_SIDECAR_CONFIG_SHA256,
        "deck_file_sha256": R198_DECK_FILE_SHA256,
        "deck_cards_sha256": R198_DECK_CARDS_SHA256,
        "matchup_tree_sha256": R198_MATCHUP_TREE_SHA256,
        "sizing_profile": "pure_rl_r197",
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "required_neural_passes": {"normal": 6, "forced_replan": 5},
    }
    if set(binding) != set(expected):
        raise RTPPromotionEvidenceError("candidate evaluation binding is incomplete or has aliases")
    for key, required in expected.items():
        observed = binding.get(key)
        if key.endswith("_sha256"):
            observed = _sha256(observed, f"candidate_evaluation_binding.{key}")
        elif key in {"max_neural_passes", "max_action_combos"}:
            observed = _integer(observed, f"candidate_evaluation_binding.{key}")
        elif key == "required_neural_passes":
            observed = _mapping(observed, f"candidate_evaluation_binding.{key}")
        if observed != required:
            raise RTPPromotionEvidenceError(f"candidate evaluation binding differs at {key}")
    frozen = _mapping(receipt.get("frozen_artifacts"), "evaluation frozen_artifacts")
    shared = _mapping(frozen.get("shared_artifacts"), "evaluation shared_artifacts")
    for artifact, field in (
        ("parent_checkpoint", "parent_checkpoint_sha256"),
        ("deck", "deck_file_sha256"),
        ("matchup_tree", "matchup_tree_sha256"),
    ):
        if _sha256(
            _mapping(shared.get(artifact), f"evaluation {artifact}").get("sha256"),
            f"evaluation {artifact}.sha256",
        ) != expected[field]:
            raise RTPPromotionEvidenceError(
                f"candidate evaluation binding is not tied to frozen {artifact} bytes"
            )
    arms = _mapping(frozen.get("arms"), "evaluation arms")
    for arm in (DIRECT_BRIDGE_ARM, "recursive_rtp"):
        arm_spec = _mapping(arms.get(arm), f"evaluation arm {arm}")
        if _sha256(
            _mapping(arm_spec.get("rtp_sidecar"), f"evaluation arm {arm} sidecar").get("sha256"),
            f"evaluation arm {arm} sidecar sha256",
        ) != expected["sidecar_sha256"]:
            raise RTPPromotionEvidenceError("candidate evaluation binding is not tied to its sidecar")
        profile = _mapping(arm_spec.get("profile"), f"evaluation arm {arm} profile")
        for field in ("sizing_profile", "max_neural_passes", "max_action_combos"):
            if profile.get(field) != expected[field]:
                raise RTPPromotionEvidenceError("candidate evaluation binding differs from arm profile")
    return expected


def _validate_expected_candidate_bindings(
    receipt: Mapping[str, Any],
    *,
    expected_parent_checkpoint_sha256: str | None,
    expected_sidecar_sha256: str | None,
    expected_sidecar_config_sha256: str | None,
    expected_deck_file_sha256: str | None,
    expected_deck_cards_sha256: str | None,
    expected_matchup_tree_sha256: str | None,
) -> None:
    """Cross-bind evaluator inputs to the promotion currently being consumed."""

    binding = _candidate_evaluation_binding(receipt)
    expected_values = (
        ("parent_checkpoint_sha256", expected_parent_checkpoint_sha256),
        ("sidecar_sha256", expected_sidecar_sha256),
        ("sidecar_config_sha256", expected_sidecar_config_sha256),
        ("deck_file_sha256", expected_deck_file_sha256),
        ("deck_cards_sha256", expected_deck_cards_sha256),
        ("matchup_tree_sha256", expected_matchup_tree_sha256),
    )
    for field, requested in expected_values:
        if requested is not None and binding[field] != _sha256(requested, field):
            raise RTPPromotionEvidenceError(
                f"evaluation receipt is bound to a different {field}"
            )


def validate_r198_evaluation_receipt(
    receipt_path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_local_evidence: bool = True,
    expected_parent_checkpoint_sha256: str | None = None,
    expected_sidecar_sha256: str | None = None,
    expected_sidecar_config_sha256: str | None = None,
    expected_deck_file_sha256: str | None = None,
    expected_deck_cards_sha256: str | None = None,
    expected_matchup_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a semantically promotable v2 receipt or raise fail-closed.

    ``require_local_evidence=False`` is only for a sealed submission copy whose
    original 3,000 transcripts remain in the immutable evaluation archive.  It
    still verifies the receipt's canonical content, every identity shape, all
    fixed r198 gates, and promotion isolation; it simply does not dereference
    archive paths that are intentionally outside the package.
    """

    evidence_cache: _EvidenceReadCache = {}
    material = _read_immutable_evidence(
        receipt_path, "evaluation receipt", evidence_cache=evidence_cache
    )
    actual_digest = material.sha256
    if expected_sha256 is not None and actual_digest != _sha256(expected_sha256, "evaluation receipt SHA"):
        raise RTPPromotionEvidenceError("evaluation receipt digest mismatch")
    try:
        receipt = _mapping(
            json.loads(material.payload.decode("utf-8")), "evaluation receipt"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RTPPromotionEvidenceError("evaluation receipt is not valid JSON") from exc
    if receipt.get("schema") != EVALUATION_RECEIPT_SCHEMA:
        raise RTPPromotionEvidenceError("evaluation receipt schema is not r198 v2")
    status = receipt.get("status")
    if status not in {"ready_for_separate_promotion_review", "hold"}:
        raise RTPPromotionEvidenceError("evaluation receipt is not ready for separate review")
    ready_for_review = status == "ready_for_separate_promotion_review"
    material = {
        key: value
        for key, value in receipt.items()
        if key not in {"created_at_utc", "receipt_input_sha256"}
    }
    if _sha256(receipt.get("receipt_input_sha256"), "receipt_input_sha256") != canonical_digest(material):
        raise RTPPromotionEvidenceError("evaluation receipt canonical digest mismatch")
    # Retain the historical shallow HOLD rejection for incomplete receipts.
    # A full compiler HOLD has validated rows and reaches the audit path below;
    # a hand-written/minimal hold has no evidence to audit.
    if not ready_for_review and "validated_rows" not in receipt:
        raise RTPPromotionEvidenceError("evaluation receipt is not ready for separate review")
    decision = _mapping(receipt.get("promotion_decision"), "promotion_decision")
    if (
        not isinstance(decision.get("eligible_for_separate_promotion_review"), bool)
        or decision.get("self_promotion_performed") is not False
        or decision.get("serving_change_authorized") is not False
        or decision.get("eligible_for_separate_promotion_review") is not ready_for_review
    ):
        raise RTPPromotionEvidenceError("evaluation receipt has invalid promotion isolation")
    isolation = _mapping(receipt.get("evaluation_isolation"), "evaluation_isolation")
    for field in (
        "training_eligible",
        "replay_eligible",
        "formal_gate",
        "serving_change_authorized",
        "self_promotion_allowed",
    ):
        if isolation.get(field) is not False:
            raise RTPPromotionEvidenceError(f"evaluation isolation must keep {field} false")
    results = _mapping(receipt.get("results"), "evaluation results")
    if results.get("in_memory") is True:
        raise RTPPromotionEvidenceError("in-memory evaluation results are non-promotable")
    _immutable_identity(
        results,
        "evaluation results",
        require_local=require_local_evidence,
        evidence_cache=evidence_cache,
    )
    # A held receipt is never promotion authority, but it can still carry
    # fully sealed row evidence.  Audit that evidence before returning the
    # fixed HOLD outcome so a malformed over-cap row cannot hide behind the
    # known absent-target gate.
    _validate_gate_table(
        receipt.get("promotion_gates"), require_passed=ready_for_review
    )
    if "validated_rows" not in receipt:
        if ready_for_review:
            raise RTPPromotionEvidenceError(
                "evaluation receipt is non-promotable: trusted counterfactual candidate targets are absent"
            )
        raise RTPPromotionEvidenceError("evaluation receipt is not ready for separate review")
    rows = _validate_panel_and_rows(
        receipt,
        require_local_evidence=require_local_evidence,
        evidence_cache=evidence_cache,
    )
    _validate_efficacy_semantics(
        receipt,
        rows,
        require_local_evidence=require_local_evidence,
        evidence_cache=evidence_cache,
    )
    _validate_expected_candidate_bindings(
        receipt,
        expected_parent_checkpoint_sha256=expected_parent_checkpoint_sha256,
        expected_sidecar_sha256=expected_sidecar_sha256,
        expected_sidecar_config_sha256=expected_sidecar_config_sha256,
        expected_deck_file_sha256=expected_deck_file_sha256,
        expected_deck_cards_sha256=expected_deck_cards_sha256,
        expected_matchup_tree_sha256=expected_matchup_tree_sha256,
    )
    # The fixed r197 candidate deliberately has no trustworthy alternative
    # action targets.  This immutable HOLD survives a successful row audit;
    # nothing above can turn evaluation telemetry into promotion authority.
    raise RTPPromotionEvidenceError(
        "evaluation receipt is non-promotable: trusted counterfactual candidate targets are absent"
    )


__all__ = [
    "DIRECT_BRIDGE_ARM",
    "EVALUATION_RECEIPT_SCHEMA",
    "R198PackagedEvaluationCapability",
    "RTPPromotionEvidenceError",
    "arm_r198_packaged_evaluation_capability",
    "canonical_digest",
    "read_r198_immutable_json_object",
    "resolve_r198_packaged_evaluation_capability",
    "validate_r198_evaluation_receipt",
]
