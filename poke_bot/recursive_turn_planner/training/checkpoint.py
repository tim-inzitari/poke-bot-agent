"""Sidecar checkpoints for Recursive Turn Planner shadow training."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import torch

from poke_bot.recursive_turn_planner.config import RTPConfig
from poke_bot.recursive_turn_planner.planner import (
    RecursiveTurnPlanner,
    required_recursive_passes,
)
from poke_bot.recursive_turn_planner.profiles import (
    PURE_RL_R197_MAX_ACTION_COMBOS,
    get_profile,
)
from poke_bot.rtp_evaluation_promotion import (
    RTPPromotionEvidenceError,
    read_r198_immutable_json_object,
    resolve_r198_packaged_evaluation_capability,
    validate_r198_evaluation_receipt,
)


RTP_SHADOW_TRAIN_SCHEMA = "poke_bot.recursive_turn_planner.shadow_train/v1"
RTP_PROMOTION_SCHEMA = "poke_bot.rtp_promotion/v1"
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROMOTION_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "status",
        "specialist_id",
        "parent_checkpoint_sha256",
        "sidecar_sha256",
        "sidecar_config_sha256",
        "max_neural_passes",
        "required_neural_passes",
        "deck_file_sha256",
        "deck_cards_sha256",
        "matchup_tree_sha256",
        "evaluation_receipt_path",
        "evaluation_receipt_sha256",
        "identity_gate_passed",
        "planner_activation_gate_passed",
        "reliability_gate_passed",
        "heldout_efficacy_gate_passed",
        "robustness_gate_passed",
        "latency_gate_passed",
        "serving_eligible",
        "action_authority_enabled",
        "created_at_utc",
    }
)


def _normalize_sha256(
    value: object,
    *,
    label: str,
    required: bool = False,
) -> str:
    digest = str(value or "").strip().lower()
    if not digest:
        if required:
            raise ValueError(f"RTP checkpoint requires {label}")
        return ""
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"RTP checkpoint has invalid {label}: {value!r}")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("RTP checkpoint value is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_true(receipt: Mapping[str, object], field: str) -> None:
    if receipt.get(field) is not True:
        raise ValueError(f"RTP promotion receipt does not pass {field}")


def _validate_promotion_receipt(
    *,
    promotion_receipt: Path | str | None,
    expected_promotion_receipt_digest: Optional[str],
    checkpoint_path: Path,
    payload: Mapping[str, object],
    config: RTPConfig,
    expected_parent_digest: str,
) -> None:
    """Verify the immutable authority proof for a shadow sidecar.

    A sidecar cannot promote itself: its root authority flags remain false.
    The separately supplied receipt is checksum-bound to both the sidecar
    bytes and its serialized configuration, then must attest every gate.
    """

    if promotion_receipt is None:
        raise ValueError("serving-qualified RTP load requires a promotion receipt")
    expected_receipt_digest = _normalize_sha256(
        expected_promotion_receipt_digest,
        label="expected_promotion_receipt_digest",
        required=True,
    )
    try:
        promotion_identity, receipt = read_r198_immutable_json_object(
            promotion_receipt,
            label="RTP promotion receipt",
            expected_sha256=expected_receipt_digest,
        )
    except RTPPromotionEvidenceError as exc:
        raise ValueError(f"RTP promotion receipt is not immutable evidence: {exc}") from exc
    receipt_path = Path(str(promotion_identity["path"]))
    missing = sorted(_PROMOTION_REQUIRED_FIELDS.difference(receipt))
    if missing:
        raise ValueError("RTP promotion receipt is incomplete: " + ", ".join(missing))
    if receipt.get("schema") != RTP_PROMOTION_SCHEMA:
        raise ValueError("RTP promotion receipt schema is not recognized")
    if receipt.get("status") != "accepted":
        raise ValueError("RTP promotion receipt is not accepted")
    if not isinstance(receipt.get("specialist_id"), str) or not str(
        receipt["specialist_id"]
    ).strip():
        raise ValueError("RTP promotion receipt has no specialist_id")
    if not isinstance(receipt.get("created_at_utc"), str) or not str(
        receipt["created_at_utc"]
    ).strip():
        raise ValueError("RTP promotion receipt has no creation timestamp")
    if not isinstance(receipt.get("evaluation_receipt_path"), str) or not str(
        receipt["evaluation_receipt_path"]
    ).strip():
        raise ValueError("RTP promotion receipt has no evaluation receipt path")

    receipt_parent = _normalize_sha256(
        receipt.get("parent_checkpoint_sha256"),
        label="promotion.parent_checkpoint_sha256",
        required=True,
    )
    if receipt_parent != expected_parent_digest:
        raise ValueError("RTP promotion receipt parent digest mismatch")
    if _normalize_sha256(
        receipt.get("sidecar_sha256"),
        label="promotion.sidecar_sha256",
        required=True,
    ) != _sha256_file(checkpoint_path):
        raise ValueError("RTP promotion receipt sidecar digest mismatch")
    raw_config = payload.get("config")
    if _normalize_sha256(
        receipt.get("sidecar_config_sha256"),
        label="promotion.sidecar_config_sha256",
        required=True,
    ) != _canonical_json_sha256(raw_config):
        raise ValueError("RTP promotion receipt config digest mismatch")
    for field in (
        "deck_file_sha256",
        "deck_cards_sha256",
        "matchup_tree_sha256",
        "evaluation_receipt_sha256",
    ):
        _normalize_sha256(receipt.get(field), label=f"promotion.{field}", required=True)
    try:
        packaged_capability = resolve_r198_packaged_evaluation_capability()
        if packaged_capability is not None:
            if (
                packaged_capability.promotion_receipt_path != str(receipt_path)
                or packaged_capability.promotion_receipt_sha256
                != expected_receipt_digest
            ):
                # A stale/unrelated package capability is not authority for a
                # source-side load.  Fall back to the promotion wrapper's
                # local immutable archive rather than letting ambient state
                # suppress that evidence check.
                packaged_capability = None
        if packaged_capability is not None:
            evaluation_path = Path(packaged_capability.evaluation_receipt_path)
        else:
            evaluation_path = Path(str(receipt.get("evaluation_receipt_path") or ""))
        validate_r198_evaluation_receipt(
            evaluation_path,
            expected_sha256=str(receipt["evaluation_receipt_sha256"]),
            # A sealed submission copy is hash-bound to this promotion
            # wrapper/profile but intentionally keeps the 3,000 immutable
            # execution records in the external evaluation archive.
            require_local_evidence=packaged_capability is None,
            expected_parent_checkpoint_sha256=expected_parent_digest,
            expected_sidecar_sha256=_sha256_file(checkpoint_path),
            expected_sidecar_config_sha256=_canonical_json_sha256(raw_config),
            expected_deck_file_sha256=str(receipt["deck_file_sha256"]),
            expected_deck_cards_sha256=str(receipt["deck_cards_sha256"]),
            expected_matchup_tree_sha256=str(receipt["matchup_tree_sha256"]),
        )
    except RTPPromotionEvidenceError as exc:
        raise ValueError(f"RTP promotion evaluation receipt is non-promotable: {exc}") from exc

    for field in (
        "identity_gate_passed",
        "planner_activation_gate_passed",
        "reliability_gate_passed",
        "heldout_efficacy_gate_passed",
        "robustness_gate_passed",
        "latency_gate_passed",
        "serving_eligible",
        "action_authority_enabled",
    ):
        _require_true(receipt, field)
    if isinstance(receipt.get("max_neural_passes"), bool) or not isinstance(
        receipt.get("max_neural_passes"), int
    ):
        raise ValueError("RTP promotion receipt max_neural_passes is invalid")
    if int(receipt["max_neural_passes"]) != int(config.max_neural_passes):
        raise ValueError("RTP promotion receipt max_neural_passes mismatch")
    if config.sizing_profile == "pure_rl_r197":
        if isinstance(receipt.get("max_action_combos"), bool) or not isinstance(
            receipt.get("max_action_combos"), int
        ):
            raise ValueError("RTP promotion receipt max_action_combos is invalid")
        if int(receipt["max_action_combos"]) != PURE_RL_R197_MAX_ACTION_COMBOS:
            raise ValueError("RTP promotion receipt max_action_combos mismatch")
    expected_required = {
        "normal": required_recursive_passes(config),
        "forced_replan": required_recursive_passes(config, force_recurse=True),
    }
    if receipt.get("required_neural_passes") != expected_required:
        raise ValueError("RTP promotion receipt required_neural_passes mismatch")


def _serialized_config(config: RTPConfig) -> dict[str, Any]:
    """Persist every planner/executor-relevant configuration field exactly."""

    return {
        "schema": config.schema,
        "sizing_profile": config.sizing_profile,
        "d_model": config.d_model,
        "dynamics_width": config.dynamics_width,
        "num_plan_candidates": config.num_plan_candidates,
        "max_recursion_depth": config.max_recursion_depth,
        "max_neural_passes": config.max_neural_passes,
        "max_plan_length": config.max_plan_length,
        "complexity_option_threshold": config.complexity_option_threshold,
        "complexity_entropy_threshold": config.complexity_entropy_threshold,
        "skip_trivial_decisions": config.skip_trivial_decisions,
        "online_sim_verify_budget": config.online_sim_verify_budget,
        "repair_budget": config.repair_budget,
        "compute_cost_penalty": config.compute_cost_penalty,
        "option_batch_hint": config.option_batch_hint,
        "prefer_option_hidden": config.prefer_option_hidden,
        "policy_aid_cap": config.policy_aid_cap,
        "default_subgoals": tuple(config.default_subgoals),
    }


def _config_from_payload(raw: object) -> RTPConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("RTP checkpoint config must be an object")
    raw_profile = str(raw.get("sizing_profile") or "").strip().lower()
    if raw_profile:
        try:
            defaults = get_profile(raw_profile).to_config()
        except KeyError:
            # The earliest shadow sidecars used ``trained`` rather than a
            # registered profile and defaulted to the d=96 lean topology.
            defaults = RTPConfig(
                sizing_profile=raw_profile,
                d_model=96,
                dynamics_width=192,
                num_plan_candidates=6,
            )
    else:
        defaults = RTPConfig(
            sizing_profile="trained",
            d_model=96,
            dynamics_width=192,
            num_plan_candidates=6,
        )

    def value(name: str, default: Any) -> Any:
        return raw[name] if name in raw else default

    default_subgoals = value("default_subgoals", defaults.default_subgoals)
    if not isinstance(default_subgoals, (list, tuple)):
        raise ValueError("RTP checkpoint default_subgoals must be a sequence")
    return RTPConfig(
        schema=str(value("schema", defaults.schema)),
        sizing_profile=str(value("sizing_profile", defaults.sizing_profile)),
        d_model=int(value("d_model", defaults.d_model)),
        dynamics_width=int(value("dynamics_width", defaults.dynamics_width)),
        num_plan_candidates=int(
            value("num_plan_candidates", defaults.num_plan_candidates)
        ),
        max_recursion_depth=int(
            value("max_recursion_depth", defaults.max_recursion_depth)
        ),
        max_neural_passes=int(
            value("max_neural_passes", defaults.max_neural_passes)
        ),
        max_plan_length=int(value("max_plan_length", defaults.max_plan_length)),
        complexity_option_threshold=int(
            value("complexity_option_threshold", defaults.complexity_option_threshold)
        ),
        complexity_entropy_threshold=float(
            value("complexity_entropy_threshold", defaults.complexity_entropy_threshold)
        ),
        skip_trivial_decisions=bool(
            value("skip_trivial_decisions", defaults.skip_trivial_decisions)
        ),
        online_sim_verify_budget=int(
            value("online_sim_verify_budget", defaults.online_sim_verify_budget)
        ),
        repair_budget=int(value("repair_budget", defaults.repair_budget)),
        compute_cost_penalty=float(
            value("compute_cost_penalty", defaults.compute_cost_penalty)
        ),
        option_batch_hint=int(value("option_batch_hint", defaults.option_batch_hint)),
        prefer_option_hidden=bool(
            value("prefer_option_hidden", defaults.prefer_option_hidden)
        ),
        policy_aid_cap=float(value("policy_aid_cap", defaults.policy_aid_cap)),
        default_subgoals=tuple(str(item) for item in default_subgoals),
    )


def _config_mismatches(actual: RTPConfig, expected: RTPConfig) -> list[str]:
    return [
        name
        for name in RTPConfig.__dataclass_fields__
        if getattr(actual, name) != getattr(expected, name)
    ]


def save_rtp_checkpoint(
    planner: RecursiveTurnPlanner,
    path: Path | str,
    *,
    metrics: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
    parent_checkpoint_sha256: Optional[str] = None,
    shadow_only: Optional[bool] = None,
    research_only: Optional[bool] = None,
    serving_eligible: bool = False,
    action_authority_enabled: bool = False,
) -> dict[str, Any]:
    """Write a non-authoritative planner sidecar.

    A sidecar is always saved without serving/action authority.  Promotion is
    intentionally external and immutable: a serving bridge validates a
    separate receipt which binds this file, its exact config, and all gates.
    ``shadow_only``/``research_only`` are explicit root metadata; existing
    shadow callers that already carry either marker in ``extra`` retain it.
    """

    metadata = dict(extra or {})
    if shadow_only is None:
        shadow_only = bool(metadata.get("shadow_only", False))
    if research_only is None:
        research_only = bool(metadata.get("research_only", False))
    if serving_eligible or action_authority_enabled:
        raise ValueError(
            "RTP sidecar authority is granted only by an immutable promotion receipt"
        )
    if parent_checkpoint_sha256 is None:
        parent_checkpoint_sha256 = metadata.get("parent_checkpoint_sha256")
    parent_digest = _normalize_sha256(
        parent_checkpoint_sha256,
        label="parent_checkpoint_sha256",
        required=False,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": RTP_SHADOW_TRAIN_SCHEMA,
        "research_only": bool(research_only),
        "shadow_only": bool(shadow_only),
        "serving_eligible": False,
        "action_authority_enabled": False,
        "parent_checkpoint_sha256": parent_digest,
        "generated_at_unix": time.time(),
        "config": _serialized_config(planner.config),
        "state_dict": {k: v.detach().cpu() for k, v in planner.state_dict().items()},
        "metrics": dict(metrics or {}),
        "extra": metadata,
        "inventory": planner.inventory(),
    }
    torch.save(payload, out)
    receipt = {
        "schema": RTP_SHADOW_TRAIN_SCHEMA + ".receipt",
        "checkpoint_path": str(out.resolve()),
        "d_model": planner.config.d_model,
        "parameters": int(sum(p.numel() for p in planner.parameters())),
        "metrics": dict(metrics or {}),
        "parent_checkpoint_sha256": parent_digest,
        "research_only": bool(research_only),
        "shadow_only": bool(shadow_only),
        "serving_eligible": False,
        "action_authority_enabled": False,
        "required_neural_passes_normal": required_recursive_passes(planner.config),
        "required_neural_passes_forced_replan": required_recursive_passes(
            planner.config, force_recurse=True
        ),
    }
    receipt_path = out.with_suffix(out.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load_rtp_checkpoint(
    path: Path | str,
    *,
    device: str | torch.device = "cpu",
    planner: Optional[RecursiveTurnPlanner] = None,
    expected_parent_digest: Optional[str] = None,
    expected_config: Optional[RTPConfig] = None,
    promotion_receipt: Path | str | None = None,
    expected_promotion_receipt_digest: Optional[str] = None,
    serving_qualified: bool = False,
) -> RecursiveTurnPlanner:
    """Safely load a planner sidecar, optionally under a serving contract.

    A serving-qualified load is intentionally stricter than shadow/research
    loading: it requires a parent digest, a checksum-bound external promotion
    receipt, an exact runtime config match, and enough passes to complete the
    current recursive plan skeleton.  It never falls back to pickle-enabled
    deserialization.
    """

    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"RTP checkpoint safe load failed: {checkpoint_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("RTP checkpoint payload must be an object")
    if payload.get("schema") != RTP_SHADOW_TRAIN_SCHEMA:
        raise ValueError("RTP checkpoint schema is not recognized")

    cfg = _config_from_payload(payload.get("config"))
    expected_digest = _normalize_sha256(
        expected_parent_digest,
        label="expected_parent_digest",
        required=serving_qualified,
    )
    actual_digest = _normalize_sha256(
        payload.get("parent_checkpoint_sha256"),
        label="parent_checkpoint_sha256",
        required=bool(expected_digest) or serving_qualified,
    )
    if expected_digest and actual_digest != expected_digest:
        raise ValueError(
            "RTP checkpoint parent digest mismatch: "
            f"checkpoint={actual_digest} expected={expected_digest}"
        )
    if serving_qualified:
        if expected_config is None:
            raise ValueError("serving-qualified RTP load requires expected_config")
        if payload.get("research_only") is True:
            raise ValueError("serving-qualified RTP checkpoint is research-only")
        if payload.get("shadow_only") is not True:
            raise ValueError(
                "serving-qualified RTP checkpoint is not explicitly shadow-only"
            )
        if payload.get("serving_eligible") is not False:
            raise ValueError(
                "RTP sidecar must remain non-authoritative; use its promotion receipt"
            )
        if payload.get("action_authority_enabled") is not False:
            raise ValueError(
                "RTP sidecar must remain non-authoritative; use its promotion receipt"
            )
        mismatches = _config_mismatches(cfg, expected_config)
        if mismatches:
            raise ValueError(
                "RTP checkpoint serving config mismatch: " + ", ".join(mismatches)
            )
        if cfg.sizing_profile == "pure_rl_r197":
            r197_mismatches = _config_mismatches(
                cfg,
                get_profile("pure_rl_r197").to_config(),
            )
            if r197_mismatches:
                raise ValueError(
                    "pure_rl_r197 serving config must exactly match the "
                    "authorized 256-pass profile: " + ", ".join(r197_mismatches)
                )
        required = required_recursive_passes(cfg)
        if int(cfg.max_neural_passes) < required:
            raise ValueError(
                "RTP checkpoint cannot complete a recursive plan: "
                f"max_neural_passes={cfg.max_neural_passes} required={required}"
            )
        _validate_promotion_receipt(
            promotion_receipt=promotion_receipt,
            expected_promotion_receipt_digest=expected_promotion_receipt_digest,
            checkpoint_path=checkpoint_path,
            payload=payload,
            config=cfg,
            expected_parent_digest=expected_digest,
        )

    if planner is None:
        planner = RecursiveTurnPlanner(cfg)
    elif planner.config != cfg:
        raise ValueError("RTP checkpoint config does not match supplied planner")
    state = payload.get("state_dict")
    if state is None:
        # r195-style sidecars used this legacy field name.  It remains
        # research-loadable, but receives the same safe and strict validation.
        state = payload.get("planner_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"RTP checkpoint missing state_dict: {path}")
    if any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("RTP checkpoint state_dict must contain only tensor entries")
    try:
        planner.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise ValueError("RTP checkpoint state_dict is incompatible with its config") from exc
    planner.to(torch.device(device))
    return planner
