"""Competition submission agent: history-conditioned policy, fail-closed actions.

Hard constraints:
  - No ``__file__`` at import time (isolated tarball / Kaggle).
  - Deck from ``deck.csv`` next to ``main.py`` or ``/kaggle_simulations/agent/``.
  - Deterministically honor the packaged turn-order profile before importing
    cg or loading the model.
  - Info-set only (features.assert_info_set inside the policy runtime).
  - Fail-closed: illegal selects -> legal random fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_PROCESS_STARTED = time.monotonic()
_AGENT_DIR_CANDIDATES = (
    Path.cwd(),
    Path("/kaggle_simulations/agent"),
)
_RUNTIME_PROFILE_SCHEMA = "poke_bot.submission_runtime_profile/v1"
_RTP_SIDECAR_SCHEMA = "poke_bot.recursive_turn_planner.shadow_train/v1"
_RTP_PROMOTION_SCHEMA = "poke_bot.rtp_promotion/v1"
_RTP_R197_SPECIALIST_ID = "alakazam"
_RTP_R197_SIZING_PROFILE = "pure_rl_r197"
_RTP_R198_EXACT_MAX_NEURAL_PASSES = 256
_RTP_ABSOLUTE_MAX_NEURAL_PASSES = 256
_RTP_R198_EXACT_MAX_ACTION_COMBOS = 1024
_RTP_R197_REQUIRED_NEURAL_PASSES = {
    "normal": 6,
    "forced_replan": 5,
}
_SUBMISSION_NO_PROGRESS_ESCAPE_TURNS = 512
_RTP_PROMOTION_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "status",
        "specialist_id",
        "parent_checkpoint_sha256",
        "sidecar_sha256",
        "sidecar_config_sha256",
        "max_neural_passes",
        "max_action_combos",
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


def _agent_dir() -> Path:
    for directory in _AGENT_DIR_CANDIDATES:
        if (directory / "deck.csv").is_file():
            return directory
    return Path.cwd()


def _sha256_file(path: Path) -> str:
    """Return the canonical SHA-256 identity without buffering model bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    """Checksum a JSON object in the one package-independent representation."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("RTP binding payload is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _deck_cards_sha256(path: Path) -> str:
    """Bind the ordered 60-card serving list, not merely its CSV bytes."""

    cards: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cards.append(int(line.split(",", 1)[0]))
        except ValueError as exc:
            raise RuntimeError("packaged deck contains a non-card row") from exc
    if len(cards) != 60:
        raise RuntimeError(f"packaged deck must have 60 cards, got {len(cards)}")
    return _canonical_json_sha256(cards)


def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value or "")
    if (
        len(digest) != 71
        or not digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise ValueError(f"runtime profile has invalid {field}")
    return digest


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"packaged {label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"packaged {label} must contain an object")
    return payload


def _runtime_profile_mode(profile: Mapping[str, object]) -> str:
    """Resolve only explicit new modes, retaining immutable r195 profiles."""

    if profile.get("schema") != _RUNTIME_PROFILE_SCHEMA:
        raise ValueError("submission runtime profile schema changed")
    mode = profile.get("rtp_mode")
    if mode in {"off", "direct", "recursive"}:
        return str(mode)
    # Historical r195 packages predate the three-arm contract.  They remain
    # reconstructable exactly, but no newly built profile may use these names.
    if mode is None:
        legacy = profile.get("recursive_turn_planner")
        if legacy == "disabled":
            return "legacy_off"
        if legacy == "enabled":
            return "legacy_recursive"
    raise ValueError("unsupported submitted recursive-turn-planner profile")


def _assert_profile_model_identity(profile: Mapping[str, object]) -> str:
    model = _agent_dir() / "model.pt"
    if not model.is_file():
        raise FileNotFoundError("model.pt is required")
    expected = _require_sha256(
        profile.get("model_checkpoint_sha256"),
        field="model_checkpoint_sha256",
    )
    actual = _sha256_file(model)
    if actual != expected:
        raise RuntimeError("packaged model digest does not match runtime profile")
    return actual


def _assert_no_packaged_rtp_sidecar() -> None:
    if (_agent_dir() / "rtp_shadow_planner.pt").exists():
        raise RuntimeError("non-recursive RTP package contains a sidecar")


def _apply_runtime_profile() -> dict[str, object]:
    """Apply an optional checksum-packaged serving profile before agent import."""
    path = _agent_dir() / "runtime_profile.json"
    if not path.is_file():
        return {}
    profile = _read_json_object(path, label="runtime profile")
    mode = _runtime_profile_mode(profile)
    if mode == "legacy_off":
        # This is an exact submitted-entrypoint override, not a host default.
        # It deliberately wins over any inherited Kaggle/worker environment.
        os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] = "0"
        for name in (
            "POKEBOT_RTP_CHECKPOINT",
            "POKEBOT_RTP_ALLOW_UNTRAINED",
            "POKEBOT_RTP_SIZING_PROFILE",
            "POKEBOT_RTP_SPECIALIST_ID",
            "POKEBOT_RTP_SERVING_QUALIFIED",
            "POKEBOT_RTP_PARENT_CHECKPOINT_SHA256",
            "POKEBOT_RTP_PROMOTION_RECEIPT",
            "POKEBOT_RTP_PROMOTION_RECEIPT_SHA256",
            "POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT",
        ):
            os.environ.pop(name, None)
    elif mode == "legacy_recursive":
        sidecar = (_agent_dir() / "rtp_shadow_planner.pt").resolve()
        if not sidecar.is_file():
            raise FileNotFoundError("enabled RTP submission lacks its sidecar")
        expected = str(profile.get("rtp_checkpoint_sha256") or "")
        actual = _sha256_file(sidecar)
        if not expected.startswith("sha256:") or actual != expected:
            raise ValueError("submitted RTP sidecar checksum changed")
        os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] = "1"
        os.environ["POKEBOT_RTP_CHECKPOINT"] = str(sidecar)
        os.environ["POKEBOT_RTP_SPECIALIST_ID"] = str(
            profile.get("specialist_id") or "alakazam"
        )
        os.environ.pop("POKEBOT_RTP_SERVING_QUALIFIED", None)
        os.environ.pop("POKEBOT_RTP_PARENT_CHECKPOINT_SHA256", None)
        os.environ.pop("POKEBOT_RTP_PROMOTION_RECEIPT", None)
        os.environ.pop("POKEBOT_RTP_PROMOTION_RECEIPT_SHA256", None)
        os.environ.pop("POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT", None)
    elif mode == "off":
        if (
            profile.get("recursive_turn_planner") != "disabled"
            or profile.get("display") != "NO RTP"
            or profile.get("rtp_sidecar_packaged") is not False
        ):
            raise ValueError("explicit off RTP runtime profile changed")
        _assert_profile_model_identity(profile)
        _assert_no_packaged_rtp_sidecar()
        os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] = "0"
        for name in (
            "POKEBOT_RTP_CHECKPOINT",
            "POKEBOT_RTP_ALLOW_UNTRAINED",
            "POKEBOT_RTP_SIZING_PROFILE",
            "POKEBOT_RTP_SPECIALIST_ID",
            "POKEBOT_RTP_SERVING_QUALIFIED",
            "POKEBOT_RTP_PARENT_CHECKPOINT_SHA256",
            "POKEBOT_RTP_PROMOTION_RECEIPT",
            "POKEBOT_RTP_PROMOTION_RECEIPT_SHA256",
            "POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT",
        ):
            os.environ.pop(name, None)
    elif mode == "direct":
        if (
            profile.get("recursive_turn_planner") != "enabled"
            or profile.get("display") != "DIRECT RTP"
            or profile.get("rtp_sidecar_packaged") is not True
            or profile.get("rtp_direct_bridge_only") is not True
            or profile.get("rtp_sizing_profile") != _RTP_R197_SIZING_PROFILE
            or profile.get("specialist_id") != _RTP_R197_SPECIALIST_ID
        ):
            raise ValueError("direct RTP runtime profile changed")
        model_digest = _assert_profile_model_identity(profile)
        parent_digest = _require_sha256(
            profile.get("parent_checkpoint_sha256"),
            field="parent_checkpoint_sha256",
        )
        if parent_digest != model_digest:
            raise RuntimeError("direct RTP parent is not the packaged model")
        sidecar = (_agent_dir() / "rtp_shadow_planner.pt").resolve()
        if not sidecar.is_file():
            raise FileNotFoundError("direct RTP submission lacks its sidecar")
        if _sha256_file(sidecar) != _require_sha256(
            profile.get("rtp_checkpoint_sha256"),
            field="rtp_checkpoint_sha256",
        ):
            raise RuntimeError("packaged direct RTP sidecar digest changed")
        if (
            (_agent_dir() / "rtp_promotion_receipt.json").exists()
            or (_agent_dir() / "rtp_evaluation_receipt.json").exists()
        ):
            raise RuntimeError("direct RTP package must not carry promotion authority")
        _require_sha256(profile.get("rtp_config_sha256"), field="rtp_config_sha256")
        if _require_exact_int(profile, "max_neural_passes") != (
            _RTP_R198_EXACT_MAX_NEURAL_PASSES
        ):
            raise RuntimeError("direct RTP requires exact 256 neural passes")
        if _require_exact_int(profile, "max_action_combos") != (
            _RTP_R198_EXACT_MAX_ACTION_COMBOS
        ):
            raise RuntimeError("direct RTP requires exact 1024 action combinations")
        if profile.get("required_neural_passes") != _RTP_R197_REQUIRED_NEURAL_PASSES:
            raise RuntimeError("direct RTP requires exact normal=6/forced-replan=5 passes")
        os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] = "1"
        os.environ["POKEBOT_RTP_CHECKPOINT"] = str(sidecar)
        os.environ["POKEBOT_RTP_SIZING_PROFILE"] = _RTP_R197_SIZING_PROFILE
        os.environ["POKEBOT_RTP_SPECIALIST_ID"] = _RTP_R197_SPECIALIST_ID
        os.environ["POKEBOT_RTP_PARENT_CHECKPOINT_SHA256"] = parent_digest
        os.environ.pop("POKEBOT_RTP_ALLOW_UNTRAINED", None)
        os.environ.pop("POKEBOT_RTP_SERVING_QUALIFIED", None)
        os.environ.pop("POKEBOT_RTP_PROMOTION_RECEIPT", None)
        os.environ.pop("POKEBOT_RTP_PROMOTION_RECEIPT_SHA256", None)
        os.environ.pop("POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT", None)
    elif mode == "recursive":
        if (
            profile.get("recursive_turn_planner") != "enabled"
            or profile.get("display") != "RTP"
            or profile.get("rtp_sidecar_packaged") is not True
            or profile.get("rtp_sizing_profile") != _RTP_R197_SIZING_PROFILE
            or profile.get("specialist_id") != _RTP_R197_SPECIALIST_ID
            or profile.get("rtp_promotion_receipt_file")
            != "rtp_promotion_receipt.json"
            or profile.get("rtp_evaluation_receipt_file")
            != "rtp_evaluation_receipt.json"
        ):
            raise ValueError("recursive RTP runtime profile changed")
        model_digest = _assert_profile_model_identity(profile)
        parent_digest = _require_sha256(
            profile.get("parent_checkpoint_sha256"),
            field="parent_checkpoint_sha256",
        )
        if parent_digest != model_digest:
            raise RuntimeError("recursive RTP parent is not the packaged model")
        sidecar = (_agent_dir() / "rtp_shadow_planner.pt").resolve()
        if not sidecar.is_file():
            raise FileNotFoundError("recursive RTP submission lacks its sidecar")
        if _sha256_file(sidecar) != _require_sha256(
            profile.get("rtp_checkpoint_sha256"),
            field="rtp_checkpoint_sha256",
        ):
            raise RuntimeError("packaged RTP sidecar digest changed")
        _require_sha256(profile.get("rtp_config_sha256"), field="rtp_config_sha256")
        _require_sha256(
            profile.get("rtp_promotion_receipt_sha256"),
            field="rtp_promotion_receipt_sha256",
        )
        _require_sha256(
            profile.get("rtp_evaluation_receipt_sha256"),
            field="rtp_evaluation_receipt_sha256",
        )
        _require_sha256(profile.get("deck_cards_sha256"), field="deck_cards_sha256")
        _require_sha256(profile.get("matchup_tree_sha256"), field="matchup_tree_sha256")
        if _require_exact_int(profile, "max_neural_passes") != (
            _RTP_R198_EXACT_MAX_NEURAL_PASSES
        ):
            raise RuntimeError("recursive RTP requires exact 256 neural passes")
        if _require_exact_int(profile, "max_action_combos") != (
            _RTP_R198_EXACT_MAX_ACTION_COMBOS
        ):
            raise RuntimeError("recursive RTP requires exact 1024 action combinations")
        if profile.get("required_neural_passes") != _RTP_R197_REQUIRED_NEURAL_PASSES:
            raise RuntimeError("recursive RTP requires exact normal=6/forced-replan=5 passes")
        os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] = "1"
        os.environ["POKEBOT_RTP_CHECKPOINT"] = str(sidecar)
        os.environ["POKEBOT_RTP_SIZING_PROFILE"] = _RTP_R197_SIZING_PROFILE
        os.environ["POKEBOT_RTP_SPECIALIST_ID"] = _RTP_R197_SPECIALIST_ID
        os.environ["POKEBOT_RTP_SERVING_QUALIFIED"] = "1"
        os.environ["POKEBOT_RTP_PARENT_CHECKPOINT_SHA256"] = parent_digest
        os.environ["POKEBOT_RTP_PROMOTION_RECEIPT"] = str(
            (_agent_dir() / "rtp_promotion_receipt.json").resolve()
        )
        os.environ["POKEBOT_RTP_PROMOTION_RECEIPT_SHA256"] = _require_sha256(
            profile.get("rtp_promotion_receipt_sha256"),
            field="rtp_promotion_receipt_sha256",
        )
        try:
            from poke_bot.rtp_evaluation_promotion import (
                RTPPromotionEvidenceError,
                arm_r198_packaged_evaluation_capability,
            )

            packaged_capability = arm_r198_packaged_evaluation_capability(
                package_root=_agent_dir(), runtime_profile=profile
            )
        except RTPPromotionEvidenceError as exc:
            raise RuntimeError(
                "packaged RTP promotion/evaluation evidence is not sealed: " + str(exc)
            ) from exc
        os.environ["POKEBOT_RTP_PACKAGED_EVALUATION_RECEIPT"] = (
            packaged_capability.evaluation_receipt_path
        )
        os.environ.pop("POKEBOT_RTP_ALLOW_UNTRAINED", None)
    return profile


def _require_exact_bool(
    payload: Mapping[str, object], field: str, *, expected: bool = True
) -> None:
    if payload.get(field) is not expected:
        raise RuntimeError(
            f"RTP promotion receipt has invalid {field}={payload.get(field)!r}"
        )


def _require_exact_int(
    payload: Mapping[str, object], field: str
) -> int:
    value = payload.get(field)
    if type(value) is not int:
        raise RuntimeError(f"RTP binding has non-integer {field}")
    return int(value)


def _assert_r197_sidecar_config(
    config: Mapping[str, object], *, expected_max_neural_passes: int
) -> None:
    """Reject a merely shape-compatible sidecar on the recursive policy path."""

    if expected_max_neural_passes > _RTP_ABSOLUTE_MAX_NEURAL_PASSES:
        raise RuntimeError("recursive RTP neural-pass budget is outside its hard ceiling")
    if expected_max_neural_passes != _RTP_R198_EXACT_MAX_NEURAL_PASSES:
        raise RuntimeError(
            "recursive RTP requires the revision-198 exact neural-pass budget"
        )
    expected = {
        "schema": "poke_bot.recursive_turn_planner/v1",
        "sizing_profile": _RTP_R197_SIZING_PROFILE,
        "d_model": 96,
        "dynamics_width": 192,
        "num_plan_candidates": 4,
        "max_recursion_depth": 2,
        "max_neural_passes": expected_max_neural_passes,
        "max_plan_length": 12,
        "complexity_option_threshold": 8,
        "complexity_entropy_threshold": 1.5,
        "prefer_option_hidden": True,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise RuntimeError(
                f"recursive RTP sidecar config changed: {field}={config.get(field)!r}"
            )


def _assert_direct_rtp_binding(
    profile: Mapping[str, object], *, model_digest: str
) -> dict[str, object]:
    """Bind B to C's inert r197 sidecar without granting C's authority.

    The direct arm deliberately uses the same learned sidecar bytes/config as
    recursive evaluation, then replaces only the recursion decision after the
    bridge has been constructed.  It never packages or reads a promotion
    receipt and never sets serving-qualified authority.
    """

    import torch
    from dataclasses import fields

    from poke_bot.recursive_turn_planner.config import RTPConfig
    from poke_bot.recursive_turn_planner.planner import RecursiveTurnPlanner
    agent_dir = _agent_dir()
    sidecar = agent_dir / "rtp_shadow_planner.pt"
    if not sidecar.is_file():
        raise FileNotFoundError("direct RTP package lacks its bound sidecar")
    matchup_tree = agent_dir / "matchup_tree.json"
    if not matchup_tree.is_file():
        raise FileNotFoundError("direct RTP package lacks its matchup tree")
    profile_deck_digest = _require_sha256(
        profile.get("deck_cards_sha256"), field="deck_cards_sha256"
    )
    profile_tree_digest = _require_sha256(
        profile.get("matchup_tree_sha256"), field="matchup_tree_sha256"
    )
    if _deck_cards_sha256(agent_dir / "deck.csv") != profile_deck_digest:
        raise RuntimeError("direct RTP package deck cards changed")
    if _sha256_file(matchup_tree) != profile_tree_digest:
        raise RuntimeError("direct RTP package matchup tree changed")
    try:
        sidecar_payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 — malformed package bytes are fatal.
        raise RuntimeError("direct RTP sidecar safe load failed") from exc
    if not isinstance(sidecar_payload, Mapping):
        raise RuntimeError("direct RTP sidecar payload is not an object")
    if sidecar_payload.get("schema") != _RTP_SIDECAR_SCHEMA:
        raise RuntimeError("direct RTP sidecar schema changed")
    if sidecar_payload.get("research_only") is not False:
        raise RuntimeError("direct RTP sidecar is research-only")
    if sidecar_payload.get("shadow_only") is not True:
        raise RuntimeError("direct RTP sidecar is not explicitly shadow-only")
    if (
        sidecar_payload.get("serving_eligible") is not False
        or sidecar_payload.get("action_authority_enabled") is not False
    ):
        raise RuntimeError("direct RTP sidecar must remain promotion-inert")
    sidecar_parent = _require_sha256(
        sidecar_payload.get("parent_checkpoint_sha256"),
        field="sidecar.parent_checkpoint_sha256",
    )
    config = sidecar_payload.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("direct RTP sidecar lacks a config object")
    state_dict = sidecar_payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("direct RTP sidecar lacks a state dictionary")
    config_dict = dict(config)
    config_fields = {field.name for field in fields(RTPConfig)}
    unknown_config_fields = sorted(set(config_dict).difference(config_fields))
    if unknown_config_fields:
        raise RuntimeError(
            "direct RTP sidecar config has unknown fields: "
            + ", ".join(unknown_config_fields)
        )
    try:
        strict_probe = RecursiveTurnPlanner(RTPConfig(**config_dict))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("direct RTP sidecar config cannot construct a planner") from exc
    expected_state = strict_probe.state_dict()
    if set(state_dict) != set(expected_state):
        raise RuntimeError("direct RTP sidecar state dictionary is not exact")
    for name, expected_tensor in expected_state.items():
        actual_tensor = state_dict[name]
        if getattr(actual_tensor, "shape", None) != expected_tensor.shape:
            raise RuntimeError(f"direct RTP sidecar tensor shape changed: {name}")
    if (
        strict_probe.required_recursive_passes() != _RTP_R197_REQUIRED_NEURAL_PASSES["normal"]
        or strict_probe.required_recursive_passes(force_recurse=True)
        != _RTP_R197_REQUIRED_NEURAL_PASSES["forced_replan"]
    ):
        raise RuntimeError("direct RTP sidecar required neural passes changed")

    config_digest = _canonical_json_sha256(config_dict)
    sidecar_digest = _sha256_file(sidecar)
    profile_max_neural_passes = _require_exact_int(profile, "max_neural_passes")
    profile_max_action_combos = _require_exact_int(profile, "max_action_combos")
    _assert_r197_sidecar_config(
        config_dict, expected_max_neural_passes=profile_max_neural_passes
    )
    if (
        sidecar_parent != model_digest
        or _require_sha256(
            profile.get("parent_checkpoint_sha256"),
            field="parent_checkpoint_sha256",
        )
        != model_digest
        or _require_sha256(
            profile.get("rtp_checkpoint_sha256"),
            field="rtp_checkpoint_sha256",
        )
        != sidecar_digest
        or _require_sha256(
            profile.get("rtp_config_sha256"), field="rtp_config_sha256"
        )
        != config_digest
        or profile_max_neural_passes != _RTP_R198_EXACT_MAX_NEURAL_PASSES
        or profile_max_action_combos != _RTP_R198_EXACT_MAX_ACTION_COMBOS
        or profile.get("required_neural_passes") != _RTP_R197_REQUIRED_NEURAL_PASSES
    ):
        raise RuntimeError("direct RTP package binding changed")
    return config_dict


def _assert_recursive_rtp_binding(
    profile: Mapping[str, object], *, model_digest: str
) -> dict[str, object]:
    """Verify the exact sidecar, promotion proof, and r197 package bindings.

    ``main.py`` repeats this after unpacking because a Kaggle submission can be
    assembled without our local build script.  The validation is intentionally
    literal: historical r195 aliases and a plausibly shaped shadow sidecar do
    not gain action authority through this path.
    """

    import torch
    from dataclasses import fields

    from poke_bot.recursive_turn_planner.config import RTPConfig
    from poke_bot.recursive_turn_planner.planner import RecursiveTurnPlanner
    from poke_bot.rtp_evaluation_promotion import (
        RTPPromotionEvidenceError,
        read_r198_immutable_json_object,
        resolve_r198_packaged_evaluation_capability,
        validate_r198_evaluation_receipt,
    )

    agent_dir = _agent_dir()
    sidecar = agent_dir / "rtp_shadow_planner.pt"
    promotion_path = agent_dir / "rtp_promotion_receipt.json"
    evaluation_path = agent_dir / "rtp_evaluation_receipt.json"
    matchup_tree = agent_dir / "matchup_tree.json"
    if not matchup_tree.is_file():
        raise FileNotFoundError("recursive RTP package lacks its matchup tree")
    profile_deck_digest = _require_sha256(
        profile.get("deck_cards_sha256"), field="deck_cards_sha256"
    )
    profile_tree_digest = _require_sha256(
        profile.get("matchup_tree_sha256"), field="matchup_tree_sha256"
    )
    if _deck_cards_sha256(agent_dir / "deck.csv") != profile_deck_digest:
        raise RuntimeError("recursive RTP package deck cards changed")
    packaged_deck_file_digest = _sha256_file(agent_dir / "deck.csv")
    if _require_sha256(
        profile.get("deck_file_sha256"), field="deck_file_sha256"
    ) != packaged_deck_file_digest:
        raise RuntimeError("recursive RTP package deck file changed")
    if _sha256_file(matchup_tree) != profile_tree_digest:
        raise RuntimeError("recursive RTP package matchup tree changed")

    try:
        promotion_identity, promotion = read_r198_immutable_json_object(
            promotion_path,
            label="packaged RTP promotion receipt",
            expected_sha256=_require_sha256(
                profile.get("rtp_promotion_receipt_sha256"),
                field="rtp_promotion_receipt_sha256",
            ),
        )
        packaged_capability = resolve_r198_packaged_evaluation_capability()
    except RTPPromotionEvidenceError as exc:
        raise RuntimeError(
            "packaged RTP promotion/evaluation evidence is not sealed: " + str(exc)
        ) from exc
    if packaged_capability is None:
        raise RuntimeError("recursive RTP package has no sealed evaluation capability")
    if (
        packaged_capability.promotion_receipt_path != promotion_identity["path"]
        or packaged_capability.promotion_receipt_sha256 != promotion_identity["sha256"]
        or packaged_capability.evaluation_receipt_path
        != os.path.abspath(os.fspath(evaluation_path))
        or packaged_capability.evaluation_receipt_sha256
        != _require_sha256(
            profile.get("rtp_evaluation_receipt_sha256"),
            field="rtp_evaluation_receipt_sha256",
        )
    ):
        raise RuntimeError("recursive RTP package sealed evaluation capability changed")
    missing = sorted(_RTP_PROMOTION_REQUIRED_FIELDS.difference(promotion))
    if missing:
        raise RuntimeError("RTP promotion receipt is incomplete: " + ", ".join(missing))
    if (
        promotion.get("schema") != _RTP_PROMOTION_SCHEMA
        or promotion.get("status") != "accepted"
        or promotion.get("specialist_id") != _RTP_R197_SPECIALIST_ID
    ):
        raise RuntimeError("RTP promotion receipt is not an accepted r197 receipt")
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
        _require_exact_bool(promotion, field)
    if not isinstance(promotion.get("created_at_utc"), str) or not str(
        promotion.get("created_at_utc")
    ).strip():
        raise RuntimeError("RTP promotion receipt has no creation timestamp")
    if not isinstance(promotion.get("evaluation_receipt_path"), str) or not str(
        promotion.get("evaluation_receipt_path")
    ).strip():
        raise RuntimeError("RTP promotion receipt has no evaluation receipt path")
    try:
        validate_r198_evaluation_receipt(
            packaged_capability.evaluation_receipt_path,
            expected_sha256=_require_sha256(
                promotion.get("evaluation_receipt_sha256"),
                field="promotion.evaluation_receipt_sha256",
            ),
            # The immutable source archive is intentionally not duplicated in
            # a Kaggle package.  The package still revalidates the full v2
            # receipt content and every identity/gate, while the build
            # boundary above reopens the external evidence.
            require_local_evidence=False,
            expected_parent_checkpoint_sha256=model_digest,
            expected_sidecar_sha256=_sha256_file(sidecar),
            expected_sidecar_config_sha256=_require_sha256(
                promotion.get("sidecar_config_sha256"),
                field="promotion.sidecar_config_sha256",
            ),
            expected_deck_file_sha256=packaged_deck_file_digest,
            expected_deck_cards_sha256=profile_deck_digest,
            expected_matchup_tree_sha256=profile_tree_digest,
        )
    except RTPPromotionEvidenceError as exc:
        raise RuntimeError(f"packaged RTP evaluation receipt is non-promotable: {exc}") from exc

    try:
        sidecar_payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 — malformed package bytes are fatal.
        raise RuntimeError("recursive RTP sidecar safe load failed") from exc
    if not isinstance(sidecar_payload, Mapping):
        raise RuntimeError("recursive RTP sidecar payload is not an object")
    if sidecar_payload.get("schema") != _RTP_SIDECAR_SCHEMA:
        raise RuntimeError("recursive RTP sidecar schema changed")
    if sidecar_payload.get("research_only") is not False:
        raise RuntimeError("recursive RTP sidecar is research-only")
    if sidecar_payload.get("shadow_only") is not True:
        raise RuntimeError("recursive RTP sidecar is not explicitly shadow-only")
    # A sidecar is a learned artifact, not its own promotion authority.  The
    # immutable external r197 promotion receipt is the only serving grant.
    if (
        sidecar_payload.get("serving_eligible") is not False
        or sidecar_payload.get("action_authority_enabled") is not False
    ):
        raise RuntimeError("recursive RTP sidecar must remain promotion-inert")
    sidecar_parent = _require_sha256(
        sidecar_payload.get("parent_checkpoint_sha256"),
        field="sidecar.parent_checkpoint_sha256",
    )
    config = sidecar_payload.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("recursive RTP sidecar lacks a config object")
    state_dict = sidecar_payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("recursive RTP sidecar lacks a state dictionary")
    config_dict = dict(config)
    config_fields = {field.name for field in fields(RTPConfig)}
    unknown_config_fields = sorted(set(config_dict).difference(config_fields))
    if unknown_config_fields:
        raise RuntimeError(
            "recursive RTP sidecar config has unknown fields: "
            + ", ".join(unknown_config_fields)
        )
    try:
        strict_probe = RecursiveTurnPlanner(RTPConfig(**config_dict))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("recursive RTP sidecar config cannot construct a planner") from exc
    expected_state = strict_probe.state_dict()
    if set(state_dict) != set(expected_state):
        raise RuntimeError("recursive RTP sidecar state dictionary is not exact")
    for name, expected_tensor in expected_state.items():
        actual_tensor = state_dict[name]
        if getattr(actual_tensor, "shape", None) != expected_tensor.shape:
            raise RuntimeError(
                f"recursive RTP sidecar tensor shape changed: {name}"
            )
    if (
        strict_probe.required_recursive_passes() != _RTP_R197_REQUIRED_NEURAL_PASSES["normal"]
        or strict_probe.required_recursive_passes(force_recurse=True)
        != _RTP_R197_REQUIRED_NEURAL_PASSES["forced_replan"]
    ):
        raise RuntimeError("recursive RTP sidecar required neural passes changed")
    config_digest = _canonical_json_sha256(config_dict)
    sidecar_digest = _sha256_file(sidecar)
    profile_max_neural_passes = _require_exact_int(
        profile, "max_neural_passes"
    )
    profile_max_action_combos = _require_exact_int(profile, "max_action_combos")
    receipt_max_neural_passes = _require_exact_int(
        promotion, "max_neural_passes"
    )
    receipt_max_action_combos = _require_exact_int(
        promotion, "max_action_combos"
    )
    _assert_r197_sidecar_config(
        config_dict, expected_max_neural_passes=profile_max_neural_passes
    )
    if (
        sidecar_parent != model_digest
        or _require_sha256(
            profile.get("parent_checkpoint_sha256"),
            field="parent_checkpoint_sha256",
        )
        != model_digest
        or _require_sha256(
            promotion.get("parent_checkpoint_sha256"),
            field="promotion.parent_checkpoint_sha256",
        )
        != model_digest
        or _require_sha256(
            profile.get("rtp_checkpoint_sha256"),
            field="rtp_checkpoint_sha256",
        )
        != sidecar_digest
        or _require_sha256(
            promotion.get("sidecar_sha256"), field="promotion.sidecar_sha256"
        )
        != sidecar_digest
        or _require_sha256(
            profile.get("rtp_config_sha256"), field="rtp_config_sha256"
        )
        != config_digest
        or _require_sha256(
            promotion.get("sidecar_config_sha256"),
            field="promotion.sidecar_config_sha256",
        )
        != config_digest
        or receipt_max_neural_passes != profile_max_neural_passes
        or receipt_max_action_combos != profile_max_action_combos
        or profile.get("required_neural_passes") != _RTP_R197_REQUIRED_NEURAL_PASSES
        or promotion.get("required_neural_passes")
        != _RTP_R197_REQUIRED_NEURAL_PASSES
        or _require_sha256(
            promotion.get("deck_file_sha256"),
            field="promotion.deck_file_sha256",
        )
        != packaged_deck_file_digest
        or _require_sha256(
            promotion.get("deck_cards_sha256"),
            field="promotion.deck_cards_sha256",
        )
        != profile_deck_digest
        or _require_sha256(
            promotion.get("matchup_tree_sha256"),
            field="promotion.matchup_tree_sha256",
        )
        != profile_tree_digest
        or _require_sha256(
            promotion.get("evaluation_receipt_sha256"),
            field="promotion.evaluation_receipt_sha256",
        )
        != _sha256_file(evaluation_path)
    ):
        raise RuntimeError("recursive RTP package binding changed")
    if profile_max_neural_passes != _RTP_R198_EXACT_MAX_NEURAL_PASSES:
        raise RuntimeError(
            "recursive RTP receipt did not authorize the exact 256 neural-pass budget"
        )
    if profile_max_action_combos != _RTP_R198_EXACT_MAX_ACTION_COMBOS:
        raise RuntimeError(
            "recursive RTP receipt did not authorize the exact 1024 action combinations"
        )
    return config_dict


def _force_direct_rtp_bridge(policy: object) -> None:
    """Make the B arm use the bridge's legal-combo direct policy only."""

    bridge = getattr(policy, "_rtp_bridge", None)
    planner = getattr(bridge, "planner", None)
    if bridge is None or planner is None:
        raise RuntimeError("direct RTP package did not create its bridge")

    def direct_only(_memory: object, *, policy_logits: object = None):
        return False, {
            "submission_rtp_mode": "direct",
            "recursion_forced_disabled": True,
            "policy_logits_present": policy_logits is not None,
        }

    original_plan_turn = planner.plan_turn

    def direct_plan_turn(
        memory: object,
        *,
        policy_logits: object = None,
        force_recurse: object = None,
    ):
        # ``RTPAgentBridge`` may request ``force_recurse=True`` while repairing
        # an already-persisted program.  B must remain a no-recursion control,
        # so discard that request and route every plan entry through the
        # direct-only complexity gate above.
        del force_recurse
        return original_plan_turn(
            memory,
            policy_logits=policy_logits,
            force_recurse=None,
        )

    # The bridge remains unchanged; this package-only override is deliberately
    # narrow so B−A measures the full-combo bridge while C−B measures every
    # recursive plan and forced-replan path.
    setattr(planner, "should_recurse", direct_only)
    setattr(planner, "plan_turn", direct_plan_turn)
    setattr(bridge, "_submission_direct_only", True)


def _assert_live_recursive_config(policy: object, expected_config: Mapping[str, object]) -> None:
    """Ensure bridge/executor and sidecar did not diverge after loading."""

    bridge = getattr(policy, "_rtp_bridge", None)
    planner = getattr(bridge, "planner", None)
    bridge_config = getattr(bridge, "config", None)
    planner_config = getattr(planner, "config", None)
    if bridge is None or planner is None or bridge_config is None or planner_config is None:
        raise RuntimeError("recursive RTP package did not create its bridge")
    for field, expected in expected_config.items():
        if field == "schema":
            bridge_value = getattr(bridge_config, "schema", None)
            planner_value = getattr(planner_config, "schema", None)
        else:
            bridge_value = getattr(bridge_config, field, None)
            planner_value = getattr(planner_config, field, None)
        if bridge_value != expected or planner_value != expected:
            raise RuntimeError(
                f"recursive RTP live config diverged at {field}: "
                f"bridge={bridge_value!r} planner={planner_value!r}"
            )
    if getattr(bridge, "max_action_combos", None) != _RTP_R198_EXACT_MAX_ACTION_COMBOS:
        raise RuntimeError("recursive RTP bridge did not bind exact 1024 action combinations")


def _read_deck() -> list[int]:
    path = _agent_dir() / "deck.csv"
    deck: list[int] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        deck.append(int(line.split(",")[0]))
        if len(deck) >= 60:
            break
    if len(deck) != 60:
        raise ValueError(f"deck.csv must have 60 cards, got {len(deck)}")
    return deck


_DECK: list[int] | None = None
_MODEL = None
_CLOCK = None
_POLICY = None
_SEARCH_BUDGET = None
_SEARCH_CONFIG = None
_GAME_COUNT = 0
_RNG = random.Random(0)
_NO_PROGRESS_ESCAPE_TURNS = 0
_NO_PROGRESS_LAST_SEEN_TURN: int | None = None
_NO_PROGRESS_LAST_PROGRESS_TURN: int | None = None
_NO_PROGRESS_LAST_SIGNATURE: tuple[Any, ...] | None = None
_NO_PROGRESS_ESCAPE_LATCHED = False
_TRAINED_DORMANT_MATCHUP_ADAPTER_SCHEMA = (
    "poke_bot.trained_dormant_matchup_adapter/v1"
)


def _checkpoint_has_trained_matchup_adapter_bank(payload: object) -> bool:
    """Return whether this immutable checkpoint requires packaged activation.

    The serialized adapter flag remains deliberately dormant so fitting cannot
    silently change ordinary checkpoint loads.  A submitted package instead
    proves its serving activation with the shipped public tree and entry point.
    """

    if not isinstance(payload, Mapping):
        return False
    extra = payload.get("extra")
    if not isinstance(extra, Mapping):
        return False
    dormant = extra.get("dormant_matchup_adapter_bank")
    return bool(
        isinstance(dormant, Mapping)
        and dormant.get("schema") == _TRAINED_DORMANT_MATCHUP_ADAPTER_SCHEMA
        and dormant.get("zero_output") is False
    )


def _assert_trained_matchup_tree_binding(
    *, checkpoint_payload: Mapping[str, object], matchup_tree: Path
) -> None:
    """Require the shipped tree to bind every enabled route to this checkpoint.

    This duplicates the builder's final package guard intentionally: Kaggle
    runs ``main.py``, not the local build command, so a manually assembled
    archive must not be able to arm a trained bank with a merely plausible
    public tree.
    """

    from poke_bot.matchup_adapter_routes import (
        require_runtime_route_binding,
        resolve_matchup_adapter_route_contract,
    )
    from poke_bot.public_matchup_router import PublicMatchupDecisionTree

    extra = checkpoint_payload.get("extra")
    if not isinstance(extra, Mapping):
        raise TypeError("trained matchup adapter checkpoint lacks metadata")
    adapter_config = extra.get("matchup_adapter_config")
    if not isinstance(adapter_config, Mapping):
        raise TypeError("trained matchup adapter checkpoint lacks route contract")
    try:
        tree = PublicMatchupDecisionTree.from_path(
            matchup_tree, require_runtime_enabled=True
        )
        route_contract = resolve_matchup_adapter_route_contract(adapter_config)
        tree_payload = json.loads(matchup_tree.read_text(encoding="utf-8"))
        runtime = dict(tree_payload.get("runtime_contract") or {})
        require_runtime_route_binding(runtime, route_contract)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "packaged matchup adapter tree is not runtime-bound to the checkpoint"
        ) from exc

    checkpoint_archetype = str(
        checkpoint_payload.get("archetype_id") or ""
    ).strip().casefold()
    if (
        not checkpoint_archetype
        or checkpoint_archetype not in tree.runtime_accepted_archetype_ids
        or tuple(tree.targets) != route_contract.target_ids
        or tuple(tree.route_physical_slots) != route_contract.physical_slots
        or tree.adapter_format != route_contract.adapter_format
        or tree.slot_registry_digest != route_contract.slot_registry_digest
    ):
        raise RuntimeError(
            "packaged matchup adapter tree does not match the checkpoint route contract"
        )

    state = checkpoint_payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("trained matchup adapter checkpoint lacks model state")

    def output_nonzero(value: Any) -> bool | None:
        try:
            return int(value.detach().count_nonzero().item()) > 0
        except (AttributeError, RuntimeError, TypeError):
            return None

    invalid_routes: list[str] = []
    for archetype_id in sorted(tree.runtime_accepted_archetype_ids):
        slot = route_contract.physical_slot_by_target[archetype_id]
        prefix = f"matchup_adapter_bank.experts.{slot}.up."
        weight_nonzero = output_nonzero(state.get(prefix + "weight"))
        bias_nonzero = output_nonzero(state.get(prefix + "bias"))
        if weight_nonzero is None or bias_nonzero is None:
            invalid_routes.append(f"{archetype_id}@{slot}:missing-output")
        elif not (weight_nonzero or bias_nonzero):
            invalid_routes.append(f"{archetype_id}@{slot}:zero-output")
    if invalid_routes:
        raise RuntimeError(
            "packaged matchup tree accepts adapter route(s) without a verified "
            "nonzero output projection: "
            + ", ".join(invalid_routes)
        )


def _assert_trained_matchup_runtime(
    *, model, policy, matchup_tree: Path
) -> None:
    """Fail closed unless the exact package activated its frozen adapter bank."""

    if getattr(policy, "matchup_adapter_runtime", None) is not True:
        raise RuntimeError("packaged matchup adapter runtime was not enabled")
    bank = getattr(model, "matchup_adapter_bank", None)
    if bank is None or getattr(bank, "enabled", None) is not True:
        raise RuntimeError("packaged matchup adapter bank was not enabled")
    if any(parameter.requires_grad for parameter in bank.parameters()):
        raise RuntimeError("packaged matchup adapter bank must remain frozen")
    router = getattr(policy, "_matchup_adapter_shadow_router", None)
    tree = getattr(router, "tree", None)
    expected_digest = "sha256:" + hashlib.sha256(matchup_tree.read_bytes()).hexdigest()
    if (
        tree is None
        or getattr(tree, "runtime_enabled", None) is not True
        or getattr(tree, "digest", None) != expected_digest
    ):
        raise RuntimeError("packaged matchup adapter tree activation changed")


def _turn_order_preference() -> str:
    """Read the immutable packaged preference without importing the runtime."""

    path = _agent_dir() / "turn_order_profile.json"
    if not path.is_file():
        return "first_if_allowed"
    payload = json.loads(path.read_text())
    preference = str(payload.get("turn_order_preference") or "")
    if preference not in {"first_if_allowed", "second_if_allowed"}:
        raise RuntimeError("invalid packaged turn-order preference")
    return preference


def _turn_order_choice(obs_dict: dict) -> list[int] | None:
    """Resolve IsFirst directly from the wire enum without runtime imports."""

    selection = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not isinstance(selection, dict):
        return None
    context = selection.get("context")
    normalized_context = "".join(
        character for character in str(context).lower() if character.isalnum()
    )
    if context != 41 and normalized_context != "isfirst":
        return None
    options = list(selection.get("option") or [])
    desired_type = (
        "yes" if _turn_order_preference() == "first_if_allowed" else "no"
    )
    desired_integer = 1 if desired_type == "yes" else 2
    matches = [
        index
        for index, option in enumerate(options)
        if isinstance(option, dict)
        and (
            option.get("type") == desired_integer
            or str(option.get("type") or "").strip().lower() == desired_type
        )
    ]
    return matches if len(matches) == 1 else []


def _go_first_choice(obs_dict: dict) -> list[int] | None:
    """Backward-compatible alias for the packaged turn-order resolver."""

    return _turn_order_choice(obs_dict)


def _ensure_agent_path() -> None:
    agent_dir = str(_agent_dir())
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)


def _ensure_runtime():
    global _DECK, _MODEL, _CLOCK, _POLICY, _SEARCH_BUDGET, _SEARCH_CONFIG
    global _NO_PROGRESS_ESCAPE_TURNS
    if _DECK is None:
        _DECK = _read_deck()
    if _MODEL is None:
        runtime_profile = _apply_runtime_profile()
        raw_escape_turns = runtime_profile.get("no_progress_escape_turns")
        if raw_escape_turns is not None:
            if (
                isinstance(raw_escape_turns, bool)
                or int(raw_escape_turns) != _SUBMISSION_NO_PROGRESS_ESCAPE_TURNS
            ):
                raise RuntimeError("submission no-progress escape contract changed")
            _NO_PROGRESS_ESCAPE_TURNS = int(raw_escape_turns)
        _ensure_agent_path()
        # Vendored ``cg/`` sits directly beside this entry point. The shared
        # runtime path resolver otherwise looks only for repository/Kaggle
        # development layouts that do not exist inside the submitted tarball.
        os.environ.setdefault("CG_LIB_PATH", str(_agent_dir()))
        import torch

        from poke_bot import checkpoint as checkpoint_mod
        from poke_bot.agent import PolicyAgent
        from poke_bot.belief import EmpiricalDeckPosterior
        from poke_bot.checkpoint import (
            assert_trusted_policy_checkpoint,
            checkpoint_digest,
        )
        from poke_bot.dormant_adapter_compat import validate_zero_dormant_checkpoint
        from poke_bot.submission_budget import SubmissionSearchBudget
        from poke_bot.train import load_model_from_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = _agent_dir() / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError("model.pt is required")
        assert_trusted_policy_checkpoint(checkpoint)
        checkpoint_payload = checkpoint_mod.load_checkpoint(
            checkpoint, map_location="cpu"
        )
        derivative_checkpoint = checkpoint_payload.get("schema") == (
            "poke_bot.alakazam_rule_derivative_composite_candidate_initialization/v1"
        )
        if derivative_checkpoint:
            derivative_authority = (
                checkpoint_payload.get("goal_revision"),
                checkpoint_payload.get("goal_contract_sha256"),
            )
            if (
                derivative_authority
                not in {
                    (
                        9,
                        "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8",
                    ),
                    (
                        10,
                        "sha256:91e9c60e87fe093446ef9979f64464be18e77a5041afe82164c4c6ca80d2225f",
                    ),
                }
                or runtime_profile is None
                or runtime_profile.get("public_rule_semantic_projection")
                != "enabled"
                or runtime_profile.get("public_rule_semantic_projection_gate")
                != 1.0
                or runtime_profile.get("model_checkpoint_sha256")
                != _sha256_file(checkpoint)
            ):
                raise RuntimeError(
                    "rule-derivative checkpoint lacks its exact submitted runtime binding"
                )
        runtime_mode = (
            _runtime_profile_mode(runtime_profile) if runtime_profile else "default_off"
        )
        direct_rtp_config: dict[str, object] | None = None
        recursive_rtp_config: dict[str, object] | None = None
        if runtime_mode == "direct":
            # Validate the exact inert sidecar before PolicyAgent can load it;
            # direct-only behavior is applied only after that bridge exists.
            direct_rtp_config = _assert_direct_rtp_binding(
                runtime_profile,
                model_digest=_sha256_file(checkpoint),
            )
        elif runtime_mode == "recursive":
            # Verify sidecar + promotion bindings before the policy can encode
            # or load an unbound planner.  This is intentionally on the first
            # non-turn-order policy path rather than build-only validation.
            recursive_rtp_config = _assert_recursive_rtp_binding(
                runtime_profile,
                model_digest=_sha256_file(checkpoint),
            )
        trained_matchup_adapter_bank = _checkpoint_has_trained_matchup_adapter_bank(
            checkpoint_payload
        )
        matchup_tree = _agent_dir() / "matchup_tree.json"
        if trained_matchup_adapter_bank and not matchup_tree.is_file():
            raise FileNotFoundError(
                "trained matchup adapter checkpoint requires matchup_tree.json"
            )
        if trained_matchup_adapter_bank:
            # Serving activation is an explicit package-only override.  The
            # immutable checkpoint itself must remain a validated frozen,
            # serialized-dormant bank.
            validate_zero_dormant_checkpoint(checkpoint, allow_trained=True)
            _assert_trained_matchup_tree_binding(
                checkpoint_payload=checkpoint_payload,
                matchup_tree=matchup_tree,
            )
        if trained_matchup_adapter_bank and matchup_tree.is_file():
            # The shipped tree is itself runtime-gated and consumes only
            # cumulative public opponent cards. PolicyAgent validates the
            # artifact before enabling the frozen trained adapter bank.
            os.environ["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"
            os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(matchup_tree)
        else:
            # A package may carry an inert tree solely for the shared queue's
            # immutable bundle ABI.  Without a trained bank it must not turn
            # routing on, nor inherit another package's activation state.
            os.environ.pop("POKEBOT_MATCHUP_ADAPTER_RUNTIME", None)
            os.environ.pop("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", None)
        model = load_model_from_checkpoint(checkpoint, device=device)
        model.eval()
        _MODEL = model
        semantic_projection = None
        if derivative_checkpoint:
            from poke_bot.alakazam_rule_derivative_model_r298 import (
                R298PublicRuleSemanticProjection,
                R298SemanticProjectionConfig,
            )

            semantic_projection = R298PublicRuleSemanticProjection(
                R298SemanticProjectionConfig(
                    **dict(
                        checkpoint_payload[
                            "public_rule_semantic_projection_config"
                        ]
                    )
                )
            ).to(device)
            semantic_projection.load_state_dict(
                checkpoint_payload["public_rule_semantic_projection_state_dict"],
                strict=True,
            )
            semantic_projection.requires_grad_(False)
            semantic_projection.eval()
        derivative_policy_kwargs = {
            "public_rule_semantic_projection": semantic_projection,
            "public_rule_semantic_projection_enabled": derivative_checkpoint,
            "public_rule_semantic_projection_gate": 1.0,
            "strict_runtime": derivative_checkpoint,
        }
        search_config_path = _agent_dir() / "search_config.json"
        belief_decks_path = _agent_dir() / "belief_decks.json"
        search_enabled = (
            os.environ.get("POKEBOT_SUBMISSION_SEARCH_DISABLE", "0") != "1"
            and search_config_path.is_file()
            and belief_decks_path.is_file()
        )
        if search_enabled:
            _SEARCH_CONFIG = json.loads(search_config_path.read_text())
            _SEARCH_BUDGET = SubmissionSearchBudget.from_config(
                _SEARCH_CONFIG,
                started_at=_PROCESS_STARTED,
            )
            if _SEARCH_CONFIG.get("enabled") is not True:
                # Canonical competition mode is the frozen policy-only path.
                # The digest-bound belief-MCTS implementation below remains
                # dormant for a separately validated future experiment.
                _POLICY = PolicyAgent(
                    model=model,
                    deck=_DECK,
                    use_mcts=False,
                    **derivative_policy_kwargs,
                )
                _CLOCK = None
            else:
                belief_payload = json.loads(belief_decks_path.read_text())
                deck_hypotheses = belief_payload.get("deck_lists") or ()
                if (
                    _SEARCH_CONFIG.get("algorithm")
                    != "public_history_root_sampled_belief_mcts"
                    or _SEARCH_CONFIG.get("leaf_evaluator")
                    != "trained_checkpoint_policy_value_head"
                    or _SEARCH_CONFIG.get("leaf_evaluator_checkpoint")
                    != "submission_model_pt"
                    or _SEARCH_CONFIG.get("require_trained_state_evaluator")
                    is not True
                    or _SEARCH_CONFIG.get("search_failure_behavior")
                    != "greedy_current_decision_then_retry"
                    or _SEARCH_CONFIG.get(
                        "game_wide_greedy_only_for_time_budget"
                    )
                    is not True
                    or _SEARCH_CONFIG.get("fallback")
                    != "frozen_model_greedy_policy"
                    or _SEARCH_CONFIG.get("oracle_inputs_allowed") is not False
                    or int(_SEARCH_CONFIG.get("lane_count", 1)) not in (1, 8)
                    or (
                        int(_SEARCH_CONFIG.get("lane_count", 1)) == 8
                        and _SEARCH_CONFIG.get("search_runtime")
                        != "native_handle_eight_lane_belief_forest"
                    )
                    or belief_payload.get("schema")
                    != "poke_bot.submission_belief_decks/v1"
                    or belief_payload.get("anonymous") is not True
                    or belief_payload.get("contains_opponent_identity") is not False
                    or int(belief_payload.get("deck_count") or 0)
                    != len(deck_hypotheses)
                    or len(deck_hypotheses) < 8
                    or any(
                        len(deck) != 60
                        or any(int(card) <= 0 for card in deck)
                        for deck in deck_hypotheses
                    )
                ):
                    raise RuntimeError("submission belief-deck prior changed")
                posterior = EmpiricalDeckPosterior(deck_hypotheses)
                model_digest = checkpoint_digest(checkpoint)
                _POLICY = PolicyAgent(
                    model=model,
                    deck=_DECK,
                    use_mcts=True,
                    belief_mcts=True,
                    belief_posterior=posterior,
                    checkpoint_digest=model_digest,
                    model_generation=0,
                    game_time_budget_s=float(
                        _SEARCH_CONFIG["total_search_budget_s"]
                    ),
                    game_watchdog_reserve_s=0.0,
                    expected_search_decisions=int(
                        _SEARCH_CONFIG["expected_search_decisions"]
                    ),
                    max_sims=int(_SEARCH_CONFIG["minimum_sims"]),
                    min_trusted_sims=int(_SEARCH_CONFIG["minimum_sims"]),
                    move_time_s=float(_SEARCH_CONFIG["maximum_move_s"]),
                    belief_mcts_lanes=int(_SEARCH_CONFIG.get("lane_count", 1)),
                    **derivative_policy_kwargs,
                )
                _CLOCK = _POLICY.clock
        else:
            _POLICY = PolicyAgent(
                model=model,
                deck=_DECK,
                use_mcts=False,
                **derivative_policy_kwargs,
            )
            _CLOCK = None
        if runtime_mode in {"legacy_off", "off"}:
            if (
                _POLICY.use_recursive_turn_planner is not False
                or _POLICY._rtp_bridge is not None
                or os.environ.get("POKEBOT_USE_RECURSIVE_TURN_PLANNER") != "0"
                or os.environ.get("POKEBOT_RTP_CHECKPOINT")
            ):
                raise RuntimeError("NO RTP submitted runtime profile was not enforced")
        elif runtime_mode == "direct":
            _force_direct_rtp_bridge(_POLICY)
            if (
                _POLICY.use_recursive_turn_planner is not True
                or _POLICY._rtp_bridge is None
                or not os.environ.get("POKEBOT_RTP_CHECKPOINT")
                or direct_rtp_config is None
                or os.environ.get("POKEBOT_RTP_ALLOW_UNTRAINED")
                or os.environ.get("POKEBOT_RTP_SERVING_QUALIFIED")
                or os.environ.get("POKEBOT_RTP_PROMOTION_RECEIPT")
                or os.environ.get("POKEBOT_RTP_PROMOTION_RECEIPT_SHA256")
            ):
                raise RuntimeError("direct RTP submitted runtime profile was not enforced")
            _assert_live_recursive_config(_POLICY, direct_rtp_config)
        elif runtime_mode == "recursive":
            if (
                _POLICY.use_recursive_turn_planner is not True
                or _POLICY._rtp_bridge is None
                or not os.environ.get("POKEBOT_RTP_CHECKPOINT")
                or recursive_rtp_config is None
            ):
                raise RuntimeError("recursive RTP submitted runtime profile was not enforced")
            _assert_live_recursive_config(_POLICY, recursive_rtp_config)
        if trained_matchup_adapter_bank:
            _assert_trained_matchup_runtime(
                model=model,
                policy=_POLICY,
                matchup_tree=matchup_tree,
            )
    return _DECK, _MODEL, _POLICY


def _fail_closed(obs_dict: dict, preferred: list[int]) -> list[int]:
    selection = obs_dict.get("select") if obs_dict else None
    if selection is None:
        return preferred
    option_count = len(selection.get("option") or [])
    if option_count <= 0:
        return []
    minimum = int(selection.get("minCount", 0) or 0)
    maximum = min(int(selection.get("maxCount", 0) or 0), option_count)
    minimum = max(0, min(minimum, maximum))
    clean: list[int] = []
    for raw in preferred:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= index < option_count and index not in clean:
            clean.append(index)
    if minimum <= len(clean) <= maximum and clean:
        return clean[:maximum]
    if maximum <= 0:
        return []
    count = _RNG.randint(minimum, maximum) if maximum >= minimum else maximum
    return _RNG.sample(range(option_count), count) if count > 0 else []


def _submission_progress_pokemon(card: Any) -> tuple[Any, ...] | None:
    if not isinstance(card, Mapping):
        return None
    return (
        card.get("id"),
        card.get("hp"),
        card.get("maxHp"),
        tuple(sorted(str(value) for value in (card.get("energies") or []))),
        tuple(sorted(str(value) for value in (card.get("energyCards") or []))),
    )


def _submission_win_progress_signature(obs_dict: dict) -> tuple[Any, ...]:
    """Public win progress; hand/deck/discard recycling is intentionally inert."""

    current = dict(obs_dict.get("current") or {})
    rows: list[tuple[Any, ...]] = []
    for raw_player in current.get("players") or []:
        player = raw_player if isinstance(raw_player, Mapping) else {}
        rows.append(
            (
                len(player.get("prize") or []),
                tuple(
                    value
                    for value in (
                        _submission_progress_pokemon(card)
                        for card in (player.get("active") or [])
                    )
                    if value is not None
                ),
                tuple(
                    value
                    for value in (
                        _submission_progress_pokemon(card)
                        for card in (player.get("bench") or [])
                    )
                    if value is not None
                ),
                bool(player.get("poisoned")),
                bool(player.get("burned")),
                bool(player.get("asleep")),
                bool(player.get("paralyzed")),
                bool(player.get("confused")),
            )
        )
    return tuple(rows)


def _reset_submission_no_progress_escape() -> None:
    global _NO_PROGRESS_LAST_SEEN_TURN, _NO_PROGRESS_LAST_PROGRESS_TURN
    global _NO_PROGRESS_LAST_SIGNATURE, _NO_PROGRESS_ESCAPE_LATCHED

    _NO_PROGRESS_LAST_SEEN_TURN = None
    _NO_PROGRESS_LAST_PROGRESS_TURN = None
    _NO_PROGRESS_LAST_SIGNATURE = None
    _NO_PROGRESS_ESCAPE_LATCHED = False


def _legal_end_turn_choice(obs_dict: dict) -> list[int] | None:
    selection = obs_dict.get("select") if obs_dict else None
    if not isinstance(selection, Mapping):
        return None
    context = selection.get("context")
    if not (
        context == 0
        or str(context or "").strip().lower() in {"main", "selectcontext.main"}
    ):
        return None
    options = selection.get("option") or []
    minimum = int(selection.get("minCount", 0) or 0)
    maximum = min(int(selection.get("maxCount", 0) or 0), len(options))
    if not minimum <= 1 <= maximum:
        return None
    matches = [
        index
        for index, option in enumerate(options)
        if isinstance(option, Mapping)
        and (
            option.get("type") == 14
            or str(option.get("type") or "").strip().lower()
            in {"end", "end_turn", "optiontype.end"}
        )
    ]
    return [matches[0]] if matches else None


def _submission_no_progress_escape(obs_dict: dict) -> list[int] | None:
    """Latch a package-only legal END escape after an extreme public stall."""

    global _NO_PROGRESS_LAST_SEEN_TURN, _NO_PROGRESS_LAST_PROGRESS_TURN
    global _NO_PROGRESS_LAST_SIGNATURE, _NO_PROGRESS_ESCAPE_LATCHED

    if _NO_PROGRESS_ESCAPE_TURNS <= 0:
        return None
    current = dict(obs_dict.get("current") or {})
    if int(current.get("result", -1)) != -1:
        return None
    turn = int(current.get("turn", 0) or 0)
    if turn != _NO_PROGRESS_LAST_SEEN_TURN:
        signature = _submission_win_progress_signature(obs_dict)
        if signature != _NO_PROGRESS_LAST_SIGNATURE:
            _NO_PROGRESS_LAST_SIGNATURE = signature
            _NO_PROGRESS_LAST_PROGRESS_TURN = turn
        _NO_PROGRESS_LAST_SEEN_TURN = turn
        if (
            _NO_PROGRESS_LAST_PROGRESS_TURN is not None
            and turn - _NO_PROGRESS_LAST_PROGRESS_TURN
            >= _NO_PROGRESS_ESCAPE_TURNS
        ):
            _NO_PROGRESS_ESCAPE_LATCHED = True
    if not _NO_PROGRESS_ESCAPE_LATCHED:
        return None
    return _legal_end_turn_choice(obs_dict)


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point."""

    global _GAME_COUNT
    turn_order = _turn_order_choice(obs_dict)
    if turn_order is not None:
        return _fail_closed(obs_dict, turn_order)

    deck, _model, policy = _ensure_runtime()
    _ensure_agent_path()
    from cg.api import to_observation_class

    observation = to_observation_class(obs_dict)
    if observation.select is None:
        _reset_submission_no_progress_escape()
        if policy is not None:
            policy.reset_game()
        if _SEARCH_BUDGET is not None:
            if _GAME_COUNT > 0:
                _SEARCH_BUDGET.reset()
            _GAME_COUNT += 1
        return list(deck)

    escape_action = _submission_no_progress_escape(obs_dict)
    if escape_action is not None:
        return _fail_closed(obs_dict, escape_action)

    try:
        if _SEARCH_BUDGET is None:
            action = policy.trusted_search_or_greedy_select(
                obs_dict,
                search=False,
            )
        else:
            plan = _SEARCH_BUDGET.plan(obs_dict)
            policy.max_sims = plan.max_sims or policy.max_sims
            policy.move_time_s = plan.move_time_s or policy.move_time_s
            prior_result = policy.last_result
            started = time.monotonic()
            action = policy.trusted_search_or_greedy_select(
                obs_dict,
                search=plan.search,
            )
            elapsed = time.monotonic() - started
            if plan.search:
                result = (
                    policy.last_result
                    if policy.last_result is not prior_result
                    else None
                )
                _SEARCH_BUDGET.record_search(
                    elapsed_s=elapsed,
                    completed_sims=(
                        int(result.sims_run) if result is not None else 0
                    ),
                    succeeded=(
                        result is not None
                        and policy.last_search_fallback_reason is None
                    ),
                )
    except Exception:
        action = []
    return _fail_closed(obs_dict, action)
