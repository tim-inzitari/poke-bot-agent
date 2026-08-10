"""Causal, read-only replay-step inference for the standalone inspector.

The cache/provenance layer resolves a submission to an immutable bundle and
checkpoint before this module is called.  This module then replays only the
acting seat's *masked* observations, recreates the legal factorized stage, and
reports model activations without writing replay or checkpoint data.

It intentionally does not try to infer another Kaggle participant's weights
from a replay.  A replay can be evaluated only with a separately verified
checkpoint supplied by the caller.
"""

from __future__ import annotations

import gc
import importlib
import math
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch

from poke_bot import checkpoint, deck_guides, features
from poke_bot.model import (
    DECISION_FUSION_V3_MAX_RELIABILITY,
    DECISION_FUSION_V3_MIN_RELIABILITY,
)
from poke_bot.replay_import import extract_setup_decks

from .timeline import ReplayTimelineError, assert_active_select_actor

INFERENCE_SCHEMA = "poke_bot.replay_model_inspector.inference/v1"
UNAVAILABLE_TARGET_MASK_REASON = (
    "Kaggle replay rows do not carry training target masks; no mask is invented"
)
_MODEL_TRACE_LOCK = threading.RLock()
_SUBMITTED_RUNTIME_ACTIVATION_BASIS = "checksum_bound_submitted_startup"
_CAUSAL_REEVALUATION_BASIS = "checksum_bound_causal_re_evaluation"


class ReplayInspectionError(RuntimeError):
    """The request violates the read-only inference contract."""


class ReplayStepUnavailable(ReplayInspectionError):
    """A replay/model format cannot reconstruct the requested model decision."""

    def __init__(
        self, reason: str, *, code: str = "unavailable", **details: Any
    ) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.code = str(code)
        self.details = dict(details)


@dataclass(frozen=True)
class LoadedModel:
    """One already checksum-verified CPU evaluation model."""

    model: torch.nn.Module
    checkpoint_path: Path
    checkpoint_digest: str


class VerifiedCpuModelCache:
    """Keep at most one checksum-verified evaluation model resident."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded: LoadedModel | None = None

    @property
    def loaded(self) -> LoadedModel | None:
        with self._lock:
            return self._loaded

    def clear(self) -> None:
        """Release only the in-memory model; never delete a checkpoint file."""

        with self._lock:
            self._loaded = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _device() -> torch.device:
        requested = (
            os.environ.get("POKEBOT_REPLAY_INSPECTOR_DEVICE", "cpu").strip().casefold()
        )
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise ReplayInspectionError(
                    "CUDA was requested for replay inspection but is unavailable"
                )
            return torch.device("cuda:0")
        raise ReplayInspectionError(
            "POKEBOT_REPLAY_INSPECTOR_DEVICE must be cpu or cuda"
        )

    def load(
        self,
        checkpoint_path: str | Path,
        expected_digest: str,
    ) -> LoadedModel:
        """Verify, lazily load, and retain exactly one immutable checkpoint.

        The digest is checked before and after deserialization so a path changed
        during load cannot silently become the resident model.  The provenance
        layer is responsible for configured-root/symlink policy; this class
        nevertheless resolves a concrete existing file and refuses a missing
        expected digest.
        """

        expected = str(expected_digest or "").strip()
        if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", expected) is None:
            raise ReplayInspectionError(
                "a sha256 checkpoint digest is required before model loading"
            )
        try:
            resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ReplayInspectionError("checkpoint path cannot be resolved") from exc
        if not resolved.is_file():
            raise ReplayInspectionError("checkpoint path is not a regular file")

        with self._lock:
            current = self._loaded
            if (
                current is not None
                and current.checkpoint_path == resolved
                and current.checkpoint_digest == expected
            ):
                return current

            before = checkpoint.checkpoint_digest(resolved)
            if before != expected:
                raise ReplayInspectionError(
                    "checkpoint checksum does not match the provenance binding"
                )

            # Drop the old model before allocating the next one.  This is a
            # resource bound, not an eviction of any on-disk artifact.
            self._loaded = None
            del current
            gc.collect()
            # Loading the training module is intentionally lazy: a caller that
            # receives an already verified fixture model should not need the
            # training CLI's optional progress/reporting dependencies.
            from poke_bot.train import load_model_from_checkpoint

            device = self._device()
            model = load_model_from_checkpoint(resolved, device=device)
            model.to(device)
            model.eval()

            after = checkpoint.checkpoint_digest(resolved)
            if after != expected:
                del model
                gc.collect()
                raise ReplayInspectionError(
                    "checkpoint changed while being loaded; refusing the model"
                )
            self._loaded = LoadedModel(model, resolved, expected)
            return self._loaded


class ReplayInferenceEngine:
    """Convenience facade used by the localhost API around one model cache."""

    def __init__(self, cache: VerifiedCpuModelCache | None = None) -> None:
        self.cache = cache or VerifiedCpuModelCache()

    def inspect(
        self,
        *,
        checkpoint_path: str | Path,
        expected_checkpoint_digest: str,
        replay: Mapping[str, Any],
        acting_seat: int,
        env_step: int,
        factorized_stage: int = 0,
        own_deck: Sequence[int] | None = None,
        router: Any | None = None,
        router_factory: Callable[[], Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        submitted_runtime_activation: Mapping[str, Any] | None = None,
        head_scales: Mapping[str, float] | None = None,
        allow_setup_prompt_model_forward: bool = False,
    ) -> dict[str, Any]:
        loaded = self.cache.load(checkpoint_path, expected_checkpoint_digest)
        return inspect_replay_step(
            model=loaded.model,
            replay=replay,
            acting_seat=acting_seat,
            env_step=env_step,
            factorized_stage=factorized_stage,
            own_deck=own_deck,
            router=router,
            router_factory=router_factory,
            checkpoint_digest=loaded.checkpoint_digest,
            checkpoint_path=loaded.checkpoint_path,
            provenance=provenance,
            submitted_runtime_activation=submitted_runtime_activation,
            head_scales=head_scales,
            allow_setup_prompt_model_forward=allow_setup_prompt_model_forward,
        )


def _availability(
    available: bool, reason: str | None = None, **extra: Any
) -> dict[str, Any]:
    result: dict[str, Any] = {"available": bool(available)}
    if reason is not None:
        result["reason"] = str(reason)
    result.update(extra)
    return result


def _provenance_mapping(provenance: Any | None) -> dict[str, Any]:
    """Accept the catalog's mapping/dataclass form without trusting paths from it."""

    if provenance is None:
        return {}
    if isinstance(provenance, Mapping):
        return dict(provenance)
    to_dict = getattr(provenance, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if is_dataclass(provenance):
        value = asdict(provenance)
        if isinstance(value, Mapping):
            return dict(value)
    raise ReplayInspectionError(
        "inference provenance must be a mapping or catalog record"
    )


def _adapter_output_evidence(adapter_bank: Any) -> dict[str, Any]:
    """Verify that the materialized bank has a non-zero output projection.

    Both supported adapter formats initialize every ``up`` projection to an
    exact zero.  A non-zero value in one of those tensors is therefore the
    checkpoint-local evidence that the otherwise dormant bank contains a
    learned/materialized policy residual.  Randomly initialized ``down``
    tensors alone are deliberately insufficient.
    """

    state_dict = getattr(adapter_bank, "state_dict", None)
    if not callable(state_dict):
        return {
            "available": False,
            "nonzero": None,
            "reason": "adapter bank does not expose checkpoint tensors",
        }
    output_tensors = [
        value.detach()
        for name, value in state_dict().items()
        if isinstance(value, torch.Tensor) and name.endswith(("up.weight", "up.bias"))
    ]
    if not output_tensors:
        return {
            "available": False,
            "nonzero": None,
            "reason": "adapter bank has no recognized output-projection tensors",
        }
    nonzero_count = sum(int(value.count_nonzero().item()) for value in output_tensors)
    provenance = getattr(adapter_bank, "dormant_provenance", None)
    provenance = provenance if isinstance(provenance, Mapping) else {}
    return {
        "available": True,
        "nonzero": nonzero_count > 0,
        "output_tensor_count": len(output_tensors),
        "nonzero_output_element_count": nonzero_count,
        "method": "checkpoint_adapter_up_projection_nonzero_inventory",
        "dormant_checkpoint_schema": provenance.get("schema"),
        "dormant_checkpoint_zero_output": provenance.get("zero_output"),
    }


@dataclass
class _AdapterRuntimeRestoration:
    adapter_bank: Any | None
    previous_enabled: bool | None
    previous_requires_grad: tuple[bool, ...]
    state_changed: bool


def _begin_submitted_runtime_activation(
    model: torch.nn.Module,
    request: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], _AdapterRuntimeRestoration]:
    """Apply the submitted PolicyAgent startup state for one locked trace.

    The serialized training flag is intentionally not serving authority.  A
    caller must supply the exact-runtime/tree evidence established by the
    provenance boundary.  Missing or malformed evidence forces an exact
    request-local bypass, including when a fixture/checkpoint happened to
    deserialize with ``enabled=True``.
    """

    adapter_bank = getattr(model, "matchup_adapter_bank", None)
    raw_enabled = getattr(adapter_bank, "enabled", None)
    previous_enabled = raw_enabled if isinstance(raw_enabled, bool) else None
    parameters = (
        tuple(adapter_bank.parameters())
        if adapter_bank is not None
        and callable(getattr(adapter_bank, "parameters", None))
        else ()
    )
    restoration = _AdapterRuntimeRestoration(
        adapter_bank=adapter_bank,
        previous_enabled=previous_enabled,
        previous_requires_grad=tuple(
            bool(parameter.requires_grad) for parameter in parameters
        ),
        state_changed=False,
    )
    base: dict[str, Any] = {
        "scope": "single_checksum_bound_trace_request",
        "evaluation_basis": _CAUSAL_REEVALUATION_BASIS,
        "historical_activation_recorded": False,
        "requested": request is not None,
        "applied": False,
        "serialized_enabled_before_request": previous_enabled,
        "cached_model_state_restored_after_request": False,
    }

    reason: str | None = None
    if request is None:
        reason = "checksum-bound submitted startup activation evidence was not supplied"
    elif not isinstance(request, Mapping):
        reason = "submitted runtime activation evidence is not a mapping"
    else:
        basis = str(request.get("basis") or "")
        tree_digest = str(request.get("matchup_tree_sha256") or "")
        runtime_digest = str(request.get("runtime_source_tree_sha256") or "")
        base.update(
            {
                "basis": basis or None,
                "matchup_tree_sha256": tree_digest or None,
                "runtime_source_tree_sha256": runtime_digest or None,
                "submitted_startup_behavior": request.get("submitted_startup_behavior"),
            }
        )
        required_flags = (
            "runtime_parity_verified",
            "runtime_identity_verified",
            "matchup_tree_verified",
            "submitted_startup_behavior_verified",
        )
        missing_flags = [
            name for name in required_flags if request.get(name) is not True
        ]
        if basis != _SUBMITTED_RUNTIME_ACTIVATION_BASIS:
            reason = "submitted runtime activation basis is not checksum-bound"
        elif missing_flags:
            reason = (
                "submitted runtime activation evidence is incomplete: "
                + ", ".join(missing_flags)
            )
        elif re.fullmatch(r"sha256:[0-9a-f]{64}", tree_digest) is None:
            reason = "verified matchup tree digest is missing or malformed"
        elif re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_digest) is None:
            reason = "verified submitted runtime source digest is missing or malformed"

    output_evidence = (
        _adapter_output_evidence(adapter_bank)
        if adapter_bank is not None
        else {
            "available": False,
            "nonzero": None,
            "reason": "checkpoint has no matchup adapter bank",
        }
    )
    base["trained_nonzero_bank_evidence"] = output_evidence
    if reason is None and adapter_bank is None:
        reason = "checkpoint has no matchup adapter bank"
    if reason is None and output_evidence.get("available") is not True:
        reason = str(
            output_evidence.get("reason")
            or "trained matchup adapter evidence is unavailable"
        )
    if reason is None and output_evidence.get("nonzero") is not True:
        reason = "checkpoint matchup adapter output projections are all zero"

    # A cached model is shared across requests.  Exact startup evidence enables
    # the bank only for this inference span; every other case forces the safe
    # base-policy path.  The caller holds _MODEL_TRACE_LOCK until restoration.
    if adapter_bank is not None and previous_enabled is not None:
        should_enable = reason is None
        if previous_enabled != should_enable:
            adapter_bank.enabled = should_enable
            restoration.state_changed = True
        if should_enable:
            for parameter in parameters:
                if parameter.requires_grad:
                    parameter.requires_grad_(False)
                    restoration.state_changed = True
            base.update(
                {
                    "status": "applied",
                    "applied": True,
                    "reason": None,
                }
            )
            return base, restoration

    base.update(
        {
            "status": "unavailable",
            "reason": reason
            or "adapter bank does not expose a mutable runtime enabled flag",
        }
    )
    return base, restoration


def _restore_submitted_runtime_activation(
    restoration: _AdapterRuntimeRestoration,
    payload: dict[str, Any],
) -> None:
    adapter_bank = restoration.adapter_bank
    if adapter_bank is not None and restoration.previous_enabled is not None:
        adapter_bank.enabled = restoration.previous_enabled
        parameters = tuple(adapter_bank.parameters())
        for parameter, requires_grad in zip(
            parameters, restoration.previous_requires_grad, strict=True
        ):
            parameter.requires_grad_(requires_grad)
    payload["cached_model_state_restored_after_request"] = True


def _unavailable_payload(
    error: ReplayStepUnavailable,
    *,
    checkpoint_digest: str | None,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": INFERENCE_SCHEMA,
        "availability": _availability(False, error.reason, code=error.code),
        "provenance": {
            **_json_safe(_provenance_mapping(provenance)),
            "checkpoint_digest": checkpoint_digest,
        },
        "unavailable": {"code": error.code, "reason": error.reason, **error.details},
    }


def _json_safe(value: Any) -> Any:
    """Turn replay metadata into deterministic JSON-friendly primitives."""

    if isinstance(value, torch.Tensor):
        return _tensor_to_json(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if is_dataclass(value):
        return _json_safe(asdict(value))
    # Replay JSON should already consist of primitives.  This fallback keeps an
    # inspector response serializable without pretending an opaque object was a
    # model input feature.
    return str(value)


def _tensor_to_json(tensor: torch.Tensor) -> Any:
    """Serialize an activation while representing non-finite values explicitly."""

    values = tensor.detach().to(device="cpu", dtype=torch.float64).tolist()

    def convert(item: Any) -> Any:
        if isinstance(item, list):
            return [convert(value) for value in item]
        value = float(item)
        return value if math.isfinite(value) else None

    return convert(values)


def _state_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < 1 or tensor.size(0) != 1:
        raise ReplayInspectionError("expected one replay-stage state batch")
    return tensor[0]


def _option_tensor(tensor: torch.Tensor, option_count: int) -> torch.Tensor:
    if tensor.dim() < 2 or tensor.size(0) != 1:
        raise ReplayInspectionError("expected one replay-stage option batch")
    return tensor[0, :option_count]


def _legal_vector(tensor: torch.Tensor, option_count: int) -> torch.Tensor:
    """Accept either a batched option tensor or its already-unbatched vector."""

    if tensor.dim() == 1:
        if tensor.size(0) < option_count:
            raise ReplayInspectionError("legal option tensor is narrower than expected")
        return tensor[:option_count]
    return _option_tensor(tensor, option_count)


def _normalization(
    tensor: torch.Tensor,
    *,
    kind: str,
    option_count: int | None = None,
) -> dict[str, Any]:
    """Return a normalized activation only when one is mathematically defined."""

    if kind == "tanh":
        return {
            "kind": "tanh",
            "defined": True,
            "values": _tensor_to_json(torch.tanh(tensor)),
        }
    if kind == "sigmoid":
        return {
            "kind": "sigmoid_independent_bernoulli",
            "defined": True,
            "values": _tensor_to_json(torch.sigmoid(tensor)),
        }
    if kind == "softmax_state":
        return {
            "kind": "softmax_over_head_classes",
            "defined": True,
            "values": _tensor_to_json(torch.softmax(tensor, dim=-1)),
        }
    if kind == "softmax_options":
        if option_count is None:
            raise ReplayInspectionError("option normalization requires option count")
        # ``_head_payload`` has already stripped the single batch axis.  The
        # factor heads are scalar per legal candidate, so candidate axis zero
        # is the exact cross-entropy normalization axis.
        values = tensor
        if values.dim() != 1 or int(values.numel()) != int(option_count):
            raise ReplayInspectionError("option softmax head has an invalid shape")
        return {
            "kind": "softmax_over_legal_candidates",
            "defined": True,
            "values": _tensor_to_json(torch.softmax(values, dim=0)),
        }
    return {
        "kind": "not_globally_defined",
        "defined": False,
        "reason": "this head is a regression or mixed typed output",
        "values": None,
    }


def _head_payload(
    *,
    name: str,
    tensor: torch.Tensor,
    source_kind: str,
    option_count: int,
    normalization: str = "none",
    module_name: str | None = None,
    fusion_input: torch.Tensor | None = None,
    base_output: torch.Tensor | None = None,
    h10_residual: torch.Tensor | None = None,
    h10_reason: str | None = None,
) -> dict[str, Any]:
    if source_kind not in {"state", "option"}:
        raise ReplayInspectionError(f"unknown head source kind: {source_kind}")
    display = (
        _option_tensor(tensor, option_count)
        if source_kind == "option"
        else _state_tensor(tensor)
    )
    record: dict[str, Any] = {
        "availability": _availability(True),
        "name": name,
        "module_name": module_name,
        "source_kind": source_kind,
        "shape": [int(value) for value in display.shape],
        "raw_values": _tensor_to_json(display),
        "normalization": _normalization(
            display,
            kind=normalization,
            option_count=option_count,
        ),
        "mask": {
            "legal_candidate_mask": [True] * option_count
            if source_kind == "option"
            else None,
            "training_target_mask": _availability(
                False, UNAVAILABLE_TARGET_MASK_REASON
            ),
        },
    }
    if fusion_input is not None:
        fusion_display = (
            _option_tensor(fusion_input, option_count)
            if source_kind == "option"
            else _state_tensor(fusion_input)
        )
        record["fusion_input_values"] = _tensor_to_json(fusion_display)
    if base_output is not None:
        base_display = (
            _option_tensor(base_output, option_count)
            if source_kind == "option"
            else _state_tensor(base_output)
        )
        record["base_output_values"] = _tensor_to_json(base_display)
    if h10_residual is not None:
        residual_display = (
            _option_tensor(h10_residual, option_count)
            if source_kind == "option"
            else _state_tensor(h10_residual)
        )
        record["h10_residual"] = {
            "availability": _availability(True),
            "values": _tensor_to_json(residual_display),
        }
    else:
        record["h10_residual"] = {
            "availability": _availability(
                False,
                h10_reason or "this checkpoint has no H10 residual for the head",
            )
        }
    return record


def _head_unavailable(
    name: str, reason: str, *, source_kind: str | None = None
) -> dict[str, Any]:
    return {
        "availability": _availability(False, reason),
        "name": name,
        "source_kind": source_kind,
        "raw_values": None,
        "normalization": {
            "kind": "unavailable",
            "defined": False,
            "reason": reason,
            "values": None,
        },
        "mask": {
            "legal_candidate_mask": None,
            "training_target_mask": _availability(False, reason),
        },
    }


def _component_output(
    model: torch.nn.Module,
    *,
    module_name: str,
    residual_name: str,
    source: torch.Tensor,
    reshape: Callable[[torch.Tensor], torch.Tensor] | None = None,
    squeeze_last: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Evaluate base + optional H10 residual without manufacturing a branch."""

    module = getattr(model, module_name, None)
    if not callable(module):
        raise ReplayInspectionError(
            f"architecture advertised unavailable {module_name}"
        )
    base = module(source)
    residual: torch.Tensor | None = None
    residuals = getattr(model, "h10_head_residuals", None)
    h10_enabled = bool(getattr(model, "h10_capacity_enabled", False))
    if h10_enabled and residuals is not None and residual_name in residuals:
        residual_module = residuals[residual_name]
        residual = residual_module(source)
    combined = base if residual is None else base + residual
    if squeeze_last:
        if base.size(-1) != 1 or (residual is not None and residual.size(-1) != 1):
            raise ReplayInspectionError(
                f"{module_name} is not a scalar-per-option strategic head"
            )
        base = base.squeeze(-1)
        if residual is not None:
            residual = residual.squeeze(-1)
        combined = combined.squeeze(-1)
    if reshape is not None:
        base = reshape(base)
        if residual is not None:
            residual = reshape(residual)
        combined = reshape(combined)
    return base, residual, combined


def _build_head_records(
    *,
    model: torch.nn.Module,
    state: torch.Tensor,
    option_hidden: torch.Tensor,
    option_count: int,
    state_sources: Mapping[str, torch.Tensor] | None,
    option_sources: Mapping[str, torch.Tensor] | None,
) -> dict[str, dict[str, Any]]:
    """Expose every architecture-present learned head and explicit absences."""

    records: dict[str, dict[str, Any]] = {}
    value_head = getattr(model, "value_head", None)
    if callable(value_head):
        raw_value = value_head(state)
        fusion_value = None if state_sources is None else state_sources.get("value")
        records["value"] = _head_payload(
            name="value",
            tensor=raw_value,
            source_kind="state",
            option_count=option_count,
            normalization="tanh",
            module_name="value_head",
            fusion_input=fusion_value,
            h10_reason="value has no H10 strategic residual branch",
        )
    else:
        records["value"] = _head_unavailable(
            "value", "checkpoint has no value_head", source_kind="state"
        )

    belief_method = getattr(model, "belief_aux_logits", None)
    belief: Mapping[str, torch.Tensor] = {}
    if callable(belief_method):
        maybe_belief = belief_method(state)
        if isinstance(maybe_belief, Mapping):
            belief = maybe_belief
    belief_specs = (
        ("archetype", "aux_logits", "softmax_state"),
        ("opponent_hand", "opp_hand_logits", "sigmoid"),
        ("opponent_remainder", "opp_remainder_logits", "sigmoid"),
        ("lethal_threat", "lethal_threat_logits", "sigmoid"),
        ("prize_race", "prize_race_pred", "none"),
    )
    for head_name, key, normalizer in belief_specs:
        value = belief.get(key)
        if not isinstance(value, torch.Tensor):
            records[head_name] = _head_unavailable(
                head_name,
                f"checkpoint does not expose belief output {key}",
                source_kind="state",
            )
            continue
        records[head_name] = _head_payload(
            name=head_name,
            tensor=value,
            source_kind="state",
            option_count=option_count,
            normalization=normalizer,
            module_name={
                "archetype": "aux_head",
                "opponent_hand": "opp_hand_head",
                "opponent_remainder": "opp_remainder_head",
                "lethal_threat": "lethal_threat_head",
                "prize_race": "prize_race_head",
            }[head_name],
            fusion_input=None
            if state_sources is None
            else state_sources.get(head_name),
            h10_reason="H10 residuals are defined only for strategic expanded heads",
        )

    expanded_enabled = bool(getattr(model, "expanded_heads_enabled", False))
    expanded_reason = "checkpoint predates expanded strategic heads"
    option_specs = (
        ("action_q", "action_q_head", "none"),
        ("action_type", "action_type_head", "softmax_options"),
        ("action_target", "action_target_head", "softmax_options"),
        ("action_resource", "action_resource_head", "softmax_options"),
        ("action_utility", "action_utility_head", "none"),
    )
    state_specs = (
        ("tactical_outcomes", "tactical_outcome_head", "tactical_outcome", "none"),
        ("opponent_response", "opponent_response_head", "opponent_response", "sigmoid"),
        ("resource_forecast", "resource_forecast_head", "resource_forecast", "none"),
        ("game_phase", "game_phase_head", "game_phase", "softmax_state"),
        (
            "outcome_distribution",
            "outcome_distribution_head",
            "outcome_distribution",
            "softmax_state",
        ),
        ("remaining_turns", "remaining_turns_head", "remaining_turns", "none"),
    )
    if not expanded_enabled:
        for name, _module, _normalizer in option_specs:
            records[name] = _head_unavailable(
                name, expanded_reason, source_kind="option"
            )
        for name, _module, _residual, _normalizer in state_specs:
            records[name] = _head_unavailable(
                name, expanded_reason, source_kind="state"
            )
        records["setup_board_outcome"] = _head_unavailable(
            "setup_board_outcome", expanded_reason, source_kind="option"
        )
        records["combo_state"] = _head_unavailable(
            "combo_state", expanded_reason, source_kind="option"
        )
        return records

    for name, module_name, normalizer in option_specs:
        try:
            base, residual, combined = _component_output(
                model,
                module_name=module_name,
                residual_name=name,
                source=option_hidden,
                squeeze_last=name != "action_utility",
            )
            records[name] = _head_payload(
                name=name,
                tensor=combined,
                source_kind="option",
                option_count=option_count,
                normalization=normalizer,
                module_name=module_name,
                fusion_input=None
                if option_sources is None
                else option_sources.get(name),
                base_output=base,
                h10_residual=residual,
            )
        except ReplayInspectionError as exc:
            records[name] = _head_unavailable(name, str(exc), source_kind="option")

    for name, module_name, residual_name, normalizer in state_specs:
        reshape = (
            (lambda value: value.reshape(*value.shape[:-1], 3, 6))
            if name == "tactical_outcomes"
            else None
        )
        try:
            base, residual, combined = _component_output(
                model,
                module_name=module_name,
                residual_name=residual_name,
                source=state,
                reshape=reshape,
            )
            records[name] = _head_payload(
                name=name,
                tensor=combined,
                source_kind="state",
                option_count=option_count,
                normalization=normalizer,
                module_name=module_name,
                fusion_input=None if state_sources is None else state_sources.get(name),
                base_output=base,
                h10_residual=residual,
            )
        except ReplayInspectionError as exc:
            records[name] = _head_unavailable(name, str(exc), source_kind="state")

    optional_specs = (
        ("setup_board_outcome", "setup_board_outcome_head", "setup_board_outcome"),
        ("combo_state", "combo_state_head", "combo_state"),
    )
    for name, module_name, residual_name in optional_specs:
        enabled = bool(getattr(model, f"{name}_head_enabled", False))
        if not enabled:
            records[name] = _head_unavailable(
                name,
                f"checkpoint architecture does not enable {name}_head",
                source_kind="option",
            )
            continue
        try:
            base, residual, combined = _component_output(
                model,
                module_name=module_name,
                residual_name=residual_name,
                source=option_hidden,
            )
            records[name] = _head_payload(
                name=name,
                tensor=combined,
                source_kind="option",
                option_count=option_count,
                normalization="none",
                module_name=module_name,
                fusion_input=None
                if option_sources is None
                else option_sources.get(name),
                base_output=base,
                h10_residual=residual,
            )
        except ReplayInspectionError as exc:
            records[name] = _head_unavailable(name, str(exc), source_kind="option")
    return records


def _margin(logits: torch.Tensor, target: int, option_count: int) -> float:
    values = (
        logits[:option_count]
        if logits.dim() == 1 and logits.numel() >= option_count
        else _option_tensor(logits, option_count)
    )
    if option_count <= 1:
        return 0.0
    others = torch.cat((values[:target], values[target + 1 :]))
    return float((values[target] - torch.max(others)).item())


def _reliability_payload(fusion: Any, name: str) -> tuple[dict[str, Any], float]:
    """Return a V3 learned multiplier or explicit fixed/no-parameter metadata."""

    typed_centered = bool(getattr(fusion, "typed_output_centered_routes", False))
    if not typed_centered:
        return (
            {
                "availability": _availability(
                    False,
                    "this Fusion schema has no learned route-reliability parameter",
                ),
                "kind": "fixed_unity",
                "raw_log_value": None,
                "effective_multiplier": 1.0,
            },
            1.0,
        )
    params = getattr(fusion, "dedicated_route_log_reliability", {})
    if name not in params:
        return (
            {
                "availability": _availability(
                    False,
                    "route has no learned reliability parameter",
                ),
                "kind": "unavailable",
                "raw_log_value": None,
                "effective_multiplier": None,
            },
            1.0,
        )
    raw_log = float(params[name].detach().cpu().item())
    min_multiplier = float(DECISION_FUSION_V3_MIN_RELIABILITY)
    max_multiplier = float(DECISION_FUSION_V3_MAX_RELIABILITY)
    clamped_log = min(max(raw_log, math.log(min_multiplier)), math.log(max_multiplier))
    before_action_type_cap = math.exp(clamped_log)
    effective = before_action_type_cap
    action_cap: float | None = None
    if name == "action_type":
        action_cap = float(
            getattr(fusion, "action_type_reliability_cap", max_multiplier)
        )
        effective = min(effective, action_cap)
    return (
        {
            "availability": _availability(True),
            "kind": "learned_positive_bounded",
            "raw_log_value": raw_log,
            "clamped_log_value": clamped_log,
            "bounds": [min_multiplier, max_multiplier],
            "before_action_type_cap": before_action_type_cap,
            "action_type_cap": action_cap,
            "effective_multiplier": effective,
        },
        effective,
    )


def _route_records(
    *,
    fusion: Any,
    option_hidden: torch.Tensor,
    state_sources: Mapping[str, torch.Tensor],
    option_sources: Mapping[str, torch.Tensor],
    option_count: int,
    runtime_active: bool,
) -> dict[str, Any]:
    if not bool(getattr(fusion, "dedicated_routes_enabled", False)):
        return {
            "availability": _availability(
                False, "checkpoint has no dedicated per-head Fusion routes"
            ),
            "routes": {},
        }
    routes = getattr(fusion, "dedicated_routes", {})
    records: dict[str, Any] = {}
    for name in tuple(getattr(fusion, "required_heads", ())):
        if name not in routes:
            records[name] = {
                "availability": _availability(False, "dedicated route is absent"),
            }
            continue
        if name in option_sources:
            source = option_sources[name]
            option_conditioned = True
        elif name in state_sources:
            source = state_sources[name]
            option_conditioned = False
        else:
            records[name] = {
                "availability": _availability(False, "Fusion source is absent"),
            }
            continue
        typed = fusion._option_conditioned_source(
            source,
            batch_size=option_hidden.size(0),
            option_count=option_hidden.size(1),
            already_option_conditioned=option_conditioned,
        )
        raw_delta = routes[name](option_hidden, typed)
        reliability, multiplier = _reliability_payload(fusion, name)
        weighted_delta = raw_delta * multiplier
        record: dict[str, Any] = {
            "availability": _availability(True),
            "source_kind": "option" if option_conditioned else "state",
            "typed_source_values": _tensor_to_json(_option_tensor(typed, option_count)),
            "raw_route_delta": _tensor_to_json(_option_tensor(raw_delta, option_count)),
            "reliability": reliability,
            "reliability_weighted_delta": _tensor_to_json(
                _option_tensor(weighted_delta, option_count)
            ),
            "runtime_contribution": (
                _tensor_to_json(_option_tensor(weighted_delta, option_count))
                if runtime_active
                else None
            ),
            "runtime_active": bool(runtime_active),
        }
        if not runtime_active:
            record["runtime_contribution_reason"] = (
                "dedicated routes are architecture-present but runtime-disabled"
            )
        records[name] = record
    return {"availability": _availability(True), "routes": records}


def _ablate(
    name: str,
    *,
    state_sources: Mapping[str, torch.Tensor],
    option_sources: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    changed_state = dict(state_sources)
    changed_option = dict(option_sources)
    if name in changed_state:
        changed_state[name] = torch.zeros_like(changed_state[name])
    elif name in changed_option:
        changed_option[name] = torch.zeros_like(changed_option[name])
    else:
        raise ReplayInspectionError(f"Fusion has no source named {name}")
    return changed_state, changed_option


def _loo_effect(
    *,
    full: torch.Tensor,
    ablated: torch.Tensor,
    target: int,
    option_count: int,
) -> dict[str, Any]:
    """Measure a causal full-policy minus no-head counterfactual.

    Raw head activations are predictions, not action influence.  The policy
    fields here instead come from two complete final-logit passes: the actual
    head-source set and the same set with exactly one source zeroed.  Keeping
    both probability vectors makes every displayed summary reconstructable.
    """

    full_values = _legal_vector(full, option_count)
    ablated_values = _legal_vector(ablated, option_count)
    effect = full_values - ablated_values
    choice = int(torch.argmax(full_values).item())
    ablated_choice = int(torch.argmax(ablated_values).item())
    result: dict[str, Any] = {
        "effect_logits": _tensor_to_json(effect),
        "mean_absolute_logit_effect": float(effect.abs().mean().item()),
        "max_absolute_logit_effect": float(effect.abs().max().item()),
        "full_policy_logits": _tensor_to_json(full_values),
        "policy_without_head_logits": _tensor_to_json(ablated_values),
        "probability_normalization": "softmax_over_legal_candidates",
        "recorded_target_margin_effect": _margin(full, target, option_count)
        - _margin(ablated, target, option_count),
        "changes_model_choice": choice != ablated_choice,
        "ablated_model_choice_index": ablated_choice,
    }
    if not bool(
        torch.isfinite(full_values).all() and torch.isfinite(ablated_values).all()
    ):
        reason = (
            "exact leave-one-head-out final logits contain a non-finite legal "
            "option, so probability influence is unavailable"
        )
        result.update(
            {
                "full_policy_probabilities": None,
                "policy_without_head_probabilities": None,
                "effect_probabilities": None,
                "policy_influence": {
                    "availability": _availability(False, reason),
                    "method": "exact_leave_one_head_out_final_policy_recomputation",
                    "sign_convention": "full_policy_minus_policy_without_head",
                },
            }
        )
        return result

    full_probabilities = torch.softmax(full_values, dim=-1)
    without_head_probabilities = torch.softmax(ablated_values, dim=-1)
    probability_effect = full_probabilities - without_head_probabilities
    helped_index = int(torch.argmax(probability_effect).item())
    hurt_index = int(torch.argmin(probability_effect).item())
    result.update(
        {
            "full_policy_probabilities": _tensor_to_json(full_probabilities),
            "policy_without_head_probabilities": _tensor_to_json(
                without_head_probabilities
            ),
            "effect_probabilities": _tensor_to_json(probability_effect),
            "policy_influence": {
                "availability": _availability(True),
                "method": "exact_leave_one_head_out_final_policy_recomputation",
                "sign_convention": "full_policy_minus_policy_without_head",
                "selected_option_index": choice,
                "selected_option_probability_delta": float(
                    probability_effect[choice].item()
                ),
                "selected_option_logit_delta": float(effect[choice].item()),
                "maximum_absolute_option_probability_delta": float(
                    probability_effect.abs().max().item()
                ),
                # The total-variation distance is a bounded whole-policy
                # summary.  Unlike an activation norm, it is computed from
                # the two actual final conditional policies.
                "total_variation_distance": float(
                    0.5 * probability_effect.abs().sum().item()
                ),
                "most_helped_option": {
                    "index": helped_index,
                    "probability_delta": float(probability_effect[helped_index].item()),
                },
                "most_hurt_option": {
                    "index": hurt_index,
                    "probability_delta": float(probability_effect[hurt_index].item()),
                },
            },
        }
    )
    return result


def _head_availability_reason(record: Mapping[str, Any], fallback: str) -> str:
    availability = record.get("availability")
    if isinstance(availability, Mapping):
        reason = availability.get("reason")
        if reason not in (None, ""):
            return str(reason)
    return fallback


def _attach_head_policy_influence(
    heads: Mapping[str, dict[str, Any]], fusion_payload: Mapping[str, Any]
) -> None:
    """Attach the actual final-policy LOO metrics to every head record.

    An unavailable route or a runtime-disabled Fusion path is not reported as
    a zero effect: it is an unavailable causal attribution with its reason.
    """

    all_ablations = fusion_payload.get("leave_one_out")
    all_ablations = all_ablations if isinstance(all_ablations, Mapping) else {}
    fusion_availability = fusion_payload.get("availability")
    fusion_reason = _head_availability_reason(
        fusion_payload,
        "checkpoint has no causal decision-fusion attribution path",
    )
    for name, head in heads.items():
        head_availability = head.get("availability")
        if (
            isinstance(head_availability, Mapping)
            and head_availability.get("available") is False
        ):
            head["policy_influence"] = {
                "availability": _availability(
                    False,
                    _head_availability_reason(head, "head is unavailable"),
                )
            }
            continue
        if (
            isinstance(fusion_availability, Mapping)
            and fusion_availability.get("available") is False
        ):
            head["policy_influence"] = {
                "availability": _availability(False, fusion_reason)
            }
            continue
        ablation = all_ablations.get(name)
        if not isinstance(ablation, Mapping):
            head["policy_influence"] = {
                "availability": _availability(
                    False,
                    "checkpoint has no exact leave-one-head-out source for this head",
                )
            }
            continue
        runtime_path = ablation.get("runtime_path")
        if not isinstance(runtime_path, Mapping):
            head["policy_influence"] = {
                "availability": _availability(
                    False,
                    "runtime leave-one-head-out final-policy path is unavailable",
                )
            }
            continue
        runtime_availability = runtime_path.get("availability")
        if (
            isinstance(runtime_availability, Mapping)
            and runtime_availability.get("available") is False
        ):
            head["policy_influence"] = {
                "availability": _availability(
                    False,
                    _head_availability_reason(
                        runtime_path,
                        "runtime leave-one-head-out final-policy path is unavailable",
                    ),
                )
            }
            continue
        influence = runtime_path.get("policy_influence")
        if isinstance(influence, Mapping):
            head["policy_influence"] = dict(influence)
        else:
            head["policy_influence"] = {
                "availability": _availability(
                    False,
                    "runtime leave-one-head-out result has no final-policy probability metrics",
                )
            }


def _post_fusion_policy_logits(
    *,
    model: torch.nn.Module,
    option_hidden: torch.Tensor,
    state: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Apply the model's last policy transform after decision fusion.

    H10/Fusion checkpoints normally return ``logits`` unchanged here.  A
    checkpoint with action-authoritative latent lookahead, however, adds its
    causal ``policy_aid`` *after* Fusion.  Leave-one-head-out effects need the
    same final path as the policy a player would have received; otherwise the
    delta itself is right but a reported margin or choice flip can be wrong.
    """

    transform = getattr(model, "latent_aided_policy_logits", None)
    if not callable(transform):
        return logits
    final = transform(option_hidden, state, logits)
    if not isinstance(final, torch.Tensor) or final.shape != logits.shape:
        raise ReplayInspectionError(
            "model latent policy transform returned an incompatible logit tensor"
        )
    return final


def _fusion_payload(
    *,
    model: torch.nn.Module,
    option_hidden: torch.Tensor,
    state: torch.Tensor,
    base_logits: torch.Tensor,
    state_sources: Mapping[str, torch.Tensor] | None,
    option_sources: Mapping[str, torch.Tensor] | None,
    target: int,
    option_count: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Evaluate actual and counterfactual Fusion paths plus exact ablations."""

    fusion = getattr(model, "decision_fusion", None)
    if fusion is None or state_sources is None or option_sources is None:
        return (
            {
                "availability": _availability(
                    False, "checkpoint has no causal decision-fusion architecture"
                ),
                "routes": {
                    "availability": _availability(False, "Fusion unavailable"),
                    "routes": {},
                },
                "leave_one_out": {},
            },
            base_logits,
        )
    runtime_enabled = bool(getattr(model, "decision_fusion_runtime_enabled", False))
    dedicated_enabled = bool(getattr(fusion, "dedicated_routes_enabled", False))
    dedicated_runtime = bool(
        getattr(model, "decision_fusion_dedicated_routes_runtime_enabled", False)
    )
    actual_dedicated_active = dedicated_enabled and dedicated_runtime
    counterfactual_dedicated_active = dedicated_enabled
    legacy_logits = fusion(
        option_hidden,
        base_logits,
        state_sources=dict(state_sources),
        option_sources=dict(option_sources),
        dedicated_routes_active=False,
    )
    counterfactual_logits = fusion(
        option_hidden,
        base_logits,
        state_sources=dict(state_sources),
        option_sources=dict(option_sources),
        dedicated_routes_active=counterfactual_dedicated_active,
    )
    actual_fusion_logits = (
        fusion(
            option_hidden,
            base_logits,
            state_sources=dict(state_sources),
            option_sources=dict(option_sources),
            dedicated_routes_active=actual_dedicated_active,
        )
        if runtime_enabled
        else base_logits
    )
    counterfactual_final_logits = _post_fusion_policy_logits(
        model=model,
        option_hidden=option_hidden,
        state=state,
        logits=counterfactual_logits,
    )
    actual_final_logits = _post_fusion_policy_logits(
        model=model,
        option_hidden=option_hidden,
        state=state,
        logits=actual_fusion_logits,
    )
    route_data = _route_records(
        fusion=fusion,
        option_hidden=option_hidden,
        state_sources=state_sources,
        option_sources=option_sources,
        option_count=option_count,
        runtime_active=bool(runtime_enabled and actual_dedicated_active),
    )
    loo: dict[str, Any] = {}
    for name in tuple(getattr(fusion, "required_heads", ())):
        try:
            changed_state, changed_option = _ablate(
                name, state_sources=state_sources, option_sources=option_sources
            )
            counterfactual_ablated = fusion(
                option_hidden,
                base_logits,
                state_sources=changed_state,
                option_sources=changed_option,
                dedicated_routes_active=counterfactual_dedicated_active,
            )
            counterfactual_ablated_final = _post_fusion_policy_logits(
                model=model,
                option_hidden=option_hidden,
                state=state,
                logits=counterfactual_ablated,
            )
            record: dict[str, Any] = {
                "availability": _availability(True),
                "counterfactual_all_routes": _loo_effect(
                    full=counterfactual_final_logits,
                    ablated=counterfactual_ablated_final,
                    target=target,
                    option_count=option_count,
                ),
                "counterfactual_all_routes_before_latent": _loo_effect(
                    full=counterfactual_logits,
                    ablated=counterfactual_ablated,
                    target=target,
                    option_count=option_count,
                ),
            }
            if runtime_enabled:
                actual_ablated = fusion(
                    option_hidden,
                    base_logits,
                    state_sources=changed_state,
                    option_sources=changed_option,
                    dedicated_routes_active=actual_dedicated_active,
                )
                actual_ablated_final = _post_fusion_policy_logits(
                    model=model,
                    option_hidden=option_hidden,
                    state=state,
                    logits=actual_ablated,
                )
                record["runtime_path"] = _loo_effect(
                    full=actual_final_logits,
                    ablated=actual_ablated_final,
                    target=target,
                    option_count=option_count,
                )
                record["runtime_path_before_latent"] = _loo_effect(
                    full=actual_fusion_logits,
                    ablated=actual_ablated,
                    target=target,
                    option_count=option_count,
                )
            else:
                record["runtime_path"] = {
                    "availability": _availability(
                        False,
                        "decision fusion is runtime-disabled; flat policy is the actual path",
                    )
                }
            loo[name] = record
        except ReplayInspectionError as exc:
            loo[name] = {"availability": _availability(False, str(exc))}
    return (
        {
            "availability": _availability(True),
            "schema": _json_safe(
                getattr(model, "decision_fusion_inventory", dict)().get("schema")
            )
            if callable(getattr(model, "decision_fusion_inventory", None))
            else None,
            "runtime_enabled": runtime_enabled,
            "dedicated_routes_runtime_enabled": dedicated_runtime,
            "base_logits": _tensor_to_json(_option_tensor(base_logits, option_count)),
            "legacy_residual": _tensor_to_json(
                _option_tensor(legacy_logits - base_logits, option_count)
            ),
            "legacy_fusion_logits": _tensor_to_json(
                _option_tensor(legacy_logits, option_count)
            ),
            "counterfactual_all_routes_logits": _tensor_to_json(
                _option_tensor(counterfactual_logits, option_count)
            ),
            "counterfactual_all_routes_final_logits": _tensor_to_json(
                _option_tensor(counterfactual_final_logits, option_count)
            ),
            "actual_fusion_logits": _tensor_to_json(
                _option_tensor(actual_fusion_logits, option_count)
            ),
            "actual_final_logits": _tensor_to_json(
                _option_tensor(actual_final_logits, option_count)
            ),
            "routes": route_data,
            "leave_one_out": loo,
        },
        actual_fusion_logits,
    )


def _decision_influence_payload(
    *,
    model: torch.nn.Module,
    option_hidden: torch.Tensor,
    state: torch.Tensor,
    base_logits: torch.Tensor,
    baseline_final_logits: torch.Tensor,
    state_sources: Mapping[str, torch.Tensor] | None,
    option_sources: Mapping[str, torch.Tensor] | None,
    requested_scales: Mapping[str, float] | None,
    target: int,
    option_count: int,
    fusion_payload: Mapping[str, Any],
    final_logit_bonus: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Recompute the exact decision path with bounded head-source scales.

    This is a decision-only hypothetical.  A scale multiplies the named typed
    Fusion source before the learned nonlinear legacy/dedicated Fusion modules
    run; it does not scale a displayed LOO delta, mutate parameters, or change
    any training-loss coefficient.
    """

    fusion = getattr(model, "decision_fusion", None)
    policy_bonus = (
        torch.zeros_like(_legal_vector(baseline_final_logits, option_count))
        if final_logit_bonus is None
        else _legal_vector(final_logit_bonus, option_count)
    )
    baseline = _legal_vector(baseline_final_logits, option_count) + policy_bonus
    base: dict[str, Any] = {
        "schema": "poke_bot.replay_model_inspector.decision_influence/v1",
        "mode": "decision_only_counterfactual",
        "method": "exact_nonlinear_fusion_source_recomputation",
        "training_weight": False,
        "checkpoint_mutated": False,
        "historical": False,
        "reproduction_status": "recomputed_not_historical",
        "default_scale": 1.0,
        "scale_bounds": [0.0, 2.0],
        "requested_scales": dict(requested_scales or {}),
    }
    if fusion is None or state_sources is None or option_sources is None:
        return {
            **base,
            "availability": _availability(
                False, "checkpoint has no causal decision-fusion architecture"
            ),
            "eligible_heads": [],
        }
    if not bool(getattr(model, "decision_fusion_runtime_enabled", False)):
        return {
            **base,
            "availability": _availability(
                False,
                "decision fusion is runtime-disabled; head scaling cannot change the actual policy path",
            ),
            "eligible_heads": [],
        }

    # Newer Fusion implementations distinguish physically present routes from
    # the active serving inventory.  The exact submitted runtime predates that
    # property and exposes only ``required_heads``; absence of the newer name
    # must not collapse a valid runtime inventory to an empty tuple.
    raw_head_inventory = getattr(fusion, "active_required_heads", None)
    head_inventory_source = "active_required_heads"
    if raw_head_inventory is None:
        raw_head_inventory = getattr(fusion, "required_heads", ())
        head_inventory_source = "required_heads"
    eligible_heads = tuple(
        name
        for name in raw_head_inventory
        if name in state_sources or name in option_sources
    )
    base["head_inventory_source"] = head_inventory_source
    routes_wrapper = fusion_payload.get("routes")
    route_records = (
        routes_wrapper.get("routes") if isinstance(routes_wrapper, Mapping) else None
    )
    participating_heads = tuple(
        name
        for name in eligible_heads
        if isinstance(route_records, Mapping)
        and isinstance(route_records.get(name), Mapping)
        and bool(route_records[name].get("runtime_active"))
    )
    route_count = len(participating_heads)
    total_delta_cap = float(getattr(fusion, "dedicated_route_total_delta_cap", 1.0))
    baseline_head_weights: dict[str, dict[str, Any]] = {}
    for name in eligible_heads:
        route = route_records.get(name) if isinstance(route_records, Mapping) else None
        reliability = route.get("reliability") if isinstance(route, Mapping) else None
        multiplier = (
            reliability.get("effective_multiplier")
            if isinstance(reliability, Mapping)
            else None
        )
        coefficient = (
            total_delta_cap * float(multiplier) / route_count
            if isinstance(multiplier, (int, float))
            and not isinstance(multiplier, bool)
            and math.isfinite(float(multiplier))
            and route_count > 0
            and name in participating_heads
            else None
        )
        baseline_head_weights[name] = {
            "source_scale": 1.0,
            "learned_route_multiplier": multiplier,
            "nominal_policy_coefficient": coefficient,
            "shared_active_route_count": route_count,
            "shared_total_delta_cap": total_delta_cap,
            "aggregation": (
                "cap_times_tanh_of_mean_reliability_weighted_route_signals"
                if bool(getattr(fusion, "typed_output_centered_routes", False))
                else "cap_times_mean_route_signals"
            ),
            "is_final_policy_contribution": False,
            "explanation": (
                "This is the head's baseline scalar before its learned route "
                "signal joins the nonlinear shared fusion. The exact policy "
                "effect is decision-specific and is measured by recomputation."
            ),
        }
    base["baseline_head_weights"] = baseline_head_weights
    invalid: list[str] = []
    clean_requested: dict[str, float] = {}
    for raw_name, raw_scale in dict(requested_scales or {}).items():
        name = str(raw_name)
        if name not in eligible_heads:
            invalid.append(name)
            continue
        if (
            isinstance(raw_scale, bool)
            or not isinstance(raw_scale, (int, float))
            or not math.isfinite(float(raw_scale))
            or not 0.0 <= float(raw_scale) <= 2.0
        ):
            invalid.append(name)
            continue
        clean_requested[name] = float(raw_scale)
    if invalid:
        return {
            **base,
            "availability": _availability(
                False,
                "one or more requested head scales are unavailable or outside 0x-2x",
            ),
            "eligible_heads": list(eligible_heads),
            "invalid_heads": sorted(set(invalid)),
        }

    effective_scales = {name: clean_requested.get(name, 1.0) for name in eligible_heads}
    changed_state = {
        name: value * effective_scales.get(name, 1.0)
        for name, value in state_sources.items()
    }
    changed_option = {
        name: value * effective_scales.get(name, 1.0)
        for name, value in option_sources.items()
    }
    dedicated_active = bool(
        getattr(fusion, "dedicated_routes_enabled", False)
        and getattr(model, "decision_fusion_dedicated_routes_runtime_enabled", False)
    )
    changed_fusion_logits = fusion(
        option_hidden,
        base_logits,
        state_sources=changed_state,
        option_sources=changed_option,
        dedicated_routes_active=dedicated_active,
    )
    changed_final = (
        _option_tensor(
            _post_fusion_policy_logits(
                model=model,
                option_hidden=option_hidden,
                state=state,
                logits=changed_fusion_logits,
            ),
            option_count,
        )
        + policy_bonus
    )
    if not bool(torch.isfinite(baseline).all() and torch.isfinite(changed_final).all()):
        return {
            **base,
            "availability": _availability(
                False, "baseline or counterfactual final logits are non-finite"
            ),
            "eligible_heads": list(eligible_heads),
            "effective_scales": effective_scales,
        }

    baseline_probabilities = torch.softmax(baseline, dim=-1)
    changed_probabilities = torch.softmax(changed_final, dim=-1)
    logit_delta = changed_final - baseline
    probability_delta = changed_probabilities - baseline_probabilities
    baseline_choice = int(torch.argmax(baseline).item())
    changed_choice = int(torch.argmax(changed_final).item())
    helped = int(torch.argmax(probability_delta).item())
    hurt = int(torch.argmin(probability_delta).item())
    all_unit = all(scale == 1.0 for scale in effective_scales.values())
    parity: dict[str, Any] = {
        "all_scales_one": all_unit,
        "one_x_matches_runtime_baseline": (
            bool(torch.equal(changed_final, baseline)) if all_unit else None
        ),
        "zero_x_matches_exact_leave_one_out": None,
    }
    zero_overrides = [name for name, scale in clean_requested.items() if scale == 0.0]
    non_unit_overrides = [
        name for name, scale in clean_requested.items() if scale != 1.0
    ]
    if len(zero_overrides) == 1 and non_unit_overrides == zero_overrides:
        head = zero_overrides[0]
        leave_one_out = fusion_payload.get("leave_one_out")
        row = leave_one_out.get(head) if isinstance(leave_one_out, Mapping) else None
        runtime_path = row.get("runtime_path") if isinstance(row, Mapping) else None
        expected = (
            runtime_path.get("policy_without_head_logits")
            if isinstance(runtime_path, Mapping)
            else None
        )
        parity["zero_x_matches_exact_leave_one_out"] = {
            "head": head,
            "matches": expected == _tensor_to_json(changed_final),
        }

    return {
        **base,
        "availability": _availability(True),
        "eligible_heads": list(eligible_heads),
        "effective_scales": effective_scales,
        "baseline": {
            "final_logits": _tensor_to_json(baseline),
            "probabilities": _tensor_to_json(baseline_probabilities),
            "selected_option_index": baseline_choice,
            "recorded_target_margin": _margin(baseline, target, option_count),
        },
        "counterfactual": {
            "final_logits": _tensor_to_json(changed_final),
            "probabilities": _tensor_to_json(changed_probabilities),
            "selected_option_index": changed_choice,
            "recorded_target_margin": _margin(changed_final, target, option_count),
        },
        "effect": {
            "sign_convention": "counterfactual_minus_baseline",
            "logit_delta": _tensor_to_json(logit_delta),
            "probability_delta": _tensor_to_json(probability_delta),
            "selected_action_changed": baseline_choice != changed_choice,
            "maximum_absolute_probability_shift": float(
                probability_delta.abs().max().item()
            ),
            "total_variation_distance": float(
                0.5 * probability_delta.abs().sum().item()
            ),
            "most_helped_option": {
                "index": helped,
                "probability_delta": float(probability_delta[helped].item()),
            },
            "most_hurt_option": {
                "index": hurt,
                "probability_delta": float(probability_delta[hurt].item()),
            },
        },
        "parity": parity,
    }


def _latent_payload(
    *,
    model: torch.nn.Module,
    option_hidden: torch.Tensor,
    state: torch.Tensor,
    option_count: int,
) -> dict[str, Any]:
    if not bool(getattr(model, "latent_lookahead_enabled", False)):
        return {
            "availability": _availability(
                False, "checkpoint has no action-conditioned latent-lookahead module"
            )
        }
    output_fn = getattr(model, "latent_lookahead_outputs", None)
    if not callable(output_fn):
        return {
            "availability": _availability(
                False, "checkpoint does not expose latent-lookahead outputs"
            )
        }
    outputs = output_fn(option_hidden, state)
    if not isinstance(outputs, Mapping):
        raise ReplayInspectionError("latent-lookahead output has an invalid type")
    result = {
        "availability": _availability(True),
        "action_authority_enabled": bool(
            getattr(model, "latent_lookahead_action_authority_enabled", False)
        ),
    }
    for name in ("predicted_next_state_latent", "continuation_value", "policy_aid"):
        value = outputs.get(name)
        if not isinstance(value, torch.Tensor):
            result[name] = {"availability": _availability(False, "output is absent")}
            continue
        result[name] = {
            "availability": _availability(True),
            "values": _tensor_to_json(_option_tensor(value, option_count)),
        }
    return result


def _model_config(model: torch.nn.Module) -> Any:
    cfg = getattr(model, "cfg", None)
    return _json_safe(cfg) if cfg is not None else None


def _model_device(model: torch.nn.Module) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _guide_shadow_payload(
    *,
    observation: Mapping[str, Any],
    candidates: Sequence[Sequence[int]],
    deck: Sequence[int],
    recorded_index: int,
    model_index: int,
) -> dict[str, Any]:
    """Evaluate an exact-runtime deck guide without granting policy authority.

    Current-deck guides are sparse, deterministic training teachers rather
    than neural policy heads.  The submitted runtime source tree is already
    checksum-attested by the caller, so this diagnostic may ask every guide
    implementation in that exact tree whether it recognizes the submitted
    deck.  Zero or multiple matches fail closed instead of guessing a guide.
    """

    registry = getattr(deck_guides, "_GUIDES", None)
    if not isinstance(registry, Mapping):
        return {
            "availability": _availability(
                False, "exact submitted runtime exposes no deck-guide registry"
            ),
            "policy_authority": False,
            "policy_logit_delta": 0.0,
        }
    matches: list[tuple[str, Any, list[float]]] = []
    failures: list[str] = []
    for raw_id, module in registry.items():
        guide_id = str(raw_id).strip()
        scorer = getattr(module, "guide_scores", None)
        if not guide_id or not callable(scorer):
            continue
        try:
            raw_scores = scorer(
                dict(observation),
                [list(candidate) for candidate in candidates],
                deck=list(deck),
                force_enabled=True,
            )
        except TypeError:
            # Very old guide implementations may not expose the shadow-only
            # force flag.  Do not mutate process environment to enable them.
            failures.append(f"{guide_id}: shadow opt-in is unsupported")
            continue
        except Exception as exc:  # noqa: BLE001 - diagnostic remains optional
            failures.append(f"{guide_id}: {exc}")
            continue
        if raw_scores is None:
            continue
        if (
            not isinstance(raw_scores, Sequence)
            or isinstance(raw_scores, (str, bytes))
            or len(raw_scores) != len(candidates)
        ):
            failures.append(f"{guide_id}: score width does not match legal options")
            continue
        scores: list[float] = []
        valid = True
        for raw_score in raw_scores:
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                valid = False
                break
            score = float(raw_score)
            if not math.isfinite(score):
                valid = False
                break
            scores.append(score)
        if not valid:
            failures.append(f"{guide_id}: scores are not finite numbers")
            continue
        matches.append((guide_id, module, scores))
    if not matches:
        return {
            "availability": _availability(
                False,
                "no exact-runtime current-deck guide produced a safe ranking for this decision",
            ),
            "policy_authority": False,
            "policy_logit_delta": 0.0,
            "diagnostic_failures": failures,
        }
    if len(matches) != 1:
        return {
            "availability": _availability(
                False,
                "multiple exact-runtime current-deck guides matched; guide identity is ambiguous",
            ),
            "policy_authority": False,
            "policy_logit_delta": 0.0,
            "matched_guide_ids": [guide_id for guide_id, _module, _scores in matches],
        }
    guide_id, module, scores = matches[0]
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    recommended_index = int(order[0])
    margin = scores[order[0]] - scores[order[1]] if len(order) > 1 else None
    version = getattr(module, "GUIDE_VERSION", None)
    return {
        "schema": "poke_bot.replay_model_inspector.guide_shadow/v1",
        "availability": _availability(True),
        "mode": "exact_runtime_training_guide_shadow",
        "guide_id": guide_id,
        "guide_version": str(version) if version is not None else None,
        "scores": scores,
        "recommended_index": recommended_index,
        "recommended_candidate": list(candidates[recommended_index]),
        "score_margin": margin,
        "agrees_with_recorded_action": recommended_index == int(recorded_index),
        "agrees_with_model_action": recommended_index == int(model_index),
        "policy_authority": False,
        "policy_logit_delta": 0.0,
        "training_only": True,
        "historical_execution": False,
        "explanation": (
            "Shadow-only current-deck guide recommendation from the exact submitted "
            "runtime. It did not change logits or choose the production action."
        ),
    }


def _submitted_runtime_policy_payload(
    *,
    observation: Mapping[str, Any],
    candidates: Sequence[Sequence[int]],
    deck: Sequence[int],
    neural_logits: torch.Tensor,
    neural_probabilities: torch.Tensor,
    neural_index: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Apply an exact package-local post-model decision policy when present.

    Ordinary submitted runtimes have no such module and retain the neural
    policy byte-for-byte.  Revision-206 packages include
    ``poke_bot.submission_guide_policy`` and select from model log-probability
    plus a bounded normalized guide bonus.  Calling the exact runtime module
    here avoids reimplementing or guessing that package's serving rule.

    Returns the audit payload, submitted-runtime logits/probabilities/choice,
    and the option-conditioned additive bonus used by counterfactual paths.
    """

    baseline = _legal_vector(neural_logits, len(candidates))
    probabilities = _legal_vector(neural_probabilities, len(candidates))
    zero_bonus = torch.zeros_like(baseline)
    unavailable = {
        "availability": _availability(
            False, "exact submitted runtime has no package-local guide decision policy"
        ),
        "policy_authority": False,
        "applied": False,
        "selected_index": int(neural_index),
    }
    try:
        runtime_policy = importlib.import_module("poke_bot.submission_guide_policy")
    except ImportError:
        return unavailable, baseline, probabilities, int(neural_index), zero_bonus
    load_config = getattr(runtime_policy, "load_config", None)
    select_index = getattr(runtime_policy, "select_index", None)
    if not callable(load_config) or not callable(select_index):
        return (
            {
                **unavailable,
                "availability": _availability(
                    False,
                    "package-local guide decision policy has an incompatible API",
                ),
            },
            baseline,
            probabilities,
            int(neural_index),
            zero_bonus,
        )
    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001 - exact package config boundary
        return (
            {
                **unavailable,
                "availability": _availability(
                    False, f"package-local guide policy config is invalid: {exc}"
                ),
            },
            baseline,
            probabilities,
            int(neural_index),
            zero_bonus,
        )
    if config is None:
        return unavailable, baseline, probabilities, int(neural_index), zero_bonus
    try:
        selected, raw_audit = select_index(
            observation=dict(observation),
            candidates=[list(candidate) for candidate in candidates],
            model_policy=_tensor_to_json(probabilities),
            model_index=int(neural_index),
            deck=list(deck),
            config=config,
        )
    except Exception as exc:
        raise ReplayInspectionError(
            "exact submitted guide decision policy failed during reconstruction"
        ) from exc
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or not 0 <= selected < len(candidates)
        or not isinstance(raw_audit, Mapping)
    ):
        raise ReplayInspectionError(
            "exact submitted guide decision policy returned an invalid selection"
        )
    audit = _json_safe(dict(raw_audit))
    applied = bool(raw_audit.get("guide_available"))
    adjusted_scores = raw_audit.get("adjusted_log_scores")
    normalized_scores = raw_audit.get("normalized_guide_scores")
    weight = raw_audit.get("guide_logit_weight")
    bonus = zero_bonus
    submitted_logits = baseline
    submitted_probabilities = probabilities
    if applied:
        if (
            not isinstance(adjusted_scores, Sequence)
            or isinstance(adjusted_scores, (str, bytes))
            or len(adjusted_scores) != len(candidates)
            or not isinstance(normalized_scores, Sequence)
            or isinstance(normalized_scores, (str, bytes))
            or len(normalized_scores) != len(candidates)
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
        ):
            raise ReplayInspectionError(
                "exact submitted guide decision policy omitted its applied scores"
            )
        try:
            submitted_logits = torch.as_tensor(
                [float(value) for value in adjusted_scores],
                dtype=baseline.dtype,
                device=baseline.device,
            )
            bonus = torch.as_tensor(
                [float(weight) * float(value) for value in normalized_scores],
                dtype=baseline.dtype,
                device=baseline.device,
            )
        except (TypeError, ValueError) as exc:
            raise ReplayInspectionError(
                "exact submitted guide decision scores are not finite numbers"
            ) from exc
        if not bool(
            torch.isfinite(submitted_logits).all() and torch.isfinite(bonus).all()
        ):
            raise ReplayInspectionError(
                "exact submitted guide decision scores are not finite numbers"
            )
        submitted_probabilities = torch.softmax(submitted_logits, dim=-1)
        if int(torch.argmax(submitted_logits).item()) != selected:
            raise ReplayInspectionError(
                "exact submitted guide selection disagrees with its adjusted scores"
            )
    payload = {
        "schema": str(raw_audit.get("schema") or ""),
        "availability": _availability(True),
        "mode": "exact_package_local_post_model_policy",
        "policy_authority": True,
        "applied": applied,
        "historical": False,
        "reproduction_status": "recomputed_not_historical",
        "neural_selected_index": int(neural_index),
        "selected_index": int(selected),
        "changed_neural_choice": int(selected) != int(neural_index),
        "guide_logit_bonus": _tensor_to_json(bonus),
        "audit": audit,
        "explanation": (
            "The exact submitted package applied its bounded guide bonus after "
            "the neural policy. These percentages and the selected action include "
            "that production decision layer."
            if applied
            else "The exact submitted package checked its guide decision layer and "
            "fell back exactly to the neural policy for this decision."
        ),
    }
    return (
        payload,
        submitted_logits,
        submitted_probabilities,
        int(selected),
        bonus,
    )


def _apply_policy_bonus_to_fusion_loo(
    fusion_payload: dict[str, Any],
    *,
    bonus: torch.Tensor,
    target: int,
    option_count: int,
) -> None:
    """Measure every head inside the same package-local submitted policy."""

    if not bool(torch.count_nonzero(bonus).item()):
        return
    leave_one_out = fusion_payload.get("leave_one_out")
    if not isinstance(leave_one_out, Mapping):
        return
    for record in leave_one_out.values():
        if not isinstance(record, dict):
            continue
        runtime_path = record.get("runtime_path")
        if not isinstance(runtime_path, dict):
            continue
        full = runtime_path.get("full_policy_logits")
        without = runtime_path.get("policy_without_head_logits")
        if not isinstance(full, list) or not isinstance(without, list):
            continue
        try:
            full_tensor = torch.as_tensor(full, dtype=bonus.dtype, device=bonus.device)
            without_tensor = torch.as_tensor(
                without, dtype=bonus.dtype, device=bonus.device
            )
            runtime_path.update(
                _loo_effect(
                    full=full_tensor + bonus,
                    ablated=without_tensor + bonus,
                    target=target,
                    option_count=option_count,
                )
            )
            runtime_path["submitted_runtime_policy_bonus_retained"] = True
        except (ReplayInspectionError, TypeError, ValueError):
            continue


def _router_for_request(
    *, router: Any | None, router_factory: Callable[[], Any] | None
) -> Any | None:
    if router is not None and router_factory is not None:
        raise ReplayInspectionError("pass either router or router_factory, not both")
    if router_factory is not None:
        return router_factory()
    if router is None:
        return None
    fork = getattr(router, "fork", None)
    # RuntimePublicMatchupRouter has ``fork``.  Using it prevents a query from
    # carrying public-card state into another game query.
    return fork() if callable(fork) else router


def _observe_router(router: Any, obs: Mapping[str, Any], depth: int) -> None:
    observe = getattr(router, "observe", None)
    if not callable(observe):
        raise ReplayStepUnavailable(
            "provided matchup router has no observe method", code="invalid_router"
        )
    try:
        observe(obs, scope="game_root", depth=depth)
    except TypeError:
        # A small testing/custom router may only accept the observation.  The
        # production RuntimePublicMatchupRouter accepts the richer call above.
        observe(obs)


def _router_snapshot(router: Any | None) -> dict[str, Any]:
    if router is None:
        return {
            "availability": _availability(False, "no matchup router was bound"),
            "route": None,
            "reconstruction_parity": {
                "status": "unavailable",
                "reason": "no checksum-bound router artifact was supplied",
            },
        }
    route = getattr(router, "candidate_model_route", None)
    snapshot = getattr(router, "snapshot", None)
    payload: dict[str, Any] = {
        "availability": _availability(True),
        "route": None if route is None else int(route),
        "reconstruction_parity": {
            "status": "recomputed_router_path",
            "reason": (
                "route was causally reconstructed from the supplied router; "
                "the raw replay has no historical route trace"
            ),
        },
    }
    if callable(snapshot):
        try:
            payload["audit"] = _json_safe(snapshot(include_events=True))
        except TypeError:
            payload["audit"] = _json_safe(snapshot())
    return payload


def _adapter_route_capacity(adapter_bank: Any) -> int | None:
    """Return the exact physical-route capacity exposed by a bank, if any."""

    raw_capacity = getattr(adapter_bank, "slot_capacity", None)
    if isinstance(raw_capacity, int) and not isinstance(raw_capacity, bool):
        return int(raw_capacity) if raw_capacity >= 0 else None
    experts = getattr(adapter_bank, "experts", None)
    try:
        return len(experts) if experts is not None else None
    except TypeError:
        return None


def _adapter_route_details(adapter_bank: Any, route: int) -> dict[str, Any]:
    """Resolve only source-backed route metadata from the resident bank.

    V6 stores a physical slot registry while earlier banks expose an ordered
    ``expert_ids`` tuple.  This helper deliberately does not guess a matchup
    label from a replay filename, reward, or a weight tensor.
    """

    capacity = _adapter_route_capacity(adapter_bank)
    if capacity is None:
        return {
            "availability": _availability(
                False, "adapter bank does not expose a physical route capacity"
            ),
            "slot": None,
            "matched_archetype": None,
            "matched_archetype_source": None,
            "routable": None,
        }
    if route < 0 or route >= capacity:
        return {
            "availability": _availability(
                False,
                f"router route {route} is outside adapter capacity 0..{capacity - 1}",
            ),
            "slot": None,
            "matched_archetype": None,
            "matched_archetype_source": None,
            "routable": False,
        }

    registry = getattr(adapter_bank, "registry", None)
    if isinstance(registry, Mapping):
        slots = registry.get("slots")
        if isinstance(slots, Sequence) and not isinstance(slots, (str, bytes)):
            if route >= len(slots) or not isinstance(slots[route], Mapping):
                return {
                    "availability": _availability(
                        False,
                        "adapter route has no source-backed slot-registry record",
                    ),
                    "slot": route,
                    "matched_archetype": None,
                    "matched_archetype_source": None,
                    "routable": None,
                }
            row = slots[route]
            status = row.get("status")
            archetype = row.get("archetype_id")
            routable = status in {"active", "dormant"}
            if not isinstance(archetype, str) or not archetype.strip():
                return {
                    "availability": _availability(
                        False,
                        "adapter route has no source-backed archetype identity",
                    ),
                    "slot": route,
                    "matched_archetype": None,
                    "matched_archetype_source": None,
                    "routable": routable,
                    "slot_status": status,
                }
            return {
                "availability": _availability(True),
                "slot": route,
                "matched_archetype": archetype,
                "matched_archetype_source": "adapter_bank_slot_registry",
                "routable": routable,
                "slot_status": status,
            }

    expert_ids = getattr(adapter_bank, "expert_ids", None)
    if (
        isinstance(expert_ids, Sequence)
        and not isinstance(expert_ids, (str, bytes))
        and route < len(expert_ids)
        and isinstance(expert_ids[route], str)
    ):
        return {
            "availability": _availability(True),
            "slot": route,
            "matched_archetype": expert_ids[route],
            "matched_archetype_source": "adapter_bank_expert_ids",
            "routable": True,
        }
    return {
        "availability": _availability(
            False, "adapter route has no source-backed archetype identity"
        ),
        "slot": route,
        "matched_archetype": None,
        "matched_archetype_source": None,
        "routable": None,
    }


def _adapter_reliability(adapter_bank: Any) -> dict[str, Any]:
    """Expose the bank's declared residual scale without calling it learned."""

    raw_scale = getattr(adapter_bank, "residual_scale", None)
    if isinstance(raw_scale, (int, float)) and not isinstance(raw_scale, bool):
        scale = float(raw_scale)
        if math.isfinite(scale):
            return {
                "availability": _availability(True),
                "kind": "fixed_residual_scale",
                "residual_scale": scale,
            }
    return {
        "availability": _availability(
            False, "adapter bank does not expose a finite residual scale"
        )
    }


def _adapter_payload(
    *,
    model: torch.nn.Module,
    prepared: _PreparedReplayStep,
    state: torch.Tensor,
    policy_value_state: torch.Tensor,
    option_tokens: Any,
    spatial: torch.Tensor,
    final_logits: torch.Tensor,
    option_count: int,
    router_snapshot: Mapping[str, Any],
    runtime_activation: Mapping[str, Any],
    final_logit_bonus: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Describe actual Matchup Adapter use for one conditional decision.

    A present parameter bank is not evidence of usage.  The payload therefore
    carries separate runtime, route, and routability facts and only attempts a
    no-adapter policy counterfactual when the exact decision path demonstrably
    selected and applied a routable adapter route.
    """

    adapter_bank = getattr(model, "matchup_adapter_bank", None)
    route = prepared.route
    route_value = (
        route if isinstance(route, int) and not isinstance(route, bool) else None
    )
    bank_present = adapter_bank is not None
    enabled_value = getattr(adapter_bank, "enabled", None) if bank_present else None
    runtime_enabled = bool(enabled_value) if isinstance(enabled_value, bool) else None
    details = (
        _adapter_route_details(adapter_bank, route_value)
        if bank_present and route_value is not None and route_value >= 0
        else None
    )
    route_routable = details.get("routable") if isinstance(details, Mapping) else None
    decision_route_active = bool(
        bank_present
        and runtime_activation.get("applied") is True
        and runtime_enabled is True
        and route_value is not None
        and route_value >= 0
        and route_routable is True
    )
    audit = router_snapshot.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    router_archetype = audit.get("active_archetype_id")
    matched_archetype = (
        str(router_archetype)
        if isinstance(router_archetype, str) and router_archetype.strip()
        else (
            details.get("matched_archetype") if isinstance(details, Mapping) else None
        )
    )
    matched_source = (
        "router_audit"
        if isinstance(router_archetype, str) and router_archetype.strip()
        else (
            details.get("matched_archetype_source")
            if isinstance(details, Mapping)
            else None
        )
    )
    policy_influence: dict[str, Any]
    if not bank_present:
        policy_influence = {
            "availability": _availability(
                False, "checkpoint has no matchup adapter bank"
            )
        }
    elif not decision_route_active:
        activation_reason = runtime_activation.get("reason")
        if runtime_activation.get("applied") is not True:
            reason = str(
                activation_reason
                or "checksum-bound submitted startup activation is unavailable"
            )
        elif route_value is None:
            reason = "no router route was bound; policy/value state used exact bypass"
        elif runtime_enabled is False:
            reason = "matchup adapter bank is runtime-disabled for this decision"
        elif runtime_enabled is None:
            reason = "adapter runtime enabled state is unavailable"
        elif route_value < 0:
            reason = "router selected its explicit unknown/bypass route"
        elif route_routable is False:
            reason = "router route is not routable by this adapter bank"
        else:
            reason = "adapter route activation cannot be verified for this decision"
        policy_influence = {"availability": _availability(False, reason)}
    else:
        try:
            bypass_logits = model.decode_options(
                option_tokens,
                spatial,
                state,
                n_options=[option_count],
                decision_fusion_state_vec=state,
            )
            if not isinstance(bypass_logits, torch.Tensor):
                raise ReplayInspectionError(
                    "no-adapter policy counterfactual did not return logits"
                )
            bypass_final = _option_tensor(bypass_logits, option_count)
            if final_logit_bonus is not None:
                bypass_final = bypass_final + _legal_vector(
                    final_logit_bonus, option_count
                )
            effect = _loo_effect(
                full=final_logits,
                ablated=bypass_final,
                target=prepared.target_index,
                option_count=option_count,
            )
            raw_influence = effect.get("policy_influence")
            if not isinstance(raw_influence, Mapping):
                raise ReplayInspectionError(
                    "no-adapter counterfactual lacks final-policy probability metrics"
                )
            policy_influence = {
                **dict(raw_influence),
                "method": "full_final_policy_minus_policy_without_matchup_adapter_route",
                "sign_convention": "full_final_policy_minus_policy_without_matchup_adapter_route",
                "leave_one_adapter_out": effect,
            }
        except Exception as exc:  # noqa: BLE001 - preserve a read-only trace
            policy_influence = {
                "availability": _availability(
                    False,
                    "exact no-adapter final-policy counterfactual is unavailable: "
                    + str(exc),
                )
            }

    base: dict[str, Any] = {
        "availability": _availability(
            bank_present,
            None if bank_present else "checkpoint has no matchup adapter bank",
        ),
        "bank_present": bank_present,
        "runtime_enabled": runtime_enabled,
        "route": route_value,
        "decision_route_active": decision_route_active,
        "route_routable": route_routable,
        "raw_state_l2_norm": float(torch.linalg.vector_norm(state).item()),
        "policy_value_state_l2_norm": float(
            torch.linalg.vector_norm(policy_value_state).item()
        ),
        "adapter_delta_l2_norm": (
            float(torch.linalg.vector_norm(policy_value_state - state).item())
            if decision_route_active
            else None
        ),
        "route_reliability": (
            _adapter_reliability(adapter_bank)
            if bank_present
            else _availability(False, "checkpoint has no matchup adapter bank")
        ),
        "matched_archetype": matched_archetype,
        "matched_archetype_source": matched_source,
        "router_audit": _json_safe(audit) if audit else None,
        "policy_influence": policy_influence,
        # Keep this request-local dictionary by reference: the enclosing
        # ``finally`` marks restoration complete before the payload escapes.
        "runtime_activation": runtime_activation,
    }
    if isinstance(details, Mapping):
        base["slot"] = details.get("slot")
        base["slot_status"] = details.get("slot_status")
        base["route_metadata_availability"] = details.get("availability")
    else:
        base["slot"] = None
        base["slot_status"] = None
        base["route_metadata_availability"] = _availability(
            False,
            "no non-bypass adapter route was selected"
            if route_value is not None
            else "no router route was bound",
        )
    return base


@dataclass
class _PreparedReplayStep:
    observation: dict[str, Any]
    recorded_action: list[int]
    candidates: list[list[int]]
    target_index: int
    board_history: list[Any]
    action_history: list[Any]
    route: int | None
    router: Any | None
    deck: list[int]
    hypothetical_setup_prompt: bool = False


def _is_first_prompt(context: Any) -> bool:
    if isinstance(context, str):
        normalized = context.replace("_", "").replace(" ", "").casefold()
        return normalized == "isfirst"
    try:
        return int(context) == 41
    except (TypeError, ValueError):
        return False


def _recorded_action(steps: Sequence[Any], index: int, seat: int) -> list[int]:
    if index + 1 >= len(steps) or not isinstance(steps[index + 1], Sequence):
        raise ReplayStepUnavailable(
            "the replay has no next transition containing the recorded action",
            code="recorded_action_missing",
        )
    next_row = steps[index + 1]
    if seat >= len(next_row) or not isinstance(next_row[seat], Mapping):
        raise ReplayStepUnavailable(
            "the replay has no recorded action for the acting seat",
            code="recorded_action_missing",
        )
    action = next_row[seat].get("action")
    if not isinstance(action, list) or not all(
        isinstance(value, int) for value in action
    ):
        raise ReplayStepUnavailable(
            "recorded action is not an integer option-index list",
            code="recorded_action_invalid",
        )
    return [int(value) for value in action]


def _resolve_deck(
    replay: Mapping[str, Any], seat: int, own_deck: Sequence[int] | None
) -> list[int]:
    if own_deck is None:
        decks = extract_setup_decks(dict(replay))
        if seat >= len(decks):
            raise ReplayStepUnavailable("acting seat is outside replay deck data")
        own_deck = decks[seat]
    if (
        not isinstance(own_deck, Sequence)
        or isinstance(own_deck, (str, bytes))
        or len(own_deck) != 60
        or not all(isinstance(value, int) for value in own_deck)
    ):
        raise ReplayStepUnavailable(
            "the acting seat's exact 60-card deck is unavailable",
            code="deck_unavailable",
        )
    return [int(value) for value in own_deck]


def _prepare_replay_step(
    *,
    model: torch.nn.Module,
    replay: Mapping[str, Any],
    acting_seat: int,
    env_step: int,
    factorized_stage: int,
    own_deck: Sequence[int] | None,
    router: Any | None,
    router_factory: Callable[[], Any] | None,
    allow_setup_prompt_model_forward: bool,
) -> _PreparedReplayStep:
    steps = replay.get("steps")
    if not isinstance(steps, list):
        raise ReplayStepUnavailable("replay has no steps list", code="replay_invalid")
    seat = int(acting_seat)
    index_limit = int(env_step)
    stage_index = int(factorized_stage)
    if seat not in (0, 1):
        raise ReplayStepUnavailable("acting seat must be 0 or 1", code="invalid_seat")
    if index_limit < 0 or index_limit >= len(steps):
        raise ReplayStepUnavailable(
            "environment step is outside this replay", code="step_out_of_range"
        )
    if stage_index < 0:
        raise ReplayStepUnavailable(
            "factorized stage must be nonnegative", code="stage_out_of_range"
        )
    deck = _resolve_deck(replay, seat, own_deck)
    active_router = _router_for_request(router=router, router_factory=router_factory)
    board_history: list[Any] = []
    action_history: list[Any] = []
    previous_action: Any | None = None
    max_context = int(getattr(model, "max_context", 1))

    for index in range(index_limit + 1):
        row_data = steps[index]
        if not isinstance(row_data, list) or seat >= len(row_data):
            if index == index_limit:
                raise ReplayStepUnavailable(
                    "acting-seat step row is absent", code="step_unavailable"
                )
            continue
        row = row_data[seat]
        if not isinstance(row, Mapping):
            if index == index_limit:
                raise ReplayStepUnavailable(
                    "acting-seat step row is malformed", code="step_unavailable"
                )
            continue
        raw_obs = row.get("observation") or {}
        obs = dict(raw_obs) if isinstance(raw_obs, Mapping) else {}
        active_select = row.get("status") == "ACTIVE" and obs.get("select") is not None
        if not active_select:
            if index == index_limit:
                raise ReplayStepUnavailable(
                    "requested environment step is not an active model decision",
                    code="not_a_decision",
                )
            continue
        try:
            assert_active_select_actor(obs, seat=seat, step_index=index)
        except ReplayTimelineError as exc:
            raise ReplayStepUnavailable(
                "active selectable replay row has no exact archived acting-seat identity",
                code="acting_seat_identity_unavailable",
                detail=str(exc),
                blocking_step=index,
            ) from exc
        try:
            features.assert_info_set(obs)
        except Exception as exc:  # inference must never consume a leaked observation
            raise ReplayStepUnavailable(
                "masked replay observation fails the information-set contract",
                code="info_set_violation",
                detail=str(exc),
                blocking_step=index,
            ) from exc
        select = obs.get("select") or {}
        setup_prompt = _is_first_prompt(select.get("context"))
        if setup_prompt:
            if index == index_limit and not allow_setup_prompt_model_forward:
                raise ReplayStepUnavailable(
                    "IsFirst setup prompt is selected before the neural model runs",
                    code="setup_no_model_forward",
                )
            if index != index_limit:
                # submission/main.py answers IsFirst before _ensure_runtime.  The
                # PolicyAgent, matchup router, and temporal history therefore do
                # not observe this prompt; treating it as a prior neural decision
                # would shift both route hysteresis and causal context.
                continue
        action = _recorded_action(steps, index, seat)
        if active_router is not None and not setup_prompt:
            _observe_router(active_router, obs, len(board_history))
        try:
            board = features.build_board_tokens(obs, deck)
        except Exception as exc:
            raise ReplayStepUnavailable(
                "replay observation cannot be converted to model board tokens",
                code="feature_reconstruction_failed",
                detail=str(exc),
                blocking_step=index,
            ) from exc
        board_history.append(board)
        action_history.append(previous_action)
        board_history = board_history[-max_context:]
        action_history = action_history[-max_context:]

        if index == index_limit:
            try:
                stages = features.factorized_teacher_forcing_stages(obs, action)
            except Exception as exc:
                raise ReplayStepUnavailable(
                    "recorded action cannot be factorized against the legal options",
                    code="factorized_stage_unavailable",
                    detail=str(exc),
                ) from exc
            if stage_index >= len(stages):
                raise ReplayStepUnavailable(
                    "requested factorized stage is outside the recorded action",
                    code="stage_out_of_range",
                    stage_count=len(stages),
                )
            candidates, target = stages[stage_index]
            if not candidates:
                raise ReplayStepUnavailable(
                    "factorized stage has no legal candidates",
                    code="stage_unavailable",
                )
            if setup_prompt:
                # No router exists yet in the submitted entrypoint.  Keep the
                # hypothetical query on the model's exact unrouted base path.
                route_value = None
            elif active_router is None:
                route_value = None
            else:
                raw_route = getattr(active_router, "candidate_model_route", None)
                if raw_route is None:
                    raise ReplayStepUnavailable(
                        "provided matchup router has no candidate model route",
                        code="invalid_router",
                    )
                try:
                    route_value = int(raw_route)
                except (TypeError, ValueError) as exc:
                    raise ReplayStepUnavailable(
                        "provided matchup router returned an invalid model route",
                        code="invalid_router",
                    ) from exc
            return _PreparedReplayStep(
                observation=obs,
                recorded_action=action,
                candidates=[list(combo) for combo in candidates],
                target_index=int(target),
                board_history=board_history,
                action_history=action_history,
                route=route_value,
                router=None if setup_prompt else active_router,
                deck=deck,
                hypothetical_setup_prompt=setup_prompt,
            )

        try:
            previous_action = features.build_option_tokens(obs, [action])
        except Exception as exc:
            raise ReplayStepUnavailable(
                "a prior recorded action cannot be converted to causal history",
                code="causal_history_unavailable",
                detail=str(exc),
                blocking_step=index,
            ) from exc
    raise ReplayStepUnavailable(
        "requested replay step was not found", code="step_unavailable"
    )


def inspect_replay_step(
    *,
    model: torch.nn.Module,
    replay: Mapping[str, Any],
    acting_seat: int,
    env_step: int,
    factorized_stage: int = 0,
    own_deck: Sequence[int] | None = None,
    router: Any | None = None,
    router_factory: Callable[[], Any] | None = None,
    checkpoint_digest: str | None = None,
    checkpoint_path: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
    submitted_runtime_activation: Mapping[str, Any] | None = None,
    head_scales: Mapping[str, float] | None = None,
    allow_setup_prompt_model_forward: bool = False,
) -> dict[str, Any]:
    """Re-run one recorded factorized decision from causally rebuilt history.

    ``model`` should come from :class:`VerifiedCpuModelCache`.  Direct callers
    can use this pure function for fixtures, but must supply an immutable
    evaluation model themselves.  A legacy/missing architecture returns a
    structured ``available: false`` field per absent capability; it is never
    represented by a fake zero tensor.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("inspect_replay_step requires a torch.nn.Module")
    try:
        prepared = _prepare_replay_step(
            model=model,
            replay=replay,
            acting_seat=acting_seat,
            env_step=env_step,
            factorized_stage=factorized_stage,
            own_deck=own_deck,
            router=router,
            router_factory=router_factory,
            allow_setup_prompt_model_forward=allow_setup_prompt_model_forward,
        )
    except ReplayStepUnavailable as exc:
        return _unavailable_payload(
            exc, checkpoint_digest=checkpoint_digest, provenance=provenance
        )

    # A cached model is shared by every trace for its checksum.  Serialize the
    # complete evaluation-mode/runtime-activation span so no request can see
    # another request's temporary submitted startup state.
    with _MODEL_TRACE_LOCK, torch.inference_mode():
        was_training = bool(model.training)
        runtime_activation, restoration = _begin_submitted_runtime_activation(
            model, submitted_runtime_activation
        )
        try:
            model.eval()
            try:
                first_stages = features.factorized_teacher_forcing_stages(
                    prepared.observation, prepared.recorded_action
                )
                first_tokens = features.build_option_tokens(
                    prepared.observation, first_stages[0][0]
                )
                route_arg = None if prepared.route is None else [prepared.route]
                history_output = model.forward_history_batch(
                    [prepared.board_history],
                    [first_tokens],
                    n_options=[len(first_stages[0][0])],
                    previous_action_histories=[prepared.action_history],
                    matchup_routes=route_arg,
                )
                state = history_output["state_vec"]
                spatial = history_output["spatial_memory"]
                if not isinstance(state, torch.Tensor) or not isinstance(
                    spatial, torch.Tensor
                ):
                    raise ReplayInspectionError(
                        "model history output lacks state tensors"
                    )
                policy_value_state = model.matchup_policy_value_state(state, route_arg)
                option_tokens = features.build_option_tokens(
                    prepared.observation, prepared.candidates
                )
                final_logits, option_hidden = model.decode_options(
                    option_tokens,
                    spatial,
                    policy_value_state,
                    n_options=[len(prepared.candidates)],
                    return_hidden=True,
                    decision_fusion_state_vec=state,
                )
            except ReplayStepUnavailable:
                raise
            except Exception as exc:
                raise ReplayInspectionError(
                    "model inference failed while recreating the replay step"
                ) from exc
            if not isinstance(final_logits, torch.Tensor) or not isinstance(
                option_hidden, torch.Tensor
            ):
                raise ReplayInspectionError("model decode did not return tensors")
            option_count = len(prepared.candidates)
            base_logits = model.policy_head(option_hidden).squeeze(-1)
            decision_fusion = getattr(model, "decision_fusion", None)
            state_sources: Mapping[str, torch.Tensor] | None = None
            option_sources: Mapping[str, torch.Tensor] | None = None
            if decision_fusion is not None:
                source_builder = getattr(model, "decision_fusion_sources", None)
                if callable(source_builder):
                    built_state, built_option = source_builder(option_hidden, state)
                    if isinstance(built_state, Mapping) and isinstance(
                        built_option, Mapping
                    ):
                        state_sources, option_sources = built_state, built_option
            heads = _build_head_records(
                model=model,
                state=state,
                option_hidden=option_hidden,
                option_count=option_count,
                state_sources=state_sources,
                option_sources=option_sources,
            )
            fusion, actual_fusion_logits = _fusion_payload(
                model=model,
                option_hidden=option_hidden,
                state=state,
                base_logits=base_logits,
                state_sources=state_sources,
                option_sources=option_sources,
                target=prepared.target_index,
                option_count=option_count,
            )
            latent = _latent_payload(
                model=model,
                option_hidden=option_hidden,
                state=state,
                option_count=option_count,
            )
            neural_final = _option_tensor(final_logits, option_count)
            neural_probabilities = torch.softmax(neural_final, dim=-1)
            neural_choice = int(torch.argmax(neural_final).item())
            (
                submitted_runtime_policy,
                valid_final,
                probabilities,
                model_choice,
                final_logit_bonus,
            ) = _submitted_runtime_policy_payload(
                observation=prepared.observation,
                candidates=prepared.candidates,
                deck=prepared.deck,
                neural_logits=neural_final,
                neural_probabilities=neural_probabilities,
                neural_index=neural_choice,
            )
            _apply_policy_bonus_to_fusion_loo(
                fusion,
                bonus=final_logit_bonus,
                target=prepared.target_index,
                option_count=option_count,
            )
            for name, loo in dict(fusion.get("leave_one_out") or {}).items():
                if name in heads:
                    heads[name]["leave_one_out"] = loo
            _attach_head_policy_influence(heads, fusion)
            guide_shadow = _guide_shadow_payload(
                observation=prepared.observation,
                candidates=prepared.candidates,
                deck=prepared.deck,
                recorded_index=prepared.target_index,
                model_index=neural_choice,
            )
            decision_influence = _decision_influence_payload(
                model=model,
                option_hidden=option_hidden,
                state=state,
                base_logits=base_logits,
                baseline_final_logits=final_logits,
                state_sources=state_sources,
                option_sources=option_sources,
                requested_scales=head_scales,
                target=prepared.target_index,
                option_count=option_count,
                fusion_payload=fusion,
                final_logit_bonus=final_logit_bonus,
            )
            router_snapshot = _router_snapshot(prepared.router)
            adapter_payload = _adapter_payload(
                model=model,
                prepared=prepared,
                state=state,
                policy_value_state=policy_value_state,
                option_tokens=option_tokens,
                spatial=spatial,
                final_logits=valid_final,
                option_count=option_count,
                router_snapshot=router_snapshot,
                runtime_activation=runtime_activation,
                final_logit_bonus=final_logit_bonus,
            )
            model_value = history_output.get("value")
            supplied_provenance = _provenance_mapping(provenance)
            requested_status = str(
                supplied_provenance.get("reproduction_status") or ""
            ).strip()
            parity_verified = bool(
                supplied_provenance.get("runtime_parity_verified", False)
            )
            if prepared.hypothetical_setup_prompt:
                reproduction_status = "hypothetical_model_forward_not_submitted_runtime"
                reproduction_reason = (
                    "the submitted entrypoint answered this setup prompt before "
                    "the neural model; these values are an explicit hypothetical "
                    "forward pass through the checksum-bound archived model"
                )
            elif requested_status == "exact_reproduced" and parity_verified:
                reproduction_status = "exact_reproduced"
                reproduction_reason = (
                    "caller supplied a verified archived-runtime parity receipt"
                )
            else:
                reproduction_status = "recomputed_not_historical"
                reproduction_reason = (
                    "raw replays do not record historical logits; this is a "
                    "current-source recomputation unless archived runtime parity "
                    "is independently verified"
                )
            provenance_payload = {
                **_json_safe(supplied_provenance),
                "checkpoint_digest": checkpoint_digest,
                "checkpoint_path": None
                if checkpoint_path is None
                else str(checkpoint_path),
                "model_config": _model_config(model),
                "evaluation_mode": True,
                "inference_device": _model_device(model),
                "reproduction_status": reproduction_status,
                "reproduction_reason": reproduction_reason,
            }
            return {
                "schema": INFERENCE_SCHEMA,
                "availability": _availability(True),
                "provenance": provenance_payload,
                "replay": {
                    "acting_seat": int(acting_seat),
                    "env_step": int(env_step),
                    "factorized_stage": int(factorized_stage),
                    "observation": _json_safe(prepared.observation),
                    "recorded_action": list(prepared.recorded_action),
                    "recorded_target_index": prepared.target_index,
                    "recorded_target_candidate": prepared.candidates[
                        prepared.target_index
                    ],
                    "legal_candidates": prepared.candidates,
                    "legal_candidate_mask": [True] * option_count,
                    "history_length": len(prepared.board_history),
                    "deck_card_count": len(prepared.deck),
                    "router": router_snapshot,
                    "hypothetical_setup_prompt": prepared.hypothetical_setup_prompt,
                },
                "adapter": adapter_payload,
                "policy": {
                    "base_logits": _tensor_to_json(
                        _option_tensor(base_logits, option_count)
                    ),
                    "fusion_logits_before_latent": _tensor_to_json(
                        _option_tensor(actual_fusion_logits, option_count)
                    ),
                    "final_logits": _tensor_to_json(valid_final),
                    "probabilities": _tensor_to_json(probabilities),
                    "model_choice_index": model_choice,
                    "model_choice_candidate": prepared.candidates[model_choice],
                    "model_matches_recorded_target": model_choice
                    == prepared.target_index,
                    "recorded_target_margin": _margin(
                        valid_final, prepared.target_index, option_count
                    ),
                    "neural_final_logits": _tensor_to_json(neural_final),
                    "neural_probabilities": _tensor_to_json(neural_probabilities),
                    "neural_model_choice_index": neural_choice,
                    "neural_model_choice_candidate": prepared.candidates[neural_choice],
                    "submitted_runtime_policy": submitted_runtime_policy,
                },
                "value": {
                    "availability": _availability(
                        isinstance(model_value, torch.Tensor),
                        "model history output has no value"
                        if not isinstance(model_value, torch.Tensor)
                        else None,
                    ),
                    "policy_value": None
                    if not isinstance(model_value, torch.Tensor)
                    else _tensor_to_json(_state_tensor(model_value)),
                    "fusion_raw_state_value": heads["value"].get("fusion_input_values"),
                },
                "heads": heads,
                "fusion": fusion,
                "latent_lookahead": latent,
                "decision_influence": decision_influence,
                "guide_shadow": guide_shadow,
            }
        finally:
            model.train(was_training)
            _restore_submitted_runtime_activation(restoration, runtime_activation)


__all__ = [
    "INFERENCE_SCHEMA",
    "UNAVAILABLE_TARGET_MASK_REASON",
    "LoadedModel",
    "ReplayInferenceEngine",
    "ReplayInspectionError",
    "ReplayStepUnavailable",
    "VerifiedCpuModelCache",
    "inspect_replay_step",
]
