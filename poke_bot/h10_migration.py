"""Function-preserving H10-I expansion for final-submission derivatives."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import torch

from . import config
from .model import TemporalCabtTransformer, build_model


H10_SPATIAL_LAYERS = 7
H10_TEMPORAL_LAYERS = 3
H10_OPTION_LAYERS = 7
H10_FF_DIM = 2496
H10_HEAD_RESIDUAL_WIDTH = 512


def final_format_config(
    parent: config.ModelConfig,
    *,
    directional_fusion_v3: bool = False,
) -> config.ModelConfig:
    """Return the exact H10-I child config for the validated ordinary parent."""

    expected = {
        "d_model": 96,
        "n_heads": 8,
        "spatial_layers": 4,
        "temporal_layers": 1,
        "option_decoder_layers": 4,
        "ff_dim": 384,
        "max_context": 320,
        "decision_context": "history",
        "kv_cache": True,
        "expanded_heads_enabled": True,
        "decision_fusion_enabled": True,
        "decision_fusion_runtime_enabled": True,
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(parent, name)}
        for name, value in expected.items()
        if getattr(parent, name) != value
    }
    if mismatches:
        raise ValueError(f"ordinary Alakazam parent architecture changed: {mismatches}")
    return replace(
        parent,
        spatial_layers=H10_SPATIAL_LAYERS,
        temporal_layers=H10_TEMPORAL_LAYERS,
        option_decoder_layers=H10_OPTION_LAYERS,
        ff_dim=H10_FF_DIM,
        setup_board_outcome_head_enabled=True,
        combo_state_head_enabled=True,
        decision_fusion_dedicated_routes_enabled=True,
        decision_fusion_dedicated_routes_runtime_enabled=True,
        decision_fusion_typed_output_centered_routes_enabled=bool(
            directional_fusion_v3
        ),
        decision_fusion_action_type_reliability_cap=(
            0.25 if directional_fusion_v3 else 1.0
        ),
        h10_capacity_enabled=True,
        h10_head_residual_width=H10_HEAD_RESIDUAL_WIDTH,
    )


def _expanded_tensor(source: torch.Tensor, target: torch.Tensor, key: str) -> torch.Tensor:
    """Copy a tensor, widening only a transformer feed-forward dimension."""

    if source.shape == target.shape:
        return source.detach().clone()
    incoming = key.endswith(("linear1.weight", "linear1.bias", "ff.0.weight", "ff.0.bias"))
    outgoing = key.endswith(("linear2.weight", "ff.3.weight"))
    if incoming and target.shape[0] > source.shape[0] and target.shape[1:] == source.shape[1:]:
        result = target.detach().clone()
        old = int(source.shape[0])
        result[:old].copy_(source)
        clone_rows = torch.arange(old, int(target.shape[0])) % old
        result[old:].copy_(source.index_select(0, clone_rows))
        return result
    if outgoing and target.ndim == 2 and target.shape[1] > source.shape[1] and target.shape[0] == source.shape[0]:
        result = torch.zeros_like(target)
        result[:, : source.shape[1]].copy_(source)
        return result
    raise ValueError(
        f"unsupported inherited tensor shape change for {key}: "
        f"{tuple(source.shape)} -> {tuple(target.shape)}"
    )


def _clone_block(
    *,
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    source_prefix: str,
    target_prefix: str,
    zero_suffixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_key, source_tensor in source_state.items():
        if not source_key.startswith(source_prefix):
            continue
        suffix = source_key[len(source_prefix) :]
        target_key = target_prefix + suffix
        if target_key not in target_state:
            raise ValueError(f"appended block target lacks {target_key}")
        target_state[target_key] = _expanded_tensor(
            source_tensor, target_state[target_key], target_key
        )
        rows.append({"target": target_key, "source": source_key})
    if not rows:
        raise ValueError(f"appended block source prefix is empty: {source_prefix}")
    for suffix in zero_suffixes:
        key = target_prefix + suffix
        if key not in target_state:
            raise ValueError(f"zero-safe block tensor is absent: {key}")
        target_state[key] = torch.zeros_like(target_state[key])
    return rows


def migrate_state(
    parent: TemporalCabtTransformer,
    child: TemporalCabtTransformer,
) -> dict[str, Any]:
    """Load an exact parent-preserving H10 state into ``child``."""

    source = dict(parent.state_dict())
    target = {name: tensor.detach().clone() for name, tensor in child.state_dict().items()}
    inherited: list[dict[str, Any]] = []
    for key, tensor in source.items():
        if key not in target:
            raise ValueError(f"H10 child omitted inherited tensor {key}")
        target[key] = _expanded_tensor(tensor, target[key], key)
        inherited.append(
            {
                "target": key,
                "source": key,
                "source_shape": list(tensor.shape),
                "target_shape": list(target[key].shape),
            }
        )

    appended: list[dict[str, Any]] = []
    for index in range(4, H10_SPATIAL_LAYERS):
        appended.extend(
            _clone_block(
                source_state=source,
                target_state=target,
                source_prefix="spatial_encoder.layers.3.",
                target_prefix=f"spatial_encoder.layers.{index}.",
                zero_suffixes=(
                    "self_attn.out_proj.weight",
                    "self_attn.out_proj.bias",
                    "linear2.weight",
                    "linear2.bias",
                ),
            )
        )
    for index in range(1, H10_TEMPORAL_LAYERS):
        appended.extend(
            _clone_block(
                source_state=source,
                target_state=target,
                source_prefix="temporal_blocks.0.",
                target_prefix=f"temporal_blocks.{index}.",
                zero_suffixes=(
                    "attn.out.weight",
                    "attn.out.bias",
                    "ff.3.weight",
                    "ff.3.bias",
                ),
            )
        )
    for index in range(4, H10_OPTION_LAYERS):
        appended.extend(
            _clone_block(
                source_state=source,
                target_state=target,
                source_prefix="option_decoder.3.",
                target_prefix=f"option_decoder.{index}.",
                zero_suffixes=(
                    "cross.out_proj.weight",
                    "cross.out_proj.bias",
                    "ff.3.weight",
                    "ff.3.bias",
                ),
            )
        )

    inherited_targets = {row["target"] for row in inherited}
    appended_targets = {row["target"] for row in appended}
    new_keys = sorted(set(target) - inherited_targets - appended_targets)
    allowed_new_prefixes = (
        "setup_board_outcome_head.",
        "combo_state_head.",
        "decision_fusion.dedicated_routes.",
        "decision_fusion.dedicated_route_log_reliability.",
        "h10_head_residuals.",
    )
    unexpected = [
        key for key in new_keys if not key.startswith(allowed_new_prefixes)
    ]
    if unexpected:
        raise ValueError(f"unclassified H10 tensors: {unexpected}")
    zero_safe = []
    for key in new_keys:
        if (
            key.startswith("decision_fusion.dedicated_routes.")
            and ".network.2." in key
        ) or (
            key.startswith("h10_head_residuals.")
            and ".network.2." in key
        ):
            if torch.count_nonzero(target[key]).item() != 0:
                raise ValueError(f"new H10 output is not zero-safe: {key}")
            zero_safe.append(key)
    child.load_state_dict(target, strict=True)
    return {
        "inherited_tensors": inherited,
        "appended_block_tensors": appended,
        "new_tensors": new_keys,
        "zero_safe_output_tensors": sorted(zero_safe),
    }


@torch.no_grad()
def step_zero_parity(
    parent: TemporalCabtTransformer,
    child: TemporalCabtTransformer,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Exercise short/full histories and the exact fused option path."""

    parent.eval()
    child.eval()
    generator = torch.Generator(device="cpu").manual_seed(0xA1A_CA2A)
    maxima: dict[str, float] = {}

    def record(name: str, left: torch.Tensor, right: torch.Tensor) -> None:
        maxima[name] = float((left.float() - right.float()).abs().max().item())
        if not torch.allclose(left, right, atol=atol, rtol=rtol):
            raise ValueError(f"H10 step-zero parity failed for {name}: {maxima[name]}")

    tokens = torch.randn(2, parent.num_board_tokens, parent.d_model, generator=generator)
    parent_spatial = parent.spatial_norm(parent.spatial_encoder(tokens))
    child_spatial = child.spatial_norm(child.spatial_encoder(tokens))
    record("spatial", parent_spatial, child_spatial)

    short = torch.randn(2, 7, parent.d_model, generator=generator)
    parent_short, _ = parent.temporal_encode(short, append=False, return_all=True)
    child_short, _ = child.temporal_encode(short, append=False, return_all=True)
    record("temporal_short", parent_short, child_short)

    full = torch.randn(1, 320, parent.d_model, generator=generator)
    parent_full, _ = parent.temporal_encode(full, append=False, return_all=True)
    child_full, _ = child.temporal_encode(full, append=False, return_all=True)
    record("temporal_full_320", parent_full, child_full)

    option_tokens = torch.randn(2, 5, parent.d_model, generator=generator)
    counts = torch.tensor([5, 3])
    parent_logits, parent_hidden = parent._decode_option_tokens(
        option_tokens,
        parent_spatial,
        parent_short[:, -1, :],
        n_options=counts,
        return_hidden=True,
    )
    child_logits, child_hidden = child._decode_option_tokens(
        option_tokens,
        child_spatial,
        child_short[:, -1, :],
        n_options=counts,
        return_hidden=True,
    )
    record("option_hidden", parent_hidden, child_hidden)
    finite = torch.isfinite(parent_logits)
    if not torch.equal(finite, torch.isfinite(child_logits)):
        raise ValueError("H10 step-zero legal masks changed")
    record("policy_logits", parent_logits[finite], child_logits[finite])
    if not torch.equal(parent_logits.argmax(dim=-1), child_logits.argmax(dim=-1)):
        raise ValueError("H10 step-zero greedy actions changed")

    state = parent_short[:, -1, :]
    record("value", parent.value_head(state), child.value_head(state))
    parent_option = parent.expanded_option_logits(parent_hidden)
    child_option = child.expanded_option_logits(child_hidden)
    for name in parent_option:
        record(f"typed_option.{name}", parent_option[name], child_option[name])
    parent_state = parent.expanded_state_logits(state)
    child_state = child.expanded_state_logits(state)
    for name in parent_state:
        record(f"typed_state.{name}", parent_state[name], child_state[name])

    return {
        "passed": True,
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_differences": maxima,
        "legal_masks_exact": True,
        "greedy_actions_exact": True,
        "full_320_decision_history_passed": True,
    }


def build_h10_child(
    parent: TemporalCabtTransformer,
    *,
    directional_fusion_v3: bool = False,
) -> tuple[TemporalCabtTransformer, dict[str, Any]]:
    cfg = final_format_config(
        parent.cfg,
        directional_fusion_v3=directional_fusion_v3,
    )
    child = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=int(parent.aux_head[-1].out_features),
        encoder_vocab=int(parent.encoder_vocab),
        decoder_vocab=int(parent.decoder_vocab),
        belief_card_vocab=int(parent.belief_card_vocab),
    )
    migration = migrate_state(parent, child)
    parity = step_zero_parity(parent, child)
    return child, {
        "schema": "poke_bot.h10_function_preserving_migration/v1",
        "parent_model_config": asdict(parent.cfg),
        "child_model_config": asdict(cfg),
        "migration": migration,
        "step_zero_parity": parity,
        "learned_parameters": int(sum(p.numel() for p in child.parameters())),
        "directional_fusion_v3": bool(directional_fusion_v3),
    }
