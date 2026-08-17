#!/usr/bin/env python3
"""Warm-start a one-layer game-bounded temporal policy from a state evaluator.

Every temporal training sequence is one acting seat's complete causal game
record. Games are never concatenated and no decision may attend across an
episode boundary. The default 320-token rolling window retains the newest
within-game history and covers about 99.15% of measured games without truncation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint
from poke_bot.model import build_model
from poke_bot.pure_rl.model_profile import (
    PURE_RL_FULL_GAME_MAX_CONTEXT,
    count_params,
    validate_param_budget,
)
from poke_bot.train import load_model_from_checkpoint


def _zero_temporal_residual_outputs(model: nn.Module) -> list[str]:
    """Make each fresh residual block an exact identity at initialization."""
    zeroed: list[str] = []
    for block_i, block in enumerate(model.temporal_blocks):
        for name, parameter in (
            (f"temporal_blocks.{block_i}.attn.out.weight", block.attn.out.weight),
            (f"temporal_blocks.{block_i}.attn.out.bias", block.attn.out.bias),
        ):
            nn.init.zeros_(parameter)
            zeroed.append(name)
        final_ff = next(
            layer for layer in reversed(block.ff) if isinstance(layer, nn.Linear)
        )
        nn.init.zeros_(final_ff.weight)
        zeroed.append(f"temporal_blocks.{block_i}.ff.final.weight")
        if final_ff.bias is not None:
            nn.init.zeros_(final_ff.bias)
            zeroed.append(f"temporal_blocks.{block_i}.ff.final.bias")
    return zeroed


@torch.no_grad()
def _single_timestep_parity(
    teacher: nn.Module,
    student: nn.Module,
) -> float:
    """Prove the identity-initialized temporal block preserves state behavior."""
    generator = torch.Generator(device="cpu").manual_seed(20260720)
    probe = torch.randn(3, 1, int(teacher.d_model), generator=generator)
    expected, _ = teacher.temporal_encode(
        probe, kv_cache=None, append=False, return_all=True
    )
    actual, _ = student.temporal_encode(
        probe, kv_cache=None, append=False, return_all=True
    )
    return float((expected - actual).abs().max().item())


def build_temporal_history_seed(
    source: Path,
    output: Path,
    *,
    max_context: int = PURE_RL_FULL_GAME_MAX_CONTEXT,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    max_context = int(max_context)
    if not source.is_file():
        raise FileNotFoundError(source)
    if max_context <= 0:
        raise ValueError("temporal warm-start max_context must be positive")

    trusted = checkpoint.assert_trusted_policy_checkpoint(source)
    payload = checkpoint.load_checkpoint(source, map_location="cpu")
    teacher = load_model_from_checkpoint(source, device=torch.device("cpu"))
    if teacher.decision_context != "stateless" or teacher.cfg.temporal_layers != 0:
        raise ValueError(
            "temporal history warm-start requires a zero-layer stateless source"
        )

    history_cfg = replace(
        teacher.cfg,
        temporal_layers=1,
        decision_context="history",
        kv_cache=True,
        max_context=max_context,
        temporal_pos="rope",
    )
    student = build_model(
        history_cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=teacher.aux_head[-1].out_features,
        encoder_vocab=teacher.encoder_vocab,
        decoder_vocab=teacher.decoder_vocab,
        belief_card_vocab=teacher.belief_card_vocab,
    )

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    transferred: list[str] = []
    initialized: list[str] = []
    for key, value in student_state.items():
        candidate = teacher_state.get(key)
        if candidate is not None and tuple(candidate.shape) == tuple(value.shape):
            student_state[key] = candidate.detach().cpu().clone()
            transferred.append(key)
        else:
            initialized.append(key)
    student.load_state_dict(student_state, strict=True)
    zeroed = _zero_temporal_residual_outputs(student)
    teacher.eval()
    student.eval()
    parity_max_abs = _single_timestep_parity(teacher, student)
    if parity_max_abs > 1e-6:
        raise RuntimeError(
            "identity temporal warm-start changed single-step state output: "
            f"max_abs={parity_max_abs:.3g}"
        )

    n_params = count_params(student)
    validate_param_budget(n_params)
    source_digest = checkpoint.checkpoint_digest(source)
    extra = dict(payload.get("extra") or {})
    extra.update(
        {
            "pure_rl": True,
            "smoke": False,
            "model_profile": asdict(history_cfg),
            "temporal_history_warm_start": {
                "source": str(source),
                "source_digest": source_digest,
                "sequence_scope": "acting_seat_game_bounded_rolling_suffix",
                "causal": True,
                "cross_game_attention": False,
                "simulator_game_step_limit": 4000,
                "max_context": max_context,
                "older_prefix_truncated_after": max_context,
                "temporal_layers": 1,
                "kv_cache": True,
                "transferred_tensor_count": len(transferred),
                "initialized_tensor_count": len(initialized),
                "identity_zeroed_tensors": zeroed,
                "single_timestep_parity_max_abs": parity_max_abs,
                "optimizer_state_reset": True,
            },
        }
    )
    out_payload = checkpoint.build_checkpoint(
        model=student,
        step=int(payload.get("step", 0)),
        epoch=int(payload.get("epoch", 0)),
        rl_iteration=int(payload.get("rl_iteration", 0)),
        best_metric=payload.get("best_metric"),
        model_config=history_cfg,
        archetype_id=payload.get("archetype_id"),
        model_id=output.stem,
        extra=extra,
    )
    checkpoint.immutable_torch_save(out_payload, output)
    checkpoint.assert_trusted_policy_checkpoint(output)
    reloaded = load_model_from_checkpoint(output, device=torch.device("cpu"))
    if not (
        reloaded.decision_context == "history"
        and len(reloaded.temporal_blocks) == 1
        and reloaded.kv_cache_enabled
        and reloaded.max_context == max_context
    ):
        raise RuntimeError("temporal history seed failed trusted reload validation")

    return {
        "output": str(output),
        "digest": checkpoint.checkpoint_digest(output),
        "params": n_params,
        "source_digest": source_digest,
        "source_contract": trusted,
        "sequence_scope": "acting_seat_game_bounded_rolling_suffix",
        "max_context": max_context,
        "temporal_layers": 1,
        "kv_cache": True,
        "transferred_tensor_count": len(transferred),
        "initialized_tensor_count": len(initialized),
        "single_timestep_parity_max_abs": parity_max_abs,
        "model_config": asdict(history_cfg),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-context",
        type=int,
        default=PURE_RL_FULL_GAME_MAX_CONTEXT,
    )
    args = parser.parse_args(argv)
    report = build_temporal_history_seed(
        args.source,
        args.output,
        max_context=args.max_context,
    )
    print(json.dumps(report, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
