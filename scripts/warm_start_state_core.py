#!/usr/bin/env python3
"""Create a state-only Pure-RL seed from a compatible history checkpoint.

The board encoder, action decoder, embeddings, and heads are transferred when
their tensor shapes match. Temporal blocks are deliberately dropped. The
result is a new immutable checkpoint that can be behavior-cloned on ladder
replays before it starts a fresh Pure-RL lineage.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint
from poke_bot.model import build_model
from poke_bot.pure_rl.model_profile import count_params, validate_param_budget
from poke_bot.train import load_model_from_checkpoint


def build_state_seed(source: Path, output: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    teacher = load_model_from_checkpoint(source, device=torch.device("cpu"))
    state_cfg = replace(
        teacher.cfg,
        temporal_layers=0,
        decision_context="stateless",
        kv_cache=False,
    )
    student = build_model(
        state_cfg,
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

    n_params = count_params(student)
    validate_param_budget(n_params)
    source_digest = checkpoint.checkpoint_digest(source)
    payload = checkpoint.build_checkpoint(
        model=student,
        model_config=state_cfg,
        archetype_id="core",
        model_id=output.stem,
        extra={
            "pure_rl": True,
            "smoke": False,
            "model_profile": asdict(state_cfg),
            "state_core_warm_start": {
                "source": str(source),
                "source_digest": source_digest,
                "source_decision_context": teacher.decision_context,
                "transferred_tensor_count": len(transferred),
                "initialized_tensor_count": len(initialized),
                "dropped_temporal_tensor_count": sum(
                    key.startswith("temporal_blocks.") for key in teacher_state
                ),
            },
        },
    )
    checkpoint.immutable_torch_save(payload, output)
    checkpoint.assert_trusted_policy_checkpoint(output)
    reloaded = load_model_from_checkpoint(output, device=torch.device("cpu"))
    if reloaded.decision_context != "stateless":
        raise RuntimeError("state seed reloaded with the wrong decision context")

    return {
        "output": str(output),
        "digest": checkpoint.checkpoint_digest(output),
        "params": n_params,
        "source_digest": source_digest,
        "transferred_tensor_count": len(transferred),
        "initialized_tensor_count": len(initialized),
        "dropped_temporal_tensor_count": sum(
            key.startswith("temporal_blocks.") for key in teacher_state
        ),
        "model_config": asdict(state_cfg),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_state_seed(args.source, args.output)
    print(json.dumps(report, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
