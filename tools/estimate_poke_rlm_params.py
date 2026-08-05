#!/usr/bin/env python3
"""Approximate PokeRLM parameter and parameter-memory budgets.

This estimates standard transformer-style modules. It is a planning tool, not a
replacement for counting the instantiated repository model with p.numel().
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    d_model: int
    encoder_layers: int
    action_decoder_layers: int
    planner_layers: int
    dynamics_blocks: int
    embedding_rows: int
    n_heads: int = 8
    ffn_multiplier: int = 4
    dynamics_hidden_multiplier: int = 2
    plan_vocab_size: int = 256
    q_quantiles: int = 32
    q_bootstrap_heads: int = 4
    value_horizons: int = 5
    successor_dim: int = 128
    branch_types: int = 16
    belief_dim: int = 128


PROFILES = {
    "pilot_256": Profile(
        name="pilot_256",
        d_model=256,
        encoder_layers=8,
        action_decoder_layers=2,
        planner_layers=2,
        dynamics_blocks=3,
        embedding_rows=8_000,
    ),
    "base_384": Profile(
        name="base_384",
        d_model=384,
        encoder_layers=10,
        action_decoder_layers=2,
        planner_layers=3,
        dynamics_blocks=4,
        embedding_rows=9_000,
    ),
    "strong_512": Profile(
        name="strong_512",
        d_model=512,
        encoder_layers=12,
        action_decoder_layers=2,
        planner_layers=3,
        dynamics_blocks=4,
        embedding_rows=10_000,
    ),
}


def encoder_layer_params(d: int, ffn_mult: int) -> int:
    # Self-attention: QKV + output = 4D^2.
    # FFN: D -> mD -> D = 2mD^2. Remaining terms approximate biases/norms.
    return 4 * d * d + 2 * ffn_mult * d * d + (ffn_mult + 13) * d


def decoder_layer_params(d: int, ffn_mult: int) -> int:
    # Self-attention + cross-attention = 8D^2; FFN = 2mD^2.
    return 8 * d * d + 2 * ffn_mult * d * d + (ffn_mult + 19) * d


def dynamics_block_params(d: int, hidden_mult: int) -> int:
    # Residual MLP: D -> hD -> D plus approximate biases/norms.
    return 2 * hidden_mult * d * d + (hidden_mult + 5) * d


def estimate(profile: Profile) -> dict[str, int | float | str]:
    d = profile.d_model
    embeddings = profile.embedding_rows * d
    encoder = profile.encoder_layers * encoder_layer_params(d, profile.ffn_multiplier)
    action_decoder = profile.action_decoder_layers * decoder_layer_params(d, profile.ffn_multiplier)
    planner = profile.planner_layers * decoder_layer_params(d, profile.ffn_multiplier)
    dynamics = profile.dynamics_blocks * dynamics_block_params(d, profile.dynamics_hidden_multiplier)

    heads = (
        profile.plan_vocab_size * d
        + (d + 1)  # policy
        + d * (profile.q_quantiles * profile.q_bootstrap_heads)
        + profile.q_quantiles * profile.q_bootstrap_heads
        + d * profile.value_horizons
        + profile.value_horizons
        + d * profile.successor_dim
        + profile.successor_dim
        + (d + 1)  # state value
        + d * profile.branch_types
        + profile.branch_types
        + (d + 1)  # stop
        + (d + 1)  # complexity router
        + d * profile.belief_dim
        + profile.belief_dim
    )

    total = embeddings + encoder + action_decoder + planner + dynamics + heads
    attachment = action_decoder + planner + dynamics + heads

    return {
        "profile": profile.name,
        "embeddings": embeddings,
        "state_encoder": encoder,
        "action_decoder": action_decoder,
        "recursive_planner": planner,
        "latent_dynamics": dynamics,
        "heads": heads,
        "total": total,
        "new_attachment": attachment,
        "bf16_weight_mb": total * 2 / 1_000_000,
        "fp32_weight_mb": total * 4 / 1_000_000,
        "rough_16_bytes_per_param_gb": total * 16 / 1_000_000_000,
    }


def human_count(value: int) -> str:
    return f"{value / 1_000_000:.2f}M"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="base_384")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    result = estimate(profile)

    if args.json:
        print(json.dumps({"configuration": asdict(profile), "estimate": result}, indent=2))
        return 0

    print(f"PokeRLM approximate parameter budget: {profile.name}")
    print(f"  width / encoder layers: {profile.d_model} / {profile.encoder_layers}")
    for key in (
        "embeddings",
        "state_encoder",
        "action_decoder",
        "recursive_planner",
        "latent_dynamics",
        "heads",
    ):
        print(f"  {key:20s} {human_count(int(result[key])):>9s}")
    print("  " + "-" * 30)
    print(f"  {'total':20s} {human_count(int(result['total'])):>9s}")
    print(f"  {'new_attachment':20s} {human_count(int(result['new_attachment'])):>9s}")
    print()
    print(f"  BF16 weights: {result['bf16_weight_mb']:.1f} MB")
    print(f"  FP32 weights: {result['fp32_weight_mb']:.1f} MB")
    print(
        "  Rough 16-byte/parameter training-state floor: "
        f"{result['rough_16_bytes_per_param_gb']:.2f} GB"
    )
    print()
    print("Confirm the real instantiated model with:")
    print("  sum(p.numel() for p in model.parameters())")
    print("  sum(p.numel() for p in model.parameters() if p.requires_grad)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
