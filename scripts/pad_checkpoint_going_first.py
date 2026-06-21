#!/usr/bin/env python3
"""One-off: pad a 282-dim checkpoint to 283 for going_first smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# going_first sits after the first 10 legacy base features, before derived hand-tracking.
GOING_FIRST_INDEX = 10
GOING_FIRST_MEAN = 0.5
GOING_FIRST_STD = 1.0


def insert_feature_axis(array: np.ndarray, index: int, value: float) -> np.ndarray:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    return np.insert(flat, index, value)


def pad_state_dict_for_going_first(state_dict: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    out = dict(state_dict)
    # token_proj.weight: [d_model, input_dim]
    weight = out["token_proj.weight"]
    padded = torch.zeros((weight.shape[0], weight.shape[1] + 1), dtype=weight.dtype)
    padded[:, :index] = weight[:, :index]
    padded[:, index + 1 :] = weight[:, index:]
    out["token_proj.weight"] = padded

    # next_feature_head.2.weight: [input_dim, d_model]
    nf_weight = out["next_feature_head.2.weight"]
    nf_padded = torch.zeros((nf_weight.shape[0] + 1, nf_weight.shape[1]), dtype=nf_weight.dtype)
    nf_padded[:index] = nf_weight[:index]
    nf_padded[index + 1 :] = nf_weight[index:]
    out["next_feature_head.2.weight"] = nf_padded

    # next_feature_head.2.bias: [input_dim]
    nf_bias = out["next_feature_head.2.bias"]
    nf_bias_padded = torch.zeros(nf_bias.shape[0] + 1, dtype=nf_bias.dtype)
    nf_bias_padded[:index] = nf_bias[:index]
    nf_bias_padded[index + 1 :] = nf_bias[index:]
    out["next_feature_head.2.bias"] = nf_bias_padded
    return out


def pad_checkpoint(checkpoint: dict, *, index: int = GOING_FIRST_INDEX) -> dict:
    old_dim = int(checkpoint["input_dim"])
    if old_dim != 282:
        raise ValueError(f"expected 282-dim checkpoint, got {old_dim}")

    padded = dict(checkpoint)
    padded["input_dim"] = old_dim + 1
    padded["coarse_feature_dim"] = int(checkpoint.get("coarse_feature_dim", 26)) + 1
    padded["feature_mean"] = insert_feature_axis(checkpoint["feature_mean"], index, GOING_FIRST_MEAN).tolist()
    padded["feature_std"] = insert_feature_axis(checkpoint["feature_std"], index, GOING_FIRST_STD).tolist()
    padded["model_state_dict"] = pad_state_dict_for_going_first(checkpoint["model_state_dict"], index)
    return padded


def main() -> None:
    parser = argparse.ArgumentParser(description="Pad 282-dim checkpoint to 283 with neutral going_first weights.")
    parser.add_argument("--input", type=Path, default=Path("outputs/checkpoints/temporal_current.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/checkpoints/temporal_current_283_smoke.pt"))
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    padded = pad_checkpoint(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(padded, args.output)
    print(f"padded {args.input} -> {args.output} (input_dim {checkpoint['input_dim']} -> {padded['input_dim']})")


if __name__ == "__main__":
    main()
