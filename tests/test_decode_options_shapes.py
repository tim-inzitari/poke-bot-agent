"""Shape regression guard for ``TemporalCabtTransformer.decode_options``.

Regression test for the training-path crash where a per-decision 1-D
``state_vec`` (``[d_model]``) was concatenated against a 3-D
``spatial_memory`` (``[B, T, d_model]``). Both the batched inference route
(2-D state) and the per-decision training route (1-D state) must produce
matching option logits.

Run:
    /home/pokebot/miniconda3/envs/poke-bot-agent/bin/python tests/test_decode_options_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.features import SparseVector
from poke_bot.model import build_model


def _make_options(n_words: int) -> SparseVector:
    sv = SparseVector()
    for w in range(n_words):
        sv.word_start()
        sv.add((w * 7 + 3) % 32, 1.0)
    return sv


def test_decode_options_training_and_inference_shapes() -> None:
    torch.manual_seed(0)
    model = build_model(device=torch.device("cpu"))
    model.eval()
    d = model.d_model
    n_opt = 4
    opts = _make_options(n_opt)
    spatial_1 = torch.randn(1, 24, d)

    # Training route: state_vec is 1-D [d_model].
    state_1d = torch.randn(d)
    logits_train = model.decode_options(opts, spatial_1, state_1d, n_options=[n_opt])
    assert logits_train.shape == (1, n_opt), logits_train.shape

    # Inference route: state_vec is 2-D [B, d_model].
    state_2d = state_1d.unsqueeze(0)
    logits_infer = model.decode_options(opts, spatial_1, state_2d, n_options=[n_opt])
    assert logits_infer.shape == (1, n_opt), logits_infer.shape

    # Both routes must agree numerically (same underlying state).
    assert torch.allclose(logits_train, logits_infer, atol=1e-5), (
        logits_train, logits_infer,
    )

    # Batched inference: B=3 boards/options/states.
    b = 3
    batch_opts = [_make_options(n_opt) for _ in range(b)]
    spatial_b = torch.randn(b, 24, d)
    state_b = torch.randn(b, d)
    logits_b = model.decode_options(
        batch_opts, spatial_b, state_b, n_options=[n_opt] * b
    )
    assert logits_b.shape == (b, n_opt), logits_b.shape

    # Guard fires loudly on a batch mismatch.
    mismatched = False
    try:
        model.decode_options(batch_opts, torch.randn(2, 24, d), state_b, n_options=[n_opt] * b)
    except AssertionError:
        mismatched = True
    assert mismatched, "expected AssertionError on batch mismatch"

    print("OK: decode_options shapes (training 1-D, inference 2-D, batched, guard)")


if __name__ == "__main__":
    test_decode_options_training_and_inference_shapes()
