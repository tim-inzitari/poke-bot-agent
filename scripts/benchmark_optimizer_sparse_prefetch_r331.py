#!/usr/bin/env python3
"""Non-authorizing parity/throughput probe for the revision-28 feed path."""

from __future__ import annotations

import argparse
import copy
import json
import time

import torch

from poke_bot import config, features
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.model import build_model
from poke_bot.train import (
    batch_losses,
    iter_prefetched_sparse_batches,
    prepare_host_sparse_batch,
)


def sparse(words: int, offset: int) -> features.SparseVector:
    value = features.SparseVector()
    for word in range(words):
        value.word_start()
        value.add((offset + word) % 512, 1.0)
        value.add((offset * 7 + word * 13 + 1) % 512, 0.25)
    return value


def decision(index: int, options: int) -> DecisionSample:
    combos = [[option] for option in range(options)]
    return DecisionSample(
        board=sparse(features.NUM_BOARD_TOKENS, index),
        options=sparse(options, index + 17),
        action=[index % options],
        action_combo_index=index % options,
        action_combos=combos,
        env_step=index,
        action_token=sparse(1, index + 41),
        policy_stages=[
            PolicyStage(
                options=sparse(options, index + 17),
                action_combos=combos,
                target_index=index % options,
            )
        ],
    )


def sequences(games: int, decisions: int, options: int) -> list[GameSequence]:
    return [
        GameSequence(
            episode_id=f"r331-synthetic-{game}",
            seat=game % 2,
            archetype="alakazam",
            opp_archetype="mirror",
            deck=[743] * 60,
            value=1.0 if game % 2 == 0 else -1.0,
            decisions=[
                decision(game * decisions + row, options)
                for row in range(decisions)
            ],
        )
        for game in range(games)
    ]


def model(device: torch.device):
    cfg = config.ModelConfig(
        d_model=96,
        spatial_layers=2,
        temporal_layers=2,
        option_decoder_layers=2,
        n_heads=4,
        ff_dim=384,
        max_context=128,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        dropout=0.0,
    )
    return build_model(
        cfg,
        device=device,
        aux_archetype_classes=3,
        encoder_vocab=512,
        decoder_vocab=512,
        belief_card_vocab=512,
    )


def run_step(net, optimizer, games, prepared=None):
    optimizer.zero_grad(set_to_none=True)
    predictions: list[int] = []
    loss, metrics = batch_losses(
        net,
        games,
        pure_rl=True,
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        prediction_sink=predictions,
        prepared_sparse_batch=prepared,
    )
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in net.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    return loss.detach().float().cpu(), metrics, predictions, gradients


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--decisions", type=int, default=16)
    parser.add_argument("--options", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(331)
    reference = model(device)
    candidate = model(device)
    candidate.load_state_dict(reference.state_dict())
    reference.train()
    candidate.train()
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4)
    cand_optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4)
    batch = sequences(args.games, args.decisions, args.options)
    host = prepare_host_sparse_batch(
        batch,
        board_words=reference.num_board_tokens,
        card_vocab=reference.belief_card_vocab,
        pin_memory=True,
    )
    prepared = host.to_device(device, non_blocking=True)
    torch.cuda.synchronize(device)
    ref_loss, ref_metrics, ref_pred, ref_grad = run_step(
        reference, ref_optimizer, batch
    )
    cand_loss, cand_metrics, cand_pred, cand_grad = run_step(
        candidate, cand_optimizer, batch, prepared
    )
    torch.cuda.synchronize(device)
    max_grad_abs = max(
        float((ref_grad[name] - cand_grad[name]).abs().max().item())
        for name in ref_grad
    )
    max_param_abs = max(
        float((left.detach() - right.detach()).abs().max().item())
        for left, right in zip(reference.parameters(), candidate.parameters())
    )

    def timed(prefetched: bool) -> float:
        net = model(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
        batches = [batch for _ in range(args.steps)]
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if prefetched:
            iterator = iter_prefetched_sparse_batches(
                batches,
                board_words=net.num_board_tokens,
                card_vocab=net.belief_card_vocab,
                device=device,
            )
            for games, packed in iterator:
                run_step(net, optimizer, games, packed)
        else:
            for games in batches:
                run_step(net, optimizer, games)
        torch.cuda.synchronize(device)
        return time.perf_counter() - started

    reference_seconds = timed(False)
    candidate_seconds = timed(True)
    report = {
        "schema": "poke_bot.optimizer_sparse_prefetch_r331_benchmark/v1",
        "device": torch.cuda.get_device_name(device),
        "games": args.games,
        "decisions_per_game": args.decisions,
        "options_per_decision": args.options,
        "steps": args.steps,
        "loss_reference": float(ref_loss.item()),
        "loss_candidate": float(cand_loss.item()),
        "loss_abs_diff": float(abs(ref_loss.item() - cand_loss.item())),
        "selected_predictions_identical": cand_pred == ref_pred,
        "metric_rows_identical": cand_metrics.n_decisions == ref_metrics.n_decisions,
        "max_gradient_abs_diff": max_grad_abs,
        "max_parameter_abs_diff_after_update": max_param_abs,
        "reference_seconds": reference_seconds,
        "candidate_seconds": candidate_seconds,
        "speedup": reference_seconds / candidate_seconds,
        "authorizes_activation": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["selected_predictions_identical"]:
        return 2
    if report["loss_abs_diff"] > 1e-5 or max_grad_abs > 1e-4 or max_param_abs > 1e-5:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
