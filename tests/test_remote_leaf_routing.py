import torch

from poke_bot.batched_infer import LeafPacket
from poke_bot.mcts import LeafEvaluator


def test_leaf_evaluator_uses_remote_backend_without_local_model() -> None:
    calls: list[list[LeafPacket]] = []

    def remote(packets):
        calls.append(list(packets))
        return [
            LeafPacket(
                obs=packet.obs,
                your_deck=packet.your_deck,
                root_seat=packet.root_seat,
                value=0.25,
                priors=[0.75, 0.25],
                combos=[[0], [1]],
            )
            for packet in packets
        ]

    evaluator = LeafEvaluator(
        None,
        [10, 20],
        [30, 40],
        root_seat=1,
        device=torch.device("cpu"),
        leaf_backend=remote,
    )

    value, priors, combos = evaluator.evaluate_one(object())

    assert len(calls) == 1
    assert value == 0.25
    assert priors == [0.75, 0.25]
    assert combos == [[0], [1]]
