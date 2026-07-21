import torch

from poke_bot import config, features
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.model import build_model
from poke_bot.train import batch_losses, device_batch_losses


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    sv = features.SparseVector()
    for word in range(words):
        sv.word_start()
        sv.add((offset + word) % 48, 1.0 + word / 100.0)
        if word % 2:
            sv.add((offset + word + 7) % 48, 0.25)
    return sv


def _decision(index: int, *, staged: bool) -> DecisionSample:
    options = _sparse(2, index + 3)
    stages = []
    if staged:
        stages = [
            PolicyStage(options=options, action_combos=[[0], [1]], target_index=1),
            PolicyStage(
                options=_sparse(3, index + 9),
                action_combos=[[0], [1], [2]],
                target_index=2,
            ),
        ]
    return DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS, index),
        options=options,
        action=[index % 2],
        action_combo_index=index % 2,
        action_combos=[[0], [1]],
        env_step=index,
        policy_stages=stages,
    )


def _game() -> GameSequence:
    return GameSequence(
        episode_id="device-corpus",
        seat=0,
        archetype="",
        opp_archetype="unknown",
        deck=[1] * 60,
        value=1.0,
        decisions=[_decision(0, staged=True), _decision(1, staged=False)],
    )


def test_device_resident_loss_matches_existing_stateless_hard_target_path() -> None:
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=0,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="stateless",
        kv_cache=False,
        dense_card2vec=False,
        dropout=0.0,
    )
    torch.manual_seed(7)
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
    )
    model.eval()
    game = _game()
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        [game], [], device=torch.device("cpu")
    )
    sample_ids = torch.arange(corpus.train_samples)

    reference, reference_metrics = batch_losses(
        model,
        [game],
        value_weight=1.0,
        aux_weight=0.0,
        opp_hand_weight=0.0,
        opp_remainder_weight=0.0,
        lethal_threat_weight=0.0,
        prize_race_weight=0.0,
    )
    resident, resident_metrics = device_batch_losses(
        model, corpus, sample_ids, value_weight=1.0
    )

    assert corpus.decisions == 2
    assert corpus.train_samples == 3
    torch.testing.assert_close(resident, reference, rtol=1e-5, atol=1e-6)
    assert resident_metrics.n_decisions == reference_metrics.n_decisions == 3
    assert resident_metrics.policy_acc == reference_metrics.policy_acc
    assert abs(resident_metrics.policy_loss - reference_metrics.policy_loss) < 1e-5
    assert abs(resident_metrics.value_loss - reference_metrics.value_loss) < 1e-5
