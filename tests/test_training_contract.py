import torch

from poke_bot import config, features
from poke_bot.agent import PolicyAgent
from poke_bot.dataset import DecisionSample, GameSequence, PolicyStage
from poke_bot.model import build_model
from poke_bot.train import batch_losses


def _sparse(words: int, offset: int = 0) -> features.SparseVector:
    sv = features.SparseVector()
    for i in range(words):
        sv.word_start()
        sv.add((offset + i) % 32, 1.0)
    return sv


def _decision(index: int) -> DecisionSample:
    return DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS, index),
        options=_sparse(2, index + 3),
        action=[index % 2],
        action_combo_index=index % 2,
        action_combos=[[0], [1]],
        env_step=index,
        action_token=_sparse(1, index + 7),
    )


def _small_model():
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        dropout=0.0,
    )
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
    )
    model.eval()
    return model


def test_training_and_incremental_inference_share_history_contract() -> None:
    torch.manual_seed(0)
    model = _small_model()
    decisions = [_decision(0), _decision(1)]
    seq = GameSequence(
        episode_id="e",
        seat=0,
        archetype="",
        opp_archetype="unknown-baseline-id",
        deck=[1] * 60,
        value=1.0,
        decisions=decisions,
    )

    seen_temporal_lengths = []

    def _capture(_module, args):
        seen_temporal_lengths.append(args[0].shape[1])

    hook = model.temporal_blocks[0].register_forward_pre_hook(_capture)
    try:
        loss, metrics = batch_losses(
            model,
            [seq],
            aux_weight=1.0,
            history_identity_weight=1.0,
        )
    finally:
        hook.remove()

    assert torch.isfinite(loss)
    assert seen_temporal_lengths == [2]
    assert metrics.n_decisions == 2
    assert metrics.aux_loss == 0.0  # unknown opponent ids are masked
    assert metrics.history_identity_loss >= 0.0
    assert metrics.target_value_mean == 1.0

    offline_states, offline_spatial = model.encode_history(
        [d.board for d in decisions],
        return_all=True,
        previous_actions=[None, decisions[0].action_token],
    )
    cache = None
    previous_action = None
    for i, decision in enumerate(decisions):
        offline_logits = model.decode_options(
            decision.options,
            offline_spatial[i],
            offline_states[i],
            n_options=[2],
        )
        offline_value = torch.tanh(model.value_head(offline_states[i])).squeeze(-1)
        incremental = model.forward(
            decision.board,
            decision.options,
            kv_cache=cache,
            append_cache=True,
            n_options=[2],
            previous_action=previous_action,
        )
        cache = incremental["kv_cache"]
        previous_action = decision.action_token
        assert torch.allclose(
            offline_logits[0], incremental["policy_logits"][0], atol=1e-5
        )
        assert torch.allclose(
            offline_value, incremental["value"][0], atol=1e-5
        )
    assert cache is not None and cache.length == 2


def test_training_supervises_each_factorized_policy_stage() -> None:
    model = _small_model()
    decision = _decision(0)
    decision.policy_stages = [
        PolicyStage(
            options=_sparse(3, 4),
            action_combos=[[0], [1], [2]],
            target_index=1,
        ),
        PolicyStage(
            options=_sparse(2, 8),
            action_combos=[[1, 0], [1]],
            target_index=0,
        ),
    ]
    seq = GameSequence(
        episode_id="factorized",
        seat=0,
        archetype="",
        opp_archetype="unknown-baseline-id",
        deck=[1] * 60,
        value=0.0,
        decisions=[decision],
    )
    loss, metrics = batch_losses(model, [seq], aux_weight=0.0)
    assert torch.isfinite(loss)
    assert metrics.n_decisions == 2


def test_legacy_stateless_contract_rejects_kv_append() -> None:
    cfg = _small_model().cfg
    cfg.decision_context = "stateless"
    cfg.kv_cache = False
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
    )
    decision = _decision(0)
    try:
        model.forward(decision.board, decision.options, append_cache=True)
    except ValueError as exc:
        assert "stateless" in str(exc)
    else:
        raise AssertionError("KV append must be rejected")


def test_agent_reset_clears_history_and_cache() -> None:
    model = _small_model()
    agent = PolicyAgent(model=model, deck=[1] * 60, use_mcts=False)
    agent.board_history.append(_sparse(features.NUM_BOARD_TOKENS))
    agent._kv_cache = model.forward(
        _decision(0).board,
        _decision(0).options,
        append_cache=True,
        n_options=[2],
    )["kv_cache"]
    assert agent._kv_cache is not None
    agent.reset_game()
    assert agent.board_history == []
    assert agent._kv_cache is None


def test_remote_leaf_agent_honors_full_game_context_override() -> None:
    agent = PolicyAgent(
        model=None,
        deck=[1] * 60,
        use_mcts=False,
        leaf_backend=lambda _packets: [],
        max_context_override=4096,
    )
    assert agent._history_context_limit() == 4096
