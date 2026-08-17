from poke_bot.pure_rl.model_profile import (
    build_pure_rl_model,
    pure_rl_history_model_config,
    pure_rl_model_config,
)


def test_pure_rl_default_is_sub_2m_state_evaluator() -> None:
    cfg = pure_rl_model_config()
    model = build_pure_rl_model(cfg=cfg)

    assert cfg.decision_context == "stateless"
    assert cfg.temporal_layers == 0
    assert cfg.kv_cache is False
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_history_profile_is_one_layer_game_bounded_causal_model() -> None:
    cfg = pure_rl_history_model_config()
    model = build_pure_rl_model(cfg=cfg)

    assert cfg.decision_context == "history"
    assert cfg.temporal_layers == 1
    assert cfg.kv_cache is True
    assert cfg.max_context == 320
    assert model.kv_cache_enabled is True
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000
