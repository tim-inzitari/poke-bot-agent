from poke_bot.pure_rl.model_profile import build_pure_rl_model, pure_rl_model_config


def test_pure_rl_default_is_sub_2m_state_evaluator() -> None:
    cfg = pure_rl_model_config()
    model = build_pure_rl_model(cfg=cfg)

    assert cfg.decision_context == "stateless"
    assert cfg.temporal_layers == 0
    assert cfg.kv_cache is False
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000
