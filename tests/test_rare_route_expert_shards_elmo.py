from pathlib import Path


def test_rare_route_expert_shards_are_additive_and_bounded() -> None:
    source = Path("scripts/prepare_rare_route_expert_shards_elmo.py").read_text()
    assert "rare-route-expert-history-20260626-20260701" in source
    assert "--cpus" in source
    assert '"4"' in source
    assert "--memory" in source
    assert '"24g"' in source
    assert "/workspace/data/training_mixes:ro" in source
    assert "pokemon-tcg-ai-battle-episodes-index:ro" in source
    assert "_validated_jsonl(day)" in source
    assert "output_sha256" in source
    assert "collect_top_ladder_replays.py:ro" in source
    assert "featurize_bootstrap_shard.py:ro" in source
    assert "feature_shards.py:ro" in source
    assert "--compact-mode temporal-expert-v1" in source
    assert "--max-context 320" in source
    assert "--recognized-only" in source
    assert "--additive-archetype" in source
    assert "--min-recognized-seat-frac 0.0" in source
    assert "MIN_RECOGNIZED_RECORDS_PER_DAY = 5000" in source
    assert "rare_route_expert_shard_identity/v1" in source
    assert "additive_allowed_archetypes_v2" in source
    assert "EXPECTED_ADDITIVE_BY_DAY" in source
    assert "observed_additive_archetypes" in source
    assert "additive_registered_ids" in source
    assert "pokebot.managed-unit=rare-route-expert-shards-v1" in source
    assert "live_corpus_modified" in source
    assert "specialist-corpora-v1" not in source
    unit = Path(
        "deploy/systemd/pokebot-rare-route-expert-shards-v1.service"
    ).read_text()
    assert "ExecStopPost=" in unit
    assert "--filter name=pokebot-rare-expert-" in unit
