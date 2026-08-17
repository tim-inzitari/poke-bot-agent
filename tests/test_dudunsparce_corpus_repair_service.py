from pathlib import Path


def test_dudunsparce_repair_is_isolated_and_rebuilds_current_targets() -> None:
    source = Path("ops/elmo/run_dudunsparce_corpus_repair_v1.sh").read_text()
    assert "dudunsparce-verified-visual-schema-v2" in source
    assert "rare-route-expert-history-20260626-20260701" in source
    assert "--expected-records 29" in source
    for card_id in (646, 647, 648):
        assert f"--forbid-card-id {card_id}" in source
    assert "--cpus 20" in source
    assert "--memory 48g" in source
    assert "--compact-mode temporal-expert-v1" in source
    assert "--seal-protected" in source
    assert "scripts/featurize_bootstrap_shard.py:/workspace/scripts/featurize_bootstrap_shard.py:ro" in source
    assert "scripts/assemble_feature_manifest.py:/workspace/scripts/assemble_feature_manifest.py:ro" in source
    assert "poke_bot/authoritative_visual_trace.py:/workspace/poke_bot/authoritative_visual_trace.py:ro" in source
    assert "--require-target-coverage opponent_hand_rows" in source
    assert "--require-target-coverage opponent_private_prize_rows" in source
    assert "pokebot.managed-unit=dudunsparce-corpus-repair-v1" in source
