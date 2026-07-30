from __future__ import annotations

import json
from pathlib import Path

from scripts import prestage_next_specialist as prestage


def test_prestage_contract_cannot_control_live_training() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/prestage_next_specialist.py"
    ).read_text(encoding="utf-8")
    assert "systemctl" not in source
    assert "subprocess" not in source
    assert "_atomic_selector" not in source
    assert "register_specialist_runtime" not in source
    assert '"live_training_modified": False' in source
    assert "validate_corpus_source_contract" in source
    assert '"source_contract": source_contract' in source
    assert "selected expert corpus source identity changed" in source
    assert '"source": "current_runtime_tree"' in source
    assert "protocol_valid_expert_corpus_not_ready" in source
    assert "expanded_strategic_corpus_not_ready" in source
    assert "_manifest_expanded_targets" in source
    assert "_deck_guide_contract" in source
    assert '"current_deck_guide": deck_guide' in source
    assert '"terminal_preflight": terminal_preflight' in source


def test_cycle_contract_pins_read_only_prestage() -> None:
    root = Path(__file__).parents[1]
    contract = json.loads(
        (root / "ops/specialist_cycle_handoff_v1.json").read_text(
            encoding="utf-8"
        )
    )
    stage = contract["prestage"]
    assert contract["selection"]["state"] == (
        "/home/inzi/poke-bot-agent/state/specialists.yaml"
    )
    assert stage["live_training_modification_allowed"] is False
    assert stage["selector_update_allowed"] is False
    assert stage["service_control_allowed"] is False
    assert stage["required_target_coverage"] == list(prestage.TARGETS)
    assert stage["current_deck_guide_required"] is True
    assert stage["current_deck_guide_filtered_expert_rows_required"] is True
    assert stage["cpu_pack_workers"] == 4
    assert stage["cpu_pack_memory_reserve_gib"] == 12.0
    assert stage["cpu_pack_disk_reserve_gib"] == 16.0
    expanded = contract["training"]["expanded_heads"]
    assert expanded["architecture_schema"] == (
        "poke_bot.expanded_strategic_heads/v1"
    )
    assert expanded["target_schema"] == (
        "poke_bot.expanded_strategic_targets/v2"
    )
    assert stage["ladder_representatives"].endswith(
        "top_ladder_representatives.v1.json"
    )


def test_prestage_service_is_resource_bounded_and_periodic() -> None:
    root = Path(__file__).parents[1]
    service = (
        root / "deploy/systemd/pokebot-next-specialist-prestage.service"
    ).read_text(encoding="utf-8")
    timer = (
        root / "deploy/systemd/pokebot-next-specialist-prestage.timer"
    ).read_text(encoding="utf-8")
    assert "--build-cpu-pack" in service
    assert "MemoryMax=28G" in service
    assert "CPUQuota=400%" in service
    assert "Nice=15" in service
    assert (
        "WorkingDirectory=/home/inzi/poke-bot-agent-deployments/"
        "specialist-handoff-current"
    ) in service
    assert "pure-rl-resident-v41" not in service
    assert "OnActiveSec=1min" in timer
    assert "OnUnitActiveSec=30min" in timer


def test_latest20_sync_triggers_immediate_read_only_prestage() -> None:
    root = Path(__file__).parents[1]
    service = (
        root
        / "deploy/systemd/pokebot-expert-latest20-specialist-sync.service"
    ).read_text(encoding="utf-8")
    assert "OnSuccess=pokebot-next-specialist-prestage.service" in service


def test_hammer_guide_pipeline_seals_promotes_and_refreshes_prestage() -> None:
    root = Path(__file__).parents[1]
    window = (
        root
        / "deploy/systemd/pokebot-hammer-pult-guide-window-v1.service"
    ).read_text(encoding="utf-8")
    finalizer = (
        root
        / "deploy/systemd/pokebot-hammer-pult-guide-finalize-v1.service"
    ).read_text(encoding="utf-8")
    promotion = (
        root
        / "deploy/systemd/pokebot-hammer-pult-guide-promote-v1.service"
    ).read_text(encoding="utf-8")
    timer = (
        root
        / "deploy/systemd/pokebot-hammer-pult-guide-promote-v1.timer"
    ).read_text(encoding="utf-8")

    assert "OnSuccess=pokebot-hammer-pult-guide-finalize-v1.service" in window
    assert "--required-archetype hammer-pult" in window
    assert "--current-deck-guide hammer-pult" in window
    assert "--start 2026-07-04 --end 2026-07-23" in window
    assert "--specialist-id hammer-pult" in finalizer
    assert "--guide-version hammer-pult-north-star-v1" in finalizer
    assert "CURRENT_DECK_GUIDE_CORPUS_READY.json" in promotion
    assert "--specialist-id hammer-pult" in promotion
    assert (
        "ExecStartPost=/usr/bin/systemctl --user start --no-block "
        "pokebot-next-specialist-prestage.service"
    ) in promotion
    assert "OnUnitActiveSec=5min" in timer
    assert "Unit=pokebot-hammer-pult-guide-promote-v1.service" in timer


def test_missing_representative_is_an_explicit_blocker(tmp_path: Path) -> None:
    registry = tmp_path / "representatives.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_deck_representatives/v1",
                "decks": {},
            }
        ),
        encoding="utf-8",
    )
    result = prestage._representative(registry, "future-specialist")
    assert result["ready"] is False
    assert result["reason"] == "exact_60_card_representative_missing"


def test_exact_representative_is_checksum_bound(tmp_path: Path) -> None:
    registry = tmp_path / "representatives.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_deck_representatives/v1",
                "decks": {"future-specialist": {"card_ids": list(range(60))}},
            }
        ),
        encoding="utf-8",
    )
    result = prestage._representative(registry, "future-specialist")
    assert result["ready"] is True
    assert result["card_count"] == 60
    assert result["cards_sha256"].startswith("sha256:")


def test_exact_representative_accepts_canonical_logical_alias(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "representatives.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_deck_representatives/v1",
                "decks": {"festival-lead": {"card_ids": list(range(60))}},
            }
        ),
        encoding="utf-8",
    )

    result = prestage._representative(
        registry,
        "thwackey",
        logical_aliases={"festival-lead": "thwackey"},
    )

    assert result["ready"] is True
    assert result["logical_specialist_id"] == "thwackey"
    assert result["resolved_deck_id"] == "festival-lead"


def test_ready_receipt_identity_ignores_only_timer_cache_telemetry() -> None:
    left = {
        "schema": prestage.SCHEMA,
        "status": "ready",
        "created_at_utc": "2026-07-27T20:00:00Z",
        "selected_specialist": "thwackey",
        "cpu_pack": {
            "key": "pack-key",
            "payload_sha256": "sha256:" + "a" * 64,
            "cache_hit": False,
            "elapsed_sec": 14.0,
        },
    }
    right = {
        **left,
        "created_at_utc": "2026-07-27T21:00:00Z",
        "cpu_pack": {
            **left["cpu_pack"],
            "cache_hit": True,
            "elapsed_sec": 0.2,
        },
    }

    assert prestage._stable_receipt_identity(left) == (
        prestage._stable_receipt_identity(right)
    )
    right["cpu_pack"]["payload_sha256"] = "sha256:" + "b" * 64
    assert prestage._stable_receipt_identity(left) != (
        prestage._stable_receipt_identity(right)
    )


def test_deck_guide_contract_requires_materialized_expert_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path
    guide_root = root / "config/deck_guides"
    guide_root.mkdir(parents=True)
    writeup = root / "docs/deck_guides/future-specialist-expert-brief.md"
    writeup.parent.mkdir(parents=True)
    writeup.write_text(
        "# Future specialist\n\nSource: https://example.test/guide\n",
        encoding="utf-8",
    )
    writeup_sha256 = prestage.sha256(writeup)
    writeup_words = len(writeup.read_text(encoding="utf-8").split())
    (guide_root / "future-specialist.yaml").write_text(
        f"""
schema_version: poke_bot.current_deck_guide/v1
specialist_id: future-specialist
guide_version: future-v1
teacher_module: poke_bot.future
strategy_sources:
  - url: https://example.test/guide
    reviewed_at_utc: 2026-07-25T00:00:00Z
expert_writeup:
  path: docs/deck_guides/future-specialist-expert-brief.md
  sha256: {writeup_sha256}
  word_count: {writeup_words}
  maximum_words: 10000
  audience: world_champion_subject_matter_experts
  guide_identity: future-specialist
  cites_same_strategy_source_set: true
validation:
  unit_tests_passed: true
  scorer_canary_passed: true
  guide_rows_in_filtered_expert_corpus: null
""",
        encoding="utf-8",
    )
    corpus = root / "corpus"
    corpus.mkdir()
    manifest = corpus / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "totals": {
                    "decisions_kept": 10,
                    "target_coverage": {"guide_rows": 0},
                }
            }
        ),
        encoding="utf-8",
    )
    pointer = corpus / "PROTECTED_EXPERT_CORPUS.json"
    pointer.write_text(
        json.dumps(
            {
                "schema": "poke_bot.pinned_expert_corpus/v1",
                "protected": True,
                "manifest": "manifest.json",
                "manifest_sha256": prestage.sha256(manifest),
            }
        ),
        encoding="utf-8",
    )
    result = prestage._deck_guide_contract(
        root,
        "future-specialist",
        corpus_pointer=pointer,
        corpus_manifest=manifest,
    )
    assert result["implementation_ready"] is True
    assert result["targets_ready"] is False
    assert result["status"] == "staged"
    assert (
        result["reason"]
        == "current_deck_guide_corpus_binding_not_ready"
    )


def test_active_runtime_tree_comes_from_checksum_bound_registry(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "active-tree.json"
    tree.write_text(
        json.dumps(
            {
                "schema": "poke_bot.public_matchup_decision_tree/v1",
                "runtime_enabled": True,
                "runtime_contract": {
                    "schema": (
                        "poke_bot.public_matchup_tree_runtime_activation/v1"
                    ),
                    "accepted_archetype_ids": ["dudunsparce"],
                },
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "runtime-registry.json"
    registry.write_text(
        json.dumps(
            {
                "specialists": {
                    "dragapult-dusknoir": {
                        "status": "ready",
                        "matchup_runtime_tree": str(tree),
                        "matchup_runtime_tree_sha256": (
                            prestage.sha256(tree).removeprefix("sha256:")
                        ),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    resolved, source = prestage._active_runtime_tree(
        {"runtime_registry": str(registry)},
        active_id="dragapult-dusknoir",
    )

    assert resolved == tree.resolve()
    assert source == registry.resolve()


def test_missing_protocol_corpus_writes_explicit_blocked_receipt(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "prestage.json"
    result = prestage._blocked_selection_receipt(
        output=output,
        contract_path=contract,
        active_id="dragapult-dusknoir",
        completed_ids={"alakazam", "starmie"},
        assets={
            "source": "current_runtime_tree",
            "candidate_tree": tmp_path / "tree.json",
            "candidate_audit": None,
        },
        reason=(
            "no unfinished specialist currently has a "
            "protocol-valid corpus"
        ),
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == result
    assert result["status"] == "blocked"
    assert result["selected_specialist"] is None
    assert result["live_training_modified"] is False
    assert result["blockers"] == [
        "protocol_valid_expert_corpus_not_ready"
    ]
