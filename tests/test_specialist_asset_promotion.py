from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from scripts.promote_rare_route_assets_from_elmo import (
    _make_directories_owner_writable,
    _pointer_has_complete_target_coverage,
)
from scripts.resolve_specialist_assets import resolve_specialist_assets


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_specialist_assets_fall_back_until_atomic_receipt_exists(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "default-corpus"
    corpus.mkdir()
    tree = tmp_path / "default-tree.json"
    audit = tmp_path / "default-audit.json"
    _write(tree, {})
    _write(audit, {})

    result = resolve_specialist_assets(
        default_corpus_root=corpus,
        default_candidate_tree=tree,
        default_candidate_audit=audit,
        promotion_receipt=tmp_path / "absent.json",
    )

    assert result["source"] == "contract_defaults"
    assert result["corpus_root"] == corpus


def test_specialist_assets_accept_checksum_bound_boundary_promotion(
    tmp_path: Path,
) -> None:
    defaults = tmp_path / "defaults"
    defaults.mkdir()
    tree = tmp_path / "v34-tree.json"
    _write(tree, {"runtime_enabled": False})
    accepted = ["lucario", "walrein"]
    audit = tmp_path / "v34-audit.json"
    _write(
        audit,
        {
            "schema": "poke_bot.public_matchup_tree_candidate_audit/v1",
            "runtime_enabled": False,
            "artifact_sha256": _sha(tree),
            "accepted_specialist_ids": accepted,
            "accepted_count": len(accepted),
        },
    )
    corpus = tmp_path / "v2-corpus"
    for archetype in accepted:
        _write(
            corpus / archetype / "PROTECTED_EXPERT_CORPUS.json",
            {"schema": "poke_bot.pinned_expert_corpus/v1"},
        )
    receipt = tmp_path / "receipt.json"
    payload = {
        "schema": "poke_bot.rare_route_asset_promotion/v1",
        "status": "ready",
        "candidate_tree": str(tree),
        "candidate_tree_sha256": _sha(tree),
        "candidate_audit": str(audit),
        "candidate_audit_sha256": _sha(audit),
        "accepted_specialist_ids": accepted,
        "corpus_root": str(corpus),
        "ready_rare_archetype_ids": ["walrein"],
        "live_trainer_modified": False,
        "activation_policy": "specialist_boundary_only",
    }
    _write(receipt, payload)

    result = resolve_specialist_assets(
        default_corpus_root=defaults,
        default_candidate_tree=tree,
        default_candidate_audit=audit,
        promotion_receipt=receipt,
    )

    assert result["source"] == "rare_route_promotion"
    assert result["corpus_root"] == corpus
    assert result["candidate_tree"] == tree


def test_router_only_scope_keeps_canonical_corpus_generation(
    tmp_path: Path,
) -> None:
    defaults = tmp_path / "latest20-v6"
    defaults.mkdir()
    tree = tmp_path / "promoted-tree.json"
    _write(tree, {"runtime_enabled": False})
    accepted = ["dudunsparce"]
    audit = tmp_path / "promoted-audit.json"
    _write(
        audit,
        {
            "schema": "poke_bot.public_matchup_tree_candidate_audit/v1",
            "runtime_enabled": False,
            "artifact_sha256": _sha(tree),
            "accepted_specialist_ids": accepted,
            "accepted_count": 1,
        },
    )
    promoted_corpus = tmp_path / "historical-overlay"
    _write(
        promoted_corpus / "dudunsparce" / "PROTECTED_EXPERT_CORPUS.json",
        {"schema": "poke_bot.pinned_expert_corpus/v1"},
    )
    receipt = tmp_path / "receipt.json"
    _write(
        receipt,
        {
            "schema": "poke_bot.rare_route_asset_promotion/v1",
            "status": "ready",
            "candidate_tree": str(tree),
            "candidate_tree_sha256": _sha(tree),
            "candidate_audit": str(audit),
            "candidate_audit_sha256": _sha(audit),
            "accepted_specialist_ids": accepted,
            "corpus_root": str(promoted_corpus),
            "ready_rare_archetype_ids": accepted,
            "live_trainer_modified": False,
            "activation_policy": "specialist_boundary_only",
        },
    )

    result = resolve_specialist_assets(
        default_corpus_root=defaults,
        default_candidate_tree=tree,
        default_candidate_audit=audit,
        promotion_receipt=receipt,
        promotion_scope="router_only",
    )

    assert result["candidate_tree"] == tree
    assert result["corpus_root"] == defaults
    assert result["promoted_corpus_root"] == promoted_corpus


def test_elmo_builder_splits_only_the_six_additive_rare_routes() -> None:
    source = Path("scripts/prepare_rare_route_expert_shards_elmo.py").read_text()
    for archetype in (
        "dragapult-blaziken",
        "dragapult-dusknoir",
        "dudunsparce",
        "gardevoir",
        "ns-zoroark",
        "walrein",
    ):
        assert f'"{archetype}"' in source
    assert "split_expert_manifest_by_archetype.py" in source
    assert "--minimum-decisions 1" in source
    assert "live_corpus_modified" in source


def test_all22_v35_assets_are_imported_only_for_specialist_boundaries() -> None:
    unit = Path(
        "deploy/systemd/pokebot-rare-route-assets-v35-import.service"
    ).read_text()
    assert "public-matchup-tree-calibration-v35" in unit
    assert "rare-route-assets-v35-ready.json" in unit
    assert "--remote-root " in unit
    assert "--require-all-targets" in unit
    for contract_path in (
        "ops/specialist_cycle_handoff_v1.json",
        "ops/post_starmie_core_v2_handoff_v1.json",
    ):
        contract = json.loads(Path(contract_path).read_text())
        receipt = contract["runtime"]["future_assets_receipt"]
        expected = (
                "/rare-route-assets-roster18-v42-ready.json"
            if contract_path == "ops/specialist_cycle_handoff_v1.json"
            else "/rare-route-assets-v37-ready.json"
        )
        assert receipt.endswith(expected)


def test_router_only_promotion_cannot_claim_or_replace_rare_corpora() -> None:
    source = Path(
        "scripts/promote_rare_route_assets_from_elmo.py"
    ).read_text(encoding="utf-8")
    assert "--router-only" in source
    assert "router-only promotion must retain the existing protected" in source
    assert '"existing_protected_generation"' in source
    assert "if not args.router_only:" in source


def test_additive_public_history_masks_missing_private_aux_labels() -> None:
    source = Path(
        "scripts/promote_rare_route_assets_from_elmo.py"
    ).read_text(encoding="utf-8")
    assert "temporal_policy_with_masked_missing_aux" in source
    assert "Never fabricate those labels." in source
    assert "if old.joinpath(\"PROTECTED_EXPERT_CORPUS.json\").is_file()" in source
    assert "shutil.rmtree(target, ignore_errors=True)" in source


def test_merge_only_makes_copied_directories_owner_writable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "copied-generation"
    nested = root / "specialist"
    nested.mkdir(parents=True)
    shard = nested / "shard.features.npz"
    shard.write_bytes(b"immutable")
    root.chmod(0o555)
    nested.chmod(0o555)
    shard.chmod(0o444)

    _make_directories_owner_writable(root)

    assert root.stat().st_mode & stat.S_IWUSR
    assert nested.stat().st_mode & stat.S_IWUSR
    assert not shard.stat().st_mode & stat.S_IWUSR


def test_public_addition_does_not_claim_complete_hidden_target_coverage() -> None:
    partial = {
        "totals": {
            "decisions_kept": 10,
            "target_coverage": {
                "temporal_action_rows": 10,
                "opponent_hand_rows": 0,
            },
        }
    }
    complete = {
        "totals": {
            "decisions_kept": 10,
            "target_coverage": {
                name: 10
                for name in (
                    "temporal_action_rows",
                    "opponent_hand_rows",
                    "opponent_remainder_rows",
                    "opponent_private_prize_rows",
                    "lethal_threat_rows",
                    "prize_race_rows",
                )
            },
        }
    }

    assert not _pointer_has_complete_target_coverage(partial)
    assert _pointer_has_complete_target_coverage(complete)


def test_receipt_full_multihead_requires_every_row_to_have_every_target() -> None:
    source = Path(
        "scripts/promote_rare_route_assets_from_elmo.py"
    ).read_text(encoding="utf-8")
    assert 'decisions_kept = int(' in source
    assert 'int(target_coverage.get(name) or 0) == decisions_kept' in source
