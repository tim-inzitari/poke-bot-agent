from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from poke_bot import checkpoint, config, dataset, features
from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION, DecisionSample, GameSequence, PolicyStage
from poke_bot.feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    compact_temporal_expert_sequence,
)
from poke_bot.model import build_model
from poke_bot.own_deck_ledger import OwnDeckLedger
from poke_bot.own_deck_rollout_store import (
    OWN_DECK_ROLLOUT_SIDECAR_SCHEMA,
    OWN_DECK_ROLLOUT_SIDECAR_VERSION,
)
from poke_bot.own_deck_supervision import (
    OWN_DECK_SUPERVISION_SCHEMA,
    OWN_DECK_SUPERVISION_VERSION,
    terminal_conversion_target_mask,
    terminal_conversion_target_vector,
    visible_tutor_completion_target_mask,
    visible_tutor_completion_target_vector,
)
from poke_bot.r260_inzi_sidecar_index import R260InziSidecarIndex, R260InziSidecarIndexError
from poke_bot.train import TrainConfig, streaming_r260_host_rehearsal_step


def test_queries_only_selected_four_keys_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE rows (episode_id TEXT, seat INTEGER, env_step INTEGER, observation_fingerprint TEXT, payload TEXT, PRIMARY KEY(episode_id,seat,env_step,observation_fingerprint))")
    for step in range(200):
        row = {"episode_id": "ep", "seat": 0, "env_step": step, "observation_fingerprint": f"sha256:{step:064x}"[-71:]}
        db.execute("INSERT INTO rows VALUES (?,?,?,?,?)", ("ep", 0, step, row["observation_fingerprint"], json.dumps(row)))
    db.commit(); db.close(); path.chmod(0o444)
    index = R260InziSidecarIndex(path, source_manifest_sha256="sha256:" + "a" * 64, daily_meta_sha256s={"2026-07-22": "sha256:" + "b" * 64})
    sequence = SimpleNamespace(episode_id="ep", seat=0, decisions=[SimpleNamespace(env_step=17, observation_fingerprint=f"sha256:{17:064x}"[-71:])])
    rows = list(index.rows_for_sequences([sequence]))
    assert [row["env_step"] for row in rows] == [17]


def test_missing_selected_key_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "empty.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE rows (episode_id TEXT, seat INTEGER, env_step INTEGER, observation_fingerprint TEXT, payload TEXT, PRIMARY KEY(episode_id,seat,env_step,observation_fingerprint))")
    db.commit(); db.close(); path.chmod(0o444)
    index = R260InziSidecarIndex(path, source_manifest_sha256="sha256:" + "a" * 64, daily_meta_sha256s={"2026-07-22": "sha256:" + "b" * 64})
    sequence = SimpleNamespace(episode_id="missing", seat=0, decisions=[SimpleNamespace(env_step=0, observation_fingerprint="sha256:" + "0" * 64)])
    with pytest.raises(R260InziSidecarIndexError, match="missing selected"):
        list(index.rows_for_sequences([sequence]))


def test_build_ignores_hidden_transfer_remnants_but_rejects_visible_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r274 preserves dot remnants but indexes only the 20 committed days."""

    root = tmp_path / "r260-own-deck-training-dataset-staging-09848f04"
    daily = root / "daily"
    daily.mkdir(parents=True)
    first = date(2026, 7, 22)
    expected = {
        (first + timedelta(days=index)).isoformat(): "sha256:" + f"{index:064x}"
        for index in range(20)
    }
    for day in expected:
        (daily / day).mkdir()
    (daily / ".2026-08-11.r259.partial").mkdir()

    from poke_bot import r260_inzi_sidecar_index as index_module

    monkeypatch.setattr(
        index_module,
        "read_daily_meta",
        lambda _root, day: {
            "meta_sha256": expected[day],
            "source": {"manifest": {"sha256": "sha256:" + "a" * 64}},
        },
    )
    monkeypatch.setattr(
        index_module,
        "iter_daily_sidecar_rows",
        lambda *_args, **_kwargs: iter(()),
    )
    R260InziSidecarIndex.build(
        sidecar_root=root,
        output=tmp_path / "index.sqlite3",
        source_manifest_sha256="sha256:" + "a" * 64,
        daily_meta_sha256s=expected,
    )

    (daily / "2026-08-11").mkdir()

    with pytest.raises(R260InziSidecarIndexError, match="committed non-dot"):
        R260InziSidecarIndex.build(
            sidecar_root=root,
            output=tmp_path / "extra-index.sqlite3",
            source_manifest_sha256="sha256:" + "a" * 64,
            daily_meta_sha256s=expected,
        )


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sparse(words: int, offset: int) -> features.SparseVector:
    value = features.SparseVector()
    for index in range(words):
        value.word_start()
        value.add((offset + index) % 32, 1.0)
    return value


def _r260_observation() -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "looking": [],
            "players": [
                {
                    "hand": [{"id": 1, "serial": 101}],
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "prize": [None] * 6,
                    "deckCount": 52,
                },
                {
                    "hand": [],
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "prize": [None] * 6,
                    "deckCount": 54,
                },
            ],
        },
        "select": {
            "deck": [{"id": 2, "serial": 201}],
            "option": [
                {"type": 10, "area": 1, "index": 0, "playerIndex": 0},
                {"type": 10, "area": 12, "index": 0, "playerIndex": 0},
            ],
        },
    }


def _supervision() -> tuple[dict, dict]:
    tutor = {
        "selected_card_id": {"value": 2, "mask": True},
        "selected_card_ids": [2],
        "selected_card_serials": [201],
        "selected_from_visible_deck": {"value": 1.0, "mask": True},
        "selected_target_observed_after_action": {"value": 1.0, "mask": True},
        "same_actor_followup": {"value": 1.0, "mask": True},
        "same_actor_terminal_class": {"value": 1, "mask": True},
        "target_only": True,
        "provenance": "actual_visible_tutor_then_observed_public_followup",
        "mask_reason": None,
    }
    terminal = {
        "terminal_class": {"value": 1, "mask": True},
        "prize_closeout": {"value": 1.0, "mask": True},
        "opponent_knockout": {"value": 1.0, "mask": True},
        "target_only": True,
        "provenance": "actual_selected_complete_action_public_transition",
    }
    return tutor, terminal


def _sequence_and_sidecar_row(episode_id: str) -> tuple[GameSequence, dict]:
    deck = [1] * 30 + [2] * 30
    observation = _r260_observation()
    snapshot = OwnDeckLedger(deck).observe(observation)
    options = _sparse(2, 9)
    stage = PolicyStage(
        options=options,
        action_combos=[[0], [1]],
        target_index=0,
    )
    fingerprint = "sha256:" + hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
    decision = DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS, 1),
        options=options,
        action=[0],
        action_combo_index=0,
        action_combos=[[0], [1]],
        env_step=3,
        action_token=_sparse(1, 17),
        policy_stages=[stage],
        observation_fingerprint=fingerprint,
    )
    sequence = GameSequence(
        episode_id=episode_id,
        seat=0,
        archetype="alakazam",
        opp_archetype="alakazam",
        deck=deck,
        value=1.0,
        decisions=[decision],
        # Preserve the exact candidate matrix through temporal-expert shard
        # compaction; the disk sidecar must still be able to validate its
        # stage-local four-key payload against those candidates.
        factorized_policy_targets=[
            [
                {
                    "action_combos": [[0], [1]],
                    "policy": [1.0, 0.0],
                    "selected_index": 0,
                }
            ]
        ],
    )
    tutor, terminal = _supervision()
    menu = snapshot.option_features(observation, [[0], [1]])
    row = {
        "schema": OWN_DECK_ROLLOUT_SIDECAR_SCHEMA,
        "version": OWN_DECK_ROLLOUT_SIDECAR_VERSION,
        "episode_id": episode_id,
        "seat": 0,
        "env_step": 3,
        "source_date": "2026-07-22",
        "source_manifest_sha256": "sha256:" + "a" * 64,
        "deck_fingerprint": snapshot.deck_fingerprint,
        "observation_fingerprint": fingerprint,
        "ledger_observation_fingerprint": snapshot.observation_fingerprint,
        "board_feature_fingerprint": dataset._sparse_board_feature_fingerprint(decision.board),
        "ledger_snapshot": snapshot.to_dict(),
        "policy_stage_option_features": [
            {
                "stage_index": 0,
                "action_combos_fingerprint": dataset._action_combos_fingerprint([[0], [1]]),
                "candidate_count": 2,
                "selected_index": 0,
                "ledger_option_features": [list(option) for option in menu],
            }
        ],
        "supervision": {
            "schema": OWN_DECK_SUPERVISION_SCHEMA,
            "version": OWN_DECK_SUPERVISION_VERSION,
            "target_only": True,
            "visible_tutor_completion": {
                "labels": tutor,
                "vector": list(visible_tutor_completion_target_vector(tutor)),
                "mask": list(visible_tutor_completion_target_mask(tutor)),
            },
            "terminal_conversion": {
                "labels": terminal,
                "vector": list(terminal_conversion_target_vector(terminal)),
                "mask": list(terminal_conversion_target_mask(terminal)),
            },
        },
        "training_eligibility": {"active_r241": False},
    }
    return sequence, row


def _write_temporal_manifest(root: Path, sequences: list[GameSequence]) -> tuple[Path, str]:
    shard = root / "expert.features"
    compact = [compact_temporal_expert_sequence(sequence) for sequence in sequences]
    stats = {
        "records_kept": len(compact),
        "decisions_kept": sum(len(sequence.decisions) for sequence in compact),
    }
    with shard.open("wb") as stream:
        pickle.dump(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                "feature_schema": features.FEATURE_SCHEMA_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for sequence in compact:
            pickle.dump(sequence, stream, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(
            {
                "format": SHARD_FORMAT + "-footer",
                "format_version": SHARD_FORMAT_VERSION,
                "stats": stats,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                "shards": [{"path": shard.name, "sha256": _sha256_file(shard), "stats": stats}],
                "totals": stats,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, _sha256_file(manifest)


def test_streaming_r260_refresh_uses_disk_index_and_updates_all_required_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduled trainer streams only selected rows and has real gradients."""

    for name in (
        "card_vocab_size",
        "attack_vocab_size",
        "encoder_vocab_size",
        "decoder_vocab_size",
    ):
        monkeypatch.setattr(features, name, lambda: 64)
    monkeypatch.setattr(features, "decoder_binding_offset", lambda: 64)

    sequences, rows = zip(*[_sequence_and_sidecar_row(f"episode-{index}") for index in range(3)])
    manifest, manifest_digest = _write_temporal_manifest(tmp_path, list(sequences))
    source_manifest = "sha256:" + "a" * 64
    first = date(2026, 7, 22)
    daily_digests = {
        (first + timedelta(days=index)).isoformat(): "sha256:" + f"{index:064x}"
        for index in range(20)
    }
    root = tmp_path / "r260-own-deck-training-dataset"
    for day in daily_digests:
        (root / "daily" / day).mkdir(parents=True)

    from poke_bot import r260_inzi_sidecar_index as index_module

    monkeypatch.setattr(
        index_module,
        "read_daily_meta",
        lambda _root, day: {
            "meta_sha256": daily_digests[day],
            "source": {"manifest": {"sha256": source_manifest}},
        },
    )
    monkeypatch.setattr(
        index_module,
        "iter_daily_sidecar_rows",
        lambda _root, day, **_kwargs: iter(rows if day == "2026-07-22" else ()),
    )
    index = R260InziSidecarIndex.build(
        sidecar_root=root,
        output=tmp_path / "r260-index.sqlite3",
        source_manifest_sha256=source_manifest,
        daily_meta_sha256s=daily_digests,
    )
    index.assert_verified(
        expected_source_manifest_sha256=source_manifest,
        daily_meta_sha256s=daily_digests,
    )

    model_cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=False,
        dropout=0.0,
        own_deck_ledger_enabled=True,
        own_deck_ledger_runtime_enabled=False,
        expanded_heads_enabled=True,
        decision_fusion_enabled=True,
        decision_fusion_runtime_enabled=False,
        decision_fusion_dedicated_routes_enabled=True,
        decision_fusion_dedicated_routes_runtime_enabled=False,
        visible_tutor_completion_head_enabled=True,
        terminal_conversion_head_enabled=True,
        visible_tutor_completion_route_enabled=True,
        visible_tutor_completion_route_runtime_enabled=False,
        terminal_conversion_route_enabled=True,
        terminal_conversion_route_runtime_enabled=False,
        matchup_adapters_enabled=False,
    )
    model = build_model(
        model_cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )
    parent = checkpoint.atomic_torch_save(
        checkpoint.build_checkpoint(
            model=model,
            optimizer=torch.optim.AdamW(
                [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("matchup_adapter_bank.")
                ],
                lr=1e-3,
            ),
            model_config=model.cfg,
            rl_iteration=4,
            archetype_id="alakazam",
            model_id="r260-test-parent",
        ),
        tmp_path / "parent.pt",
    )
    output = tmp_path / "rehearsal.pt"
    result = streaming_r260_host_rehearsal_step(
        manifest_path=manifest,
        manifest_digest=manifest_digest,
        sidecar_index=index,
        base_ckpt=parent,
        output_path=output,
        archetype_id="alakazam",
        epochs=2,
        cfg=TrainConfig.pure_rl_defaults(
            epochs=2,
            visible_tutor_completion_loss_weight=0.025,
            terminal_conversion_loss_weight=0.025,
            collect_own_deck_promotion_metrics=True,
        ),
        seed=71,
        max_context=8,
        batch_games=1,
        manifest_workers=2,
        device=torch.device("cpu"),
    )
    assert result["rl_iteration"] == 4
    assert result["max_games_per_batch"] == 1
    assert result["manifest_workers"] == 2
    assert result["manifest_plan_scans"] == 1
    assert result["manifest_plan_reused_across_epochs"] is True
    assert result["sampled_keys"]
    assert all(result["gradient_reachability"].values())
    payload = checkpoint.load_checkpoint(output, map_location="cpu")
    assert payload["rl_iteration"] == 4
    assert payload["extra"]["r260_streaming_rehearsal"]["full_window_device_resident"] is False
    assert payload["extra"]["r260_streaming_rehearsal"]["manifest_workers"] == 2
    assert payload["extra"]["r260_streaming_rehearsal"]["manifest_plan_scans"] == 1
