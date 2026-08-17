from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import pytest
import torch

from poke_bot import cg_env, checkpoint, config, features
from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION, DecisionSample, GameSequence
from poke_bot.feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
)
from poke_bot.matchup_adapter_activation import (
    TRAINING_TICKET_SCHEMA,
    build_activation_receipt,
    build_adapter_rehearsal_authorization,
    merge_dormant_adapter_checkpoint,
)
from poke_bot.matchup_adapters import EXPERT_IDS, HIDDEN_DIM, UNKNOWN_ROUTE
from poke_bot.model import build_model
from poke_bot.pure_rl.matchup_adapter_corpus import STAGED_CORPUS_SCHEMA, sha256_file
from poke_bot.pure_rl.expert_adapter_rehearsal import (
    rehearsal_paths,
    run_or_recover_expert_adapter_rehearsal,
)
from poke_bot.pure_rl.matchup_adapter_trainer import (
    StreamingAdapterTrainConfig,
    _assert_resume_parent_state,
    iter_single_route_batches,
    load_staged_training_contract,
    train_matchup_adapters_streaming,
)


@pytest.fixture(autouse=True)
def _fake_feature_vocab(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SelectContext:
        MAIN = 0
        RECOVER_SPECIAL_CONDITION = 48

    monkeypatch.setattr(features, "_CARD_COUNT", 8)
    monkeypatch.setattr(features, "_ATTACK_COUNT", 8)
    monkeypatch.setattr(features, "_CARD_TABLE", {})
    monkeypatch.setitem(cg_env.__dict__, "SelectContext", _SelectContext)


def _digest(label: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sparse(words: int, offset: int) -> features.SparseVector:
    result = features.SparseVector()
    for word in range(words):
        result.word_start()
        result.add((offset + word) % 32, 1.0)
    return result


def _sequence(route: int, split: str) -> GameSequence:
    episode_id = f"route-{route}-{split}"
    decision = DecisionSample(
        board=_sparse(features.NUM_BOARD_TOKENS, route),
        options=_sparse(2, route + 3),
        action=[0],
        action_combo_index=route % 2,
        action_combos=[[0], [1]],
        env_step=0,
        action_token=_sparse(1, route + 7),
        matchup_adapter_oracle_route=route,
        matchup_adapter_public_route=UNKNOWN_ROUTE,
    )
    sequence = GameSequence(
        episode_id=episode_id,
        seat=0,
        archetype="alakazam",
        opp_archetype=EXPERT_IDS[route],
        deck=[1] * 60,
        value=1.0 if route % 2 == 0 else -1.0,
        decisions=[decision],
    )
    sequence.matchup_adapter_training_ticket = {
        "schema": TRAINING_TICKET_SCHEMA,
        "opponent_id": f"ladder-{EXPERT_IDS[route]}",
        "package_digest": _digest(f"package-{route}"),
        "archetype_id": EXPERT_IDS[route],
        "route": route,
        "corpus_manifest_digest": _digest("oracle-manifest"),
        "gate_contract_digest": _digest("active-gate-contract"),
        "episode_id": episode_id,
        "seat": 0,
        "identity_kind": "raw-archive-full-deck-and-package",
        "opponent_deck_digest": _digest(f"deck-{route}"),
        "source_archive_digest": _digest("archive"),
        "source_member_digest": _digest(episode_id),
        "source_feature_digest": _digest("source-feature-shard"),
        "oracle_index_digest": _digest("oracle-index"),
        "classifier_digest": _digest("classifier"),
        "package_registry_digest": _digest("registry"),
        "classifier_method": "representative_exact",
        "formal_eval": False,
        "runtime_route_authorized": False,
    }
    return sequence


def _staged_corpus(root: Path) -> Path:
    root.mkdir(parents=True)
    shards = []
    routes = []
    for route, archetype_id in enumerate(EXPERT_IDS):
        route_counts = {
            "route": route,
            "archetype_id": archetype_id,
            "episodes": 2,
            "train_episodes": 1,
            "val_episodes": 1,
            "train_sequences": 1,
            "train_decisions": 1,
            "val_sequences": 1,
            "val_decisions": 1,
        }
        routes.append(route_counts)
        for split in ("train", "val"):
            path = root / f"route_{route:02d}_{archetype_id}.{split}.features"
            with path.open("wb") as stream:
                pickle.dump(
                    {
                        "format": SHARD_FORMAT,
                        "format_version": SHARD_FORMAT_VERSION,
                        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                        "feature_schema": features.FEATURE_SCHEMA_VERSION,
                        "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                        "matchup_adapter_staged": True,
                        "offline_oracle_only": True,
                        "runtime_routes_enabled": False,
                        "route": route,
                        "archetype_id": archetype_id,
                        "split": split,
                        "corpus_manifest_digest": _digest("oracle-manifest"),
                        "gate_contract_digest": _digest("active-gate-contract"),
                        "classifier_digest": _digest("classifier"),
                        "package_registry_digest": _digest("registry"),
                        "membership_digest": _digest("membership"),
                    },
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                pickle.dump(
                    _sequence(route, split),
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                pickle.dump(
                    {
                        "format": SHARD_FORMAT + "-footer",
                        "format_version": SHARD_FORMAT_VERSION,
                        "stats": {
                            "records_total": 1,
                            "records_kept": 1,
                            "records_dropped": 0,
                            "decisions_kept": 1,
                        },
                    },
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            shards.append(
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "route": route,
                    "archetype_id": archetype_id,
                    "split": split,
                    "stats": {
                        "records_total": 1,
                        "records_kept": 1,
                        "records_dropped": 0,
                        "decisions_kept": 1,
                    },
                }
            )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": STAGED_CORPUS_SCHEMA,
                "offline_oracle_only": True,
                "runtime_routes_enabled": False,
                "source_feature_manifest_digest": _digest("feature-manifest"),
                "oracle_manifest_digest": _digest("oracle-manifest"),
                "classifier_digest": _digest("classifier"),
                "package_registry_digest": _digest("registry"),
                "active_gate_contract_digest": _digest("active-gate-contract"),
                "active_gate_contract_file_digest": _digest("active-gate-file"),
                "membership_digest": _digest("membership"),
                "split": {
                    "algorithm": "sha256-ranked-per-route-episode/v1",
                    "seed": 42,
                    "val_frac": 0.5,
                    "episode_disjoint": True,
                },
                "routes": routes,
                "shards": shards,
                "totals": {
                    "source_sequences": 14,
                    "selected_sequences": 14,
                    "gate_or_ineligible_excluded": 0,
                    "unsupported_matchup_excluded": 0,
                    "decisions": 14,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _parent_and_receipt(root: Path) -> tuple[Path, Path]:
    model_cfg = config.ModelConfig(
        d_model=HIDDEN_DIM,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=128,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        matchup_adapters_enabled=False,
        dropout=0.0,
    )
    torch.manual_seed(991)
    model = build_model(
        model_cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=16,
    )
    parent = checkpoint.atomic_torch_save(
        checkpoint.build_checkpoint(
            model=model,
            model_config=model.cfg,
            step=99,
            epoch=15,
            rl_iteration=15,
            extra={"matchup_adapters_runtime_enabled": False},
        ),
        root / "iter15-parent.pt",
    )
    state = {
        "version": 1,
        "last_completed_iteration": 15,
        "next_iteration": 16,
        "learner": {
            "path": str(parent),
            "digest": checkpoint.checkpoint_digest(parent),
        },
    }
    run = root / "run"
    (run / "commits").mkdir(parents=True)
    (run / "commits" / "iter_00015.json").write_text(
        json.dumps(state) + "\n", encoding="utf-8"
    )
    (run / "loop_state.json").write_text(
        json.dumps(state) + "\n", encoding="utf-8"
    )
    receipt = root / "activation.json"
    build_activation_receipt(run_dir=run, output_path=receipt)
    return parent, receipt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resume_parent_identity_is_device_independent(tmp_path: Path) -> None:
    parent, _receipt = _parent_and_receipt(tmp_path)
    resumed = checkpoint.load_checkpoint(parent, map_location=torch.device("cuda:0"))
    _assert_resume_parent_state(resumed, parent)


def _cfg(**overrides) -> StreamingAdapterTrainConfig:
    values = {
        "epochs": 1,
        "games_per_batch": 1,
        "max_decisions_per_batch": 4,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "value_loss_weight": 1.0,
        "grad_clip": 1.0,
        "amp": False,
        "seed": 123,
        "early_stop_patience": 2,
        "checkpoint_every_steps": 1,
        "log_every_steps": 100,
        "max_process_rss_gib": 64.0,
        "min_available_ram_gib": 0.0,
        "memory_check_every_batches": 1,
    }
    values.update(overrides)
    return StreamingAdapterTrainConfig(**values)


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_contract_and_batch_stream_are_exact_and_single_route(tmp_path: Path) -> None:
    manifest = _staged_corpus(tmp_path / "corpus")
    contract = load_staged_training_contract(manifest)
    assert contract.train_sequences == contract.val_sequences == len(EXPERT_IDS)
    inputs = contract.training_contract["inputs"]
    for field in (
        "staged_manifest_file_digest",
        "source_feature_manifest_digest",
        "oracle_manifest_digest",
        "active_gate_contract_digest",
        "active_gate_contract_file_digest",
        "classifier_digest",
        "package_registry_digest",
        "membership_digest",
        "implementation_digest",
    ):
        assert str(inputs[field]).startswith("sha256:")
    batches = list(
        iter_single_route_batches(
            manifest,
            "train",
            games_per_batch=4,
            max_decisions_per_batch=4,
            expected_sequences=len(EXPERT_IDS),
        )
    )
    assert [batch.route for batch in batches] == list(range(len(EXPERT_IDS)))
    assert all(
        len(batch.sequences) == 1
        and batch.sequences[0].decisions[0].matchup_adapter_oracle_route
        == batch.route
        for batch in batches
    )


def test_interrupted_resume_is_exact_and_never_cross_contaminates(
    tmp_path: Path,
) -> None:
    manifest = _staged_corpus(tmp_path / "corpus")
    parent, receipt = _parent_and_receipt(tmp_path / "identity")
    cfg = _cfg()
    uninterrupted = train_matchup_adapters_streaming(
        staged_manifest=manifest,
        parent_checkpoint=parent,
        activation_receipt=receipt,
        output_dir=tmp_path / "uninterrupted",
        train_config=cfg,
        device=torch.device("cpu"),
        resume=False,
    )
    paused = train_matchup_adapters_streaming(
        staged_manifest=manifest,
        parent_checkpoint=parent,
        activation_receipt=receipt,
        output_dir=tmp_path / "resumed",
        train_config=cfg,
        device=torch.device("cpu"),
        resume=False,
        stop_after_steps=1,
    )
    assert paused["status"] == "stopped_after_verified_step"
    paused_payload = checkpoint.load_checkpoint(paused["latest_path"])
    parent_payload = checkpoint.load_checkpoint(parent)
    for route in range(len(EXPERT_IDS)):
        keys = [
            key
            for key in paused_payload["model_state_dict"]
            if key.startswith(f"matchup_adapter_bank.experts.{route}.")
        ]
        changed = any(
            not torch.equal(
                paused_payload["model_state_dict"][key],
                parent_payload["model_state_dict"][key],
            )
            for key in keys
        )
        assert changed is (route == 0)
    with pytest.raises(ValueError, match="configuration contract drift"):
        train_matchup_adapters_streaming(
            staged_manifest=manifest,
            parent_checkpoint=parent,
            activation_receipt=receipt,
            output_dir=tmp_path / "resumed",
            train_config=_cfg(lr=2e-3),
            device=torch.device("cpu"),
            resume="auto",
        )
    tampered = copy.deepcopy(paused_payload)
    tampered["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.123
    checkpoint.atomic_torch_save(tampered, paused["latest_path"])
    with pytest.raises(AssertionError, match="optimizer hyperparameter drift"):
        train_matchup_adapters_streaming(
            staged_manifest=manifest,
            parent_checkpoint=parent,
            activation_receipt=receipt,
            output_dir=tmp_path / "resumed",
            train_config=cfg,
            device=torch.device("cpu"),
            resume="auto",
        )
    checkpoint.atomic_torch_save(paused_payload, paused["latest_path"])
    resumed = train_matchup_adapters_streaming(
        staged_manifest=manifest,
        parent_checkpoint=parent,
        activation_receipt=receipt,
        output_dir=tmp_path / "resumed",
        train_config=cfg,
        device=torch.device("cpu"),
        resume="auto",
    )
    assert uninterrupted["status"] == resumed["status"] == "complete"
    left = checkpoint.load_checkpoint(uninterrupted["final_path"])
    right = checkpoint.load_checkpoint(resumed["final_path"])
    for key, value in left["model_state_dict"].items():
        assert torch.equal(value, right["model_state_dict"][key]), key
    _assert_nested_equal(
        left["optimizer_state_dict"], right["optimizer_state_dict"]
    )
    _assert_nested_equal(
        left["extra"]["streaming_matchup_adapter_state"],
        right["extra"]["streaming_matchup_adapter_state"],
    )
    assert left["extra"]["matchup_adapters_runtime_enabled"] is False
    assert left["model_config"]["matchup_adapters_enabled"] is False
    assert set(left["extra"]["matchup_adapter_per_route_validation"]) == set(
        EXPERT_IDS
    )
    for key, value in parent_payload["model_state_dict"].items():
        if not key.startswith("matchup_adapter_bank."):
            assert torch.equal(value, left["model_state_dict"][key]), key

    merged = merge_dormant_adapter_checkpoint(
        parent_checkpoint=parent,
        adapter_checkpoint=uninterrupted["final_path"],
        activation_receipt=receipt,
        output_path=tmp_path / "merged.pt",
    )
    assert checkpoint.load_checkpoint(merged)["model_config"][
        "matchup_adapters_enabled"
    ] is False
    rehearsal_merged = merge_dormant_adapter_checkpoint(
        parent_checkpoint=parent,
        adapter_checkpoint=uninterrupted["final_path"],
        activation_receipt=receipt,
        output_path=tmp_path / "rehearsal-merged.pt",
        import_optimizer_state=True,
        accumulate_parent_fit=True,
    )
    rehearsal_payload = checkpoint.load_checkpoint(rehearsal_merged)
    rehearsal_extra = rehearsal_payload["extra"]
    fit = rehearsal_extra["dormant_matchup_adapter_fit"]
    assert fit["optimizer_included"] is True
    assert fit["base_frozen"] is True
    assert fit["optimizer_scope"] == "matchup_adapter_bank_only"
    assert fit["epochs"] == fit["phase_epochs"] == cfg.epochs
    assert fit["steps"] == fit["phase_steps"] == uninterrupted["step"]
    assert fit["rows"] == fit["phase_rows"] == len(EXPERT_IDS) * cfg.epochs
    assert fit["route_decisions"] == {
        archetype_id: cfg.epochs for archetype_id in EXPERT_IDS
    }
    _assert_nested_equal(
        rehearsal_extra["dormant_matchup_adapter_optimizer_state"],
        left["optimizer_state_dict"],
    )
    assert rehearsal_extra["dormant_matchup_adapter_bank"][
        "optimizer_imported"
    ] is True
    rehearsal_run = tmp_path / "rehearsal-run"
    (rehearsal_run / "commits").mkdir(parents=True)
    rehearsal_state = {
        "version": 1,
        "last_completed_iteration": 25,
        "next_iteration": 26,
        "learner": {
            "path": str(rehearsal_merged),
            "digest": checkpoint.checkpoint_digest(rehearsal_merged),
        },
    }
    serialized_rehearsal_state = json.dumps(rehearsal_state) + "\n"
    (rehearsal_run / "commits" / "iter_00025.json").write_text(
        serialized_rehearsal_state,
        encoding="utf-8",
    )
    (rehearsal_run / "loop_state.json").write_text(
        serialized_rehearsal_state,
        encoding="utf-8",
    )
    transaction_paths = rehearsal_paths(rehearsal_run, 26)
    rehearsal_authorization = transaction_paths.authorization
    build_adapter_rehearsal_authorization(
        run_dir=rehearsal_run,
        completed_iteration=25,
        output_path=rehearsal_authorization,
    )
    (rehearsal_run / "shards").mkdir()
    (rehearsal_run / "shards" / "iter_00026.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no longer clean"):
        train_matchup_adapters_streaming(
            staged_manifest=manifest,
            parent_checkpoint=rehearsal_merged,
            activation_receipt=rehearsal_authorization,
            output_dir=tmp_path / "post-boundary-denied",
            train_config=cfg,
            device=torch.device("cpu"),
            resume=False,
            restore_parent_optimizer_state=True,
        )
    continued = run_or_recover_expert_adapter_rehearsal(
        run_dir=rehearsal_run,
        before_iteration=26,
        parent_checkpoint=rehearsal_merged,
        parent_digest=checkpoint.checkpoint_digest(rehearsal_merged),
        staged_manifest=manifest,
        epochs=cfg.epochs,
        learning_rate=cfg.lr,
        games_per_batch=cfg.games_per_batch,
        max_decisions_per_batch=cfg.max_decisions_per_batch,
        seed=cfg.seed,
        device=torch.device("cpu"),
        max_process_rss_gib=cfg.max_process_rss_gib,
        min_available_ram_gib=cfg.min_available_ram_gib,
    )
    continued_payload = checkpoint.load_checkpoint(continued["checkpoint"])
    assert continued_payload["extra"][
        "dormant_matchup_adapter_fit"
    ]["optimizer_state_restored"] is True
    assert continued["reused"] is True
    recovered = run_or_recover_expert_adapter_rehearsal(
        run_dir=rehearsal_run,
        before_iteration=26,
        parent_checkpoint=rehearsal_merged,
        parent_digest=checkpoint.checkpoint_digest(rehearsal_merged),
        staged_manifest=manifest,
        epochs=cfg.epochs,
        learning_rate=cfg.lr,
        games_per_batch=cfg.games_per_batch,
        max_decisions_per_batch=cfg.max_decisions_per_batch,
        seed=cfg.seed,
        device=torch.device("cpu"),
        max_process_rss_gib=cfg.max_process_rss_gib,
        min_available_ram_gib=cfg.min_available_ram_gib,
    )
    assert recovered["checkpoint_digest"] == continued["checkpoint_digest"]
    assert recovered["reused"] is True

    with pytest.raises(FileExistsError, match="refuses existing artifacts"):
        train_matchup_adapters_streaming(
            staged_manifest=manifest,
            parent_checkpoint=parent,
            activation_receipt=receipt,
            output_dir=tmp_path / "resumed",
            train_config=cfg,
            device=torch.device("cpu"),
            resume=False,
        )


def test_memory_guard_saves_verified_zero_cursor_before_failure(
    tmp_path: Path,
) -> None:
    manifest = _staged_corpus(tmp_path / "corpus")
    parent, receipt = _parent_and_receipt(tmp_path / "identity")
    output = tmp_path / "memory-stop"
    with pytest.raises(MemoryError, match="stopped safely"):
        train_matchup_adapters_streaming(
            staged_manifest=manifest,
            parent_checkpoint=parent,
            activation_receipt=receipt,
            output_dir=output,
            train_config=_cfg(max_process_rss_gib=1e-9),
            device=torch.device("cpu"),
            resume=False,
        )
    payload = checkpoint.load_checkpoint(output / "latest.pt")
    state = payload["extra"]["streaming_matchup_adapter_state"]
    assert state["step"] == state["train_sequences_consumed"] == 0
    assert payload["extra"]["matchup_adapters_runtime_enabled"] is False
