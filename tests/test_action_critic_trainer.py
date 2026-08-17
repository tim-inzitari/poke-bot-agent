from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from poke_bot.recursive_turn_planner.recent20_overlay import (
    BASE_COMPLETION_SCHEMA,
    BASE_SCHEMA,
    MANIFEST_SCHEMA,
    OVERLAY_SCHEMA,
    base_schema_descriptor,
    canonical_bytes,
    canonical_sha256,
    sha256_file,
)
from poke_bot.action_critic_targets import TARGET_OVERLAY_SCHEMA, TARGET_SET_MANIFEST_SCHEMA
from scripts import train_alakazam_action_critic as trainer
from scripts import validate_alakazam_action_critic as validator


WINDOW_DAYS = tuple(
    [f"2026-07-{day:02d}" for day in range(23, 32)]
    + [f"2026-08-{day:02d}" for day in range(1, 12)]
)
SPLIT_BY_DAY = {
    **{day: "train" for day in WINDOW_DAYS[:14]},
    **{day: "validation" for day in WINDOW_DAYS[14:17]},
    **{day: "evaluation" for day in WINDOW_DAYS[17:]},
}


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = value if isinstance(value, bytes) else canonical_bytes(value)
    path.write_bytes(body)
    return sha256_file(path)


def _identity(tag: str) -> str:
    return "sha256:" + hashlib.sha256(tag.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _FakeActionCriticConfig:
    feature_dim: int = 40
    state_hidden_dim: int = 8
    action_hidden_dim: int = 8
    q_hidden_dim: int = 8
    max_action_stages: int = 16


class _FakeActionCriticSidecar(nn.Module):
    """The documented isolated complete-action interface, not a policy model."""

    def __init__(self, config: _FakeActionCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.output = nn.Linear(config.feature_dim * 2, 8)

    def forward(
        self,
        first_stage_legal_features: torch.Tensor,
        first_stage_legal_mask: torch.Tensor,
        selected_stage_features: torch.Tensor,
        selected_stage_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert first_stage_legal_features.ndim == 3
        assert first_stage_legal_mask.dtype is torch.bool
        assert selected_stage_features.ndim == 3
        assert selected_stage_mask.dtype is torch.bool
        state_weight = first_stage_legal_mask.unsqueeze(-1).float()
        action_weight = selected_stage_mask.unsqueeze(-1).float()
        joined = torch.cat(
            (
                (first_stage_legal_features * state_weight).sum(dim=1) / state_weight.sum(dim=1),
                (selected_stage_features * action_weight).sum(dim=1) / action_weight.sum(dim=1),
            ),
            dim=1,
        )
        raw = self.output(joined)
        return torch.cat((raw[:, :2], torch.tanh(raw[:, 2:])), dim=1)


def _fake_sidecar() -> SimpleNamespace:
    return SimpleNamespace(
        ActionCriticSidecarConfig=_FakeActionCriticConfig,
        ActionCriticSidecar=_FakeActionCriticSidecar,
        ACTION_CRITIC_OUTPUT_NAMES=(
            "V_win",
            "Q_win",
            "V_prize^1",
            "Q_prize^1",
            "V_prize^2",
            "Q_prize^2",
            "V_prize^3",
            "Q_prize^3",
        ),
    )


def test_production_sidecar_rejects_legacy_four_tensor_forward(tmp_path: Path, monkeypatch) -> None:
    """A real sidecar may not silently drop the sealed action structure."""

    class _LegacyForward(nn.Module):
        def forward(
            self,
            first_stage_legal_features: torch.Tensor,
            first_stage_legal_mask: torch.Tensor,
            selected_stage_features: torch.Tensor,
            selected_stage_mask: torch.Tensor,
        ) -> torch.Tensor:
            return torch.zeros(
                (first_stage_legal_features.shape[0], 8),
                dtype=torch.float32,
                device=first_stage_legal_features.device,
            )

    strict = _fake_sidecar()
    strict.ACTION_CRITIC_SIDECAR_SCHEMA = "poke_bot.action_critic_sidecar/v1"
    monkeypatch.setattr(trainer, "action_critic_sidecar", strict)
    row = trainer.CompleteActionExample(
        program_identity="program",
        utc_day="2026-08-06",
        episode_id="episode",
        acting_seat=0,
        env_step=0,
        stage_count=1,
        first_stage_menu=(tuple([0.1] * 40), tuple([0.2] * 40)),
        selected_stage_features=(tuple([0.1] * 40),),
        selected_option_indices=(0,),
        selected_legal_counts=(2,),
        selected_action_programs=((0,),),
        terminal_z=None,
        terminal_z_mask=False,
        win_target=1.0,
        win_target_mask=True,
        prize_targets=(0.0, None, None),
        prize_masks=(True, False, False),
    )
    with pytest.raises(trainer.ActionCriticTrainingError, match="rejected the sealed"):
        trainer.critic_predictions(_LegacyForward(), [row], device=torch.device("cpu"))


def _build_sealed_fixture(tmp_path: Path) -> dict[str, Path | str]:
    base_root = tmp_path / "base-pack"
    overlay_root = tmp_path / "complete-overlay"
    target_root = tmp_path / "target-overlay"
    packs: list[dict] = []
    overlay_descriptors: list[dict] = []
    target_descriptors: list[dict] = []
    for index, day in enumerate(WINDOW_DAYS):
        source_sha = _identity(f"source:{day}")
        program_identity = _identity(f"program:{day}")
        day_root = base_root / day
        values = [float(index + offset) / 10.0 for offset in range(80)]
        feature_bytes = struct.pack("<80f", *values)
        files: dict[str, dict] = {}
        for role, name, body in (
            ("features_f32", "features.f32", feature_bytes),
            ("decision_offsets_u64", "decision_offsets.u64", struct.pack("<2Q", 0, 2)),
            ("selected_option_u32", "selected_option.u32", struct.pack("<I", 0)),
            ("decision_key_sha256", "decision_keys.sha256", b"k" * 32),
        ):
            path = day_root / name
            files[role] = {
                "path": str(path),
                "sha256": _write(path, body),
                "size_bytes": len(body),
            }
        receipt = day_root / "receipt.json"
        receipt_sha = _write(receipt, {"schema": BASE_SCHEMA, "utc_day": day})
        packs.append(
            {
                "schema": BASE_SCHEMA,
                "source_path": f"/sealed/{day}/source.jsonl",
                "source_sha256": source_sha,
                "receipt_path": f"/sealed/{day}/receipt.json",
                "receipt_sha256": receipt_sha,
                "feature_width": 40,
                "feature_dtype": "float32_le",
                "option_occurrence_count": 2,
                "decision_occurrence_count": 1,
                "deduplicated": False,
                "files": files,
            }
        )
        overlay_row = {
            "schema": OVERLAY_SCHEMA,
            "utc_day": day,
            "source_archive_sha256": source_sha,
            "source_member": f"episode-{index}.json",
            "episode_id": f"episode-{index}",
            "acting_seat": 0,
            "env_step": index,
            "program_identity": program_identity,
            "hidden_information_fields_present": False,
            "stages": [
                {
                    "factorized_stage": 0,
                    "base_ref": {"option_start": 0, "option_count": 2},
                    "ordered_legal_action_programs": [[11], [22]],
                    "selected_option_index": 0,
                    "selected_action_program": [11],
                    "valid_option_mask": [True, True],
                }
            ],
            "selected_action_program": [11],
        }
        overlay_path = overlay_root / "objects" / f"{day}.jsonl"
        overlay_sha = _write(overlay_path, overlay_row)
        overlay_descriptors.append(
            {
                "path": str(overlay_path.relative_to(overlay_root)),
                "sha256": overlay_sha,
                "size_bytes": overlay_path.stat().st_size,
                "utc_day": day,
                "split": SPLIT_BY_DAY[day],
                "base_source_shard_sha256": source_sha,
            }
        )
        z = (-1.0, 0.0, 1.0)[index % 3]
        target_row = {
            "schema": TARGET_OVERLAY_SCHEMA,
            "owner_goal_revision": 21,
            "split": SPLIT_BY_DAY[day],
            "utc_day": day,
            "source_archive_sha256": source_sha,
            "source_member": f"episode-{index}.json",
            "episode_id": f"episode-{index}",
            "acting_seat": 0,
            "env_step": index,
            "program_identity": program_identity,
            "terminal_win": {
                "value": float(z == 1.0),
                "mask": True,
                "unavailable_reason": None,
            },
            "prize_differential": {
                "h1": {
                    "value": (-1.0, 0.0, 1.0)[(index + 1) % 3],
                    "mask": True,
                    "unavailable_reason": None,
                },
                "h2": {
                    "value": 0.0 if index % 2 else None,
                    "mask": bool(index % 2),
                    "unavailable_reason": None if index % 2 else "setup_zero",
                },
                "h3": {
                    "value": 0.5 if index % 3 else None,
                    "mask": bool(index % 3),
                    "unavailable_reason": None if index % 3 else "setup_zero",
                },
            },
            "target_only": True,
            "hidden_information_fields_present": False,
        }
        target_path = target_root / "objects" / f"{day}.jsonl"
        target_sha = _write(target_path, target_row)
        target_descriptors.append(
            {
                "path": str(target_path.relative_to(target_root)),
                "sha256": target_sha,
                "size_bytes": target_path.stat().st_size,
                "utc_day": day,
                "split": SPLIT_BY_DAY[day],
                "row_schema": target_row["schema"],
            }
        )
    completion = {
        "schema": BASE_COMPLETION_SCHEMA,
        "corpus_manifest_sha256": _identity("corpus"),
        "source_shard_count": len(packs),
        "option_occurrence_count": 2 * len(packs),
        "decision_occurrence_count": len(packs),
        "deduplicated": False,
        "packs": packs,
    }
    completion_path = tmp_path / "base-completion.json"
    completion_sha = _write(completion_path, completion)
    overlay_manifest = {
        "schema": MANIFEST_SCHEMA,
        "base_pack": {
            "completion_path": str(completion_path),
            "completion_sha256": completion_sha,
            "schema_sha256": canonical_sha256(base_schema_descriptor(completion)),
        },
        "overlay_shards": overlay_descriptors,
    }
    overlay_manifest_path = overlay_root / "manifest.json"
    overlay_manifest_sha = _write(overlay_manifest_path, overlay_manifest)
    target_manifest = {
        "schema": TARGET_SET_MANIFEST_SCHEMA,
        "complete_action_overlay_manifest_sha256": overlay_manifest_sha,
        "base_pack_completion_sha256": completion_sha,
        "target_shards": target_descriptors,
    }
    target_manifest_path = target_root / "manifest.json"
    target_manifest_sha = _write(target_manifest_path, target_manifest)
    contract = tmp_path / "critic-contract.json"
    _write(contract, {"schema": "test.critic.contract/v1", "revision": 20})
    return {
        "overlay_manifest": overlay_manifest_path,
        "overlay_sha": overlay_manifest_sha,
        "base_root": base_root,
        "completion_sha": completion_sha,
        "target_manifest": target_manifest_path,
        "target_sha": target_manifest_sha,
        "contract": contract,
    }


def _train_args(paths: dict[str, Path | str], out_dir: Path, *, epochs: int):
    return trainer.build_argument_parser().parse_args(
        [
            "--overlay-manifest",
            str(paths["overlay_manifest"]),
            "--overlay-manifest-sha256",
            str(paths["overlay_sha"]),
            "--base-pack-root",
            str(paths["base_root"]),
            "--base-completion-sha256",
            str(paths["completion_sha"]),
            "--target-manifest",
            str(paths["target_manifest"]),
            "--target-manifest-sha256",
            str(paths["target_sha"]),
            "--contract",
            str(paths["contract"]),
            "--output-dir",
            str(out_dir),
            "--epochs",
            str(epochs),
            "--batch-size",
            "4",
            "--shuffle-buffer",
            "5",
            "--hidden-width",
            "8",
            "--device",
            "cpu",
            "--test-allow-noncanonical-split",
        ]
    )


def test_streaming_complete_action_critic_train_resume_and_validate(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _build_sealed_fixture(tmp_path)
    fake = _fake_sidecar()
    monkeypatch.setattr(trainer, "action_critic_sidecar", fake)
    # The validator imports helpers from the same trainer module under pytest.
    monkeypatch.setattr(validator, "build_critic", trainer.build_critic)
    output = tmp_path / "output"

    first = trainer.train(_train_args(paths, output, epochs=1))
    assert (output / "latest.pt").is_file()
    assert (output / "latest.metrics.json").is_file()
    assert first["validation"]["complete_actions"] == 3
    assert first["validation"]["coverage"]["prize_horizons"]["1"]["available"] == 3

    # Default resume is the isolated atomic latest checkpoint; no policy state
    # or corpus cache is consulted on the second epoch.
    second = trainer.train(_train_args(paths, output, epochs=2))
    checkpoint = Path(second["checkpoint_path"])
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint_data["epoch_completed"] == 2
    assert len(checkpoint_data["epoch_history"]) == 2
    assert checkpoint_data["evaluation_split_consumed"] is False
    assert checkpoint_data["runtime_or_policy_attachment"] is False

    validation_path = tmp_path / "validation.json"
    validation_args = validator.build_argument_parser().parse_args(
        [
            "--overlay-manifest",
            str(paths["overlay_manifest"]),
            "--overlay-manifest-sha256",
            str(paths["overlay_sha"]),
            "--base-pack-root",
            str(paths["base_root"]),
            "--base-completion-sha256",
            str(paths["completion_sha"]),
            "--target-manifest",
            str(paths["target_manifest"]),
            "--target-manifest-sha256",
            str(paths["target_sha"]),
            "--contract",
            str(paths["contract"]),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-sha256",
            str(second["checkpoint_sha256"]),
            "--training-receipt",
            str(second["receipt_path"]),
            "--training-receipt-sha256",
            str(second["receipt_sha256"]),
            "--output-json",
            str(validation_path),
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--test-allow-noncanonical-split",
        ]
    )
    result = validator.validate(validation_args)
    receipt = json.loads(validation_path.read_text(encoding="utf-8"))
    assert result["checkpoint_sha256"] == second["checkpoint_sha256"]
    assert receipt["optimization_performed"] is False
    assert receipt["evaluation_split_consumed"] is False
    assert receipt["validation"]["complete_actions"] == 3
    assert receipt["validation"]["coverage"]["win"]["available"] == 3


def test_target_alignment_refuses_extra_or_misaligned_rows(tmp_path: Path) -> None:
    paths = _build_sealed_fixture(tmp_path)
    target_manifest_path = Path(paths["target_manifest"])
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    first_target = target_manifest["target_shards"][0]
    target_path = target_manifest_path.parent / first_target["path"]
    row = json.loads(target_path.read_text(encoding="utf-8"))
    row["program_identity"] = _identity("wrong-program")
    rewritten = canonical_bytes(row)
    target_path.write_bytes(rewritten)
    first_target["sha256"] = sha256_file(target_path)
    first_target["size_bytes"] = target_path.stat().st_size
    target_manifest_path.write_bytes(canonical_bytes(target_manifest))
    target_sha = sha256_file(target_manifest_path)
    args = _train_args({**paths, "target_sha": target_sha}, tmp_path / "out", epochs=1)
    args.skip_input_shard_sha256 = False
    dataset, targets, _split_days, _binding = trainer.open_sealed_inputs(args)
    with torch.no_grad():
        try:
            next(trainer.iter_complete_action_examples(dataset, targets, split="train"))
        except trainer.ActionCriticTrainingError as exc:
            assert "program identity" in str(exc)
        else:  # pragma: no cover - protects the exact alignment invariant
            raise AssertionError("misaligned target row was accepted")


def test_target_set_manifest_resolves_each_day_artifact_root(tmp_path: Path) -> None:
    paths = _build_sealed_fixture(tmp_path)
    direct = json.loads(Path(paths["target_manifest"]).read_text(encoding="utf-8"))
    target_root = Path(paths["target_manifest"]).parent
    overlay_manifest = json.loads(Path(paths["overlay_manifest"]).read_text(encoding="utf-8"))
    overlay_by_day = {
        item["utc_day"]: item for item in overlay_manifest["overlay_shards"]
    }
    contract_sha = sha256_file(Path(paths["contract"]))
    target_days: list[dict[str, object]] = []
    for descriptor in direct["target_shards"]:
        day = descriptor["utc_day"]
        day_root = target_root / "days" / day
        original = target_root / descriptor["path"]
        shard_path = day_root / "objects" / original.name
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_path.write_bytes(original.read_bytes())
        portable_descriptor = {
            **descriptor,
            "path": str(shard_path.relative_to(day_root)),
            "sha256": sha256_file(shard_path),
            "size_bytes": shard_path.stat().st_size,
            "row_count": 1,
        }
        raw = {
            "sha256": _identity(f"raw:{day}"),
            "size_bytes": 1,
            "source_archive_sha256_verified": True,
        }
        overlay = {
            "sha256": overlay_by_day[day]["sha256"],
            "size_bytes": overlay_by_day[day]["size_bytes"],
            "schema": OVERLAY_SCHEMA,
            "split": descriptor["split"],
        }
        day_manifest_path = day_root / "manifests" / "manifest.json"
        day_manifest = {
            "schema": trainer.TARGET_DAY_MANIFEST_SCHEMA,
            "owner_goal_revision": 21,
            "goal_contract": {"sha256": contract_sha},
            "utc_day": day,
            "split": descriptor["split"],
            # Portable aggregate descriptors intentionally omit the immutable
            # producer-host source paths retained by each sealed day manifest.
            "raw_episode_zip": {**raw, "path": f"/raw/{day}.zip"},
            "complete_action_overlay": {
                **overlay,
                "path": f"/overlay/{day}.jsonl",
            },
            "target_shard": portable_descriptor,
        }
        day_manifest_sha = _write(day_manifest_path, day_manifest)
        day_receipt_path = day_root / "receipts" / "receipt.json"
        day_receipt_sha = _write(
            day_receipt_path,
            {
                "schema": trainer.TARGET_DAY_RECEIPT_SCHEMA,
                "owner_goal_revision": 21,
                "manifest_path": str(day_manifest_path.relative_to(day_root)),
                "manifest_sha256": day_manifest_sha,
                "goal_contract_sha256": contract_sha,
                "complete_action_overlay_sha256": overlay["sha256"],
                "raw_episode_zip_sha256": raw["sha256"],
                "target_shard_sha256": portable_descriptor["sha256"],
                "target_shard_size_bytes": portable_descriptor["size_bytes"],
                "target_row_count": 1,
            },
        )
        target_days.append(
            {
                "utc_day": day,
                "split": descriptor["split"],
                "day_artifact_root": str(day_root.relative_to(target_root)),
                "day_manifest_path": str(day_manifest_path.relative_to(target_root)),
                "day_manifest_sha256": day_manifest_sha,
                "day_receipt_path": str(day_receipt_path.relative_to(target_root)),
                "day_receipt_sha256": day_receipt_sha,
                "raw_episode_zip": raw,
                "complete_action_overlay": overlay,
                "target_shard": portable_descriptor,
            }
        )
    aggregate = {
        "schema": TARGET_SET_MANIFEST_SCHEMA,
        "goal_contract": {"sha256": contract_sha},
        "base_pack_completion": {"sha256": paths["completion_sha"]},
        "complete_action_overlay_manifest": {"sha256": paths["overlay_sha"]},
        "target_days": target_days,
    }
    aggregate_path = target_root / "target-set.json"
    aggregate_sha = _write(aggregate_path, aggregate)
    target_set = trainer.TargetOverlay(
        aggregate_path,
        expected_sha256=aggregate_sha,
        expected_overlay_manifest_sha256=str(paths["overlay_sha"]),
        expected_base_completion_sha256=str(paths["completion_sha"]),
        expected_contract_sha256=sha256_file(Path(paths["contract"])),
        allow_test_fixture=True,
    )
    assert target_set.split_days("train") == WINDOW_DAYS[:14]
    assert len(list(target_set.iter_rows("validation"))) == 3


def test_production_sidecar_batched_complete_action_interface(tmp_path: Path) -> None:
    """The trainer uses the landed sidecar ABI, not only its test fallback."""
    model, config = trainer.build_critic(hidden_width=8)
    assert config["feature_dim"] == 40
    rows = [
        trainer.CompleteActionExample(
            program_identity=f"program-{index}",
            utc_day="2026-08-06",
            episode_id=f"episode-{index}",
            acting_seat=0,
            env_step=index,
            stage_count=1,
            first_stage_menu=(tuple([0.1 + index] * 40), tuple([0.2 + index] * 40)),
            selected_stage_features=(tuple([0.1 + index] * 40),),
            selected_option_indices=(index,),
            selected_legal_counts=(2,),
            selected_action_programs=((index,),),
            terminal_z=None,
            terminal_z_mask=False,
            win_target=float(index),
            win_target_mask=True,
            prize_targets=(0.0, None, 0.5),
            prize_masks=(True, False, True),
        )
        for index in range(2)
    ]
    prediction = trainer.critic_predictions(model, rows, device=torch.device("cpu"))
    total, components = trainer.critic_loss(
        prediction, trainer._target_batch(rows, torch.device("cpu"))
    )
    assert tuple(prediction.shape) == (2, 8)
    assert torch.isfinite(total)
    assert "VQ_prize^1" in components
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    payload = trainer._checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch_completed=1,
        optimizer_steps=1,
        model_config=config,
        trainer_config={"batch_size": 2, "learning_rate": 1.0e-3, "weight_decay": 0.0, "seed": 1},
        source={"fixture": "sealed"},
        split_days={"train": ["2026-07-23"], "validation": ["2026-08-06"], "evaluation": ["2026-08-09"]},
        epoch_history=[],
    )
    assert payload["schema"] == "poke_bot.action_critic_sidecar_checkpoint/v1"
    assert "policy_model_state_dict" not in payload
    checkpoint_path = tmp_path / "atomic-sidecar.pt"
    trainer._atomic_torch_save(checkpoint_path, payload)
    restored_model, _restored_config = trainer.build_critic(saved_config=config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1.0e-3)
    epoch, steps, history = trainer.load_resume_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
        source={"fixture": "sealed"},
        trainer_config={"batch_size": 2, "learning_rate": 1.0e-3, "weight_decay": 0.0, "seed": 1},
    )
    assert (epoch, steps, history) == (1, 1, [])
