"""Fail-closed r197 input plumbing for submission bundle construction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from poke_bot import rtp_evaluation_promotion as promotion
from scripts import handle_passed_gate as handler


ROOT = Path(__file__).resolve().parents[1]


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _seal_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o444)
    return path


def _rewrite_sealed_json(path: Path, value: object) -> Path:
    os.chmod(path, 0o644)
    return _seal_json(path, value)


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256_bytes(path.read_bytes()),
        "bytes": path.stat().st_size,
    }


def _masked_evaluation_receipt(tmp_path: Path) -> Path:
    """Make the smallest physical r198 receipt that reaches the fixed hold.

    The current r197 candidate is intentionally masked/non-promotable.  These
    consumer tests must prove that fact rather than manufacture arbitrary
    bytes which accidentally look like a promotion input.
    """

    registry = tmp_path / "research_control_registry_v1.json"
    registry.write_bytes((ROOT / "ops" / "research_control_registry_v1.json").read_bytes())
    os.chmod(registry, 0o444)
    results = _seal_json(tmp_path / "evaluation-results.json", {"rows": []})
    payload: dict[str, object] = {
        "schema": promotion.EVALUATION_RECEIPT_SCHEMA,
        "status": "ready_for_separate_promotion_review",
        "created_at_utc": "2026-08-09T00:00:00Z",
        "promotion_decision": {
            "eligible_for_separate_promotion_review": True,
            "self_promotion_performed": False,
            "serving_change_authorized": False,
        },
        "evaluation_isolation": {
            "training_eligible": False,
            "replay_eligible": False,
            "formal_gate": False,
            "serving_change_authorized": False,
            "self_promotion_allowed": False,
        },
        "results": {**_identity(results), "in_memory": False},
        "promotion_gates": {
            name: {"passed": True} for name in promotion._REQUIRED_GATES
        },
        "frozen_artifacts": {
            "opponents": [
                {"id": opponent, "content_digest": digest}
                for opponent, digest in promotion.R198_OFFICIAL_CONTROL_OPPONENTS.items()
            ]
        },
        "official_control_panel": {
            "registry": _identity(registry),
            "opponents": dict(promotion.R198_OFFICIAL_CONTROL_OPPONENTS),
        },
        "r197_source_exclusion_binding": {
            "candidate_contract_sha256": promotion.R198_CANDIDATE_CONTRACT_SHA256,
            "r197_source_disjoint": True,
            "evaluation_only": True,
            "source_identity_overlap_count": 0,
            "candidate_target_status": "masked_absent_no_fabrication",
            "trusted_counterfactual_candidate_targets_available": False,
        },
    }
    payload["receipt_input_sha256"] = promotion.canonical_digest(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at_utc", "receipt_input_sha256"}
        }
    )
    return _seal_json(tmp_path / "rtp-evaluation.json", payload)


def _recursive_inputs(tmp_path: Path) -> dict[str, object]:
    model = tmp_path / "model.pt"
    sidecar = tmp_path / "rtp_shadow_planner.pt"
    model.write_bytes(b"r197 frozen parent")
    sidecar.write_bytes(b"inert r197 sidecar")
    deck = tmp_path / "deck.csv"
    deck.write_text("".join(f"{card}\n" for card in range(1, 61)), encoding="utf-8")
    matchup_tree = tmp_path / "matchup_tree.json"
    matchup_tree.write_text('{"schema":"test-r198-tree/v1"}\n', encoding="utf-8")
    evaluation = _masked_evaluation_receipt(tmp_path)
    parent_digest = _sha256_bytes(model.read_bytes())
    deck_file_digest = _sha256_bytes(deck.read_bytes())
    deck_digest = promotion.canonical_digest(list(range(1, 61)))
    tree_digest = _sha256_bytes(matchup_tree.read_bytes())
    receipt = {
        "schema": "poke_bot.rtp_promotion/v1",
        "status": "accepted",
        "specialist_id": "alakazam",
        "parent_checkpoint_sha256": parent_digest,
        "sidecar_sha256": _sha256_bytes(sidecar.read_bytes()),
        "sidecar_config_sha256": _sha256_bytes(b"exact r197 config"),
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "required_neural_passes": {"normal": 6, "forced_replan": 5},
        "deck_file_sha256": deck_file_digest,
        "deck_cards_sha256": deck_digest,
        "matchup_tree_sha256": tree_digest,
        "evaluation_receipt_path": str(evaluation),
        "evaluation_receipt_sha256": _sha256_bytes(evaluation.read_bytes()),
        "identity_gate_passed": True,
        "planner_activation_gate_passed": True,
        "reliability_gate_passed": True,
        "heldout_efficacy_gate_passed": True,
        "robustness_gate_passed": True,
        "latency_gate_passed": True,
        "serving_eligible": True,
        "action_authority_enabled": True,
        "created_at_utc": "2026-08-09T00:00:00Z",
    }
    receipt_path = tmp_path / "promotion.json"
    _seal_json(receipt_path, receipt)
    return {
        "model": model,
        "sidecar": sidecar,
        "parent_digest": parent_digest,
        "deck_file_digest": deck_file_digest,
        "deck_digest": deck_digest,
        "tree_digest": tree_digest,
        "receipt": receipt,
        "receipt_path": receipt_path,
    }


@pytest.mark.unit
def test_recursive_rtp_plumbing_refuses_fixed_masked_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _recursive_inputs(tmp_path)
    monkeypatch.setenv(
        "POKEBOT_SUBMISSION_RTP_CHECKPOINT", str(fixture["sidecar"])
    )

    with pytest.raises(RuntimeError, match="trusted counterfactual candidate targets are absent"):
        handler._prepare_submission_rtp_environment(
            rtp_mode="recursive",
            rtp_promotion_receipt=fixture["receipt_path"],
            frozen_checkpoint=fixture["model"],
            frozen_model_digest=str(fixture["parent_digest"]),
            expected_archetype="alakazam",
            deck_file_digest=str(fixture["deck_file_digest"]),
            deck_cards_digest=str(fixture["deck_digest"]),
            matchup_tree_digest=str(fixture["tree_digest"]),
        )


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["writable", "symlink"])
def test_recursive_rtp_plumbing_rejects_nonphysical_promotion_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    fixture = _recursive_inputs(tmp_path)
    promotion_path = Path(fixture["receipt_path"])
    if kind == "writable":
        os.chmod(promotion_path, 0o644)
        supplied_path = promotion_path
        expected = "mode 0444"
    else:
        supplied_path = tmp_path / "promotion-link.json"
        supplied_path.symlink_to(promotion_path)
        expected = "symbolic link"
    monkeypatch.setenv("POKEBOT_SUBMISSION_RTP_CHECKPOINT", str(fixture["sidecar"]))

    with pytest.raises(RuntimeError, match=expected):
        handler._prepare_submission_rtp_environment(
            rtp_mode="recursive",
            rtp_promotion_receipt=supplied_path,
            frozen_checkpoint=fixture["model"],
            frozen_model_digest=str(fixture["parent_digest"]),
            expected_archetype="alakazam",
            deck_file_digest=str(fixture["deck_file_digest"]),
            deck_cards_digest=str(fixture["deck_digest"]),
            matchup_tree_digest=str(fixture["tree_digest"]),
        )


@pytest.mark.unit
@pytest.mark.parametrize("passes", [24, 32, 255, 257])
def test_recursive_rtp_plumbing_rejects_non_256_receipt_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passes: int
) -> None:
    fixture = _recursive_inputs(tmp_path)
    receipt = dict(fixture["receipt"])
    receipt["max_neural_passes"] = passes
    _rewrite_sealed_json(Path(fixture["receipt_path"]), receipt)
    monkeypatch.setenv(
        "POKEBOT_SUBMISSION_RTP_CHECKPOINT", str(fixture["sidecar"])
    )

    with pytest.raises(RuntimeError, match="exact 256-pass/1024-action"):
        handler._prepare_submission_rtp_environment(
            rtp_mode="recursive",
            rtp_promotion_receipt=fixture["receipt_path"],
            frozen_checkpoint=fixture["model"],
            frozen_model_digest=str(fixture["parent_digest"]),
            expected_archetype="alakazam",
            deck_file_digest=str(fixture["deck_file_digest"]),
            deck_cards_digest=str(fixture["deck_digest"]),
            matchup_tree_digest=str(fixture["tree_digest"]),
        )


@pytest.mark.unit
@pytest.mark.parametrize("action_combos", [256, 1023, 1025])
def test_recursive_rtp_plumbing_rejects_non_1024_receipt_action_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action_combos: int
) -> None:
    fixture = _recursive_inputs(tmp_path)
    receipt = dict(fixture["receipt"])
    receipt["max_action_combos"] = action_combos
    _rewrite_sealed_json(Path(fixture["receipt_path"]), receipt)
    monkeypatch.setenv(
        "POKEBOT_SUBMISSION_RTP_CHECKPOINT", str(fixture["sidecar"])
    )

    with pytest.raises(RuntimeError, match="exact 256-pass/1024-action"):
        handler._prepare_submission_rtp_environment(
            rtp_mode="recursive",
            rtp_promotion_receipt=fixture["receipt_path"],
            frozen_checkpoint=fixture["model"],
            frozen_model_digest=str(fixture["parent_digest"]),
            expected_archetype="alakazam",
            deck_file_digest=str(fixture["deck_file_digest"]),
            deck_cards_digest=str(fixture["deck_digest"]),
            matchup_tree_digest=str(fixture["tree_digest"]),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "required_passes",
    [
        {"normal": 5, "forced_replan": 5},
        {"normal": 6, "forced_replan": 6},
        {"normal": 6},
    ],
)
def test_recursive_rtp_plumbing_rejects_noncanonical_required_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    required_passes: dict[str, int],
) -> None:
    fixture = _recursive_inputs(tmp_path)
    receipt = dict(fixture["receipt"])
    receipt["required_neural_passes"] = required_passes
    _rewrite_sealed_json(Path(fixture["receipt_path"]), receipt)
    monkeypatch.setenv(
        "POKEBOT_SUBMISSION_RTP_CHECKPOINT", str(fixture["sidecar"])
    )

    with pytest.raises(RuntimeError, match="exact 256-pass/1024-action"):
        handler._prepare_submission_rtp_environment(
            rtp_mode="recursive",
            rtp_promotion_receipt=fixture["receipt_path"],
            frozen_checkpoint=fixture["model"],
            frozen_model_digest=str(fixture["parent_digest"]),
            expected_archetype="alakazam",
            deck_file_digest=str(fixture["deck_file_digest"]),
            deck_cards_digest=str(fixture["deck_digest"]),
            matchup_tree_digest=str(fixture["tree_digest"]),
        )


@pytest.mark.unit
def test_direct_rtp_plumbing_binds_same_sidecar_without_receipt_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _recursive_inputs(tmp_path)
    monkeypatch.setenv(
        "POKEBOT_SUBMISSION_RTP_CHECKPOINT", str(fixture["sidecar"])
    )

    env, binding = handler._prepare_submission_rtp_environment(
        rtp_mode="direct",
        rtp_promotion_receipt=None,
        frozen_checkpoint=fixture["model"],
        frozen_model_digest=str(fixture["parent_digest"]),
        expected_archetype="alakazam",
        deck_file_digest=str(fixture["deck_file_digest"]),
        deck_cards_digest=str(fixture["deck_digest"]),
        matchup_tree_digest=str(fixture["tree_digest"]),
    )

    assert env == {
        "POKEBOT_SUBMISSION_RTP_CHECKPOINT": str(
            Path(fixture["sidecar"]).resolve()
        ),
        "POKEBOT_SUBMISSION_RTP_PARENT_CHECKPOINT_SHA256": fixture[
            "parent_digest"
        ],
    }
    assert binding is not None
    assert binding["sidecar_sha256"] == fixture["receipt"]["sidecar_sha256"]
    assert binding["max_neural_passes"] == 256
    assert binding["max_action_combos"] == 1024
    assert binding["required_neural_passes"] == {"normal": 6, "forced_replan": 5}
    assert binding["deck_cards_sha256"] == fixture["deck_digest"]
    assert binding["matchup_tree_sha256"] == fixture["tree_digest"]
    assert "POKEBOT_SUBMISSION_RTP_PROMOTION_RECEIPT" not in env


@pytest.mark.unit
def test_direct_rtp_plumbing_requires_the_shared_matchup_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _recursive_inputs(tmp_path)
    monkeypatch.setenv(
        "POKEBOT_SUBMISSION_RTP_CHECKPOINT", str(fixture["sidecar"])
    )

    with pytest.raises(RuntimeError, match="requires a matchup tree"):
        handler._prepare_submission_rtp_environment(
            rtp_mode="direct",
            rtp_promotion_receipt=None,
            frozen_checkpoint=fixture["model"],
            frozen_model_digest=str(fixture["parent_digest"]),
            expected_archetype="alakazam",
            deck_file_digest=str(fixture["deck_file_digest"]),
            deck_cards_digest=str(fixture["deck_digest"]),
            matchup_tree_digest=None,
        )


@pytest.mark.unit
def test_nonrecursive_modes_reject_promotion_receipt(
    tmp_path: Path,
) -> None:
    fixture = _recursive_inputs(tmp_path)
    for mode in ("default_off", "disabled", "enabled", "off", "direct"):
        with pytest.raises(RuntimeError, match="valid only for recursive"):
            handler._prepare_submission_rtp_environment(
                rtp_mode=mode,
                rtp_promotion_receipt=fixture["receipt_path"],
                frozen_checkpoint=fixture["model"],
                frozen_model_digest=str(fixture["parent_digest"]),
                expected_archetype="alakazam",
                deck_file_digest=str(fixture["deck_file_digest"]),
                deck_cards_digest=str(fixture["deck_digest"]),
                matchup_tree_digest=str(fixture["tree_digest"]),
            )


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["default_off", "disabled", "off"])
def test_no_rtp_modes_remain_receipt_free(
    tmp_path: Path, mode: str
) -> None:
    fixture = _recursive_inputs(tmp_path)

    env, binding = handler._prepare_submission_rtp_environment(
        rtp_mode=mode,
        rtp_promotion_receipt=None,
        frozen_checkpoint=fixture["model"],
        frozen_model_digest=str(fixture["parent_digest"]),
        expected_archetype="alakazam",
        deck_file_digest=str(fixture["deck_file_digest"]),
        deck_cards_digest=str(fixture["deck_digest"]),
        matchup_tree_digest=str(fixture["tree_digest"]),
    )

    assert env == {}
    assert binding is None


@pytest.mark.unit
def test_legacy_enabled_mode_still_uses_only_its_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _recursive_inputs(tmp_path)
    monkeypatch.setenv(
        "POKEBOT_SUBMISSION_RTP_CHECKPOINT", str(fixture["sidecar"])
    )

    env, binding = handler._prepare_submission_rtp_environment(
        rtp_mode="enabled",
        rtp_promotion_receipt=None,
        frozen_checkpoint=fixture["model"],
        frozen_model_digest=str(fixture["parent_digest"]),
        expected_archetype="alakazam",
        deck_file_digest=str(fixture["deck_file_digest"]),
        deck_cards_digest=str(fixture["deck_digest"]),
        matchup_tree_digest=str(fixture["tree_digest"]),
    )

    assert env == {
        "POKEBOT_SUBMISSION_RTP_CHECKPOINT": str(
            Path(fixture["sidecar"]).resolve()
        )
    }
    assert binding is None


@pytest.mark.unit
def test_bundle_builder_forwards_recursive_receipt_without_running_a_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise only argument/env plumbing; the mocked subprocess never builds."""

    fixture = _recursive_inputs(tmp_path)
    deck = tmp_path / "deck.csv"
    deck.write_text("1\n" * 60, encoding="utf-8")
    receipt_path = Path(fixture["receipt_path"])
    captured: dict[str, object] = {}

    def fake_prepare(**kwargs: object):
        captured["prepare"] = kwargs
        return (
            {
                "POKEBOT_SUBMISSION_RTP_CHECKPOINT": str(fixture["sidecar"]),
                "POKEBOT_SUBMISSION_RTP_PARENT_CHECKPOINT_SHA256": str(
                    fixture["parent_digest"]
                ),
                "POKEBOT_SUBMISSION_RTP_PROMOTION_RECEIPT": str(receipt_path),
            },
            {"sha256": _sha256_bytes(receipt_path.read_bytes())},
        )

    class StopBeforeBuild(Exception):
        pass

    def fake_run(_command: list[str], **kwargs: object):
        captured["env"] = dict(kwargs["env"])
        raise StopBeforeBuild

    monkeypatch.setattr(handler, "_prepare_submission_rtp_environment", fake_prepare)
    monkeypatch.setattr(
        handler.checkpoint,
        "load_checkpoint",
        lambda *_args, **_kwargs: {"archetype_id": "alakazam", "model_config": {}},
    )
    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    with pytest.raises(StopBeforeBuild):
        handler.build_submission_bundle(
            repo_root=tmp_path,
            frozen_manifest={
                "model_path": str(fixture["model"]),
                "checkpoint_digest": fixture["parent_digest"],
            },
            deck_receipt={
                "path": str(deck),
                "cards_sha256": fixture["deck_digest"],
                "file_sha256": _sha256_bytes(deck.read_bytes()),
            },
            output_dir=tmp_path / "out",
            python=Path("/usr/bin/python3"),
            archetype="alakazam",
            rtp_mode="recursive",
            rtp_promotion_receipt=receipt_path,
        )

    prepared = dict(captured["prepare"])
    assert prepared["rtp_mode"] == "recursive"
    assert prepared["rtp_promotion_receipt"] == receipt_path
    env = dict(captured["env"])
    assert env["POKEBOT_SUBMISSION_RTP_MODE"] == "recursive"
    assert env["POKEBOT_SUBMISSION_RTP_CHECKPOINT"] == str(fixture["sidecar"])
    assert (
        env["POKEBOT_SUBMISSION_RTP_PARENT_CHECKPOINT_SHA256"]
        == fixture["parent_digest"]
    )
    assert env["POKEBOT_SUBMISSION_RTP_PROMOTION_RECEIPT"] == str(receipt_path)
