"""Focused r197 action-boundary checks independent of the native evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "r197_action_authority_test_module",
    ROOT / "poke_bot" / "recursive_turn_planner" / "r197_action_authority.py",
)
assert _SPEC is not None and _SPEC.loader is not None
authority = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(authority)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_digest(value: Any) -> str:
    return _digest(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    )


def _immutable_bytes(tmp_path: Path, name: str, value: bytes) -> dict[str, Any]:
    path = tmp_path / name
    path.write_bytes(value)
    os.chmod(path, 0o444)
    return {
        "path": str(path),
        "sha256": _digest(value),
        "bytes": len(value),
        "mode": 0o444,
    }


def _immutable_json(tmp_path: Path, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _immutable_bytes(
        tmp_path,
        name,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def _runtime_profile() -> dict[str, Any]:
    return {
        "sizing_profile": "pure_rl_r197",
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "serving_eligible": False,
        "action_authority_enabled": False,
    }


def _r197_config() -> dict[str, Any]:
    return {
        "schema": "poke_bot.recursive_turn_planner/v1",
        "sizing_profile": "pure_rl_r197",
        "d_model": 96,
        "dynamics_width": 192,
        "num_plan_candidates": 4,
        "max_recursion_depth": 2,
        "max_neural_passes": 256,
        "max_plan_length": 12,
        "complexity_option_threshold": 8,
        "complexity_entropy_threshold": 1.5,
        "skip_trivial_decisions": True,
        "online_sim_verify_budget": 0,
        "repair_budget": 1,
        "compute_cost_penalty": 0.01,
        "option_batch_hint": 64,
        "prefer_option_hidden": True,
        "policy_aid_cap": 0.25,
        "default_subgoals": (
            "establish_attacker",
            "find_resource",
            "maximize_draw",
            "preserve_escape",
            "reach_damage_threshold",
            "setup_next_turn",
        ),
    }


def _valid_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, str], Path]:
    """Create a sealed local analogue while retaining the exact r198 semantics."""

    parent = _immutable_bytes(tmp_path, "parent.pt", b"r198-parent")
    sidecar = _immutable_bytes(tmp_path, "sidecar.pt", b"r198-sidecar")
    # For a unit test of topology/process binding, replace the immutable
    # candidate hashes consistently.  Production never patches these module
    # constants; its source snapshot supplies the fixed published identities.
    monkeypatch.setattr(authority, "R198_PARENT_CHECKPOINT_SHA256", parent["sha256"])
    monkeypatch.setattr(authority, "R198_SIDECAR_SHA256", sidecar["sha256"])

    runtime = _immutable_json(
        tmp_path,
        "candidate-runtime.json",
        {
            "schema": authority.CANDIDATE_RUNTIME_CONTRACT_SCHEMA,
            "status": "sealed",
            "no_symlinks": True,
            "all_paths_read_only": True,
            "candidate_id": authority.R198_CANDIDATE_ID,
            "candidate_contract_sha256": authority.R198_CANDIDATE_CONTRACT_SHA256,
            "artifacts": {"parent_checkpoint": parent, "sidecar": sidecar},
        },
    )
    cell = {
        "cell_id": "cell-000000",
        "evaluation_case_id": "case-000000",
        "opponent_id": "iono",
        "candidate_seat": 0,
    }
    manifest = _immutable_json(
        tmp_path,
        "manifest.json",
        {
            "schema": authority.EVALUATION_MANIFEST_SCHEMA,
            "shared_artifacts": {"parent_checkpoint": parent},
            "candidate_evaluation_binding": {
                "candidate_contract_sha256": authority.R198_CANDIDATE_CONTRACT_SHA256,
                "parent_checkpoint_sha256": parent["sha256"],
                "sidecar_sha256": sidecar["sha256"],
                "sizing_profile": authority.R198_PROFILE,
                "max_neural_passes": authority.R198_MAX_NEURAL_PASSES,
                "max_action_combos": authority.R198_MAX_ACTION_COMBOS,
            },
            "arms": {
                authority.R198_DIRECT_ARM: {
                    "rtp_sidecar": sidecar,
                    "profile": _runtime_profile(),
                },
                authority.R198_RECURSIVE_ARM: {
                    "rtp_sidecar": sidecar,
                    "profile": _runtime_profile(),
                },
            },
            "schedule": [cell],
        },
    )
    authority_file = _immutable_json(
        tmp_path,
        "authority.json",
        {
            "schema": authority.EVALUATION_AUTHORITY_SCHEMA,
            "status": "authorized_evaluation_only",
            "manifest_sha256": manifest["sha256"],
            "evaluation_only": True,
            "training_eligible": False,
            "replay_eligible": False,
            "serving_change_authorized": False,
            "selector_change_authorized": False,
            "action_authority_authorized": False,
            "kaggle_submission_authorized": False,
        },
    )
    nonce = "a" * 48
    fence_payload = {
        "schema": authority.EVALUATION_ACTION_FENCE_SCHEMA,
        "status": "authorized_evaluation_only",
        "manifest_sha256": manifest["sha256"],
        "evaluation_authority": authority_file,
        **cell,
        "arm": authority.R198_RECURSIVE_ARM,
        "launch_nonce": nonce,
        "candidate_parent_checkpoint_sha256": parent["sha256"],
        "action_attached_rtp_sidecar_sha256": sidecar["sha256"],
        "complexity_probe_sidecar_sha256": sidecar["sha256"],
        "rtp_action_attachment_enabled": True,
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
    }
    fence_payload["runtime_contract_sha256"] = _canonical_digest(fence_payload)
    fence = _immutable_json(tmp_path, "fence.json", fence_payload)
    context = {
        "schema": authority.EVALUATION_ACTION_EXECUTION_SCHEMA,
        "status": "authorized_evaluation_only",
        "execution_kind": "evaluation_action_execution",
        "manifest": manifest,
        "evaluation_authority": authority_file,
        "runtime_contract": runtime,
        "action_fence": fence,
        **cell,
        "arm": authority.R198_RECURSIVE_ARM,
        "launch_nonce": nonce,
        "process": authority._current_process_identity(),
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "action_authority_enabled": False,
        "submission_eligible": False,
        "promotion_eligible": False,
        "kaggle_submission_authorized": False,
    }
    environment = {
        authority.RUNTIME_CONTRACT_ENV: runtime["path"],
        authority.RUNTIME_CONTRACT_SHA256_ENV: runtime["sha256"],
        authority.ACTION_FENCE_ENV: fence["path"],
        authority.ACTION_FENCE_SHA256_ENV: fence["sha256"],
        authority.LAUNCH_NONCE_ENV: nonce,
        authority.PROCESS_ID_ENV: context["process"]["process_id"],
        authority.PROCESS_START_TICKS_ENV: context["process"]["process_start_ticks"],
    }
    return context, environment, Path(sidecar["path"])


def _validate(context: dict[str, Any], environment: dict[str, str], sidecar: Path) -> dict[str, Any]:
    return authority.validate_evaluation_action_execution(
        context,
        config=_r197_config(),
        max_action_combos=1024,
        expected_parent_digest=authority.R198_PARENT_CHECKPOINT_SHA256,
        checkpoint_path=sidecar,
        environment=environment,
    )


@pytest.mark.unit
def test_r197_evaluation_action_execution_accepts_exact_sealed_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, environment, sidecar = _valid_context(tmp_path, monkeypatch)
    validated = _validate(context, environment, sidecar)
    assert validated["execution_kind"] == "evaluation_action_execution"
    assert validated["arm"] == authority.R198_RECURSIVE_ARM


@pytest.mark.unit
def test_r197_runner_generated_context_matches_consumer_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent-generated B/C map is accepted without an ambient opt-in."""

    from poke_bot import rtp_three_arm_evaluation_runner as runner

    context, environment, sidecar = _valid_context(tmp_path, monkeypatch)
    generated = runner._evaluation_action_execution_context(
        manifest_identity=context["manifest"],
        authority={"identity": context["evaluation_authority"]},
        runtime_contract=context["runtime_contract"],
        action_fence={"identity": context["action_fence"]},
        cell={
            "cell_id": context["cell_id"],
            "evaluation_case_id": context["evaluation_case_id"],
            "opponent_id": context["opponent_id"],
            "candidate_seat": context["candidate_seat"],
        },
        arm=context["arm"],
        launch_nonce=context["launch_nonce"],
    )

    assert runner.WORKER_REQUEST_SCHEMA == (
        "poke_bot.recursive_turn_planner.three_arm_evaluation_worker_request/v1"
    )
    assert runner.EXECUTION_RECEIPT_SCHEMA == (
        "poke_bot.recursive_turn_planner.three_arm_execution_receipt/v1"
    )
    assert runner.EVALUATION_ACTION_EXECUTION_SCHEMA == authority.EVALUATION_ACTION_EXECUTION_SCHEMA
    assert generated["process"] == authority._current_process_identity()
    assert _validate(generated, environment, sidecar)["arm"] == authority.R198_RECURSIVE_ARM


@pytest.mark.unit
def test_r197_identity_parser_uses_the_exact_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later pathname mutation cannot alter a payload already verified."""

    context, _environment, _sidecar = _valid_context(tmp_path, monkeypatch)
    identity = authority._immutable_identity(context["manifest"], "test manifest")
    manifest_path = Path(identity["path"])
    os.chmod(manifest_path, 0o644)
    manifest_path.write_text('{"schema":"forged"}\n', encoding="utf-8")
    assert authority._json_payload(identity, "test manifest")["schema"] == (
        authority.EVALUATION_MANIFEST_SCHEMA
    )


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["symlink", "writable"])
def test_r197_evaluation_action_execution_rejects_nonphysical_or_writable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    context, environment, sidecar = _valid_context(tmp_path, monkeypatch)
    if mutation == "symlink":
        link = tmp_path / "fence-link.json"
        link.symlink_to(context["action_fence"]["path"])
        context["action_fence"]["path"] = str(link)
        environment[authority.ACTION_FENCE_ENV] = str(link)
    elif mutation == "writable":
        os.chmod(context["runtime_contract"]["path"], 0o644)
    else:  # pragma: no cover - the parametrization is exhaustive.
        raise AssertionError(mutation)
    with pytest.raises(authority.R197ActionAuthorityError):
        _validate(context, environment, sidecar)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation", ["missing", "arm", "cell", "pid", "pid_env", "nonce", "digest"]
)
def test_r197_evaluation_action_execution_rejects_mismatched_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    context, environment, sidecar = _valid_context(tmp_path, monkeypatch)
    copied = json.loads(json.dumps(context))
    if mutation == "missing":
        copied.pop("action_fence")
    elif mutation == "arm":
        copied["arm"] = "no_rtp"
    elif mutation == "cell":
        copied["cell_id"] = "cell-other"
    elif mutation == "pid":
        copied["process"]["process_id"] = "0"
    elif mutation == "pid_env":
        environment = dict(environment, **{authority.PROCESS_ID_ENV: "0"})
    elif mutation == "nonce":
        environment = dict(environment, **{authority.LAUNCH_NONCE_ENV: "b" * 48})
    elif mutation == "digest":
        copied["action_fence"]["sha256"] = "sha256:" + "0" * 64
    else:  # pragma: no cover - the parametrization is exhaustive.
        raise AssertionError(mutation)
    with pytest.raises(authority.R197ActionAuthorityError):
        _validate(copied, environment, sidecar)


@pytest.mark.unit
def test_r197_generic_direct_invocation_has_no_action_exception() -> None:
    with pytest.raises(authority.R197ActionAuthorityError, match="evaluation_action_execution"):
        authority.assert_r197_action_selection_authorized(
            serving_qualified=False,
            serving_promotion_validated=False,
            evaluation_action_execution=None,
            config=_r197_config(),
            max_action_combos=1024,
            expected_parent_digest=authority.R198_PARENT_CHECKPOINT_SHA256,
            checkpoint_path=None,
        )


@pytest.mark.unit
def test_r197_accepted_serving_promotion_remains_an_independent_path() -> None:
    accepted = authority.assert_r197_action_selection_authorized(
        serving_qualified=True,
        serving_promotion_validated=True,
        evaluation_action_execution=None,
        config=None,
        max_action_combos=None,
        expected_parent_digest=None,
        checkpoint_path=None,
    )
    assert accepted == {"mode": "serving_qualified_promotion"}


@pytest.mark.unit
def test_policy_bridge_rejects_generic_r197_selection_context() -> None:
    """The PolicyAgent forwarding path cannot turn a generic map into a bypass."""

    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from poke_bot.agent import PolicyAgent
    from poke_bot.recursive_turn_planner.r197_action_authority import (
        R197ActionAuthorityError,
    )

    model = SimpleNamespace(
        d_model=96,
        latent_lookahead=None,
        latent_lookahead_enabled=False,
        eval=lambda: None,
        parameters=lambda: iter((torch.zeros(1),)),
    )
    agent = PolicyAgent(
        model=model,  # type: ignore[arg-type]
        deck=[1] * 60,
        use_recursive_turn_planner=True,
        rtp_sizing_profile="pure_rl_r197",
        matchup_adapter_shadow=False,
        device=torch.device("cpu"),
    )
    assert agent._rtp_bridge is not None
    with pytest.raises(R197ActionAuthorityError, match="evaluation_action_execution"):
        agent._rtp_bridge._require_r197_action_selection_authority()

    with pytest.raises(ValueError, match="evaluation_action_execution requires"):
        PolicyAgent(
            model=model,  # type: ignore[arg-type]
            deck=[1] * 60,
            use_recursive_turn_planner=False,
            rtp_evaluation_action_execution={},
            matchup_adapter_shadow=False,
            device=torch.device("cpu"),
        )
