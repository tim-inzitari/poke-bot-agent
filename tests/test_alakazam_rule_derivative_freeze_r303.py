"""Focused checks for the non-activating rev5/r303 freeze producer."""

from __future__ import annotations

import copy
import json

import pytest

import poke_bot.alakazam_rule_derivative_freeze_r303 as freeze
from poke_bot.alakazam_checklist_provenance_r298 import (
    ChecklistProvenanceError,
    revision_5_consumer_migration_schema_manifest_r298,
    validate_revision_5_consumer_migration_r298,
)


def _migration_assertions() -> dict[str, bool]:
    definition = revision_5_consumer_migration_schema_manifest_r298()["definition"]
    return {name: True for name in definition["required_assertions"]}


def test_r303_migration_payload_matches_the_checklist_owners_closed_schema() -> None:
    payload = freeze.build_revision_5_consumer_migration_payload(
        consumer_rebind_assertions=_migration_assertions(),
        validated_at_utc="2026-08-12T20:00:00Z",
    )
    definition = revision_5_consumer_migration_schema_manifest_r298()["definition"]

    assert set(payload) == set(definition["required_receipt_fields"])
    assert payload["goal_contract_sha256"] == freeze.R303_CONTRACT_SHA256
    assert payload["goal_revision"] == 5
    assert payload["root_handoff_revision"] == 303
    assert payload["blind_goal_or_contract_hash_substitution_allowed"] is False
    assert payload["revision_4_catalog_receipt_alone_satisfies_revision_5_schema_freeze"] is False
    assert payload["default_zero_and_inert"] is True
    assert payload["runtime_wired"] is False
    assert payload["production_or_inzi_authority"] is False
    assert freeze.validate_revision_5_consumer_migration_payload_for_issue(payload) == payload


def test_r303_migration_payload_rejects_missing_or_false_assertions() -> None:
    assertions = _migration_assertions()
    assertions.pop(next(iter(assertions)))
    with pytest.raises(freeze.R303FreezeError, match="assertion inventory"):
        freeze.build_revision_5_consumer_migration_payload(
            consumer_rebind_assertions=assertions,
            validated_at_utc="2026-08-12T20:00:00Z",
        )

    assertions = _migration_assertions()
    assertions[next(iter(assertions))] = False
    with pytest.raises(freeze.R303FreezeError, match="assertion is not passed"):
        freeze.build_revision_5_consumer_migration_payload(
            consumer_rebind_assertions=assertions,
            validated_at_utc="2026-08-12T20:00:00Z",
        )


def test_r303_migration_payload_rejects_stale_authority() -> None:
    payload = freeze.build_revision_5_consumer_migration_payload(
        consumer_rebind_assertions=_migration_assertions(),
        validated_at_utc="2026-08-12T20:00:00Z",
    )
    stale = copy.deepcopy(payload)
    stale["goal_contract_sha256"] = "sha256:" + "0" * 64

    with pytest.raises(freeze.R303FreezeError, match="stale or foreign"):
        freeze.validate_revision_5_consumer_migration_payload_for_issue(stale)


def test_r303_freeze_consumes_live_rev5_adapter_target_and_auxiliary_bindings() -> None:
    adapter = freeze._adapter_config_binding()
    target = freeze._target_and_auxiliary_binding()

    assert adapter["runtime_wired"] is False
    assert adapter["all_gates_exact_zero"] is True
    assert adapter["config"]["sha256"] == (
        "sha256:5a4af2db79be6e2cd976a450b1694a3d53989e272c3906754ade49a31e5860b2"
    )
    assert target["all_trainable_selected_action_masks_false_before_handoff"] is True
    assert target["opponent_belief_masked_unavailable"] is True
    assert target["policy_feature_eligible"] is False
    assert target["runtime_wired"] is False


def test_r303_freeze_does_not_let_an_unpinned_checklist_migration_anchor_pass(tmp_path) -> None:
    # The raw catalog is deliberately irrelevant to this assertion: the
    # producer reaches the owner-pinned checklist boundary only after its
    # typed catalog binding validates.  Directly exercising the checklist
    # helper is the stable proof that a local lookalike migration cannot arm
    # the public-rule freeze while its config says not_issued_fail_closed.
    payload = freeze.build_revision_5_consumer_migration_payload(
        consumer_rebind_assertions=_migration_assertions(),
        validated_at_utc="2026-08-12T20:00:00Z",
    )
    assert payload["schema"] == (
        "poke_bot.alakazam_checklist_provenance_r298_consumer_migration/v1"
    )
    # The owner validator deliberately requires a *file* whose exact digest
    # was subsequently pinned in the checklist config; the builder cannot
    # fabricate that authority.  This remains true after a real anchor exists
    # because this lookalike file has different bytes/path identity.
    proposal = tmp_path / "lookalike-migration.json"
    proposal.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ChecklistProvenanceError):
        validate_revision_5_consumer_migration_r298(proposal)
