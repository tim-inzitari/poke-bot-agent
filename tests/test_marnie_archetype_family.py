from __future__ import annotations

import copy
import json

import pytest
import torch

from poke_bot.archetype_family_activation import (
    FamilyActivationError,
    boundary_decision,
    validate_atomic_migration,
)
from poke_bot.archetype_loss_contract import (
    ArchetypeLossContractError,
    MaskedObjective,
    guide_gradient_allowed,
    macro_list_loss,
    validate_loss_contract,
)
from poke_bot.specialist_archetype_family import (
    ArchetypeFamilyError,
    canonical_counts,
    cluster_variants,
    digest_json,
    family_probabilities,
    hamilton_quotas,
    multiset_digest,
    ordered_digest,
    schedule_variants,
    split_clusters,
    swap_distance,
    validate_manifest,
)


def _cards(offset: int = 0) -> list[int]:
    # Marnie line plus unrelated positive card IDs. Three replacements create
    # a distinct >2-swap cluster while preserving classification.
    return [646, 647, 648] + [1000 + index + offset * 100 for index in range(57)]


def _manifest(cluster_count: int = 13) -> dict:
    rows = []
    for index in range(cluster_count):
        cards = _cards(index)
        rows.append(
            {
                "family_id": "marnie-s-grimmsnarl-ex",
                "variant_id": f"v{index}",
                "card_ids": cards,
                "card_counts": [[card, count] for card, count in canonical_counts(cards)],
                "ordered_digest": ordered_digest(cards),
                "multiset_digest": multiset_digest(cards),
                "provenance": {"source": f"episode-{index}"},
                "legality": {"legal": True},
                "classification": {"archetype_id": "marnie-s-grimmsnarl-ex"},
                "cluster_id": f"c{index}",
                "split": "train" if index < 7 else ("dev" if index < 10 else "locked"),
                "training_weight": 0.0,
                "capability_mask": {"core_setup_continuity": True},
                "package": index == 0,
                "measurement": index == 0,
            }
        )
    payload = {
        "schema": "poke_bot.specialist_archetype_families/v1",
        "family_id": "marnie-s-grimmsnarl-ex",
        "variants": rows,
    }
    payload["artifact_sha256"] = digest_json(payload)
    return payload


def test_manifest_digests_dedup_clusters_and_split_leakage() -> None:
    payload = _manifest()
    assert validate_manifest(payload, require_activation_ready=True)
    assert swap_distance(payload["variants"][0]["card_ids"], payload["variants"][1]["card_ids"]) == 57
    assert len(set(cluster_variants(payload["variants"]).values())) == 13
    splits = split_clusters([f"c{i}" for i in range(13)], package_cluster_id="c0", seed="fixed")
    assert splits["c0"] == "train"
    assert set(splits.values()) == {"train", "dev", "locked"}

    duplicate = copy.deepcopy(payload)
    duplicate["variants"][1]["card_ids"] = list(duplicate["variants"][0]["card_ids"])
    duplicate["variants"][1]["card_counts"] = copy.deepcopy(duplicate["variants"][0]["card_counts"])
    duplicate["variants"][1]["ordered_digest"] = duplicate["variants"][0]["ordered_digest"]
    duplicate["variants"][1]["multiset_digest"] = duplicate["variants"][0]["multiset_digest"]
    duplicate.pop("artifact_sha256")
    duplicate["artifact_sha256"] = digest_json(duplicate)
    with pytest.raises(ArchetypeFamilyError, match="duplicate canonical"):
        validate_manifest(duplicate)


def test_deterministic_hamilton_seats_derangement_and_package_cap() -> None:
    payload = _manifest()
    probs = family_probabilities(payload)
    assert probs["v0"] == pytest.approx(0.20)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert sum(hamilton_quotas(1024, probs).values()) == 1024
    first = schedule_variants(payload, games=1024, checksum_seed="abc")
    second = schedule_variants(payload, games=1024, checksum_seed="abc")
    assert first == second
    assert all(row["variant_id"] != row["opponent_variant_id"] for row in first)
    package_count = sum(row["variant_id"] == "v0" for row in first)
    assert package_count in {204, 205}
    for variant in probs:
        seats = [row["seat"] for row in first if row["variant_id"] == variant]
        assert abs(seats.count(0) - seats.count(1)) <= 1


def test_loss_contract_masking_macro_average_and_guide_authority() -> None:
    contract = json.loads(
        open("config/archetype_loss_contracts/marnie-s-grimmsnarl-ex.v1.json", encoding="utf-8").read()
    )
    validate_loss_contract(contract)
    assert guide_gradient_allowed(contract, "action_q")
    assert not guide_gradient_allowed(contract, "available_but_unauthorized")
    bad = copy.deepcopy(contract)
    bad["residual_objectives"]["core_setup_continuity"]["weight"] = 0.051
    with pytest.raises(ArchetypeLossContractError):
        validate_loss_contract(bad)

    values = torch.tensor([1.0, 1.0, 3.0], requires_grad=True)
    loss = macro_list_loss(
        [MaskedObjective("x", values, torch.ones(3, dtype=torch.bool), 1.0, "cap")],
        variant_ids=["common", "common", "rare"],
        family_applicable=torch.ones(3, dtype=torch.bool),
        capabilities={"cap": torch.ones(3, dtype=torch.bool)},
    )
    assert loss.item() == pytest.approx(2.0)  # equal-list macro, not row mean
    loss.backward()
    assert torch.isfinite(values.grad).all()

    masked = torch.tensor([2.0], requires_grad=True)
    zero = macro_list_loss(
        [MaskedObjective("x", masked, torch.zeros(1, dtype=torch.bool), 1.0, "cap")],
        variant_ids=["one"],
        family_applicable=torch.ones(1, dtype=torch.bool),
        capabilities={"cap": torch.ones(1, dtype=torch.bool)},
    )
    zero.backward()
    assert zero.item() == 0.0 and masked.grad.item() == 0.0


def test_boundary_defers_started_collection_and_atomic_allowlist() -> None:
    assert boundary_decision(trigger_valid=False, study_passed=True, committed_iteration=9, next_collection_started=False, already_paused_for_commit=False)["action"] == "continue_unchanged"
    assert boundary_decision(trigger_valid=True, study_passed=True, committed_iteration=9, next_collection_started=True, already_paused_for_commit=False) == {"action": "defer", "target_iteration": 11}
    assert boundary_decision(trigger_valid=True, study_passed=True, committed_iteration=10, next_collection_started=False, already_paused_for_commit=False)["action"] == "pause_for_atomic_activation"
    validate_atomic_migration({}, {"family_manifest": "m", "selected_loss_vector": "l"})
    with pytest.raises(FamilyActivationError):
        validate_atomic_migration({}, {"family_manifest": "m", "selected_loss_vector": "l", "router_format": 7})
